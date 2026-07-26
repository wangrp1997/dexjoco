"""Dual-arm collision avoidance via discrete CBF-QP (OSQP).

Not waypoint detours. At each control step we solve:

  min  ||p_r - p_r*||^2 + w_l ||p_l - p_l*||^2
  s.t. n_k^T (Δp_r - Δp_l) >= -γ h_k   for active geom pairs
       |Δp| <= δ_max

where h = signed_geom_distance - d_safe, n from MuJoCo mj_geomDistance fromto.
When the left hand is securing the tray, w_l is large / left frozen so the
right arm absorbs the avoidance (tray must not be yanked).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import osqp
import scipy.sparse as sp

from interaction_retarget.constants import LEFT_HAND_ROOT, RIGHT_HAND_ROOT, TRAY_BODY
from interaction_retarget.sim.settle import vec_to_arm_action

# Collision proxies: distal arm links + palms (enough for dual-arm clearance).
_RIGHT_COLLIDE_BODIES = (
    "link5_right",
    "link6_right",
    "link7_right",
    "allegro_palm_right",
    "ff_tip_right",
    "mf_tip_right",
    "rf_tip_right",
    "th_tip_right",
)
_LEFT_COLLIDE_BODIES = (
    "link5_left",
    "link6_left",
    "link7_left",
    "allegro_palm_left",
    "ff_tip_left",
    "mf_tip_left",
    "rf_tip_left",
    "th_tip_left",
)


def _body_geom_ids(model: mujoco.MjModel, body_name: str) -> list[int]:
    try:
        bid = int(model.body(body_name).id)
    except KeyError:
        return []
    return [gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) == bid]


def _subtree_geom_ids(model: mujoco.MjModel, root_name: str) -> list[int]:
    try:
        root = int(model.body(root_name).id)
    except KeyError:
        return []
    bodies = {root}
    changed = True
    while changed:
        changed = False
        for bid in range(model.nbody):
            if int(model.body_parentid[bid]) in bodies and bid not in bodies:
                bodies.add(bid)
                changed = True
    return [gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in bodies]


@dataclass
class CBFQPDebug:
    n_constraints: int = 0
    min_h: float = 1e9
    left_frozen: bool = False
    status: str = "ok"


class DualArmCBFQP:
    """Online dual-arm CBF-QP filter for mocap pos targets."""

    def __init__(
        self,
        raw_env,
        *,
        d_safe: float = 0.06,
        activate_margin: float = 0.10,
        gamma: float = 0.6,
        w_left_free: float = 4.0,
        w_left_holding: float = 400.0,
        w_right: float = 1.0,
        delta_max: float = 0.04,
        distmax: float = 0.25,
        max_pairs: int = 24,
        polish: bool = False,
    ) -> None:
        self.model = raw_env._model
        self.d_safe = float(d_safe)
        self.activate_margin = float(activate_margin)
        self.gamma = float(gamma)
        self.w_left_free = float(w_left_free)
        self.w_left_holding = float(w_left_holding)
        self.w_right = float(w_right)
        self.delta_max = float(delta_max)
        self.distmax = float(distmax)
        self.max_pairs = int(max_pairs)
        self.polish = bool(polish)

        self._right_geoms: list[int] = []
        for name in _RIGHT_COLLIDE_BODIES:
            self._right_geoms.extend(_body_geom_ids(self.model, name))
        self._left_geoms: list[int] = []
        for name in _LEFT_COLLIDE_BODIES:
            self._left_geoms.extend(_body_geom_ids(self.model, name))
        # Fallback: whole hands if named bodies missing.
        if not self._right_geoms:
            self._right_geoms = _subtree_geom_ids(self.model, RIGHT_HAND_ROOT)
        if not self._left_geoms:
            self._left_geoms = _subtree_geom_ids(self.model, LEFT_HAND_ROOT)

        self._tray_geoms = _subtree_geom_ids(self.model, TRAY_BODY)
        self._tray_body = int(self.model.body(TRAY_BODY).id)
        self._tray_rest_z: float | None = None
        self.last_debug = CBFQPDebug()

    def reset_reference(self, raw_env) -> None:
        self._tray_rest_z = float(raw_env._data.xpos[self._tray_body, 2])

    def left_holding_tray(self, raw_env) -> bool:
        data = raw_env._data
        if self._tray_rest_z is None:
            self.reset_reference(raw_env)
        tray_z = float(data.xpos[self._tray_body, 2])
        if tray_z > float(self._tray_rest_z) + 0.015:
            return True
        # Contact between left hand geoms and tray.
        left_set = set(self._left_geoms)
        tray_set = set(self._tray_geoms)
        for i in range(int(data.ncon)):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in left_set and g2 in tray_set) or (g2 in left_set and g1 in tray_set):
                return True
        return False

    def _active_constraints(self, raw_env) -> list[tuple[np.ndarray, float, str]]:
        """Return list of (n_world[3], h, kind) for active pairs."""
        model = self.model
        data = raw_env._data
        fromto = np.zeros(6, dtype=np.float64)
        out: list[tuple[np.ndarray, float, str]] = []

        # Right ↔ left
        for gr in self._right_geoms:
            for gl in self._left_geoms:
                d = float(
                    mujoco.mj_geomDistance(model, data, gr, gl, self.distmax, fromto)
                )
                h = d - self.d_safe
                if h >= self.activate_margin:
                    continue
                p1 = fromto[0:3].copy()
                p2 = fromto[3:6].copy()
                delta = p2 - p1
                nrm = float(np.linalg.norm(delta))
                if nrm < 1e-8:
                    # Degenerate: use body centers.
                    br = int(model.geom_bodyid[gr])
                    bl = int(model.geom_bodyid[gl])
                    delta = data.xpos[bl] - data.xpos[br]
                    nrm = float(np.linalg.norm(delta))
                    if nrm < 1e-8:
                        continue
                n = delta / nrm  # points right→left along closest segment (geom1=right)
                # h increases when right moves opposite to n or left along n.
                # Constraint uses n^T (Δp_r - Δp_l) — wait:
                # If geom1=right, geom2=left, fromto is right→left, n points to left.
                # Moving right along -n or left along +n increases separation.
                # ∂d/∂p_r ≈ -n, ∂d/∂p_l ≈ +n  →  -n·Δp_r + n·Δp_l = n·(Δp_l - Δp_r)
                # We store n_eff so n_eff·(Δp_r - Δp_l) >= -γ h  with n_eff = -n
                out.append((-n, h, "rl"))

        # Right ↔ tray (tray static)
        for gr in self._right_geoms:
            for gt in self._tray_geoms:
                d = float(
                    mujoco.mj_geomDistance(model, data, gr, gt, self.distmax, fromto)
                )
                h = d - self.d_safe
                if h >= self.activate_margin:
                    continue
                p1 = fromto[0:3].copy()
                p2 = fromto[3:6].copy()
                delta = p2 - p1
                nrm = float(np.linalg.norm(delta))
                if nrm < 1e-8:
                    continue
                n = delta / nrm  # right → tray
                # ∂d/∂p_r ≈ -n (tray fixed) → want -n·Δp_r >= -γ h → n_eff=-n
                out.append((-n, h, "rt"))

        # Keep the most violated / closest pairs.
        out.sort(key=lambda x: x[1])
        return out[: self.max_pairs]

    def filter(
        self,
        raw_env,
        right23: np.ndarray,
        left23: np.ndarray,
        *,
        freeze_left: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return CBF-safe mocap actions (pos filtered; quat/hand unchanged)."""
        right = vec_to_arm_action(right23).copy()
        left = vec_to_arm_action(left23).copy()
        holding = self.left_holding_tray(raw_env) if freeze_left is None else bool(freeze_left)

        # Current mocap / command positions as start of the step.
        p_r0 = read_mocap_or_cmd(raw_env, "right", right)
        p_l0 = read_mocap_or_cmd(raw_env, "left", left)
        p_r_star = right[0:3].copy()
        p_l_star = left[0:3].copy()

        cons = self._active_constraints(raw_env)
        dbg = CBFQPDebug(
            n_constraints=len(cons),
            min_h=float(min((c[1] for c in cons), default=1e9)),
            left_frozen=holding,
        )

        if not cons:
            self.last_debug = dbg
            return right, left

        # Decision vars: x = [p_r(3), p_l(3)]
        # min ||p_r - p_r*||^2 * w_r + ||p_l - p_l*||^2 * w_l
        w_r = self.w_right
        w_l = self.w_left_holding if holding else self.w_left_free
        P = sp.diags([w_r, w_r, w_r, w_l, w_l, w_l], format="csc")
        q = -np.concatenate([w_r * p_r_star, w_l * p_l_star])

        # Inequality: n_eff·(p_r - p_l - (p_r0 - p_l0)) >= -γ h
        # → n_eff·p_r - n_eff·p_l >= -γ h + n_eff·(p_r0 - p_l0)
        # For tray: n_eff·(p_r - p_r0) >= -γ h → n_eff·p_r >= -γ h + n_eff·p_r0
        a_rows: list[np.ndarray] = []
        l_b: list[float] = []
        u_b: list[float] = []
        for n_eff, h, kind in cons:
            row = np.zeros(6, dtype=np.float64)
            if kind == "rl":
                row[0:3] = n_eff
                row[3:6] = -n_eff
                rhs = -self.gamma * h + float(np.dot(n_eff, p_r0 - p_l0))
            else:  # right-tray
                row[0:3] = n_eff
                rhs = -self.gamma * h + float(np.dot(n_eff, p_r0))
            a_rows.append(row)
            l_b.append(rhs)
            u_b.append(np.inf)

        # Box: |p - p*| <= delta_max (stay near nominal) AND |p - p0| <= 2*delta_max
        for i in range(6):
            row = np.zeros(6, dtype=np.float64)
            row[i] = 1.0
            star = p_r_star[i] if i < 3 else p_l_star[i - 3]
            a_rows.append(row)
            l_b.append(star - self.delta_max)
            u_b.append(star + self.delta_max)

        if holding:
            # Hard-ish: left stays at nominal (tray grasp).
            for i in range(3):
                row = np.zeros(6, dtype=np.float64)
                row[3 + i] = 1.0
                a_rows.append(row)
                l_b.append(p_l_star[i] - 1e-4)
                u_b.append(p_l_star[i] + 1e-4)

        A = sp.csc_matrix(np.stack(a_rows, axis=0))
        l_arr = np.asarray(l_b, dtype=np.float64)
        u_arr = np.asarray(u_b, dtype=np.float64)

        solver = osqp.OSQP()
        try:
            solver.setup(
                P=P,
                q=q,
                A=A,
                l=l_arr,
                u=u_arr,
                verbose=False,
                polish=False,
                warm_start=False,
            )
            res = solver.solve()
        except Exception as ex:  # noqa: BLE001
            dbg.status = f"osqp_error:{ex}"
            self.last_debug = dbg
            return right, left

        if res.x is None or not np.all(np.isfinite(res.x)):
            dbg.status = f"osqp_fail:{res.info.status}"
            self.last_debug = dbg
            # Fallback: keep left nominal; nudge right away in XY only.
            mid = p_l0 - p_r0
            nxy = mid.copy()
            nxy[2] = 0.0
            nn = float(np.linalg.norm(nxy))
            if nn > 1e-6 and dbg.min_h < 0.0:
                right[0:3] = p_r_star - (nxy / nn) * min(self.delta_max, 0.015)
            left[0:3] = p_l_star
            return right, left

        right[0:3] = res.x[0:3]
        left[0:3] = p_l_star if holding else res.x[3:6]
        dbg.status = str(res.info.status)
        self.last_debug = dbg
        return right, left


def read_mocap_or_cmd(raw_env, side: str, cmd23: np.ndarray) -> np.ndarray:
    """Prefer live mocap target pos; fall back to command."""
    cmd23 = vec_to_arm_action(cmd23)
    try:
        if side == "right":
            mid = int(raw_env._mocap_right_id)
        else:
            mid = int(raw_env._mocap_left_id)
        return np.asarray(raw_env._data.mocap_pos[mid], dtype=np.float64).copy()
    except Exception:  # noqa: BLE001
        return cmd23[0:3].copy()
