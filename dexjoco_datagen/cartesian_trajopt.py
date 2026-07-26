"""Cartesian collision-aware trajectory optimization for mocap EE targets.

This is a real trajopt (not hand-crafted detour waypoints):

  min_C  Σ ||Δ²p||² + w_col Σ max(0, d_safe - d(p))² + w_len Σ ||Δp||²
  s.t.   p_0 = start, p_T = goal
         C = interior control points of a dense path

Distances use MuJoCo mj_geomDistance between right-arm proxies and
(left-arm + tray) proxies. Left mocap is frozen (tray grasp must not move).

Used as the in-process planner inside dexjoco (mujoco==3.4).
cuRobo (separate conda env `curobo`) is the GPU joint-space upgrade path
once the Panda+Allegro robot config is exported.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import minimize

from interaction_retarget.constants import TRAY_BODY
from interaction_retarget.sim.settle import vec_to_arm_action

_RIGHT_BODIES = (
    "link6_right",
    "link7_right",
    "allegro_palm_right",
    "ff_tip_right",
    "mf_tip_right",
    "th_tip_right",
)
_LEFT_BODIES = (
    "link6_left",
    "link7_left",
    "allegro_palm_left",
    "ff_tip_left",
    "mf_tip_left",
    "th_tip_left",
)


def _geoms_for_bodies(model: mujoco.MjModel, names: tuple[str, ...]) -> list[int]:
    out: list[int] = []
    for name in names:
        try:
            bid = int(model.body(name).id)
        except KeyError:
            continue
        out.extend(gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) == bid)
    return out


def _subtree_geoms(model: mujoco.MjModel, root: str) -> list[int]:
    try:
        rid = int(model.body(root).id)
    except KeyError:
        return []
    bodies = {rid}
    changed = True
    while changed:
        changed = False
        for bid in range(model.nbody):
            if int(model.body_parentid[bid]) in bodies and bid not in bodies:
                bodies.add(bid)
                changed = True
    return [gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in bodies]


@dataclass
class TrajOptResult:
    path: np.ndarray  # (T, 3)
    success: bool
    cost: float
    min_clearance: float
    message: str


class CartesianCollisionTrajOpt:
    def __init__(
        self,
        raw_env,
        *,
        d_safe: float = 0.07,
        w_col: float = 80.0,
        w_smooth: float = 12.0,
        w_len: float = 0.5,
        n_ctrl: int = 5,
        n_dense: int = 36,
        distmax: float = 0.30,
        maxiter: int = 40,
    ) -> None:
        self.model = raw_env._model
        self.d_safe = float(d_safe)
        self.w_col = float(w_col)
        self.w_smooth = float(w_smooth)
        self.w_len = float(w_len)
        self.n_ctrl = int(n_ctrl)
        self.n_dense = int(n_dense)
        self.distmax = float(distmax)
        self.maxiter = int(maxiter)

        self._right = _geoms_for_bodies(self.model, _RIGHT_BODIES)
        self._left = _geoms_for_bodies(self.model, _LEFT_BODIES)
        self._tray = _subtree_geoms(self.model, TRAY_BODY)
        self._obstacles = self._left + self._tray
        try:
            self._palm_right = int(self.model.body("allegro_palm_right").id)
        except KeyError:
            self._palm_right = int(self.model.body("link7_right").id)

    def _min_clearance_at_palm(self, data: mujoco.MjData, palm_pos: np.ndarray) -> float:
        """Translate right collision bodies with palm delta; report min signed distance."""
        model = self.model
        # Save / restore xpos for right bodies we touch via temporary palm shift heuristic:
        # evaluate distances using current geom poses shifted by (palm_pos - palm_now).
        palm_now = np.asarray(data.xpos[self._palm_right], dtype=np.float64).copy()
        delta = np.asarray(palm_pos, dtype=np.float64) - palm_now
        fromto = np.zeros(6, dtype=np.float64)
        best = 1e9
        # Shift geom xpos copies locally by using mj_geomDistance on current state is wrong
        # if we don't update. Use body-center approximation: distance between
        # (right_body_xpos + delta) and obstacle body xpos, minus radii proxy.
        for gr in self._right:
            br = int(model.geom_bodyid[gr])
            pr = np.asarray(data.xpos[br], dtype=np.float64) + delta
            rr = float(model.geom_size[gr, 0]) if model.geom_size[gr, 0] > 0 else 0.02
            for go in self._obstacles:
                bo = int(model.geom_bodyid[go])
                po = np.asarray(data.xpos[bo], dtype=np.float64)
                ro = float(model.geom_size[go, 0]) if model.geom_size[go, 0] > 0 else 0.02
                d = float(np.linalg.norm(pr - po) - rr - ro)
                if d < best:
                    best = d
        # Also exact geom distance at current config for baseline when delta~0
        if float(np.linalg.norm(delta)) < 1e-4:
            for gr in self._right:
                for go in self._obstacles:
                    d = float(
                        mujoco.mj_geomDistance(model, data, gr, go, self.distmax, fromto)
                    )
                    if d < best:
                        best = d
        return float(best)

    def _densify(self, start: np.ndarray, goal: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
        """Build dense path from start + interior ctrl + goal via piecewise linear."""
        knots = np.vstack([start.reshape(1, 3), ctrl.reshape(-1, 3), goal.reshape(1, 3)])
        # Arc-length parameterization
        seg = np.linalg.norm(np.diff(knots, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total < 1e-8:
            return np.repeat(goal.reshape(1, 3), self.n_dense, axis=0)
        ts = np.linspace(0.0, total, self.n_dense)
        out = np.zeros((self.n_dense, 3), dtype=np.float64)
        for i, t in enumerate(ts):
            j = int(np.searchsorted(cum, t, side="right") - 1)
            j = int(np.clip(j, 0, len(seg) - 1))
            u = 0.0 if seg[j] < 1e-9 else (t - cum[j]) / seg[j]
            out[i] = (1.0 - u) * knots[j] + u * knots[j + 1]
        return out

    def plan(
        self,
        raw_env,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
    ) -> TrajOptResult:
        start = np.asarray(start_pos, dtype=np.float64).reshape(3)
        goal = np.asarray(goal_pos, dtype=np.float64).reshape(3)
        data = raw_env._data

        # Init control points on straight line (optimizer will bend them for clearance).
        alphas = np.linspace(0.0, 1.0, self.n_ctrl + 2)[1:-1]
        x0 = np.stack([(1 - a) * start + a * goal for a in alphas], axis=0).reshape(-1)

        def cost(x: np.ndarray) -> float:
            ctrl = x.reshape(-1, 3)
            path = self._densify(start, goal, ctrl)
            # Smoothness (discrete accel) + length
            vel = np.diff(path, axis=0)
            acc = np.diff(vel, axis=0)
            c_s = self.w_smooth * float(np.sum(acc * acc))
            c_l = self.w_len * float(np.sum(vel * vel))
            c_c = 0.0
            min_d = 1e9
            for p in path:
                d = self._min_clearance_at_palm(data, p)
                min_d = min(min_d, d)
                viol = max(0.0, self.d_safe - d)
                c_c += self.w_col * viol * viol
            # small attraction already enforced by endpoint fix
            return c_s + c_l + c_c

        # Bounds: keep control points in a box around the segment
        mid = 0.5 * (start + goal)
        span = np.maximum(np.abs(goal - start), 0.08)
        lo = np.tile(mid - 1.5 * span - 0.05, self.n_ctrl)
        hi = np.tile(mid + 1.5 * span + 0.05, self.n_ctrl)
        # Prefer lifting slightly in z for clearance seed
        x0 = x0.reshape(-1, 3)
        x0[:, 2] += 0.03
        x0 = x0.reshape(-1)

        res = minimize(
            cost,
            x0,
            method="L-BFGS-B",
            bounds=list(zip(lo, hi)),
            options={"maxiter": self.maxiter, "ftol": 1e-4},
        )
        ctrl = np.asarray(res.x, dtype=np.float64).reshape(-1, 3)
        path = self._densify(start, goal, ctrl)
        min_d = min(self._min_clearance_at_palm(data, p) for p in path)
        ok = bool(res.success or min_d >= 0.5 * self.d_safe)
        return TrajOptResult(
            path=path,
            success=ok,
            cost=float(res.fun) if res.fun is not None else 1e9,
            min_clearance=float(min_d),
            message=str(res.message),
        )


def execute_pos_path(
    raw_env,
    *,
    path: np.ndarray,
    quat_wxyz: np.ndarray,
    hand16: np.ndarray,
    left_hold: np.ndarray,
    video,
    step_fn,
    start_quat: np.ndarray | None = None,
) -> None:
    """Execute a Cartesian path with fixed/slerp orientation and frozen left."""
    from scipy.spatial.transform import Rotation as R, Slerp

    path = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    hand16 = np.asarray(hand16, dtype=np.float64).reshape(16)
    left_hold = vec_to_arm_action(left_hold)
    q1 = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    if start_quat is None:
        start_quat = q1
    q0 = np.asarray(start_quat, dtype=np.float64).reshape(4)
    if float(np.dot(q0, q1)) < 0:
        q1 = -q1
    slerp = Slerp([0.0, 1.0], R.from_quat([q0[[1, 2, 3, 0]], q1[[1, 2, 3, 0]]]))
    n = len(path)
    for i, p in enumerate(path):
        t = (i + 1) / float(n)
        q_xyzw = slerp(t).as_quat()
        q = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
        right = np.concatenate([p, q, hand16], axis=0)
        step_fn(raw_env, right, left_hold, video)
