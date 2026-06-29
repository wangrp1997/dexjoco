"""Run hybrid_insert after bimanual lift until env insert success."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from hybrid_insert.assembly_contacts import AssemblyContactLabeler
from hybrid_insert.config import HybridInsertConfig
from hybrid_insert.geometry import (
    hole_opening_axis,
    in_approach_cylinder,
    peg_insert_end_pos,
    tip_socket_distance,
    toward_socket_delta,
)
from hybrid_insert.integration import EvalHybridInsert

from interaction_retarget.constants import PEG_BODY
from interaction_retarget.grasp.approach import interpolate_arm_only
from interaction_retarget.grasp.repair import _step_side
from interaction_retarget.sim.replay import policy_dual_arm_to_raw, rotvec_dual_arm_to_policy
from interaction_retarget.sim.settle import read_arm_action, vec_to_arm_action

_SOCKET_SITE = "industreal_tray_insert_round_peg_8mm_socket_site"
_BOTTOM_GEOM = "industreal_tray_insert_round_peg_8mm_bottom_contact"


def dual_arm23_to_action44(left23: np.ndarray, right23: np.ndarray) -> np.ndarray:
    left23 = np.asarray(left23, dtype=np.float64).reshape(23)
    right23 = np.asarray(right23, dtype=np.float64).reshape(23)
    r_rot = R.from_quat(right23[3:7], scalar_first=True).as_rotvec()
    l_rot = R.from_quat(left23[3:7], scalar_first=True).as_rotvec()
    return np.concatenate(
        [right23[0:3], r_rot, right23[7:23], left23[0:3], l_rot, left23[7:23]],
        dtype=np.float64,
    )


def action44_to_raw_dict(action44: np.ndarray) -> dict[str, np.ndarray]:
    action46 = rotvec_dual_arm_to_policy(np.asarray(action44, dtype=np.float64).reshape(44))
    return policy_dual_arm_to_raw(action46)


@dataclass
class InsertReport:
    success: bool
    steps: int
    insert_ok: bool
    handoff: bool
    phase: str
    peg_lift_m: float
    fail_reason: str = ""


def _env_step_info(env, action) -> dict:
    out = env.step(action)
    return out[4] if len(out) == 5 else out[3]


def _peg_lift_m(raw_env, peg_rest_z: float) -> float:
    peg_id = int(raw_env._model.body(PEG_BODY).id)
    return float(raw_env._data.xpos[peg_id, 2]) - float(peg_rest_z)


def _insert_geometry(raw_env) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    data = raw_env._data
    model = raw_env._model
    peg_id = int(model.body(PEG_BODY).id)
    socket_id = int(model.site(_SOCKET_SITE).id)
    bottom_id = int(model.geom(_BOTTOM_GEOM).id)
    tip = peg_insert_end_pos(data.xpos[peg_id], data.xmat[peg_id])
    socket = np.asarray(data.site_xpos[socket_id], dtype=np.float64)
    hole_axis = hole_opening_axis(
        socket,
        data.site_xmat[socket_id],
        np.asarray(data.geom_xpos[bottom_id], dtype=np.float64),
    )
    dist = tip_socket_distance(tip, socket)
    return tip, socket, hole_axis, dist


def _insert_peg_lift(
    raw_env,
    left: np.ndarray,
    right: np.ndarray,
    *,
    target_dz: float = 0.07,
    steps: int = 72,
) -> np.ndarray:
    """Vertical peg lift via mocap ramp (left frozen)."""
    left = vec_to_arm_action(left)
    right = vec_to_arm_action(right)
    hand = right[7:23].copy()
    z0 = float(right[2])
    z1 = z0 + float(target_dz)
    goal = right.copy()
    goal[2] = z1
    for i in range(max(int(steps), 1)):
        u = (i + 1) / max(int(steps), 1)
        active = interpolate_arm_only(right, goal, u, hand=hand)
        _step_side(raw_env, side="right", active23=active, hold_right=active, hold_left=left)
    return vec_to_arm_action(read_arm_action(raw_env, "right"))


def _try_force_handoff(
    env,
    raw_env,
    hybrid: EvalHybridInsert,
    labeler: AssemblyContactLabeler,
    *,
    cfg: HybridInsertConfig,
    peg_rest_z: float,
) -> bool:
    if hybrid.controller is None or hybrid.controller.active:
        return hybrid.controller is not None and hybrid.controller.active
    outcome = labeler.compute(raw_env)
    tip, socket, hole_axis, dist = _insert_geometry(raw_env)
    peg_dz = _peg_lift_m(raw_env, peg_rest_z)
    in_cyl = in_approach_cylinder(
        tip,
        socket,
        hole_axis,
        xy_tol_m=cfg.approach_xy_m,
        z_min_m=cfg.approach_z_min_m,
        z_max_m=cfg.approach_z_max_m,
    )
    ready = (
        outcome.tray_contact_count >= 2
        and outcome.peg_contact_count >= 3
        and peg_dz >= cfg.lift_ready_m - 0.015
        and (in_cyl or dist < 0.12)
    )
    if not ready:
        return False
    left = read_arm_action(raw_env, "left")
    right = read_arm_action(raw_env, "right")
    policy44 = dual_arm23_to_action44(left, right)
    hybrid.controller._handoff_streak = int(cfg.handoff_confirm_frames)  # noqa: SLF001
    hybrid.controller.update_handoff(raw_env, policy44)
    return bool(hybrid.controller.active)


def _privileged_approach_to_handoff(
    env,
    raw_env,
    hybrid: EvalHybridInsert,
    labeler: AssemblyContactLabeler,
    *,
    cfg: HybridInsertConfig,
    max_steps: int = 900,
) -> None:
    """Move peg toward socket + lift (skill replay has no learned policy motion)."""
    peg_rest_z = float(labeler._peg_rest_z)  # noqa: SLF001

    for step in range(int(max_steps)):
        outcome = labeler.compute(raw_env)
        left = vec_to_arm_action(read_arm_action(raw_env, "left"))
        right = vec_to_arm_action(read_arm_action(raw_env, "right"))
        tip, socket, hole_axis, dist = _insert_geometry(raw_env)
        peg_dz = _peg_lift_m(raw_env, peg_rest_z)
        in_cyl = in_approach_cylinder(
            tip,
            socket,
            hole_axis,
            xy_tol_m=cfg.approach_xy_m,
            z_min_m=cfg.approach_z_min_m,
            z_max_m=cfg.approach_z_max_m,
        )

        policy44 = dual_arm23_to_action44(left, right)
        hybrid.observe(env, policy44)
        if hybrid.controller is not None and hybrid.controller.active:
            return

        delta = toward_socket_delta(tip, socket, gain=0.55, max_step_m=0.005)
        if peg_dz < cfg.lift_ready_m:
            delta[2] += min(0.004, cfg.lift_ready_m - peg_dz)
        norm = float(np.linalg.norm(delta))
        if norm > 0.006:
            delta = delta * (0.006 / norm)
        right[0:3] = right[0:3] + delta

        policy44 = dual_arm23_to_action44(left, right)
        hybrid.observe(env, policy44)
        if hybrid.controller is not None and hybrid.controller.active:
            return
        action46 = rotvec_dual_arm_to_policy(np.asarray(policy44, dtype=np.float64).reshape(44))
        _env_step_info(env, action46.astype(np.float32))
        hybrid.observe(env, policy44)
        if hybrid.controller is not None and hybrid.controller.active:
            return

        if step % 80 == 0:
            print(
                f"skill_insert: approach step={step} peg_dz={peg_dz:.3f} "
                f"dist={dist*1000:.1f}mm in_cyl={in_cyl} "
                f"tray_ok={outcome.tray_ok} peg_ok={outcome.peg_ok}",
                flush=True,
            )

    _try_force_handoff(env, raw_env, hybrid, labeler, cfg=cfg, peg_rest_z=peg_rest_z)


def run_hybrid_insert(
    env,
    raw_env,
    *,
    max_steps: int = 2500,
    insert_cfg: HybridInsertConfig | None = None,
    settle_steps: int = 40,
    tray_rest_z: float | None = None,
    peg_rest_z: float | None = None,
) -> InsertReport:
    """Step sim with privileged geometry insert until env ``succeed`` or budget."""
    cfg = insert_cfg or HybridInsertConfig(handoff_confirm_frames=3, approach_xy_m=0.10)
    hybrid = EvalHybridInsert(
        task="bimanual_assembly",
        enabled=True,
        config=cfg,
    )
    labeler = AssemblyContactLabeler(raw_env)
    if tray_rest_z is not None and peg_rest_z is not None:
        labeler._tray_rest_z = float(tray_rest_z)  # noqa: SLF001
        labeler._peg_rest_z = float(peg_rest_z)  # noqa: SLF001
    else:
        labeler.reset_reference(raw_env)
    peg_rest_z = float(labeler._peg_rest_z)  # noqa: SLF001

    hybrid.on_reset(env)
    if hybrid.controller is not None and tray_rest_z is not None:
        hybrid.controller._peg_rest_z = float(peg_rest_z)  # noqa: SLF001
        if hybrid.controller._labeler is not None:
            hybrid.controller._labeler._tray_rest_z = float(tray_rest_z)  # noqa: SLF001
            hybrid.controller._labeler._peg_rest_z = float(peg_rest_z)  # noqa: SLF001

    for _ in range(max(int(settle_steps), 1)):
        left = read_arm_action(raw_env, "left")
        right = read_arm_action(raw_env, "right")
        raw_env.step(action44_to_raw_dict(dual_arm23_to_action44(left, right)))

    from interaction_retarget.grasp.lift import execute_peg_lift
    from interaction_retarget.grasp.locked_hold import enforce_locked_passive
    from interaction_retarget.sim.contact import AssemblyContactDetector

    left = vec_to_arm_action(read_arm_action(raw_env, "left"))
    right = vec_to_arm_action(read_arm_action(raw_env, "right"))
    lift_det = AssemblyContactDetector(raw_env)
    lift_det._peg_rest_z = float(peg_rest_z)  # noqa: SLF001
    if tray_rest_z is not None:
        lift_det._tray_rest_z = float(tray_rest_z)  # noqa: SLF001
    if _peg_lift_m(raw_env, peg_rest_z) < cfg.lift_ready_m - 0.005:
        right = execute_peg_lift(
            raw_env,
            grasp_right=right,
            hold_left=left,
            detector=lift_det,
            for_insert=True,
            steps=100,
        )
        enforce_locked_passive(raw_env, locked_left=left, locked_right=right, n_substeps=20)
        left = vec_to_arm_action(read_arm_action(raw_env, "left"))
    outcome0 = labeler.compute(raw_env)
    peg_dz0 = _peg_lift_m(raw_env, peg_rest_z)
    print(
        f"skill_insert: after peg lift dz={peg_dz0:.3f} "
        f"tray_ok={outcome0.tray_ok} peg_ok={outcome0.peg_ok} "
        f"peg_c={outcome0.peg_contact_count} tray_c={outcome0.tray_contact_count}",
        flush=True,
    )

    _privileged_approach_to_handoff(env, raw_env, hybrid, labeler, cfg=cfg, max_steps=900)
    _try_force_handoff(env, raw_env, hybrid, labeler, cfg=cfg, peg_rest_z=peg_rest_z)

    success = False
    insert_ok = False
    step = 0
    for step in range(int(max_steps)):
        left = read_arm_action(raw_env, "left")
        right = read_arm_action(raw_env, "right")
        policy44 = dual_arm23_to_action44(left, right)
        hybrid.observe(env, policy44)
        merged = hybrid.merge(env, policy44)
        action46 = rotvec_dual_arm_to_policy(np.asarray(merged, dtype=np.float64).reshape(44))
        info = _env_step_info(env, action46.astype(np.float32))
        outcome = labeler.compute(raw_env)
        insert_ok = bool(outcome.insert_ok)
        if info.get("succeed"):
            success = True
            break
        if hybrid.controller is not None and hybrid.controller.phase_name == "RELEASE":
            if insert_ok:
                success = True
                break

    phase = hybrid.controller.phase_name if hybrid.controller else "disabled"
    handoff = bool(hybrid.controller and hybrid.controller.handoff_happened)
    peg_dz = _peg_lift_m(raw_env, peg_rest_z)

    fail_reason = "ok" if success else "insert_timeout"
    if not handoff and not success:
        fail_reason = "handoff_never"
    elif handoff and not success:
        fail_reason = f"insert_incomplete_{phase.lower()}"

    return InsertReport(
        success=bool(success),
        steps=int(step),
        insert_ok=insert_ok,
        handoff=handoff,
        phase=phase,
        peg_lift_m=peg_dz,
        fail_reason=fail_reason,
    )
