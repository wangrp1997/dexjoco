"""Contact repair + open-loop grasp stability check (spider-style, phase-1 step 4).

After Laplacian IK, nudge the active hand (finger close / mocap push) until fingertip
contact counts meet demo thresholds, then hold the pose in sim to verify contact persists
and the object does not drop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from interaction_retarget.constants import (
    CONTACT_WINDOW,
    MIN_GRASP_CONTACT_COUNT,
    ON_TABLE_MARGIN_M,
    PEG_BODY,
    TRAY_BODY,
)
from interaction_retarget.grasp.distill import load_canonical_grasp
from interaction_retarget.grasp.ik import GraspIkResult
from interaction_retarget.laplacian import laplacian_coordinates
from interaction_retarget.sim.contact import AssemblyContactDetector, FrameContact
from interaction_retarget.sim.hand_geom import hand_keypoints_world
from interaction_retarget.sim.settle import settle_bimanual_actions, vec_to_arm_action
from interaction_retarget.transforms import object_to_world

ObjectName = Literal["tray", "peg"]
Side = Literal["left", "right"]

_REPAIR_FINGER_DELTAS = (-0.05, -0.03, -0.02, -0.01, -0.008, -0.004, 0.004, 0.008, 0.01, 0.02, 0.03, 0.05)
_REPAIR_MOCAP_STEPS_M = (0.001, 0.002, 0.005, 0.008, 0.012)
_OBJECT_DROP_MARGIN_M = 0.010


@dataclass
class SideContactMetrics:
    side: Side
    object_name: ObjectName
    contact_count: int
    has_contact: bool
    object_z: float
    object_rest_z: float
    on_table: bool


@dataclass
class GraspRepairResult:
    action_right: np.ndarray
    action_left: np.ndarray
    tray: SideContactMetrics
    peg: SideContactMetrics
    repair_iters: int
    hold_steps: int
    stable_tray: bool
    stable_peg: bool
    success: bool


@dataclass
class OpenLoopGraspReport:
    ik_tray_success: bool
    ik_peg_success: bool
    repair: GraspRepairResult
    tray_laplacian_rmse_m: float
    peg_laplacian_rmse_m: float


def bimanual_actions_to_dict(right23: np.ndarray, left23: np.ndarray) -> dict[str, np.ndarray]:
    right23 = vec_to_arm_action(right23)
    left23 = vec_to_arm_action(left23)
    return {"right": right23, "left": left23}


def _side_for_object(object_name: ObjectName) -> Side:
    return "left" if object_name == "tray" else "right"


def _object_body(object_name: ObjectName) -> str:
    return TRAY_BODY if object_name == "tray" else PEG_BODY


def _hand_bodies(side: Side) -> tuple[str, ...]:
    from interaction_retarget.constants import LEFT_HAND_BODIES, RIGHT_HAND_BODIES

    return LEFT_HAND_BODIES if side == "left" else RIGHT_HAND_BODIES


def _hand_joint_bounds(model, side: Side) -> tuple[np.ndarray, np.ndarray]:
    names = (
        [
            "ffj0_right", "ffj1_right", "ffj2_right", "ffj3_right",
            "mfj0_right", "mfj1_right", "mfj2_right", "mfj3_right",
            "rfj0_right", "rfj1_right", "rfj2_right", "rfj3_right",
            "thj0_right", "thj1_right", "thj2_right", "thj3_right",
        ]
        if side == "right"
        else [
            "rfj0_left", "rfj1_left", "rfj2_left", "rfj3_left",
            "mfj0_left", "mfj1_left", "mfj2_left", "mfj3_left",
            "ffj0_left", "ffj1_left", "ffj2_left", "ffj3_left",
            "thj0_left", "thj1_left", "thj2_left", "thj3_left",
        ]
    )
    lo, hi = [], []
    for name in names:
        jnt = model.joint(name)
        lo.append(float(model.jnt_range[jnt.id, 0]))
        hi.append(float(model.jnt_range[jnt.id, 1]))
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def _snapshot_sim(raw_env) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = raw_env._data
    return (
        data.qpos.copy(),
        data.qvel.copy(),
        data.mocap_pos.copy(),
        data.mocap_quat.copy(),
    )


def _restore_sim(raw_env, state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    qpos, qvel, mocap_pos, mocap_quat = state
    data = raw_env._data
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.mocap_pos[:] = mocap_pos
    data.mocap_quat[:] = mocap_quat
    mujoco.mj_forward(raw_env._model, data)


def _contact_count(contact: FrameContact, object_name: ObjectName) -> int:
    return int(contact.tray_contact_count if object_name == "tray" else contact.peg_contact_count)


def _side_metrics(
    raw_env,
    detector: AssemblyContactDetector,
    contact: FrameContact,
    *,
    object_name: ObjectName,
) -> SideContactMetrics:
    side = _side_for_object(object_name)
    body_id = raw_env._model.body(_object_body(object_name)).id
    z = float(raw_env._data.xpos[body_id, 2])
    rest_z = float(detector._tray_rest_z if object_name == "tray" else detector._peg_rest_z)  # noqa: SLF001
    count = _contact_count(contact, object_name)
    return SideContactMetrics(
        side=side,
        object_name=object_name,
        contact_count=count,
        has_contact=count > 0,
        object_z=z,
        object_rest_z=rest_z,
        on_table=(z - rest_z) <= ON_TABLE_MARGIN_M,
    )


def _step_bimanual(raw_env, right23: np.ndarray, left23: np.ndarray) -> None:
    raw_env.step(bimanual_actions_to_dict(right23, left23))


def _step_side(
    raw_env,
    *,
    side: Side,
    active23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
) -> None:
    """Step only the active arm; the other arm holds its command (async agents)."""
    active23 = vec_to_arm_action(active23)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    if side == "left":
        _step_bimanual(raw_env, hold_right, active23)
    else:
        _step_bimanual(raw_env, active23, hold_left)


def _apply_and_measure_side(
    raw_env,
    detector: AssemblyContactDetector,
    *,
    side: Side,
    active23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    object_name: ObjectName,
    hold_steps: int = 1,
) -> tuple[FrameContact, int]:
    """Step active arm; return last contact frame and min contact count during hold."""
    counts: list[int] = []
    contact: FrameContact | None = None
    for _ in range(max(int(hold_steps), 1)):
        _step_side(
            raw_env,
            side=side,
            active23=active23,
            hold_right=hold_right,
            hold_left=hold_left,
        )
        contact = detector.compute(raw_env)
        counts.append(_contact_count(contact, object_name))
    assert contact is not None
    return contact, min(counts)


def _apply_and_measure(
    raw_env,
    detector: AssemblyContactDetector,
    *,
    right23: np.ndarray,
    left23: np.ndarray,
    hold_steps: int = 1,
) -> FrameContact:
    for _ in range(int(hold_steps)):
        _step_bimanual(raw_env, right23, left23)
    return detector.compute(raw_env)


def _close_fingers(action23: np.ndarray, delta: float, hand_lo: np.ndarray, hand_hi: np.ndarray) -> np.ndarray:
    action23 = vec_to_arm_action(action23)
    hand = np.clip(action23[7:23] + delta, hand_lo, hand_hi)
    return np.concatenate([action23[0:7], hand], axis=0)


def _nudge_mocap_toward_object(
    raw_env,
    action23: np.ndarray,
    *,
    object_name: ObjectName,
    step_m: float,
) -> np.ndarray:
    action23 = vec_to_arm_action(action23)
    body_id = raw_env._model.body(_object_body(object_name)).id
    obj_pos = np.asarray(raw_env._data.xpos[body_id], dtype=np.float64)
    palm_pos = action23[0:3]
    delta = obj_pos - palm_pos
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return action23
    pos = palm_pos + (delta / norm) * float(step_m)
    return np.concatenate([pos, action23[3:23]], axis=0)


def laplacian_rmse(
    raw_env,
    canonical: dict,
    *,
    object_name: ObjectName,
) -> float:
    from interaction_retarget.grasp.ik import interaction_metrics_obj_frame

    side = _side_for_object(object_name)
    model = raw_env._model
    data = raw_env._data
    obj_body = _object_body(object_name)
    _, metrics = interaction_metrics_obj_frame(
        model,
        data,
        side=side,
        obj_body=obj_body,
        target_hand_obj=canonical["hand_points_obj"],
        target_obj_samples_obj=canonical["object_samples_obj"],
        target_laplacian=canonical["laplacian_coords"],
        adjacency=canonical["adjacency"],
    )
    return metrics["laplacian_rmse_m"]


def repair_side_grasp(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    action_right: np.ndarray,
    action_left: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    detector: AssemblyContactDetector,
    max_iters: int = 24,
    min_contact_count: int = MIN_GRASP_CONTACT_COUNT,
    hold_steps: int = 1,
    max_laplacian_drift_m: float = 0.045,
    canonical: dict | None = None,
    require_on_table: bool = True,
    finger_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy repair for one hand; the other arm holds at hold_right/hold_left."""
    model = raw_env._model
    right23 = vec_to_arm_action(action_right)
    left23 = vec_to_arm_action(action_left)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    lo, hi = _hand_joint_bounds(model, side)

    contact, count = _apply_and_measure_side(
        raw_env,
        detector,
        side=side,
        active23=left23 if side == "left" else right23,
        hold_right=hold_right,
        hold_left=hold_left,
        object_name=object_name,
        hold_steps=hold_steps,
    )
    baseline_lap = laplacian_rmse(raw_env, canonical, object_name=object_name) if canonical else None
    sim0 = _snapshot_sim(raw_env)

    iters = 0
    for _ in range(int(max_iters)):
        if count >= min_contact_count:
            break

        candidates: list[tuple[int, np.ndarray, np.ndarray]] = []

        def _maybe_add(new_left: np.ndarray, new_right: np.ndarray) -> None:
            _restore_sim(raw_env, sim0)
            c, tc = _apply_and_measure_side(
                raw_env,
                detector,
                side=side,
                active23=new_left if side == "left" else new_right,
                hold_right=hold_right,
                hold_left=hold_left,
                object_name=object_name,
                hold_steps=hold_steps,
            )
            if canonical is not None:
                lap = laplacian_rmse(raw_env, canonical, object_name=object_name)
                if baseline_lap is not None and lap > baseline_lap + max_laplacian_drift_m:
                    return
            metrics = _side_metrics(raw_env, detector, c, object_name=object_name)
            if require_on_table and not metrics.on_table:
                return
            candidates.append((tc, new_left.copy(), new_right.copy()))

        active23 = left23 if side == "left" else right23
        for delta in _REPAIR_FINGER_DELTAS:
            updated = _close_fingers(active23, delta, lo, hi)
            if side == "left":
                _maybe_add(updated, right23)
            else:
                _maybe_add(left23, updated)
        if not finger_only:
            for step in _REPAIR_MOCAP_STEPS_M:
                updated = _nudge_mocap_toward_object(raw_env, active23, object_name=object_name, step_m=step)
                if side == "left":
                    _maybe_add(updated, right23)
                else:
                    _maybe_add(left23, updated)

        if not candidates:
            break

        _, new_left, new_right = max(candidates, key=lambda x: x[0])
        left23, right23 = new_left, new_right
        _restore_sim(raw_env, sim0)
        contact, count = _apply_and_measure_side(
            raw_env,
            detector,
            side=side,
            active23=left23 if side == "left" else right23,
            hold_right=hold_right,
            hold_left=hold_left,
            object_name=object_name,
            hold_steps=hold_steps,
        )
        sim0 = _snapshot_sim(raw_env)
        iters += 1

    return right23, left23, iters


def repair_bimanual_grasp(
    raw_env,
    *,
    action_right: np.ndarray,
    action_left: np.ndarray,
    detector: AssemblyContactDetector,
    max_iters: int = 24,
    min_contact_count: int = MIN_GRASP_CONTACT_COUNT,
    hold_steps: int = 1,
    max_laplacian_drift_m: float = 0.045,
    canonical_tray: dict | None = None,
    canonical_peg: dict | None = None,
    require_on_table: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Async repair: left/tray first (right frozen), then right/peg (left frozen)."""
    right23 = vec_to_arm_action(action_right)
    left23 = vec_to_arm_action(action_left)
    right_hold = right23.copy()

    right23, left23, iters_left = repair_side_grasp(
        raw_env,
        side="left",
        object_name="tray",
        action_right=right23,
        action_left=left23,
        hold_right=right_hold,
        hold_left=left23,
        detector=detector,
        max_iters=max_iters,
        min_contact_count=min_contact_count,
        hold_steps=hold_steps,
        max_laplacian_drift_m=max_laplacian_drift_m,
        canonical=canonical_tray,
        require_on_table=require_on_table,
    )
    left_hold = left23.copy()

    right23, left23, iters_right = repair_side_grasp(
        raw_env,
        side="right",
        object_name="peg",
        action_right=right23,
        action_left=left23,
        hold_right=right23,
        hold_left=left_hold,
        detector=detector,
        max_iters=max_iters,
        min_contact_count=min_contact_count,
        hold_steps=hold_steps,
        max_laplacian_drift_m=max_laplacian_drift_m,
        canonical=canonical_peg,
        require_on_table=require_on_table,
    )
    return right23, left23, iters_left + iters_right


def verify_grasp_hold(
    raw_env,
    detector: AssemblyContactDetector,
    *,
    action_right: np.ndarray,
    action_left: np.ndarray,
    hold_steps: int = 12,
    warmup_steps: int = 10,
    contact_window: int = CONTACT_WINDOW,
    min_contact_count: int = MIN_GRASP_CONTACT_COUNT,
) -> GraspRepairResult:
    """Hold grasp pose in sim; require sustained contact and no object drop."""
    right23 = vec_to_arm_action(action_right)
    left23 = vec_to_arm_action(action_left)

    for _ in range(max(int(warmup_steps), 0)):
        settle_bimanual_actions(
            raw_env,
            right23=right23,
            left23=left23,
            n_substeps=1,
        )

    tray_counts: list[int] = []
    peg_counts: list[int] = []
    tray_z_delta: list[float] = []
    peg_z_delta: list[float] = []

    for _ in range(int(hold_steps)):
        settle_bimanual_actions(
            raw_env,
            right23=right23,
            left23=left23,
            n_substeps=1,
        )
        contact = detector.compute(raw_env)
        tray_m = _side_metrics(raw_env, detector, contact, object_name="tray")
        peg_m = _side_metrics(raw_env, detector, contact, object_name="peg")
        tray_counts.append(tray_m.contact_count)
        peg_counts.append(peg_m.contact_count)
        tray_z_delta.append(tray_m.object_z - tray_m.object_rest_z)
        peg_z_delta.append(peg_m.object_z - peg_m.object_rest_z)

    contact = detector.compute(raw_env)
    tray = _side_metrics(raw_env, detector, contact, object_name="tray")
    peg = _side_metrics(raw_env, detector, contact, object_name="peg")

    def _stable(counts: list[int], z_delta: list[float]) -> bool:
        if len(counts) < contact_window:
            return False
        recent_counts = counts[-contact_window:]
        recent_z = z_delta[-contact_window:]
        if any(z < -_OBJECT_DROP_MARGIN_M for z in recent_z):
            return False
        return all(c >= min_contact_count for c in recent_counts)

    tray_dropped = tray.object_z < (tray.object_rest_z - _OBJECT_DROP_MARGIN_M)
    peg_dropped = peg.object_z < (peg.object_rest_z - _OBJECT_DROP_MARGIN_M)

    stable_tray = _stable(tray_counts, tray_z_delta) and not tray_dropped
    stable_peg = _stable(peg_counts, peg_z_delta) and not peg_dropped
    success = (
        stable_tray
        and stable_peg
        and tray.contact_count >= min_contact_count
        and peg.contact_count >= min_contact_count
    )

    return GraspRepairResult(
        action_right=right23,
        action_left=left23,
        tray=tray,
        peg=peg,
        repair_iters=0,
        hold_steps=int(hold_steps),
        stable_tray=stable_tray,
        stable_peg=stable_peg,
        success=success,
    )


def repair_and_verify(
    raw_env,
    *,
    action_right: np.ndarray,
    action_left: np.ndarray,
    detector: AssemblyContactDetector,
    canonical_tray: dict | None = None,
    canonical_peg: dict | None = None,
    max_repair_iters: int = 24,
    hold_steps: int = 12,
    repair_hold_steps: int = 1,
    min_contact_count: int = MIN_GRASP_CONTACT_COUNT,
    require_on_table: bool = True,
) -> GraspRepairResult:
    right23, left23, repair_iters = repair_bimanual_grasp(
        raw_env,
        action_right=action_right,
        action_left=action_left,
        detector=detector,
        max_iters=max_repair_iters,
        min_contact_count=min_contact_count,
        hold_steps=repair_hold_steps,
        canonical_tray=canonical_tray,
        canonical_peg=canonical_peg,
        require_on_table=require_on_table,
    )
    result = verify_grasp_hold(
        raw_env,
        detector,
        action_right=right23,
        action_left=left23,
        hold_steps=hold_steps,
        min_contact_count=min_contact_count,
    )
    result.repair_iters = repair_iters
    return result


def merge_ik_results(tray: GraspIkResult, peg: GraspIkResult) -> tuple[np.ndarray, np.ndarray]:
    """Combine per-object IK into one bimanual grasp (left←tray, right←peg)."""
    return vec_to_arm_action(peg.action_right), vec_to_arm_action(tray.action_left)


def blend_arm_actions(
    warm23: np.ndarray,
    ik23: np.ndarray,
    *,
    finger_alpha: float = 0.45,
    arm_alpha: float = 0.12,
) -> np.ndarray:
    """Keep most of demo mocap (contact-safe), mix in IK finger shape."""
    warm23 = vec_to_arm_action(warm23)
    ik23 = vec_to_arm_action(ik23)
    out = warm23.copy()
    out[0:7] = warm23[0:7] * (1.0 - arm_alpha) + ik23[0:7] * arm_alpha
    out[7:23] = warm23[7:23] * (1.0 - finger_alpha) + ik23[7:23] * finger_alpha
    return out


def merge_ik_with_warm_start(
    tray: GraspIkResult,
    peg: GraspIkResult,
    *,
    warm_right: np.ndarray,
    warm_left: np.ndarray,
    finger_alpha: float = 0.45,
    arm_alpha: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    ik_right, ik_left = merge_ik_results(tray, peg)
    right23 = blend_arm_actions(warm_right, ik_right, finger_alpha=finger_alpha, arm_alpha=arm_alpha)
    left23 = blend_arm_actions(warm_left, ik_left, finger_alpha=finger_alpha, arm_alpha=arm_alpha)
    return right23, left23


def ramp_bimanual_actions(
    raw_env,
    *,
    action_right: np.ndarray,
    action_left: np.ndarray,
    steps: int = 5,
) -> None:
    right23 = vec_to_arm_action(action_right)
    left23 = vec_to_arm_action(action_left)
    for _ in range(int(steps)):
        _step_bimanual(raw_env, right23, left23)


def ramp_side_actions(
    raw_env,
    *,
    side: Side,
    action_right: np.ndarray,
    action_left: np.ndarray,
    steps: int = 5,
) -> None:
    """Ramp one arm to target; the other holds its command (async)."""
    right23 = vec_to_arm_action(action_right)
    left23 = vec_to_arm_action(action_left)
    for _ in range(int(steps)):
        _step_side(
            raw_env,
            side=side,
            active23=left23 if side == "left" else right23,
            hold_right=right23,
            hold_left=left23,
        )


def ramp_grasp_sequential(
    raw_env,
    *,
    home_right: np.ndarray,
    home_left: np.ndarray,
    action_right: np.ndarray,
    action_left: np.ndarray,
    steps: int = 5,
) -> None:
    """Left reach first (right at home), then right (left frozen)."""
    steps = max(int(steps), 1)
    right_hold = vec_to_arm_action(home_right)
    left_hold = vec_to_arm_action(home_left)
    left_target = vec_to_arm_action(action_left)
    right_target = vec_to_arm_action(action_right)

    for _ in range(steps):
        _step_side(raw_env, side="left", active23=left_target, hold_right=right_hold, hold_left=left_hold)
    left_hold = left_target.copy()
    for _ in range(steps):
        _step_side(raw_env, side="right", active23=right_target, hold_right=right_hold, hold_left=left_hold)


def evaluate_openloop_grasp(
    raw_env,
    *,
    tray_ik: GraspIkResult,
    peg_ik: GraspIkResult,
    detector: AssemblyContactDetector,
    canonical_tray: dict,
    canonical_peg: dict,
    hold_steps: int = 12,
    max_repair_iters: int = 24,
    ramp_steps: int = 5,
    repair_hold_steps: int = 1,
    warm_right: np.ndarray | None = None,
    warm_left: np.ndarray | None = None,
    finger_alpha: float = 0.45,
    arm_alpha: float = 0.12,
) -> OpenLoopGraspReport:
    if warm_right is not None and warm_left is not None:
        right23, left23 = merge_ik_with_warm_start(
            tray_ik,
            peg_ik,
            warm_right=warm_right,
            warm_left=warm_left,
            finger_alpha=finger_alpha,
            arm_alpha=arm_alpha,
        )
    else:
        right23, left23 = merge_ik_results(tray_ik, peg_ik)
    from interaction_retarget.sim.settle import read_arm_action

    ramp_grasp_sequential(
        raw_env,
        home_right=read_arm_action(raw_env, "right"),
        home_left=read_arm_action(raw_env, "left"),
        action_right=right23,
        action_left=left23,
        steps=ramp_steps,
    )
    repair = repair_and_verify(
        raw_env,
        action_right=right23,
        action_left=left23,
        detector=detector,
        canonical_tray=canonical_tray,
        canonical_peg=canonical_peg,
        hold_steps=hold_steps,
        max_repair_iters=max_repair_iters,
        repair_hold_steps=repair_hold_steps,
    )
    return OpenLoopGraspReport(
        ik_tray_success=tray_ik.success,
        ik_peg_success=peg_ik.success,
        repair=repair,
        tray_laplacian_rmse_m=laplacian_rmse(raw_env, canonical_tray, object_name="tray"),
        peg_laplacian_rmse_m=laplacian_rmse(raw_env, canonical_peg, object_name="peg"),
    )
