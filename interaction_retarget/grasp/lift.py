"""Lift execution: GenHand grasp→liftup (arm moves, fingers fixed) + object-frame waypoints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from interaction_retarget.constants import CONTACT_WINDOW, MIN_GRASP_CONTACT_COUNT, PEG_BODY, TRAY_BODY
from interaction_retarget.grasp.approach import interpolate_arm_only
from interaction_retarget.grasp.lift_reference import default_demo_lift_path, load_demo_lift_reference
from interaction_retarget.grasp.repair import (
    _close_fingers,
    _hand_joint_bounds,
    side_contact_count,
    _step_side,
)
from interaction_retarget.grasp.staged_grasp import prepare_lift_squeeze, re_squeeze_fc
from interaction_retarget.grasp.locked_hold import enforce_locked_passive
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.settle import read_arm_action, settle_bimanual_actions, vec_to_arm_action
from interaction_retarget.transforms import mocap_world_from_object_frame, relative_mocap_in_object_frame
from interaction_retarget.tpsr.config import TpsrConfig
from interaction_retarget.bench.verify import verify_side_hold
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.tpsr.grasp_filter import GraspFilter, grasp_filter_cfg_from_tpsr

Side = Literal["left", "right"]
ObjectName = Literal["tray", "peg"]
DEFAULT_TRAY_LIFT_M = 0.05
DEFAULT_PEG_LIFT_M = 0.05
PEG_LIFT_INSERT_M = 0.08
_LIFT_MIN_CONTACT = max(2, MIN_GRASP_CONTACT_COUNT - 1)
_SUBSTEPS_PER_WAYPOINT = 3


def _object_body(side: Side) -> str:
    return TRAY_BODY if side == "left" else PEG_BODY


def _object_name(side: Side) -> ObjectName:
    return "tray" if side == "left" else "peg"


def _segment_keys(side: Side) -> tuple[str, str, str]:
    obj = _object_name(side)
    return f"{obj}_mocap_pos_obj", f"{obj}_mocap_quat_obj", f"{obj}_lift_num_demo_frames"


def _object_pose(raw_env, side: Side) -> tuple[np.ndarray, np.ndarray]:
    body = _object_body(side)
    obj_id = raw_env._model.body(body).id
    data = raw_env._data
    return (
        np.asarray(data.xpos[obj_id], dtype=np.float64),
        np.asarray(data.xquat[obj_id], dtype=np.float64),
    )


def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _waypoint_action23(
    obj_pos: np.ndarray,
    obj_quat: np.ndarray,
    *,
    pos_obj: np.ndarray,
    quat_obj: np.ndarray,
    hand: np.ndarray,
) -> np.ndarray:
    pos_w, quat_w = mocap_world_from_object_frame(pos_obj, quat_obj, obj_pos, obj_quat)
    return np.concatenate([pos_w, quat_w, np.asarray(hand, dtype=np.float64).reshape(16)], axis=0)


def _lift_steps_from_ref(
    lift_ref: dict[str, Any],
    side: Side,
    *,
    n_waypoints: int,
    default_steps: int,
    max_steps: int | None = None,
) -> int:
    """Execute every demo waypoint; optional cap only for --fast validation."""
    n = max(int(n_waypoints), 1) * _SUBSTEPS_PER_WAYPOINT
    if max_steps is not None:
        n = min(n, max(int(max_steps), 1))
    return n


def verify_lift_fc(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    tpsr_cfg: TpsrConfig,
    hold_right: np.ndarray | None = None,
    hold_left: np.ndarray | None = None,
    detector: AssemblyContactDetector | None = None,
) -> tuple[bool, float]:
    """Dexonomy zero-wrench QP; DexGraspBench fc_mocap hold if QP feasible but above thre."""
    gf_cfg = grasp_filter_cfg_from_tpsr(tpsr_cfg, ho_collision_thre_m=-0.006)
    res = GraspFilter(gf_cfg).forward(raw_env, side=side, object_name=object_name)
    qp_err = float(res.max_qp_error)
    if res.ok:
        return True, qp_err
    if (
        qp_err < float(tpsr_cfg.qp_hold_soft_thre)
        and hold_right is not None
        and hold_left is not None
        and detector is not None
    ):
        hold = verify_side_hold(
            raw_env,
            object_name=object_name,
            action_right=hold_right,
            action_left=hold_left,
            detector=detector,
            tpsr_cfg=tpsr_cfg,
        )
        if hold.stable:
            return True, qp_err
    return False, qp_err


def _resqueeze_dex(
    raw_env,
    *,
    side: Side,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    canonical: dict,
    object_name: ObjectName,
    tpsr_cfg: TpsrConfig,
) -> tuple[np.ndarray, np.ndarray]:
    hold_right, hold_left, _, _ = re_squeeze_fc(
        raw_env,
        side=side,
        object_name=object_name,
        canonical=canonical,
        hold_right=hold_right,
        hold_left=hold_left,
        tpsr_cfg=replace(tpsr_cfg, require_qp_fc=False),
        max_rounds=4,
    )
    return hold_right, hold_left


def _maybe_squeeze_on_contact_loss(
    raw_env,
    *,
    side: Side,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    detector: AssemblyContactDetector,
    object_name: ObjectName,
    canonical: dict | None,
    tpsr_cfg: TpsrConfig | None,
) -> tuple[np.ndarray, np.ndarray]:
    if side_contact_count(detector, raw_env, object_name=object_name) >= _LIFT_MIN_CONTACT:
        return hold_right, hold_left
    if canonical is None or tpsr_cfg is None:
        return hold_right, hold_left
    return _resqueeze_dex(
        raw_env,
        side=side,
        hold_right=hold_right,
        hold_left=hold_left,
        canonical=canonical,
        object_name=object_name,
        tpsr_cfg=tpsr_cfg,
    )


def _resqueeze_at_lift_end(
    raw_env,
    *,
    side: Side,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    detector: AssemblyContactDetector,
    object_name: ObjectName,
    canonical: dict | None,
    tpsr_cfg: TpsrConfig | None,
) -> tuple[np.ndarray, np.ndarray]:
    if side_contact_count(detector, raw_env, object_name=object_name) >= _LIFT_MIN_CONTACT:
        return hold_right, hold_left
    if canonical is None or tpsr_cfg is None:
        return hold_right, hold_left
    return _resqueeze_dex(
        raw_env,
        side=side,
        hold_right=hold_right,
        hold_left=hold_left,
        canonical=canonical,
        object_name=object_name,
        tpsr_cfg=tpsr_cfg,
    )


def execute_arm_lift(
    raw_env,
    *,
    side: Side,
    grasp_arm: np.ndarray,
    hold_other: np.ndarray,
    lift_ref: dict[str, Any] | None = None,
    detector: AssemblyContactDetector | None = None,
    lift_height_m: float | None = None,
    steps: int | None = None,
    hold_steps: int = 8,
    pre_lift_settle: int = 12,
    lift_exec_cap: int | None = None,
    lock_passive_arm: bool = False,
    object_z_only: bool = False,
    canonical: dict | None = None,
    tpsr_cfg: TpsrConfig | None = None,
) -> np.ndarray:
    """GenHand lift: Dexonomy squeeze+FC → one substep per demo waypoint."""
    object_name = _object_name(side)
    hold_right = vec_to_arm_action(hold_other if side == "left" else grasp_arm)
    hold_left = vec_to_arm_action(grasp_arm if side == "left" else hold_other)
    if canonical is not None and tpsr_cfg is not None:
        hold_right, hold_left, _, _ = prepare_lift_squeeze(
            raw_env,
            side=side,
            object_name=object_name,
            canonical=canonical,
            grasp_arm=grasp_arm,
            hold_right=hold_right,
            hold_left=hold_left,
            tpsr_cfg=tpsr_cfg,
            settle_steps=pre_lift_settle,
        )
    grasp_arm = vec_to_arm_action(read_arm_action(raw_env, side))
    hold_other = vec_to_arm_action(hold_right if side == "left" else hold_left)
    hand = grasp_arm[7:23].copy()
    lo, hi = _hand_joint_bounds(raw_env._model, side)
    fallback = DEFAULT_TRAY_LIFT_M if side == "left" else DEFAULT_PEG_LIFT_M

    pos_key, quat_key, _ = _segment_keys(side)
    use_waypoints = (
        lift_ref is not None
        and pos_key in lift_ref
        and quat_key in lift_ref
        and lift_ref[pos_key].shape[0] >= 2
    )
    passive_frozen = vec_to_arm_action(hold_other if side == "left" else hold_left)

    if use_waypoints:
        pos_obj = np.asarray(lift_ref[pos_key], dtype=np.float64)
        quat_obj = np.asarray(lift_ref[quat_key], dtype=np.float64)
        # Segment-relative in object frame (lift_start = origin) for smaller incremental motion.
        pos_seg = pos_obj - pos_obj[0:1]
        if object_z_only:
            pos_seg = np.column_stack(
                [
                    np.zeros(pos_seg.shape[0]),
                    np.zeros(pos_seg.shape[0]),
                    pos_seg[:, 2],
                ]
            )
        quat_seg = quat_obj.copy()
        n_wp = int(pos_seg.shape[0])
        n_exec = _lift_steps_from_ref(
            lift_ref,
            side,
            n_waypoints=n_wp,
            default_steps=steps or n_wp,
            max_steps=lift_exec_cap,
        )
        if lift_exec_cap is not None:
            n_use = min(n_wp, max(int(lift_exec_cap), 2))
            if n_use < n_wp:
                idx = np.linspace(0, n_wp - 1, n_use, dtype=int)
                pos_seg = pos_seg[idx]
                quat_seg = quat_seg[idx]
            n_wp = int(pos_seg.shape[0])
            substeps = 1
        else:
            substeps = max(_SUBSTEPS_PER_WAYPOINT, n_exec // max(n_wp, 1))
        obj0, quat0 = _object_pose(raw_env, side)
        grasp_pos_obj, grasp_quat_obj = relative_mocap_in_object_frame(
            grasp_arm[0:3], grasp_arm[3:7], obj0, quat0
        )
        track_pos: list[float] = []
        track_demo: list[float] = []
        meta_key = f"_last_{object_name}_lift_meta"
        raw_env.__dict__[meta_key] = {  # noqa: SLF001
            "start_mocap_obj": np.asarray(grasp_pos_obj, dtype=np.float64).copy(),
            "waypoints": int(n_wp),
            "substeps_per_wp": int(substeps),
        }
        hold_left = grasp_arm if side == "left" else passive_frozen
        hold_right = passive_frozen if side == "left" else grasp_arm
        prev23 = grasp_arm.copy()
        for i in range(n_wp):
            obj_pos, obj_quat = _object_pose(raw_env, side)
            target23 = _waypoint_action23(
                obj_pos,
                obj_quat,
                pos_obj=grasp_pos_obj + pos_seg[i],
                quat_obj=quat_seg[i],
                hand=hand,
            )
            track_demo.append(float(np.linalg.norm(pos_seg[i])))
            for sub in range(substeps):
                alpha = (sub + 1) / substeps
                active = interpolate_arm_only(prev23, target23, alpha, hand=hand)
                _step_side(
                    raw_env,
                    side=side,
                    active23=active,
                    hold_right=hold_right,
                    hold_left=hold_left,
                )
                if lock_passive_arm:
                    if side == "left":
                        enforce_locked_passive(
                            raw_env,
                            locked_left=vec_to_arm_action(read_arm_action(raw_env, "left")),
                            locked_right=passive_frozen,
                            n_substeps=2,
                        )
                        hold_right = passive_frozen
                    else:
                        enforce_locked_passive(
                            raw_env,
                            locked_left=passive_frozen,
                            locked_right=vec_to_arm_action(read_arm_action(raw_env, "right")),
                            n_substeps=2,
                        )
                        hold_left = passive_frozen
                if detector is not None:
                    cc = side_contact_count(detector, raw_env, object_name=object_name)
                    if cc < _LIFT_MIN_CONTACT:
                        side_action = vec_to_arm_action(read_arm_action(raw_env, side))
                        tightened = _close_fingers(side_action, -0.012, lo, hi)
                        hand = tightened[7:23].copy()
                        if side == "left":
                            settle_bimanual_actions(
                                raw_env, right23=passive_frozen, left23=tightened, n_substeps=2
                            )
                            hold_left = vec_to_arm_action(read_arm_action(raw_env, "left"))
                            hold_right = passive_frozen
                        else:
                            settle_bimanual_actions(
                                raw_env, right23=tightened, left23=passive_frozen, n_substeps=2
                            )
                            hold_right = vec_to_arm_action(read_arm_action(raw_env, "right"))
                            hold_left = passive_frozen
            prev23 = vec_to_arm_action(read_arm_action(raw_env, side))
            act_pos, _ = relative_mocap_in_object_frame(
                prev23[0:3], prev23[3:7], *_object_pose(raw_env, side)
            )
            track_pos.append(float(np.linalg.norm(act_pos - (grasp_pos_obj + pos_seg[i]))))
        lift_arm = vec_to_arm_action(read_arm_action(raw_env, side))
        if detector is not None and canonical is not None and tpsr_cfg is not None:
            hold_right, hold_left = _resqueeze_at_lift_end(
                raw_env,
                side=side,
                hold_right=hold_right,
                hold_left=hold_left,
                detector=detector,
                object_name=object_name,
                canonical=canonical,
                tpsr_cfg=tpsr_cfg,
            )
            lift_arm = vec_to_arm_action(read_arm_action(raw_env, side))
        if side == "left":
            raw_env._last_tray_lift_meta = raw_env.__dict__[meta_key]  # noqa: SLF001
            end_pos, _ = relative_mocap_in_object_frame(
                lift_arm[0:3], lift_arm[3:7], *_object_pose(raw_env, side)
            )
            raw_env._last_tray_lift_meta["end_mocap_obj"] = np.asarray(end_pos, dtype=np.float64)
        if meta_key in raw_env.__dict__:
            arr = np.asarray(track_pos, dtype=np.float64)
            raw_env.__dict__[meta_key]["waypoint_rmse_m"] = float(  # noqa: SLF001
                np.sqrt(np.mean(arr * arr)) if arr.size else 0.0
            )
            raw_env.__dict__[meta_key]["max_waypoint_err_m"] = float(  # noqa: SLF001
                np.max(arr) if arr.size else 0.0
            )
    else:
        grasp_arm = vec_to_arm_action(grasp_arm)
        dz = float(lift_height_m if lift_height_m is not None else fallback)
        if lift_ref is not None:
            dkey = f"{object_name}_mocap_delta_world"
            if dkey in lift_ref:
                dz = float(np.asarray(lift_ref[dkey], dtype=np.float64).reshape(3)[2])
        start_pos = grasp_arm[0:3].copy()
        start_quat = grasp_arm[3:7].copy()
        hand = grasp_arm[7:23].copy()
        n_exec = max(int(steps or 20), 1)
        hold_left = grasp_arm if side == "left" else passive_frozen
        hold_right = passive_frozen if side == "left" else grasp_arm
        for step in range(n_exec):
            u = _smoothstep((step + 1) / n_exec)
            target = grasp_arm.copy()
            target[0:2] = start_pos[0:2]
            target[2] = start_pos[2] + dz * u
            target[3:7] = start_quat
            target[7:23] = hand
            _step_side(
                raw_env,
                side=side,
                active23=target,
                hold_right=hold_right,
                hold_left=hold_left,
            )
            if lock_passive_arm:
                if side == "left":
                    enforce_locked_passive(
                        raw_env,
                        locked_left=vec_to_arm_action(read_arm_action(raw_env, side)),
                        locked_right=passive_frozen,
                        n_substeps=2,
                    )
                    hold_right = passive_frozen
                else:
                    enforce_locked_passive(
                        raw_env,
                        locked_left=passive_frozen,
                        locked_right=vec_to_arm_action(read_arm_action(raw_env, side)),
                        n_substeps=2,
                    )
                    hold_left = passive_frozen
            if detector is not None:
                cc = side_contact_count(detector, raw_env, object_name=object_name)
                if cc < _LIFT_MIN_CONTACT:
                    side_action = vec_to_arm_action(read_arm_action(raw_env, side))
                    tightened = _close_fingers(side_action, -0.015, lo, hi)
                    hand = tightened[7:23].copy()
                    if side == "left":
                        settle_bimanual_actions(
                            raw_env, right23=passive_frozen, left23=tightened, n_substeps=3
                        )
                        hold_left = vec_to_arm_action(read_arm_action(raw_env, "left"))
                        hold_right = passive_frozen
                    else:
                        settle_bimanual_actions(
                            raw_env, right23=tightened, left23=passive_frozen, n_substeps=3
                        )
                        hold_right = vec_to_arm_action(read_arm_action(raw_env, "right"))
                        hold_left = passive_frozen
        lift_arm = vec_to_arm_action(read_arm_action(raw_env, side))
        if detector is not None and canonical is not None and tpsr_cfg is not None:
            hold_right, hold_left = _resqueeze_at_lift_end(
                raw_env,
                side=side,
                hold_right=hold_right,
                hold_left=hold_left,
                detector=detector,
                object_name=object_name,
                canonical=canonical,
                tpsr_cfg=tpsr_cfg,
            )
            lift_arm = vec_to_arm_action(read_arm_action(raw_env, side))

    if side == "left":
        for _ in range(max(int(hold_steps), 1)):
            settle_bimanual_actions(raw_env, right23=hold_other, left23=lift_arm, n_substeps=1)
        return read_arm_action(raw_env, "left")
    for _ in range(max(int(hold_steps), 1)):
        settle_bimanual_actions(raw_env, right23=lift_arm, left23=hold_other, n_substeps=1)
    return read_arm_action(raw_env, "right")


def execute_tray_lift(
    raw_env,
    *,
    grasp_left: np.ndarray,
    hold_right: np.ndarray,
    lift_ref: dict[str, Any] | None = None,
    detector: AssemblyContactDetector | None = None,
    lift_height_m: float = DEFAULT_TRAY_LIFT_M,
    steps: int | None = None,
    hold_steps: int = 8,
    pre_lift_settle: int = 12,
    lift_exec_cap: int | None = None,
    lock_passive_arm: bool = False,
    object_z_only: bool = False,
    canonical: dict | None = None,
    tpsr_cfg: TpsrConfig | None = None,
) -> np.ndarray:
    return execute_arm_lift(
        raw_env,
        side="left",
        grasp_arm=grasp_left,
        hold_other=hold_right,
        lift_ref=lift_ref,
        detector=detector,
        lift_height_m=lift_height_m,
        steps=steps,
        hold_steps=hold_steps,
        pre_lift_settle=pre_lift_settle,
        lift_exec_cap=lift_exec_cap,
        lock_passive_arm=lock_passive_arm,
        object_z_only=object_z_only,
        canonical=canonical,
        tpsr_cfg=tpsr_cfg,
    )


def execute_peg_lift(
    raw_env,
    *,
    grasp_right: np.ndarray,
    hold_left: np.ndarray,
    lift_ref: dict[str, Any] | None = None,
    detector: AssemblyContactDetector | None = None,
    lift_height_m: float = DEFAULT_PEG_LIFT_M,
    steps: int | None = None,
    hold_steps: int = 8,
    pre_lift_settle: int = 16,
    lift_exec_cap: int | None = None,
    object_z_only: bool = False,
    canonical: dict | None = None,
    tpsr_cfg: TpsrConfig | None = None,
) -> np.ndarray:
    """Per-demo peg lift; left arm locked so tray pose does not drift."""
    return execute_arm_lift(
        raw_env,
        side="right",
        grasp_arm=grasp_right,
        hold_other=hold_left,
        lift_ref=lift_ref,
        detector=detector,
        lift_height_m=lift_height_m,
        steps=steps,
        hold_steps=hold_steps,
        pre_lift_settle=pre_lift_settle,
        lift_exec_cap=lift_exec_cap,
        lock_passive_arm=True,
        object_z_only=object_z_only,
        canonical=canonical,
        tpsr_cfg=tpsr_cfg,
    )


def hold_tray_before_peg(
    raw_env,
    *,
    left_hold: np.ndarray,
    right_home: np.ndarray,
    detector: AssemblyContactDetector,
    lift_ref: dict[str, Any] | None = None,
    hold_steps: int | None = None,
    max_hold_steps: int = 48,
    warmup_steps: int = 10,
) -> tuple[np.ndarray, bool, int]:
    """Demo gap: left holds tray at lift_end while right stays home. Returns min contact in window."""
    left_hold = vec_to_arm_action(left_hold)
    right_home = vec_to_arm_action(right_home)
    lo, hi = _hand_joint_bounds(raw_env._model, "left")
    if hold_steps is None and lift_ref is not None and "tray_hold_steps_before_peg" in lift_ref:
        hold_steps = int(lift_ref["tray_hold_steps_before_peg"])
    steps = max(int(CONTACT_WINDOW) + 2, min(int(hold_steps or 24), int(max_hold_steps)))
    tray_id = raw_env._model.body(TRAY_BODY).id
    rest_z = float(detector._tray_rest_z)  # noqa: SLF001

    for _ in range(max(int(warmup_steps), 0)):
        settle_bimanual_actions(raw_env, right23=right_home, left23=left_hold, n_substeps=1)
        from interaction_retarget.sim.video import maybe_capture_frame

        maybe_capture_frame()

    counts: list[int] = []
    z_delta: list[float] = []
    for _ in range(steps):
        settle_bimanual_actions(raw_env, right23=right_home, left23=left_hold, n_substeps=1)
        maybe_capture_frame()
        if side_contact_count(detector, raw_env, object_name="tray") < _LIFT_MIN_CONTACT:
            left_hold = _close_fingers(left_hold, -0.012, lo, hi)
            settle_bimanual_actions(raw_env, right23=right_home, left23=left_hold, n_substeps=2)
            left_hold = vec_to_arm_action(read_arm_action(raw_env, "left"))
        counts.append(side_contact_count(detector, raw_env, object_name="tray"))
        z_delta.append(float(raw_env._data.xpos[tray_id, 2]) - rest_z)

    recent_c = counts[-int(CONTACT_WINDOW) :] if len(counts) >= int(CONTACT_WINDOW) else counts
    recent_z = z_delta[-int(CONTACT_WINDOW) :] if len(z_delta) >= int(CONTACT_WINDOW) else z_delta
    stable = (
        len(counts) >= int(CONTACT_WINDOW)
        and all(c >= _LIFT_MIN_CONTACT for c in recent_c)
        and all(z >= -0.010 for z in recent_z)
        and float(raw_env._data.xpos[tray_id, 2]) >= rest_z - 0.010
    )
    return vec_to_arm_action(read_arm_action(raw_env, "left")), bool(stable), int(min(recent_c) if recent_c else 0)


def load_lift_reference(sidecar_dir: Path) -> dict[str, Any] | None:
    path = default_demo_lift_path(sidecar_dir)
    if path.is_file():
        return load_demo_lift_reference(path)
    return None
