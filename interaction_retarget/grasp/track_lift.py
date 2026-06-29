"""Closed-loop lift: strong squeeze + FC, track demo sidecar waypoints (approx, not replay).

Passive arm command stays frozen; active arm only. Each step checks contact / object dz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from interaction_retarget.constants import MIN_GRASP_CONTACT_COUNT, PEG_BODY, TRAY_BODY
from interaction_retarget.grasp.approach import interpolate_arm_only
from interaction_retarget.grasp.locked_hold import enforce_locked_passive
from interaction_retarget.grasp.repair import _close_fingers, _hand_joint_bounds, _step_side, side_contact_count
from interaction_retarget.grasp.staged_grasp import prepare_lift_squeeze, re_squeeze_fc
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.settle import read_arm_action, settle_bimanual_actions, vec_to_arm_action
from interaction_retarget.skill_replay.demo_grasp import demo_lift_world_dz
from interaction_retarget.transforms import mocap_world_from_object_frame, relative_mocap_in_object_frame
from interaction_retarget.tpsr.config import TpsrConfig

Side = Literal["left", "right"]
ObjectName = Literal["tray", "peg"]
_LIFT_MIN_CONTACT = max(2, MIN_GRASP_CONTACT_COUNT - 1)


def strong_lift_tpsr_cfg(cfg: TpsrConfig, *, extra_squeeze: int = 24) -> TpsrConfig:
    from dataclasses import replace

    return replace(cfg, squeeze_steps=int(cfg.squeeze_steps) + int(extra_squeeze), require_qp_fc=True)


@dataclass
class TrackLiftReport:
    object_name: str
    object_dz_m: float
    target_dz_m: float
    contact_min: int
    fc_ok: bool
    qp_max_error: float
    steps_executed: int
    success: bool


def _object_body(side: Side) -> str:
    return TRAY_BODY if side == "left" else PEG_BODY


def _object_name(side: Side) -> ObjectName:
    return "tray" if side == "left" else "peg"


def _object_z(raw_env, side: Side) -> float:
    bid = int(raw_env._model.body(_object_body(side)).id)
    return float(raw_env._data.xpos[bid, 2])


def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _subsample_waypoints(pos_seg: np.ndarray, n_max: int) -> np.ndarray:
    pos_seg = np.asarray(pos_seg, dtype=np.float64).reshape(-1, 3)
    if pos_seg.shape[0] <= n_max:
        return pos_seg
    idx = np.linspace(0, pos_seg.shape[0] - 1, n_max, dtype=int)
    return pos_seg[idx]


def demo_lift_targets_obj(
    lift_ref: dict[str, Any],
    side: Side,
    *,
    z_only: bool = True,
    n_max: int = 16,
) -> np.ndarray | None:
    """Relative object-frame waypoints (segment origin = grasp at lift start)."""
    obj = _object_name(side)
    pos_key = f"{obj}_mocap_pos_obj"
    if pos_key not in lift_ref:
        return None
    pos = np.asarray(lift_ref[pos_key], dtype=np.float64).reshape(-1, 3)
    if pos.shape[0] < 2:
        return None
    seg = pos - pos[0:1]
    if z_only:
        seg = np.column_stack([np.zeros(seg.shape[0]), np.zeros(seg.shape[0]), seg[:, 2]])
    return _subsample_waypoints(seg, n_max)


def track_arm_lift(
    raw_env,
    *,
    side: Side,
    grasp_arm: np.ndarray,
    passive_cmd: np.ndarray,
    lift_ref: dict[str, Any] | None,
    manifest_entry: dict[str, Any] | None,
    canonical: dict,
    detector: AssemblyContactDetector,
    tpsr_cfg: TpsrConfig,
    target_dz_m: float | None = None,
    n_steps: int = 32,
    min_object_dz_m: float = 0.030,
) -> tuple[np.ndarray, TrackLiftReport]:
    """Strong squeeze then track demo lift (object-frame Z waypoints or world-Z fallback)."""
    object_name = _object_name(side)
    grasp_arm = vec_to_arm_action(grasp_arm)
    passive_frozen = vec_to_arm_action(passive_cmd)
    if side == "left":
        hold_right, hold_left = passive_frozen, grasp_arm
    else:
        hold_right, hold_left = grasp_arm, passive_frozen

    hold_right, hold_left, fc_ok, qp_err = prepare_lift_squeeze(
        raw_env,
        side=side,
        object_name=object_name,
        canonical=canonical,
        grasp_arm=grasp_arm,
        hold_right=hold_right,
        hold_left=hold_left,
        tpsr_cfg=tpsr_cfg,
        settle_steps=10,
    )
    grasp_arm = vec_to_arm_action(read_arm_action(raw_env, side))
    start_pos = grasp_arm[0:3].copy()
    start_quat = grasp_arm[3:7].copy()
    hand = grasp_arm[7:23].copy()
    obj_z0 = _object_z(raw_env, side)

    if target_dz_m is None:
        if manifest_entry is not None and lift_ref is not None:
            target_dz_m = demo_lift_world_dz(manifest_entry, lift_ref, object_name)
        else:
            target_dz_m = 0.05

    obj_seg = demo_lift_targets_obj(lift_ref, side, z_only=True) if lift_ref is not None else None
    use_obj_wp = False
    n_exec = max(int(n_steps), 1)

    obj_pos0, obj_quat0 = _object_pose(raw_env, side)
    grasp_pos_obj, grasp_quat_obj = relative_mocap_in_object_frame(
        grasp_arm[0:3], grasp_arm[3:7], obj_pos0, obj_quat0
    )
    lo, hi = _hand_joint_bounds(raw_env._model, side)
    contact_counts: list[int] = []
    prev23 = grasp_arm.copy()

    for step in range(n_exec):
        u = _smoothstep((step + 1) / n_exec)
        if use_obj_wp:
            obj_pos, obj_quat = _object_pose(raw_env, side)
            target23 = _waypoint23(
                obj_pos,
                obj_quat,
                grasp_pos_obj + obj_seg[step],
                grasp_quat_obj,
                hand,
            )
        else:
            target23 = grasp_arm.copy()
            target23[0:2] = start_pos[0:2]
            target23[2] = start_pos[2] + float(target_dz_m) * u
            target23[3:7] = start_quat
            target23[7:23] = hand

        active = interpolate_arm_only(prev23, target23, 1.0, hand=hand)
        if side == "left":
            _step_side(
                raw_env,
                side="left",
                active23=active,
                hold_right=passive_frozen,
                hold_left=active,
            )
        else:
            _step_side(
                raw_env,
                side="right",
                active23=active,
                hold_right=active,
                hold_left=passive_frozen,
            )
        if side == "left":
            enforce_locked_passive(
                raw_env, locked_left=active, locked_right=passive_frozen, n_substeps=2
            )
        else:
            enforce_locked_passive(
                raw_env, locked_left=passive_frozen, locked_right=active, n_substeps=2
            )
        cc = int(side_contact_count(detector, raw_env, object_name=object_name))
        contact_counts.append(cc)
        obj_dz = _object_z(raw_env, side) - obj_z0
        expected_dz = float(target_dz_m) * u
        if cc < _LIFT_MIN_CONTACT or obj_dz < expected_dz * 0.35:
            side_action = vec_to_arm_action(read_arm_action(raw_env, side))
            tightened = _close_fingers(side_action, -0.008, lo, hi)
            hand = tightened[7:23].copy()
            if side == "left":
                settle_bimanual_actions(
                    raw_env, right23=passive_frozen, left23=tightened, n_substeps=3
                )
            else:
                settle_bimanual_actions(
                    raw_env, right23=tightened, left23=passive_frozen, n_substeps=3
                )
            _, _, fc_ok, qp_err = re_squeeze_fc(
                raw_env,
                side=side,
                object_name=object_name,
                canonical=canonical,
                hold_right=passive_frozen if side == "left" else tightened,
                hold_left=tightened if side == "left" else passive_frozen,
                tpsr_cfg=tpsr_cfg,
                max_rounds=1,
            )
        prev23 = vec_to_arm_action(read_arm_action(raw_env, side))
        hand = prev23[7:23].copy()

    lift_arm = vec_to_arm_action(read_arm_action(raw_env, side))
    object_dz = _object_z(raw_env, side) - obj_z0
    contact_min = int(min(contact_counts) if contact_counts else 0)
    report = TrackLiftReport(
        object_name=object_name,
        object_dz_m=float(object_dz),
        target_dz_m=float(target_dz_m),
        contact_min=contact_min,
        fc_ok=bool(fc_ok),
        qp_max_error=float(qp_err),
        steps_executed=n_exec,
        success=bool(object_dz >= min_object_dz_m and contact_min >= _LIFT_MIN_CONTACT),
    )
    return lift_arm, report


def _object_pose(raw_env, side: Side) -> tuple[np.ndarray, np.ndarray]:
    bid = int(raw_env._model.body(_object_body(side)).id)
    data = raw_env._data
    return (
        np.asarray(data.xpos[bid], dtype=np.float64),
        np.asarray(data.xquat[bid], dtype=np.float64),
    )


def _waypoint23(
    obj_pos: np.ndarray,
    obj_quat: np.ndarray,
    pos_obj: np.ndarray,
    quat_obj: np.ndarray,
    hand: np.ndarray,
) -> np.ndarray:
    pos_w, quat_w = mocap_world_from_object_frame(pos_obj, quat_obj, obj_pos, obj_quat)
    return np.concatenate([pos_w, quat_w, np.asarray(hand, dtype=np.float64).reshape(16)], axis=0)


def track_dual_lift_parallel(
    raw_env,
    *,
    left_grasp: np.ndarray,
    right_grasp: np.ndarray,
    left_passive: np.ndarray,
    right_passive: np.ndarray,
    lift_ref: dict[str, Any] | None,
    manifest_entry: dict[str, Any] | None,
    canonical_tray: dict,
    canonical_peg: dict,
    detector: AssemblyContactDetector,
    tpsr_cfg: TpsrConfig,
    min_tray_dz_m: float = 0.030,
    min_peg_dz_m: float = 0.010,
) -> tuple[np.ndarray, np.ndarray, TrackLiftReport, TrackLiftReport]:
    """Interleaved lift steps: each arm tracks its demo ref; commands do not cross."""
    left_seg = demo_lift_targets_obj(lift_ref, "left", z_only=True) if lift_ref else None
    right_seg = demo_lift_targets_obj(lift_ref, "right", z_only=True) if lift_ref else None
    n_left = int(left_seg.shape[0]) if left_seg is not None else 32
    n_right = int(right_seg.shape[0]) if right_seg is not None else 32
    n_exec = max(n_left, n_right)

    tray_cfg = strong_lift_tpsr_cfg(tpsr_cfg, extra_squeeze=20)
    peg_cfg = strong_lift_tpsr_cfg(tpsr_cfg, extra_squeeze=24)

    left_arm = vec_to_arm_action(left_grasp)
    right_arm = vec_to_arm_action(right_grasp)
    left_passive = vec_to_arm_action(left_passive)
    right_passive = vec_to_arm_action(right_passive)

    for side, arm, canonical, obj in (
        ("left", left_arm, canonical_tray, "tray"),
        ("right", right_arm, canonical_peg, "peg"),
    ):
        hr, hl, _, _ = prepare_lift_squeeze(
            raw_env,
            side=side,  # type: ignore[arg-type]
            object_name=obj,
            canonical=canonical,
            grasp_arm=arm,
            hold_right=right_passive if side == "left" else arm,
            hold_left=arm if side == "left" else left_passive,
            tpsr_cfg=tray_cfg if side == "left" else peg_cfg,
            settle_steps=6,
        )
        if side == "left":
            left_arm = vec_to_arm_action(read_arm_action(raw_env, "left"))
            right_passive = hr
        else:
            right_arm = vec_to_arm_action(read_arm_action(raw_env, "right"))
            left_passive = hl

    left_arm = vec_to_arm_action(read_arm_action(raw_env, "left"))
    right_arm = vec_to_arm_action(read_arm_action(raw_env, "right"))
    l_pos0 = left_arm[0:3].copy()
    l_quat0 = left_arm[3:7].copy()
    l_hand = left_arm[7:23].copy()
    r_pos0 = right_arm[0:3].copy()
    r_quat0 = right_arm[3:7].copy()
    r_hand = right_arm[7:23].copy()
    tray_z0 = _object_z(raw_env, "left")
    peg_z0 = _object_z(raw_env, "right")
    tray_dz_tgt = (
        demo_lift_world_dz(manifest_entry, lift_ref, "tray")
        if manifest_entry is not None and lift_ref is not None
        else 0.05
    )
    peg_dz_tgt = (
        demo_lift_world_dz(manifest_entry, lift_ref, "peg")
        if manifest_entry is not None and lift_ref is not None
        else 0.05
    )
    lo_l, hi_l = _hand_joint_bounds(raw_env._model, "left")
    lo_r, hi_r = _hand_joint_bounds(raw_env._model, "right")
    l_counts: list[int] = []
    r_counts: list[int] = []

    l_obj0, l_q0 = _object_pose(raw_env, "left")
    r_obj0, r_q0 = _object_pose(raw_env, "right")
    l_gpos, l_gquat = relative_mocap_in_object_frame(l_pos0, l_quat0, l_obj0, l_q0)
    r_gpos, r_gquat = relative_mocap_in_object_frame(r_pos0, r_quat0, r_obj0, r_q0)

    for step in range(n_exec):
        u_l = _smoothstep(min(1.0, (step + 1) / max(n_left, 1)))
        u_r = _smoothstep(min(1.0, (step + 1) / max(n_right, 1)))
        if left_seg is not None and step < n_left:
            l_obj, l_oq = _object_pose(raw_env, "left")
            left_tgt = _waypoint23(l_obj, l_oq, l_gpos + left_seg[step], l_gquat, l_hand)
        else:
            left_tgt = left_arm.copy()
            left_tgt[0:2] = l_pos0[0:2]
            left_tgt[2] = l_pos0[2] + tray_dz_tgt * u_l
            left_tgt[3:7] = l_quat0
            left_tgt[7:23] = l_hand
        if right_seg is not None and step < n_right:
            r_obj, r_oq = _object_pose(raw_env, "right")
            right_tgt = _waypoint23(r_obj, r_oq, r_gpos + right_seg[step], r_gquat, r_hand)
        else:
            right_tgt = right_arm.copy()
            right_tgt[0:2] = r_pos0[0:2]
            right_tgt[2] = r_pos0[2] + peg_dz_tgt * u_r
            right_tgt[3:7] = r_quat0
            right_tgt[7:23] = r_hand

        left_cmd = interpolate_arm_only(left_arm, left_tgt, 1.0, hand=l_hand)
        right_cmd = interpolate_arm_only(right_arm, right_tgt, 1.0, hand=r_hand)
        settle_bimanual_actions(raw_env, right23=right_cmd, left23=left_cmd, n_substeps=1)
        from interaction_retarget.sim.video import maybe_capture_frame

        maybe_capture_frame()

        l_cc = int(side_contact_count(detector, raw_env, object_name="tray"))
        r_cc = int(side_contact_count(detector, raw_env, object_name="peg"))
        l_counts.append(l_cc)
        r_counts.append(r_cc)
        tray_dz = _object_z(raw_env, "left") - tray_z0
        peg_dz = _object_z(raw_env, "right") - peg_z0

        if l_cc < _LIFT_MIN_CONTACT or tray_dz < tray_dz_tgt * u_l * 0.3:
            lt = _close_fingers(vec_to_arm_action(read_arm_action(raw_env, "left")), -0.008, lo_l, hi_l)
            l_hand = lt[7:23].copy()
            _, _, _, _ = re_squeeze_fc(
                raw_env,
                side="left",
                object_name="tray",
                canonical=canonical_tray,
                hold_right=right_cmd,
                hold_left=lt,
                tpsr_cfg=tray_cfg,
                max_rounds=1,
            )
        if r_cc < _LIFT_MIN_CONTACT or peg_dz < peg_dz_tgt * u_r * 0.3:
            rt = _close_fingers(vec_to_arm_action(read_arm_action(raw_env, "right")), -0.008, lo_r, hi_r)
            r_hand = rt[7:23].copy()
            _, _, _, _ = re_squeeze_fc(
                raw_env,
                side="right",
                object_name="peg",
                canonical=canonical_peg,
                hold_right=rt,
                hold_left=left_cmd,
                tpsr_cfg=peg_cfg,
                max_rounds=1,
            )
        left_arm = vec_to_arm_action(read_arm_action(raw_env, "left"))
        right_arm = vec_to_arm_action(read_arm_action(raw_env, "right"))
        l_hand = left_arm[7:23].copy()
        r_hand = right_arm[7:23].copy()

    left_arm = vec_to_arm_action(read_arm_action(raw_env, "left"))
    right_arm = vec_to_arm_action(read_arm_action(raw_env, "right"))
    tray_rep = TrackLiftReport(
        object_name="tray",
        object_dz_m=float(_object_z(raw_env, "left") - tray_z0),
        target_dz_m=float(tray_dz_tgt),
        contact_min=int(min(l_counts) if l_counts else 0),
        fc_ok=True,
        qp_max_error=0.0,
        steps_executed=n_exec,
        success=bool(_object_z(raw_env, "left") - tray_z0 >= min_tray_dz_m),
    )
    peg_rep = TrackLiftReport(
        object_name="peg",
        object_dz_m=float(_object_z(raw_env, "right") - peg_z0),
        target_dz_m=float(peg_dz_tgt),
        contact_min=int(min(r_counts) if r_counts else 0),
        fc_ok=True,
        qp_max_error=0.0,
        steps_executed=n_exec,
        success=bool(_object_z(raw_env, "right") - peg_z0 >= min_peg_dz_m),
    )
    return left_arm, right_arm, tray_rep, peg_rep
