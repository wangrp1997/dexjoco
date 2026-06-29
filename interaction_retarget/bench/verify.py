"""Hold-in-sim verification with object pose stability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation as R

from interaction_retarget.constants import MIN_GRASP_CONTACT_COUNT, PEG_BODY, TRAY_BODY
from interaction_retarget.grasp.repair import side_contact_count
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.settle import settle_bimanual_actions, vec_to_arm_action
from interaction_retarget.tpsr.config import TpsrConfig
from interaction_retarget.tpsr.constraints import hole_clearance_violation_m
from interaction_retarget.bench.config import BenchConfig

ObjectName = Literal["tray", "peg"]
Side = Literal["left", "right"]


def _object_body(object_name: ObjectName) -> str:
    return TRAY_BODY if object_name == "tray" else PEG_BODY


def _side_for_object(object_name: ObjectName) -> Side:
    return "left" if object_name == "tray" else "right"


def _object_pose(raw_env, object_name: ObjectName) -> tuple[np.ndarray, np.ndarray]:
    bid = int(raw_env._model.body(_object_body(object_name)).id)
    pos = np.asarray(raw_env._data.xpos[bid], dtype=np.float64).copy()
    quat = np.asarray(raw_env._data.xquat[bid], dtype=np.float64).copy()
    return pos, quat


def _pose_delta(
    pos0: np.ndarray, quat0: np.ndarray, pos1: np.ndarray, quat1: np.ndarray
) -> tuple[float, float]:
    trans = float(np.linalg.norm(pos1 - pos0))
    r0 = R.from_quat(quat0[[1, 2, 3, 0]])
    r1 = R.from_quat(quat1[[1, 2, 3, 0]])
    angle = float((r0.inv() * r1).magnitude())
    return trans, angle


@dataclass
class BenchHoldReport:
    object_name: ObjectName
    stable: bool
    contact_ok: bool
    pose_ok: bool
    hole_ok: bool
    min_contact: int
    max_trans_m: float
    max_rot_rad: float
    hole_violation_m: float


def verify_side_hold(
    raw_env,
    *,
    object_name: ObjectName,
    action_right: np.ndarray,
    action_left: np.ndarray,
    detector: AssemblyContactDetector,
    bench_cfg: BenchConfig | None = None,
    tpsr_cfg: TpsrConfig | None = None,
) -> BenchHoldReport:
    """Hold bimanual pose; check sustained contact + object pose drift."""
    bench_cfg = bench_cfg or BenchConfig()
    tpsr_cfg = tpsr_cfg or TpsrConfig()
    side = _side_for_object(object_name)
    radius_m, length_m = (
        (tpsr_cfg.peg_insert_clearance_radius_m, tpsr_cfg.peg_insert_guard_length_m)
        if object_name == "peg"
        else (tpsr_cfg.tray_socket_clearance_radius_m, tpsr_cfg.tray_socket_guard_depth_m)
    )

    right23 = vec_to_arm_action(action_right)
    left23 = vec_to_arm_action(action_left)
    pos0, quat0 = _object_pose(raw_env, object_name)

    counts: list[int] = []
    max_trans = 0.0
    max_rot = 0.0

    for _ in range(max(int(bench_cfg.warmup_steps), 0)):
        settle_bimanual_actions(raw_env, right23=right23, left23=left23, n_substeps=1)

    for _ in range(int(bench_cfg.hold_steps)):
        settle_bimanual_actions(raw_env, right23=right23, left23=left23, n_substeps=1)
        counts.append(side_contact_count(detector, raw_env, object_name=object_name))
        pos1, quat1 = _object_pose(raw_env, object_name)
        trans, rot = _pose_delta(pos0, quat0, pos1, quat1)
        max_trans = max(max_trans, trans)
        max_rot = max(max_rot, rot)

    min_c = min(counts[-bench_cfg.contact_window :] if counts else [0])
    min_contact_gate = max(int(bench_cfg.min_contact_count), MIN_GRASP_CONTACT_COUNT)
    contact_ok = min_c >= min_contact_gate
    pose_ok = (
        max_trans <= bench_cfg.max_object_trans_m and max_rot <= bench_cfg.max_object_rot_rad
    )
    hole_v = hole_clearance_violation_m(
        raw_env,
        object_name=object_name,
        side=side,
        cfg_radius_m=radius_m,
        cfg_length_m=length_m,
    )
    hole_ok = (not bench_cfg.require_hole_clear) or hole_v <= 1e-6
    stable = contact_ok and pose_ok and hole_ok

    return BenchHoldReport(
        object_name=object_name,
        stable=stable,
        contact_ok=contact_ok,
        pose_ok=pose_ok,
        hole_ok=hole_ok,
        min_contact=min_c,
        max_trans_m=max_trans,
        max_rot_rad=max_rot,
        hole_violation_m=hole_v,
    )
