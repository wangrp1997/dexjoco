#!/usr/bin/env python3
"""Strict action-only privileged PBVS after demo handoff.

After peg_lift_end this runner may read privileged geometry, but it can only
advance simulation through 44D robot actions. Direct free-joint writes are
blocked for the remainder of each rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "dexjoco", ROOT / "embodied_grasp_insertion", ROOT.parent / "reach_insert_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.physics.grasp_metrics import peg_hand_contact_counts
from embodied_grasp_insertion.physics.grasp_metrics import REFERENCE_BODY
from embodied_grasp_insertion.simulation.full_episode_utils import make_full_env
from hybrid_insert.geometry import toward_socket_delta, wrist_rotvec_align_peg_axis
from interaction_retarget.sim.replay import (
    policy_dual_arm_to_raw,
    raw_flat_to_dict,
    rotvec_dual_arm_to_policy,
    zarr_action_to_policy46,
)
from interaction_retarget.sim.settle import settle_bimanual_actions
from reach_insert_rl.env.full_obs import current_action44, privileged_full_features
from reach_insert_rl.env.handoff_env import load_manifest_entries
from dexjoco.sim.envs import panda_bimanual_assembly_env as assembly_env_module
from dexjoco.sim.envs.assembly_geometry import names_from_raw

assembly_env_module.time.sleep = lambda _seconds: None

DEFAULT_SIDECAR = Path("/mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly")
DEFAULT_OUT = Path("/mnt/hdd/dexjoco/outputs/privileged_insert_servo_strict")
LEFT_PALM = "allegro_palm_left"


def _handoff_frame(sidecar: Path, episode: int) -> int:
    cache = sidecar / "skill_replay_cache" / f"episode_{int(episode):03d}_lift_ref.json"
    if cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        return int(payload["frames"]["peg_lift_end_frame"])
    from pose_insert.pre_insert import resolve_peg_lift_end_frame

    return int(resolve_peg_lift_end_frame({"episode_index": int(episode)}, sidecar))


def _replay_to_frame(env, frame: int) -> dict[str, Any]:
    assert env._actions is not None
    last: dict[str, Any] = {}
    for step in range(int(frame)):
        if step >= len(env._actions):
            break
        demo_action = env._actions[step]
        try:
            env._env.step(zarr_action_to_policy46(demo_action))
        except Exception:
            env._raw.step(raw_flat_to_dict(demo_action))
        env._t = step + 1
        env._hold44 = current_action44(env._raw)
        last = _measure(env, k=step, phase="demo_replay")
    return last


@dataclass
class StrictGains:
    standoff_m: float = 0.055
    target_along_m: float = -0.030
    coarse_step_m: float = 0.004
    fine_step_m: float = 0.0015
    insert_step_m: float = 0.0015
    lat_gate_m: float = 0.006
    insert_lat_gate_m: float = 0.009
    axis_gate_rad: float = 0.075
    rot_gain: float = 0.28
    max_rot_step_rad: float = 0.014
    hand_pos_step_m: float = 0.006
    hand_rot_step_rad: float = 0.020
    left_pos_step_m: float = 0.003
    left_rot_step_rad: float = 0.012
    settle_substeps: int = 40
    insert_substeps: int = 24
    jam_patience: int = 18
    unjam_steps: int = 16
    inverse_max_slip_m: float = 0.030
    inverse_max_slip_rad: float = 0.45
    max_steps: int = 700


class IllegalStateWrite(RuntimeError):
    pass


@dataclass
class ObjectAnchor:
    desired_pos: np.ndarray
    desired_rot: R
    object_in_hand_pos: np.ndarray
    object_in_hand_rot: R
    tip_in_object: np.ndarray | None = None


def _body_pose(raw, body_name: str) -> tuple[np.ndarray, R]:
    body_id = int(raw._model.body(body_name).id)
    pos = np.asarray(raw._data.xpos[body_id], dtype=np.float64).copy()
    rot = R.from_quat(
        np.asarray(raw._data.xquat[body_id], dtype=np.float64), scalar_first=True
    )
    return pos, rot


def _mocap_pose(raw, mocap_id: int) -> tuple[np.ndarray, R]:
    pos = np.asarray(raw._data.mocap_pos[int(mocap_id)], dtype=np.float64).copy()
    rot = R.from_quat(
        np.asarray(raw._data.mocap_quat[int(mocap_id)], dtype=np.float64),
        scalar_first=True,
    )
    return pos, rot


def _relative_pose(
    parent_pos: np.ndarray,
    parent_rot: R,
    child_pos: np.ndarray,
    child_rot: R,
) -> tuple[np.ndarray, R]:
    return parent_rot.inv().apply(child_pos - parent_pos), parent_rot.inv() * child_rot


def _inverse_relative_target(
    object_pos: np.ndarray,
    object_rot: R,
    object_in_hand_pos: np.ndarray,
    object_in_hand_rot: R,
) -> tuple[np.ndarray, R]:
    hand_rot = object_rot * object_in_hand_rot.inv()
    hand_pos = object_pos - hand_rot.apply(object_in_hand_pos)
    return hand_pos, hand_rot


def _compose_pose(
    parent_pos: np.ndarray,
    parent_rot: R,
    child_pos: np.ndarray,
    child_rot: R,
) -> tuple[np.ndarray, R]:
    return parent_pos + parent_rot.apply(child_pos), parent_rot * child_rot


def _bounded_pose(
    current_pos: np.ndarray,
    current_rot: R,
    target_pos: np.ndarray,
    target_rot: R,
    *,
    max_pos_m: float,
    max_rot_rad: float,
) -> tuple[np.ndarray, R]:
    delta = np.asarray(target_pos, dtype=np.float64) - np.asarray(
        current_pos, dtype=np.float64
    )
    distance = float(np.linalg.norm(delta))
    if distance > float(max_pos_m):
        delta *= float(max_pos_m) / distance
    rotation_delta = (target_rot * current_rot.inv()).as_rotvec()
    angle = float(np.linalg.norm(rotation_delta))
    if angle > float(max_rot_rad):
        rotation_delta *= float(max_rot_rad) / angle
    return current_pos + delta, R.from_rotvec(rotation_delta) * current_rot


def _mocap_from_body_target(
    mocap_pos: np.ndarray,
    mocap_rot: R,
    body_pos: np.ndarray,
    body_rot: R,
    target_body_pos: np.ndarray,
    target_body_rot: R,
) -> tuple[np.ndarray, R]:
    position = mocap_pos + (target_body_pos - body_pos)
    rotation = (target_body_rot * body_rot.inv()) * mocap_rot
    return position, rotation


def _block_object_teleport(raw):
    original = raw._set_free_joint_pose

    def blocked(*_args, **_kwargs):
        raise IllegalStateWrite("free-joint write attempted after handoff")

    raw._set_free_joint_pose = blocked
    return original


def _apply_action(
    env,
    action44: np.ndarray,
    *,
    n_substeps: int,
    capture_frame: Callable[[], None] | None = None,
) -> None:
    raw_action = policy_dual_arm_to_raw(rotvec_dual_arm_to_policy(action44))
    settle_bimanual_actions(
        env._raw,
        right23=np.asarray(raw_action["right"], dtype=np.float64),
        left23=np.asarray(raw_action["left"], dtype=np.float64),
        n_substeps=int(n_substeps),
    )
    env._hold44 = np.asarray(action44, dtype=np.float64).copy()
    env._t += 1
    if capture_frame is not None:
        capture_frame()


def _clip_right_fingers(raw, fingers: np.ndarray) -> np.ndarray:
    output = np.asarray(fingers, dtype=np.float64).reshape(16).copy()
    actuator_ids = np.asarray(raw._allegro_ctrl_ids, dtype=int)[:16]
    for index, actuator_id in enumerate(actuator_ids):
        low, high = raw._model.actuator_ctrlrange[int(actuator_id)]
        output[index] = float(np.clip(output[index], low, high))
    return output


def _cage_target(raw, fingers: np.ndarray) -> np.ndarray:
    target = np.asarray(fingers, dtype=np.float64).reshape(16).copy()
    target[4] = float(np.clip(target[4] + 0.05, -0.47, 0.47))
    target[5] = max(float(target[5]), 1.50)
    target[6] = max(float(target[6]), 1.42)
    target[7] = max(float(target[7]), 0.85)
    target[8] = float(np.clip(target[8] + 0.06, -0.47, 0.47))
    target[9] = max(float(target[9]), 1.42)
    target[10] = max(float(target[10]), 1.32)
    target[11] = max(float(target[11]), 0.55)
    return _clip_right_fingers(raw, target)


def _establish_cage(
    env,
    hold: np.ndarray,
    left_hold: np.ndarray,
    *,
    capture_frame: Callable[[], None] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    start = _clip_right_fingers(env._raw, hold[6:22])
    initial_contacts = peg_hand_contact_counts(env._raw)
    initial_classes = {
        key: int(value) for key, value in initial_contacts.by_class.items()
    }
    if (
        int(initial_contacts.total) >= 4
        and initial_classes.get("index", 0) == 1
        and initial_classes.get("thumb", 0) >= 2
    ):
        row = _measure(env, k=0, phase="keep_pinch")
        row["hand_contacts"] = int(initial_contacts.total)
        row["contact_classes"] = initial_classes
        hold[6:22] = start
        return hold, [row]
    target = _cage_target(env._raw, start)
    rows: list[dict[str, Any]] = []
    best = start.copy()
    best_contacts = int(peg_hand_contact_counts(env._raw).total)
    for step in range(12):
        alpha = float(step + 1) / 12.0
        action = hold.copy()
        action[6:22] = (1.0 - alpha) * start + alpha * target
        action[22:44] = left_hold
        _apply_action(env, action, n_substeps=12, capture_frame=capture_frame)
        contacts = peg_hand_contact_counts(env._raw)
        row = _measure(env, k=step, phase="cage")
        row["hand_contacts"] = int(contacts.total)
        row["contact_classes"] = {key: int(value) for key, value in contacts.by_class.items()}
        rows.append(row)
        if row["peg_ok"] and int(contacts.total) >= best_contacts:
            best_contacts = int(contacts.total)
            best = action[6:22].copy()
        if not row["peg_ok"]:
            break
    hold[6:22] = best
    return hold, rows


def _stabilize_left(
    raw,
    action: np.ndarray,
    anchor: ObjectAnchor,
    gains: StrictGains,
) -> None:
    names = names_from_raw(raw)
    socket_pos, socket_rot = _body_pose(raw, names.socket_body)
    palm_pos, palm_rot = _body_pose(raw, LEFT_PALM)
    mocap_pos, mocap_rot = _mocap_pose(raw, int(raw._mocap_left_id))
    socket_in_palm_pos, socket_in_palm_rot = _relative_pose(
        palm_pos, palm_rot, socket_pos, socket_rot
    )
    target_palm_pos, target_palm_rot = _inverse_relative_target(
        anchor.desired_pos,
        anchor.desired_rot,
        socket_in_palm_pos,
        socket_in_palm_rot,
    )
    target_mocap_pos, target_mocap_rot = _mocap_from_body_target(
        mocap_pos,
        mocap_rot,
        palm_pos,
        palm_rot,
        target_palm_pos,
        target_palm_rot,
    )
    target_mocap_pos, target_mocap_rot = _bounded_pose(
        mocap_pos,
        mocap_rot,
        target_mocap_pos,
        target_mocap_rot,
        max_pos_m=gains.left_pos_step_m,
        max_rot_rad=gains.left_rot_step_rad,
    )
    action[22:25] = target_mocap_pos
    action[25:28] = target_mocap_rot.as_rotvec()


def _incremental_wrist_command(
    raw,
    action: np.ndarray,
    *,
    tip: np.ndarray,
    target: np.ndarray,
    peg_axis: np.ndarray,
    hole: np.ndarray,
    rotate: bool,
    max_step: float,
    gains: StrictGains,
) -> None:
    delta = toward_socket_delta(tip, target, gain=1.0, max_step_m=max_step)
    if rotate:
        aligned = wrist_rotvec_align_peg_axis(
            peg_axis,
            hole,
            action[3:6],
            angle_tol_rad=0.025,
            gain=gains.rot_gain,
            max_step_rad=gains.max_rot_step_rad,
        )
        if aligned is not None:
            old_rot = R.from_rotvec(action[3:6])
            new_rot = R.from_rotvec(aligned)
            palm_pos, _ = _body_pose(raw, REFERENCE_BODY)
            predicted_tip = palm_pos + (new_rot * old_rot.inv()).apply(
                tip - palm_pos
            )
            delta += tip - predicted_tip
            norm = float(np.linalg.norm(delta))
            if norm > gains.fine_step_m * 1.5:
                delta *= gains.fine_step_m * 1.5 / norm
            action[3:6] = aligned
    action[:3] += delta


def _command(
    raw,
    hold: np.ndarray,
    gains: StrictGains,
    *,
    jam_steps: int,
    socket_anchor: ObjectAnchor,
    grasp_anchor: ObjectAnchor,
    force_incremental: bool,
    allow_dynamic_switch: bool,
) -> tuple[np.ndarray, str]:
    feat = privileged_full_features(raw, target_along_m=gains.target_along_m)
    tip = np.asarray(feat["tip"], dtype=np.float64)
    socket = np.asarray(feat["socket"], dtype=np.float64)
    hole = np.asarray(feat["hole"], dtype=np.float64)
    hole_u = hole / (np.linalg.norm(hole) + 1e-8)
    peg_axis = np.asarray(feat["peg_axis"], dtype=np.float64)
    lat = float(feat["lat_err"])
    along = float(feat["along"])
    axis = float(feat["axis_err"])

    action = np.asarray(hold, dtype=np.float64).copy()
    rotate = False
    if lat > gains.lat_gate_m:
        phase = "center"
        target = socket + hole_u * max(gains.standoff_m, along)
        max_step = gains.coarse_step_m if lat > 0.018 else gains.fine_step_m
    elif jam_steps >= gains.jam_patience and along < gains.standoff_m + 0.012:
        phase = "unjam"
        angle = 0.55 * float(jam_steps - gains.jam_patience)
        tangent = np.asarray(feat["lat_vec"], dtype=np.float64)
        tangent -= hole_u * float(np.dot(tangent, hole_u))
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-7:
            ref = np.array([1.0, 0.0, 0.0])
            tangent = np.cross(hole_u, ref)
            if np.linalg.norm(tangent) < 1e-7:
                tangent = np.cross(hole_u, np.array([0.0, 1.0, 0.0]))
            tangent /= np.linalg.norm(tangent) + 1e-8
        else:
            tangent /= norm
        tangent2 = np.cross(hole_u, tangent)
        radius = min(0.0010 + 0.00006 * (jam_steps - gains.jam_patience), 0.003)
        target = socket + hole_u * max(along + 0.005, 0.014)
        target += radius * (np.cos(angle) * tangent + np.sin(angle) * tangent2)
        max_step = gains.fine_step_m
    elif axis > gains.axis_gate_rad:
        phase = "axis"
        target = socket + hole_u * gains.standoff_m
        max_step = gains.fine_step_m
        rotate = True
    elif along > gains.standoff_m + 0.004:
        phase = "approach"
        target = socket + hole_u * gains.standoff_m
        max_step = gains.coarse_step_m
    elif lat > gains.insert_lat_gate_m:
        phase = "rim_center"
        target = socket + hole_u * max(along, 0.020)
        max_step = gains.fine_step_m
    else:
        phase = "insert"
        target = socket + hole_u * gains.target_along_m
        max_step = gains.insert_step_m

    target_tip = tip + toward_socket_delta(
        tip, target, gain=1.0, max_step_m=max_step
    )
    palm_pos, palm_rot = _body_pose(raw, REFERENCE_BODY)
    mocap_pos, mocap_rot = _mocap_pose(raw, int(raw._mocap_right_id))
    names = names_from_raw(raw)
    peg_pos, peg_rot = _body_pose(raw, names.peg_body)
    peg_in_palm_pos, peg_in_palm_rot = _relative_pose(
        palm_pos, palm_rot, peg_pos, peg_rot
    )
    grasp_slip_m = float(
        np.linalg.norm(peg_in_palm_pos - grasp_anchor.object_in_hand_pos)
    )
    grasp_slip_rad = float(
        np.linalg.norm(
            (peg_in_palm_rot * grasp_anchor.object_in_hand_rot.inv()).as_rotvec()
        )
    )
    slipped = (
        grasp_slip_m > gains.inverse_max_slip_m
        or grasp_slip_rad > gains.inverse_max_slip_rad
    )
    use_incremental = bool(force_incremental or slipped)
    if use_incremental:
        _incremental_wrist_command(
            raw,
            action,
            tip=tip,
            target=target,
            peg_axis=peg_axis,
            hole=hole,
            rotate=rotate,
            max_step=max_step,
            gains=gains,
        )
        return action, f"{phase}_incremental"
    tip_in_peg = peg_rot.inv().apply(tip - peg_pos)
    target_peg_rot = peg_rot
    if rotate:
        target_axis = hole_u
        desired_axis = peg_axis
        if float(np.dot(desired_axis, target_axis)) < 0.0:
            target_axis = -target_axis
        cross = np.cross(desired_axis, target_axis)
        sine = float(np.linalg.norm(cross))
        cosine = float(np.clip(np.dot(desired_axis, target_axis), -1.0, 1.0))
        if sine > 1e-9:
            angle = min(
                float(np.arctan2(sine, cosine)) * gains.rot_gain,
                gains.max_rot_step_rad,
            )
            target_peg_rot = R.from_rotvec(cross / sine * angle) * target_peg_rot

    target_tip = tip + toward_socket_delta(
        tip, target, gain=1.0, max_step_m=max_step
    )
    target_peg_pos = target_tip - target_peg_rot.apply(tip_in_peg)
    target_palm_pos, target_palm_rot = _inverse_relative_target(
        target_peg_pos,
        target_peg_rot,
        peg_in_palm_pos,
        peg_in_palm_rot,
    )
    target_mocap_pos, target_mocap_rot = _mocap_from_body_target(
        mocap_pos,
        mocap_rot,
        palm_pos,
        palm_rot,
        target_palm_pos,
        target_palm_rot,
    )
    target_mocap_pos, target_mocap_rot = _bounded_pose(
        mocap_pos,
        mocap_rot,
        target_mocap_pos,
        target_mocap_rot,
        max_pos_m=gains.hand_pos_step_m,
        max_rot_rad=gains.hand_rot_step_rad,
    )
    action[:3] = target_mocap_pos
    action[3:6] = target_mocap_rot.as_rotvec()
    grasp_anchor.desired_pos = target_peg_pos
    grasp_anchor.desired_rot = target_peg_rot
    del socket_anchor
    return action, phase


def _measure(env, *, k: int, phase: str) -> dict[str, Any]:
    feat = privileged_full_features(env._raw)
    outcome = env._labeler.compute(env._raw)
    return {
        "k": int(k),
        "phase": phase,
        "insert_ok": bool(outcome.insert_ok),
        "peg_ok": bool(outcome.peg_ok),
        "tray_ok": bool(outcome.tray_ok),
        "tip_dist_m": float(feat["tip_dist"]),
        "lat_err_m": float(feat["lat_err"]),
        "along_m": float(feat["along"]),
        "axis_err": float(feat["axis_err"]),
    }


def _release_and_verify(
    env,
    hold: np.ndarray,
    left_hold: np.ndarray,
    *,
    capture_frame: Callable[[], None] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    start_fingers = hold[6:22].copy()
    for i in range(12):
        action = hold.copy()
        action[6:22] = start_fingers * (1.0 - float(i + 1) / 12.0)
        action[22:44] = left_hold
        _apply_action(env, action, n_substeps=8, capture_frame=capture_frame)
        rows.append(_measure(env, k=i, phase="release"))
        hold = action
    feat = privileged_full_features(env._raw)
    hole_u = np.asarray(feat["hole"], dtype=np.float64)
    hole_u /= np.linalg.norm(hole_u) + 1e-8
    stable_streak = 0
    for i in range(40):
        action = hold.copy()
        action[:3] += hole_u * 0.003
        action[6:22] = 0.0
        action[22:44] = left_hold
        _apply_action(env, action, n_substeps=8, capture_frame=capture_frame)
        row = _measure(env, k=12 + i, phase="retract_verify")
        rows.append(row)
        hold = action
        stable_frame = bool(
            row["insert_ok"]
            and row["lat_err_m"] <= 0.014
            and row["along_m"] < 0.005
        )
        stable_streak = stable_streak + 1 if stable_frame else 0
        if stable_streak >= 12:
            return True, rows
    return False, rows


def run_episode(
    env,
    episode: int,
    gains: StrictGains,
    *,
    capture_frame: Callable[[], None] | None = None,
) -> dict[str, Any]:
    env.reset(episode_index=int(episode))
    env._raw.hz = 0
    handoff = _handoff_frame(Path(env.sidecar_dir), int(episode))
    handoff_info = _replay_to_frame(env, handoff)
    if capture_frame is not None:
        for _ in range(15):
            capture_frame()
    hold = current_action44(env._raw).copy()
    left_hold = hold[22:44].copy()
    original_set_pose = _block_object_teleport(env._raw)
    rows: list[dict[str, Any]] = []
    best_tip = float("inf")
    best_along = float("inf")
    previous_along = float(privileged_full_features(env._raw)["along"])
    jam_steps = 0
    success = False
    fail_reason = "max_steps"
    try:
        initial_contacts = peg_hand_contact_counts(env._raw)
        force_incremental = (
            int(initial_contacts.by_class.get("palm", 0)) == 0
            and int(initial_contacts.by_class.get("thumb", 0)) >= 3
        )
        allow_dynamic_switch = int(initial_contacts.by_class.get("palm", 0)) >= 1
        hold, cage_rows = _establish_cage(
            env, hold, left_hold, capture_frame=capture_frame
        )
        rows.extend(cage_rows)
        if cage_rows and not cage_rows[-1]["peg_ok"]:
            fail_reason = "peg_lost_during_cage"
            return {
                "episode_index": int(episode),
                "insert_ok": False,
                "handoff_frame": int(handoff),
                "handoff_info": handoff_info,
                "steps": len(rows),
                "best_tip_dist_m": float(min(row["tip_dist_m"] for row in rows)),
                "best_along_m": float(min(row["along_m"] for row in rows)),
                "fail_reason": fail_reason,
                "final": rows[-1],
                "traj_tail": rows[-20:],
            }
        names = names_from_raw(env._raw)
        peg_pos, peg_rot = _body_pose(env._raw, names.peg_body)
        right_pos, right_rot = _body_pose(env._raw, REFERENCE_BODY)
        peg_in_hand_pos, peg_in_hand_rot = _relative_pose(
            right_pos, right_rot, peg_pos, peg_rot
        )
        tip_world = np.asarray(
            privileged_full_features(env._raw)["tip"], dtype=np.float64
        )
        grasp_anchor = ObjectAnchor(
            desired_pos=peg_pos.copy(),
            desired_rot=peg_rot,
            object_in_hand_pos=peg_in_hand_pos,
            object_in_hand_rot=peg_in_hand_rot,
            tip_in_object=peg_rot.inv().apply(tip_world - peg_pos),
        )
        socket_pos, socket_rot = _body_pose(env._raw, names.socket_body)
        left_pos, left_rot = _body_pose(env._raw, LEFT_PALM)
        socket_in_hand_pos, socket_in_hand_rot = _relative_pose(
            left_pos, left_rot, socket_pos, socket_rot
        )
        socket_anchor = ObjectAnchor(
            desired_pos=socket_pos.copy(),
            desired_rot=socket_rot,
            object_in_hand_pos=socket_in_hand_pos,
            object_in_hand_rot=socket_in_hand_rot,
            tip_in_object=None,
        )
        for k in range(int(gains.max_steps)):
            action, phase = _command(
                env._raw,
                hold,
                gains,
                jam_steps=jam_steps,
                socket_anchor=socket_anchor,
                grasp_anchor=grasp_anchor,
                force_incremental=force_incremental,
                allow_dynamic_switch=allow_dynamic_switch,
            )
            action[6:22] = hold[6:22]
            action[28:44] = left_hold[6:22]
            _apply_action(
                env,
                action,
                n_substeps=(
                    gains.insert_substeps
                    if phase.startswith(("insert", "unjam"))
                    else gains.settle_substeps
                ),
                capture_frame=capture_frame,
            )
            row = _measure(env, k=k, phase=phase)
            row["jam_steps"] = int(jam_steps)
            row["controller_mode"] = (
                "incremental" if phase.endswith("_incremental") else "object_inverse"
            )
            actual_palm_pos, actual_palm_rot = _body_pose(env._raw, REFERENCE_BODY)
            actual_peg_pos, actual_peg_rot = _body_pose(env._raw, names.peg_body)
            desired_palm_pos, desired_palm_rot = _inverse_relative_target(
                grasp_anchor.desired_pos,
                grasp_anchor.desired_rot,
                grasp_anchor.object_in_hand_pos,
                grasp_anchor.object_in_hand_rot,
            )
            row["palm_target_err_m"] = float(
                np.linalg.norm(desired_palm_pos - actual_palm_pos)
            )
            row["palm_target_rot_err"] = float(
                np.linalg.norm((desired_palm_rot * actual_palm_rot.inv()).as_rotvec())
            )
            row["peg_target_err_m"] = float(
                np.linalg.norm(grasp_anchor.desired_pos - actual_peg_pos)
            )
            row["peg_target_rot_err"] = float(
                np.linalg.norm((grasp_anchor.desired_rot * actual_peg_rot.inv()).as_rotvec())
            )
            actual_rel_pos, actual_rel_rot = _relative_pose(
                actual_palm_pos, actual_palm_rot, actual_peg_pos, actual_peg_rot
            )
            row["grasp_slip_m"] = float(
                np.linalg.norm(actual_rel_pos - grasp_anchor.object_in_hand_pos)
            )
            row["grasp_slip_rad"] = float(
                np.linalg.norm(
                    (actual_rel_rot * grasp_anchor.object_in_hand_rot.inv()).as_rotvec()
                )
            )
            rows.append(row)
            hold = action
            best_tip = min(best_tip, row["tip_dist_m"])
            best_along = min(best_along, row["along_m"])

            progress = previous_along - row["along_m"]
            base_phase = phase.replace("_incremental", "")
            if base_phase == "insert" and progress < 0.00012:
                jam_steps += 1
            elif base_phase == "unjam" and jam_steps < gains.jam_patience + gains.unjam_steps:
                jam_steps += 1
            elif base_phase == "unjam":
                jam_steps = 0
            else:
                jam_steps = 0
            previous_along = row["along_m"]

            if not np.isfinite(np.asarray(env._raw._data.qpos)).all():
                fail_reason = "nonfinite"
                break
            if row["tip_dist_m"] > 0.45:
                fail_reason = "tip_diverged"
                break
            if not row["peg_ok"]:
                fail_reason = "peg_lost"
                break
            if row["insert_ok"]:
                success, release_rows = _release_and_verify(
                    env, hold, left_hold, capture_frame=capture_frame
                )
                rows.extend(release_rows)
                fail_reason = "" if success else "lost_after_release"
                break
    finally:
        env._raw._set_free_joint_pose = original_set_pose

    return {
        "episode_index": int(episode),
        "insert_ok": bool(success),
        "handoff_frame": int(handoff),
        "handoff_info": handoff_info,
        "steps": len(rows),
        "best_tip_dist_m": float(best_tip),
        "best_along_m": float(best_along),
        "fail_reason": fail_reason,
        "final": rows[-1] if rows else {},
        "traj_tail": rows[-20:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", default=[1])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    episodes = (
        sorted(int(path.parent.name.split("_")[-1]) for path in args.sidecar.glob("episode_*/meta.json"))
        if args.all
        else [int(ep) for ep in args.episodes]
    )
    gains = StrictGains(max_steps=int(args.max_steps))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _ = load_manifest_entries(args.sidecar, episode_indices=episodes)
    env = make_full_env(episodes, sidecar_dir=args.sidecar, seed=0)
    results = []
    recorder = None
    if args.video is not None:
        if len(episodes) != 1:
            parser.error("--video requires exactly one episode")
        import mujoco
        from dexjoco.data.video_writer import Mp4VideoWriter

        class NativeRecorder:
            def __init__(self, raw, path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                self.path = path
                self.raw = raw
                self.renderer = mujoco.Renderer(raw._model, height=480, width=640)
                self.writer = Mp4VideoWriter.create_h264(fps=30)
                self.writer.start(str(path))
                self.frame_count = 0

            def capture(self) -> None:
                self.renderer.update_scene(self.raw._data, camera="back")
                self.writer.write_frame(self.renderer.render())
                self.frame_count += 1

            def close(self) -> Path:
                self.writer.stop()
                self.renderer.close()
                return self.path

        recorder = NativeRecorder(env._raw, args.video)
    try:
        for episode in episodes:
            print(f"[strict] episode={episode}", flush=True)
            result = run_episode(
                env,
                episode,
                gains,
                capture_frame=None if recorder is None else recorder.capture,
            )
            results.append(result)
            print(
                f"[strict] episode={episode} ok={result['insert_ok']} "
                f"best_tip={result['best_tip_dist_m']:.4f} best_along={result['best_along_m']:.4f} "
                f"reason={result['fail_reason']}",
                flush=True,
            )
    finally:
        if recorder is not None:
            recorder.close()
        env.close()
    payload = {
        "protocol": "StrictActionOnlyPBVS",
        "rules": {
            "demo_after_handoff": False,
            "freejoint_writes_after_handoff": False,
            "snap_or_pin": False,
            "robot_action_only": True,
        },
        "gains": asdict(gains),
        "successes": sum(int(row["insert_ok"]) for row in results),
        "total": len(results),
        "results": results,
    }
    path = args.out_dir / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"successes": payload["successes"], "total": payload["total"], "path": str(path)}))


if __name__ == "__main__":
    main()
