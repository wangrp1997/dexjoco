"""Single-demo approach→grasp segment: DITTO warp + DexGraspBench staged close.

Refs:
  - DITTO warp: interaction_retarget/grasp/ditto_warp.py
  - DexGraspBench stages: refs/DexGraspBench/src/task/eval_func/tabletop_mocap.py
  - Extract pattern: interaction_retarget/grasp/lift_reference.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

from interaction_retarget.constants import PEG_BODY, TRAY_BODY
from interaction_retarget.grasp.approach import interpolate_action23, interpolate_arm_only
from interaction_retarget.grasp.ditto_warp import arm23_from_object_frame_waypoint, demo_arm_to_object_frame
from interaction_retarget.grasp.repair import _step_side, side_contact_count
from interaction_retarget.grasp.staged_grasp import execute_grasp_to_squeeze
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.replay import make_assembly_env, raw_flat_to_dict
from interaction_retarget.sim.settle import read_arm_action, vec_to_arm_action
from interaction_retarget.sim.video import maybe_capture_frame
from interaction_retarget.grasp.qpos_distill import _squeeze_frame

ObjectName = Literal["tray", "peg"]
Side = Literal["left", "right"]

TABLE_MOCAP_Z_MIN_M = 0.012
_MAX_APPROACH_WAYPOINTS = 80
_SUBSTEPS_PER_WAYPOINT = 1
_BLEND_IN_STEPS = 6
_GRASP_SNAP_STEPS = 4
_APPROACH_LOOKBACK = 80
_DEFAULT_SQUEEZE_STEPS = 24


@dataclass
class ObjFrameSegment:
    """Mocap + finger trajectory stored in object body frame."""

    mocap_pos_obj: np.ndarray
    mocap_quat_obj: np.ndarray
    hand_joint: np.ndarray
    squeeze_pos_obj: np.ndarray
    squeeze_quat_obj: np.ndarray
    squeeze_hand: np.ndarray
    grasp_pos_obj: np.ndarray
    grasp_quat_obj: np.ndarray
    grasp_hand: np.ndarray
    grasp_frame: int
    squeeze_frame: int
    demo_obj_pos: np.ndarray
    demo_obj_quat: np.ndarray


@dataclass
class DemoApproachBundle:
    episode_index: int
    zarr_path: str
    tray: ObjFrameSegment
    peg: ObjFrameSegment


def _side_object(object_name: ObjectName) -> tuple[Side, str]:
    if object_name == "tray":
        return "left", TRAY_BODY
    return "right", PEG_BODY


def _grasp_frame(timing: dict, object_name: ObjectName) -> int:
    key = "left_grasp_frame" if object_name == "tray" else "right_grasp_frame"
    return int(timing[key])


def _object_pose_raw(raw_env, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    bid = raw_env._model.body(body_name).id
    return (
        np.asarray(raw_env._data.xpos[bid], dtype=np.float64).copy(),
        np.asarray(raw_env._data.xquat[bid], dtype=np.float64).copy(),
    )


def _subsample_indices(start: int, end: int, max_pts: int) -> np.ndarray:
    start, end = int(start), int(end)
    if end <= start:
        return np.asarray([end], dtype=int)
    idx = np.arange(start, end + 1, dtype=int)
    if idx.size <= max_pts:
        return idx
    return np.unique(np.linspace(start, end, max_pts, dtype=int))


def extract_demo_approach_bundle(entry: dict[str, Any], *, seed_base: int = 0) -> DemoApproachBundle:
    """Replay demo zarr once; record approach→grasp waypoints in object frame."""
    timing = entry["timing"]
    actions, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    n_actions = len(actions)
    ep_idx = int(entry["episode_index"])

    specs: dict[str, dict] = {}
    for object_name in ("tray", "peg"):
        grasp = int(np.clip(_grasp_frame(timing, object_name), 0, n_actions - 1))  # type: ignore[arg-type]
        squeeze = _squeeze_frame(timing, object_name, n_actions=n_actions)
        start = max(0, grasp - _APPROACH_LOOKBACK)
        specs[object_name] = {
            "side": _side_object(object_name)[0],  # type: ignore[arg-type]
            "body": _side_object(object_name)[1],  # type: ignore[arg-type]
            "grasp": grasp,
            "squeeze": squeeze,
            "frame_idx": _subsample_indices(start, grasp, _MAX_APPROACH_WAYPOINTS),
        }

    end_replay = max(
        specs["tray"]["grasp"],
        specs["tray"]["squeeze"],
        specs["peg"]["grasp"],
        specs["peg"]["squeeze"],
    )
    recorded: dict[str, dict[int, tuple]] = {k: {} for k in specs}
    pos_lst: dict[str, list] = {k: [] for k in specs}
    quat_lst: dict[str, list] = {k: [] for k in specs}
    hand_lst: dict[str, list] = {k: [] for k in specs}
    demo_obj: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    env = make_assembly_env(seed=int(seed_base) + ep_idx, randomize=False)
    raw = env.unwrapped
    try:
        env.reset()
        config = CONFIG_MAPPING["bimanual_assembly"]()
        if initial_state is not None and has_restorer("bimanual_assembly"):
            restore_initial_state(env, "bimanual_assembly", config, initial_state)

        for fi, action in enumerate(actions[: end_replay + 1]):
            raw.step(raw_flat_to_dict(action))
            for object_name, spec in specs.items():
                side = spec["side"]
                body = spec["body"]
                grasp = spec["grasp"]
                squeeze = spec["squeeze"]
                frame_idx = spec["frame_idx"]
                if fi not in (grasp, squeeze, 0) and fi not in frame_idx:
                    continue
                obj_pos, obj_quat = _object_pose_raw(raw, body)
                arm23 = read_arm_action(raw, side)
                recorded[object_name][fi] = demo_arm_to_object_frame(arm23, obj_pos, obj_quat)
                if fi == 0:
                    demo_obj[object_name] = (obj_pos, obj_quat)
    finally:
        env.close()

    segments: dict[str, ObjFrameSegment] = {}
    for object_name, spec in specs.items():
        grasp = spec["grasp"]
        squeeze = spec["squeeze"]
        frame_idx = spec["frame_idx"]
        rec = recorded[object_name]
        grasp = spec["grasp"]
        for fi in frame_idx:
            if fi >= grasp or fi not in rec:
                continue
            p, q, h = rec[fi]
            pos_lst[object_name].append(p)
            quat_lst[object_name].append(q)
            hand_lst[object_name].append(h)
        if squeeze not in rec:
            raise RuntimeError(f"{object_name}: squeeze frame {squeeze} missing during extract")
        sq_p, sq_q, sq_h = rec[squeeze]
        if grasp not in rec:
            raise RuntimeError(f"{object_name}: grasp frame {grasp} missing during extract")
        g_p, g_q, g_h = rec[grasp]
        d_pos, d_quat = demo_obj.get(object_name, (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
        if pos_lst[object_name]:
            app_pos = np.stack(pos_lst[object_name], axis=0)
            app_quat = np.stack(quat_lst[object_name], axis=0)
            app_hand = np.stack(hand_lst[object_name], axis=0)
        else:
            app_pos = g_p.reshape(1, 3)
            app_quat = g_q.reshape(1, 4)
            app_hand = g_h.reshape(1, 16)
        segments[object_name] = ObjFrameSegment(
            mocap_pos_obj=app_pos,
            mocap_quat_obj=app_quat,
            hand_joint=app_hand,
            squeeze_pos_obj=sq_p,
            squeeze_quat_obj=sq_q,
            squeeze_hand=sq_h,
            grasp_pos_obj=g_p,
            grasp_quat_obj=g_q,
            grasp_hand=g_h,
            grasp_frame=grasp,
            squeeze_frame=squeeze,
            demo_obj_pos=d_pos,
            demo_obj_quat=d_quat,
        )

    return DemoApproachBundle(
        episode_index=ep_idx,
        zarr_path=str(entry["zarr_path"]),
        tray=segments["tray"],
        peg=segments["peg"],
    )


def _grasp23_from_segment(
    segment: ObjFrameSegment,
    raw_env,
    object_name: ObjectName,
) -> np.ndarray:
    _, body = _side_object(object_name)
    obj_pos, obj_quat = _object_pose_raw(raw_env, body)
    return arm23_from_object_frame_waypoint(
        segment.grasp_pos_obj,
        segment.grasp_quat_obj,
        segment.grasp_hand,
        live_obj_pos=obj_pos,
        live_obj_quat_wxyz=obj_quat,
    )


def _squeeze23_from_segment(
    segment: ObjFrameSegment,
    raw_env,
    object_name: ObjectName,
) -> np.ndarray:
    """Warp demo squeeze-frame hand pose (object frame) to current layout."""
    _, body = _side_object(object_name)
    obj_pos, obj_quat = _object_pose_raw(raw_env, body)
    return arm23_from_object_frame_waypoint(
        segment.squeeze_pos_obj,
        segment.squeeze_quat_obj,
        segment.squeeze_hand,
        live_obj_pos=obj_pos,
        live_obj_quat_wxyz=obj_quat,
    )


def execute_warped_approach_grasp(
    raw_env,
    segment: ObjFrameSegment,
    *,
    object_name: ObjectName,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    do_squeeze: bool = True,
    squeeze_steps: int = _DEFAULT_SQUEEZE_STEPS,
) -> np.ndarray:
    """DITTO warp: approach (open hand, demo-speed) → grasp snap → squeeze (DexGraspBench)."""
    side, body = _side_object(object_name)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    n_wp = int(segment.mocap_pos_obj.shape[0])
    open_hand = segment.hand_joint[0].copy()

    cur = vec_to_arm_action(read_arm_action(raw_env, side))
    obj_pos, obj_quat = _object_pose_raw(raw_env, body)
    if n_wp >= 1:
        first = arm23_from_object_frame_waypoint(
            segment.mocap_pos_obj[0],
            segment.mocap_quat_obj[0],
            open_hand,
            live_obj_pos=obj_pos,
            live_obj_quat_wxyz=obj_quat,
        )
        for bi in range(_BLEND_IN_STEPS):
            t = (bi + 1) / _BLEND_IN_STEPS
            cmd = interpolate_arm_only(cur, first, t, hand=open_hand)
            _step_side(
                raw_env,
                side=side,
                active23=cmd,
                hold_right=hold_right,
                hold_left=hold_left,
            )
            maybe_capture_frame()
        cur = vec_to_arm_action(read_arm_action(raw_env, side))

        for i in range(n_wp):
            obj_pos, obj_quat = _object_pose_raw(raw_env, body)
            tgt = arm23_from_object_frame_waypoint(
                segment.mocap_pos_obj[i],
                segment.mocap_quat_obj[i],
                open_hand,
                live_obj_pos=obj_pos,
                live_obj_quat_wxyz=obj_quat,
            )
            if float(tgt[2]) < TABLE_MOCAP_Z_MIN_M:
                tgt[2] = max(float(cur[2]), TABLE_MOCAP_Z_MIN_M)
            for sub in range(_SUBSTEPS_PER_WAYPOINT):
                t = (sub + 1) / max(_SUBSTEPS_PER_WAYPOINT, 1)
                cmd = interpolate_arm_only(cur, tgt, t, hand=open_hand)
                _step_side(
                    raw_env,
                    side=side,
                    active23=cmd,
                    hold_right=hold_right,
                    hold_left=hold_left,
                )
                maybe_capture_frame()
            cur = vec_to_arm_action(read_arm_action(raw_env, side))

    grasp23 = _grasp23_from_segment(segment, raw_env, object_name)
    for gi in range(_GRASP_SNAP_STEPS):
        t = (gi + 1) / _GRASP_SNAP_STEPS
        cmd = interpolate_action23(cur, grasp23, t)
        _step_side(
            raw_env,
            side=side,
            active23=cmd,
            hold_right=hold_right,
            hold_left=hold_left,
        )
        maybe_capture_frame()
    grasp23 = vec_to_arm_action(_grasp23_from_segment(segment, raw_env, object_name))

    if do_squeeze:
        squeeze23 = _squeeze23_from_segment(segment, raw_env, object_name)
        execute_grasp_to_squeeze(
            raw_env,
            side=side,
            grasp23=grasp23,
            squeeze23=squeeze23,
            hold_right=hold_right,
            hold_left=hold_left,
            squeeze_steps=max(int(squeeze_steps), 1),
        )
    return vec_to_arm_action(read_arm_action(raw_env, side))
