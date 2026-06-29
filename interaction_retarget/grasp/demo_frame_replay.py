"""DITTO object-frame demo replay: approach → squeeze (demo fingers) → lift (demo warp).

Each segment is extracted 1:1 from the same demo zarr and replayed with ditto_warp.
Squeeze: wrist fixed at grasp (frame 0 of squeeze track), hand qpos from demo timeline.
Lift: full demo wrist trajectory warped; fingers locked to demo squeeze-end hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

from interaction_retarget.constants import MIN_GRASP_CONTACT_COUNT, PEG_BODY, TRAY_BODY
from interaction_retarget.grasp.ditto_warp import arm23_from_object_frame_waypoint, demo_arm_to_object_frame
from interaction_retarget.transforms import (
    matrix_to_quat_wxyz,
    mocap_world_from_object_frame,
    quat_wxyz_to_matrix,
    relative_mocap_in_object_frame,
)

_SQUEEZE_SUBSTEP_MUL = 3
_SQUEEZE_HOLD_STEPS = 20
from interaction_retarget.grasp.qpos_distill import _squeeze_frame
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.grasp.repair import side_contact_count
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.replay import make_assembly_env, raw_flat_to_dict
from interaction_retarget.sim.settle import read_arm_action, settle_bimanual_actions, vec_to_arm_action
from interaction_retarget.sim.video import maybe_capture_frame

Side = Literal["left", "right"]


@dataclass
class BimanualObjFrameTrack:
    demo_frames: np.ndarray
    left_pos_obj: np.ndarray
    left_quat_obj: np.ndarray
    left_hand: np.ndarray
    right_pos_obj: np.ndarray
    right_quat_obj: np.ndarray
    right_hand: np.ndarray
    obj_z_world: np.ndarray | None = None


@dataclass
class DemoWarpTracks:
    episode_index: int
    zarr_path: str
    tray_approach: BimanualObjFrameTrack
    tray_squeeze: BimanualObjFrameTrack
    tray_lift: BimanualObjFrameTrack
    peg_approach: BimanualObjFrameTrack
    peg_squeeze: BimanualObjFrameTrack
    peg_lift: BimanualObjFrameTrack


def _object_pose_raw(raw_env, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    bid = raw_env._model.body(body_name).id
    return (
        np.asarray(raw_env._data.xpos[bid], dtype=np.float64).copy(),
        np.asarray(raw_env._data.xquat[bid], dtype=np.float64).copy(),
    )


def _frame_range(start: int, end: int) -> np.ndarray:
    start, end = int(start), int(end)
    if end < start:
        end = start
    return np.arange(start, end + 1, dtype=int)


def _lift_end_frame(
    z_by_fi: dict[int, float],
    cc_by_fi: dict[int, int],
    *,
    start: int,
    end: int,
) -> int:
    start, end = int(start), int(end)
    best_t, best_z = start, -np.inf
    for fi in range(start, end + 1):
        if cc_by_fi.get(fi, 0) < MIN_GRASP_CONTACT_COUNT:
            continue
        z = float(z_by_fi.get(fi, -np.inf))
        if z > best_z or (z == best_z and fi > best_t):
            best_z = z
            best_t = fi
    return best_t


def _object_body(side: Side) -> str:
    return TRAY_BODY if side == "left" else PEG_BODY


def _read_grasp_in_object_frame(raw_env, side: Side) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arm = vec_to_arm_action(read_arm_action(raw_env, side))
    obj_p, obj_q = _object_pose_raw(raw_env, _object_body(side))
    pos_obj, quat_obj = relative_mocap_in_object_frame(arm[0:3], arm[3:7], obj_p, obj_q)
    return pos_obj, quat_obj, arm[7:23].copy()


def _quat_rel_wxyz(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    r_from = quat_wxyz_to_matrix(q_from)
    r_to = quat_wxyz_to_matrix(q_to)
    return matrix_to_quat_wxyz(r_to @ r_from.T)


def _compose_obj_quat(base: np.ndarray, rel: np.ndarray) -> np.ndarray:
    return matrix_to_quat_wxyz(quat_wxyz_to_matrix(base) @ quat_wxyz_to_matrix(rel))


def _arm23_from_obj_frame(
    pos_obj: np.ndarray,
    quat_obj: np.ndarray,
    hand: np.ndarray,
    raw_env,
    side: Side,
) -> np.ndarray:
    obj_p, obj_q = _object_pose_raw(raw_env, _object_body(side))
    pos_w, quat_w = mocap_world_from_object_frame(pos_obj, quat_obj, obj_p, obj_q)
    return np.concatenate([pos_w, quat_w, np.asarray(hand, dtype=np.float64).reshape(16)], axis=0)


def _warp_side(
    track: BimanualObjFrameTrack,
    t: int,
    raw_env,
    side: Side,
    *,
    wrist_t: int | None = None,
    hand: np.ndarray | None = None,
) -> np.ndarray:
    tray_p, tray_q = _object_pose_raw(raw_env, TRAY_BODY)
    peg_p, peg_q = _object_pose_raw(raw_env, PEG_BODY)
    wt = int(wrist_t if wrist_t is not None else t)
    if side == "left":
        pos_obj = track.left_pos_obj[wt]
        quat_obj = track.left_quat_obj[wt]
        hand_q = track.left_hand[t] if hand is None else hand
        live_p, live_q = tray_p, tray_q
    else:
        pos_obj = track.right_pos_obj[wt]
        quat_obj = track.right_quat_obj[wt]
        hand_q = track.right_hand[t] if hand is None else hand
        live_p, live_q = peg_p, peg_q
    return vec_to_arm_action(
        arm23_from_object_frame_waypoint(
            pos_obj, quat_obj, hand_q, live_obj_pos=live_p, live_obj_quat_wxyz=live_q
        )
    )


def _settle_pair(
    raw_env,
    left23: np.ndarray,
    right23: np.ndarray,
    *,
    n_substeps: int,
) -> None:
    settle_bimanual_actions(
        raw_env,
        left23=vec_to_arm_action(left23),
        right23=vec_to_arm_action(right23),
        n_substeps=n_substeps,
    )
    maybe_capture_frame()


def _n_sub(raw_env) -> int:
    return max(int(getattr(raw_env, "_n_substeps", 1)), 1)


def replay_warp_approach(
    raw_env,
    track: BimanualObjFrameTrack,
    *,
    lock_left23: np.ndarray | None = None,
    lock_right23: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Full DITTO warp for pre-grasp approach."""
    n_sub = _n_sub(raw_env)
    lock_left23 = vec_to_arm_action(lock_left23) if lock_left23 is not None else None
    lock_right23 = vec_to_arm_action(lock_right23) if lock_right23 is not None else None
    left23 = right23 = None
    for t in range(int(track.demo_frames.shape[0])):
        tray_p, tray_q = _object_pose_raw(raw_env, TRAY_BODY)
        peg_p, peg_q = _object_pose_raw(raw_env, PEG_BODY)
        left23 = arm23_from_object_frame_waypoint(
            track.left_pos_obj[t],
            track.left_quat_obj[t],
            track.left_hand[t],
            live_obj_pos=tray_p,
            live_obj_quat_wxyz=tray_q,
        )
        right23 = arm23_from_object_frame_waypoint(
            track.right_pos_obj[t],
            track.right_quat_obj[t],
            track.right_hand[t],
            live_obj_pos=peg_p,
            live_obj_quat_wxyz=peg_q,
        )
        if lock_left23 is not None:
            left23 = lock_left23
        if lock_right23 is not None:
            right23 = lock_right23
        _settle_pair(raw_env, left23, right23, n_substeps=n_sub)
    assert left23 is not None and right23 is not None
    return vec_to_arm_action(left23), vec_to_arm_action(right23)


def replay_warp_squeeze(
    raw_env,
    track: BimanualObjFrameTrack,
    *,
    active: Side,
    lock_left23: np.ndarray | None = None,
    lock_right23: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Demo squeeze: wrist at grasp; monotonic demo finger closure + hold at max."""
    n_sub = _n_sub(raw_env) * _SQUEEZE_SUBSTEP_MUL
    lock_left23 = vec_to_arm_action(lock_left23) if lock_left23 is not None else None
    lock_right23 = vec_to_arm_action(lock_right23) if lock_right23 is not None else None
    grasp_wrist = 0
    hand_key = "left_hand" if active == "left" else "right_hand"
    hands = getattr(track, hand_key)
    hand_max = hands[0].copy()
    left23 = right23 = None

    def _apply(wl: int, wr: int, hand_t: int, hand_override: np.ndarray | None = None) -> None:
        nonlocal left23, right23
        h_left = hands[hand_t] if active == "left" and hand_override is None else track.left_hand[hand_t]
        h_right = hands[hand_t] if active == "right" and hand_override is None else track.right_hand[hand_t]
        if hand_override is not None:
            if active == "left":
                h_left = hand_override
            else:
                h_right = hand_override
        left23 = _warp_side(track, hand_t, raw_env, "left", wrist_t=wl, hand=h_left)
        right23 = _warp_side(track, hand_t, raw_env, "right", wrist_t=wr, hand=h_right)
        if lock_left23 is not None:
            left23 = lock_left23
        if lock_right23 is not None:
            right23 = lock_right23
        _settle_pair(raw_env, left23, right23, n_substeps=n_sub)

    for t in range(int(track.demo_frames.shape[0])):
        hand_max = np.maximum(hand_max, hands[t])
        wl = grasp_wrist if active == "left" else t
        wr = grasp_wrist if active == "right" else t
        _apply(wl, wr, t)

    for _ in range(_SQUEEZE_HOLD_STEPS):
        wl = grasp_wrist if active == "left" else int(track.demo_frames.shape[0]) - 1
        wr = grasp_wrist if active == "right" else int(track.demo_frames.shape[0]) - 1
        _apply(wl, wr, int(track.demo_frames.shape[0]) - 1, hand_override=hand_max.copy())

    assert left23 is not None and right23 is not None
    return vec_to_arm_action(left23), vec_to_arm_action(right23)


def replay_warp_lift(
    raw_env,
    track: BimanualObjFrameTrack,
    *,
    active: Side,
    hand_lock: np.ndarray,
    lock_left23: np.ndarray | None = None,
    lock_right23: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Demo lift: keep grasp pose, apply demo object world-Z delta (rigid lift)."""
    n_sub = _n_sub(raw_env)
    hand_lock = np.asarray(hand_lock, dtype=np.float64).reshape(16)
    lock_left23 = vec_to_arm_action(lock_left23) if lock_left23 is not None else None
    lock_right23 = vec_to_arm_action(lock_right23) if lock_right23 is not None else None

    if track.obj_z_world is None or int(track.obj_z_world.shape[0]) < 1:
        raise ValueError("lift track missing obj_z_world")

    grasp = vec_to_arm_action(read_arm_action(raw_env, active))
    pos_w0 = grasp[0:3].copy()
    quat_w0 = grasp[3:7].copy()
    z0_demo = float(track.obj_z_world[0])

    left23 = right23 = None
    for t in range(int(track.demo_frames.shape[0])):
        dz = float(track.obj_z_world[t] - z0_demo)
        active23 = grasp.copy()
        active23[0:3] = pos_w0 + np.array([0.0, 0.0, dz], dtype=np.float64)
        active23[3:7] = quat_w0
        active23[7:23] = hand_lock
        if active == "left":
            left23 = active23
            right23 = lock_right23 if lock_right23 is not None else _warp_side(track, t, raw_env, "right")
        else:
            right23 = active23
            left23 = lock_left23 if lock_left23 is not None else _warp_side(track, t, raw_env, "left")
        _settle_pair(raw_env, left23, right23, n_substeps=n_sub)
    assert left23 is not None and right23 is not None
    return vec_to_arm_action(left23), vec_to_arm_action(right23)


def extract_demo_warp_tracks(entry: dict[str, Any], *, seed_base: int = 0) -> DemoWarpTracks:
    """Offline: replay demo once, store object-frame segments (approach/squeeze/lift)."""
    timing = entry["timing"]
    actions, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    n_actions = len(actions)
    ep_idx = int(entry["episode_index"])

    tray_grasp = int(timing["left_grasp_frame"])
    tray_squeeze = _squeeze_frame(timing, "tray", n_actions=n_actions)
    tray_lift_start = int(timing["tray_lift_start"])
    peg_grasp = int(timing["right_grasp_frame"])
    peg_squeeze = _squeeze_frame(timing, "peg", n_actions=n_actions)
    peg_lift_start = int(timing["peg_lift_start"])

    tray_app = _frame_range(max(0, tray_grasp - 80), tray_grasp)
    tray_sq = _frame_range(tray_grasp, tray_squeeze)
    peg_app = _frame_range(max(0, peg_grasp - 80), peg_grasp)
    peg_sq = _frame_range(peg_grasp, peg_squeeze)

    per_frame: dict[int, dict[str, np.ndarray]] = {}
    tray_z: dict[int, float] = {}
    tray_cc: dict[int, int] = {}
    peg_z: dict[int, float] = {}
    peg_cc: dict[int, int] = {}

    env = make_assembly_env(seed=int(seed_base) + ep_idx, randomize=False)
    raw = env.unwrapped
    detector = AssemblyContactDetector(raw)
    try:
        env.reset()
        config = CONFIG_MAPPING["bimanual_assembly"]()
        if initial_state is not None and has_restorer("bimanual_assembly"):
            restore_initial_state(env, "bimanual_assembly", config, initial_state)

        end_replay = min(n_actions - 1, peg_lift_start + 150)
        for fi, action in enumerate(actions[: end_replay + 1]):
            raw.step(raw_flat_to_dict(action))
            tray_p, tray_q = _object_pose_raw(raw, TRAY_BODY)
            peg_p, peg_q = _object_pose_raw(raw, PEG_BODY)
            tray_z[fi] = float(tray_p[2])
            peg_z[fi] = float(peg_p[2])
            tray_cc[fi] = int(side_contact_count(detector, raw, object_name="tray"))
            peg_cc[fi] = int(side_contact_count(detector, raw, object_name="peg"))
            left = read_arm_action(raw, "left")
            right = read_arm_action(raw, "right")
            l_p, l_q, l_h = demo_arm_to_object_frame(left, tray_p, tray_q)
            r_p, r_q, r_h = demo_arm_to_object_frame(right, peg_p, peg_q)
            per_frame[fi] = {
                "left_pos": l_p,
                "left_quat": l_q,
                "left_hand": l_h,
                "right_pos": r_p,
                "right_quat": r_q,
                "right_hand": r_h,
            }

        tray_lift_end = _lift_end_frame(
            tray_z, tray_cc, start=tray_lift_start, end=max(tray_lift_start + 1, peg_grasp - 5)
        )
        peg_lift_end = _lift_end_frame(
            peg_z, peg_cc, start=peg_lift_start, end=min(end_replay, peg_lift_start + 150)
        )
        tray_lift = _frame_range(tray_lift_start, tray_lift_end)
        peg_lift = _frame_range(peg_lift_start, peg_lift_end)
    finally:
        env.close()

    def _build(indices: np.ndarray, *, obj_z: dict[int, float] | None = None) -> BimanualObjFrameTrack:
        rows = [per_frame[int(fi)] for fi in indices]
        z_world = None
        if obj_z is not None:
            z_world = np.asarray([float(obj_z[int(fi)]) for fi in indices], dtype=np.float64)
        return BimanualObjFrameTrack(
            demo_frames=indices.copy(),
            left_pos_obj=np.stack([r["left_pos"] for r in rows], axis=0),
            left_quat_obj=np.stack([r["left_quat"] for r in rows], axis=0),
            left_hand=np.stack([r["left_hand"] for r in rows], axis=0),
            right_pos_obj=np.stack([r["right_pos"] for r in rows], axis=0),
            right_quat_obj=np.stack([r["right_quat"] for r in rows], axis=0),
            right_hand=np.stack([r["right_hand"] for r in rows], axis=0),
            obj_z_world=z_world,
        )

    return DemoWarpTracks(
        episode_index=ep_idx,
        zarr_path=str(entry["zarr_path"]),
        tray_approach=_build(tray_app),
        tray_squeeze=_build(tray_sq),
        tray_lift=_build(tray_lift, obj_z=tray_z),
        peg_approach=_build(peg_app),
        peg_squeeze=_build(peg_sq),
        peg_lift=_build(peg_lift, obj_z=peg_z),
    )


# Back-compat alias used by older imports
replay_warped_bimanual_track = replay_warp_approach
