"""Per-demo grasp pose from zarr replay at manifest grasp frame (same env, no EGL clash)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from interaction_retarget.grasp.locked_hold import enforce_locked_passive
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.replay import raw_flat_to_dict
from interaction_retarget.sim.settle import read_arm_action, settle_bimanual_actions, vec_to_arm_action
from interaction_retarget.sim.state import restore_sim, snapshot_sim
from interaction_retarget.sim.video import maybe_capture_frame

ObjectName = Literal["tray", "peg"]


def _grasp_frame_key(object_name: ObjectName) -> str:
    return "left_grasp_frame" if object_name == "tray" else "right_grasp_frame"


def demo_lift_world_dz(
    entry: dict[str, Any],
    lift_ref: dict[str, Any],
    object_name: ObjectName,
) -> float:
    """Demo command mocap Δz over lift segment (world frame, from zarr)."""
    timing = entry["timing"]
    actions, _, _ = load_zarr_episode(entry["zarr_path"])
    if object_name == "tray":
        start = int(timing["tray_lift_start"])
        end = int(lift_ref.get("tray_lift_end_frame", timing.get("tray_lift_end", start)))
        key = "left"
    else:
        start = int(timing["peg_lift_start"])
        end = int(lift_ref.get("peg_lift_end_frame", timing.get("peg_lift_end", start)))
        key = "right"
    start = int(np.clip(start, 0, len(actions) - 1))
    end = int(np.clip(end, start, len(actions) - 1))
    z0 = float(raw_flat_to_dict(actions[start])[key][2])
    z1 = float(raw_flat_to_dict(actions[end])[key][2])
    return max(z1 - z0, 0.03)


def replay_demo_lift_vertical(
    raw_env,
    entry: dict[str, Any],
    lift_ref: dict[str, Any],
    *,
    object_name: ObjectName,
    passive23: np.ndarray,
) -> np.ndarray:
    """Lift: follow demo world-Z only; lock XY + fingers at post-grasp anchor."""
    timing = entry["timing"]
    side: Literal["left", "right"] = "left" if object_name == "tray" else "right"
    if object_name == "tray":
        start = int(timing["tray_lift_start"])
        end = int(lift_ref.get("tray_lift_end_frame", start))
    else:
        start = int(timing["peg_lift_start"])
        end = int(lift_ref.get("peg_lift_end_frame", start))
    actions, _, _ = load_zarr_episode(entry["zarr_path"])
    start = int(np.clip(start, 0, len(actions) - 1))
    end = int(np.clip(end, start, len(actions) - 1))
    passive23 = vec_to_arm_action(passive23)
    anchor = vec_to_arm_action(read_arm_action(raw_env, side))
    anchor_pos = anchor[0:3].copy()
    anchor_quat = anchor[3:7].copy()
    anchor_hand = anchor[7:23].copy()
    n_sub = max(int(getattr(raw_env, "_n_substeps", 1)), 1)
    for action in actions[start : end + 1]:
        act = raw_flat_to_dict(action)
        demo = np.asarray(act["left" if side == "left" else "right"], dtype=np.float64).reshape(23)
        cmd = anchor.copy()
        cmd[0:2] = anchor_pos[0:2]
        cmd[2] = float(demo[2])
        cmd[3:7] = anchor_quat
        cmd[7:23] = anchor_hand
        if side == "left":
            settle_bimanual_actions(raw_env, right23=passive23, left23=cmd, n_substeps=n_sub)
        else:
            settle_bimanual_actions(raw_env, right23=cmd, left23=passive23, n_substeps=n_sub)
        maybe_capture_frame()
        if side == "left":
            enforce_locked_passive(raw_env, locked_left=cmd, locked_right=passive23, n_substeps=2)
        else:
            enforce_locked_passive(raw_env, locked_left=passive23, locked_right=cmd, n_substeps=2)
    return vec_to_arm_action(read_arm_action(raw_env, side))


def replay_demo_lift_segment(
    raw_env,
    entry: dict[str, Any],
    lift_ref: dict[str, Any],
    *,
    object_name: ObjectName,
    passive23: np.ndarray,
) -> np.ndarray:
    """Replay demo zarr lift interval (world traj); passive arm locked."""
    timing = entry["timing"]
    side: Literal["left", "right"] = "left" if object_name == "tray" else "right"
    if object_name == "tray":
        start = int(timing["tray_lift_start"])
        end = int(lift_ref.get("tray_lift_end_frame", start))
    else:
        start = int(timing["peg_lift_start"])
        end = int(lift_ref.get("peg_lift_end_frame", start))
    passive23 = vec_to_arm_action(passive23)
    if side == "left":
        replay_demo_segment(
            raw_env,
            entry,
            start_frame=start,
            end_frame=end,
            lock_right=passive23,
            active_side="left",
        )
    else:
        replay_demo_segment(
            raw_env,
            entry,
            start_frame=start,
            end_frame=end,
            lock_left=passive23,
            active_side="right",
        )
    return vec_to_arm_action(read_arm_action(raw_env, side))


def apply_demo_grasp_frame(
    raw_env,
    entry: dict[str, Any],
    object_name: ObjectName,
) -> np.ndarray:
    """Replay demo zarr to grasp frame and keep env there (for plan/FC)."""
    side: Literal["left", "right"] = "left" if object_name == "tray" else "right"
    frame_key = _grasp_frame_key(object_name)
    actions, _, _ = load_zarr_episode(entry["zarr_path"])
    frame = int(np.clip(int(entry["timing"][frame_key]), 0, len(actions) - 1))
    n_sub = max(int(getattr(raw_env, "_n_substeps", 1)), 1)
    for action in actions[: frame + 1]:
        act = raw_flat_to_dict(action)
        right23 = np.asarray(act["right"], dtype=np.float64).reshape(23)
        left23 = np.asarray(act["left"], dtype=np.float64).reshape(23)
        settle_bimanual_actions(raw_env, right23=right23, left23=left23, n_substeps=n_sub)
    return vec_to_arm_action(read_arm_action(raw_env, side))


def replay_demo_segment(
    raw_env,
    entry: dict[str, Any],
    *,
    start_frame: int,
    end_frame: int,
    lock_left: np.ndarray | None = None,
    lock_right: np.ndarray | None = None,
    active_side: Literal["left", "right"] | None = None,
) -> None:
    """Replay zarr [start,end]; lock passive arm or replay only one side's demo qpos."""
    actions, _, _ = load_zarr_episode(entry["zarr_path"])
    start = int(np.clip(start_frame, 0, len(actions) - 1))
    end = int(np.clip(end_frame, start, len(actions) - 1))
    n_sub = max(int(getattr(raw_env, "_n_substeps", 1)), 1)
    lock_left = vec_to_arm_action(lock_left) if lock_left is not None else None
    lock_right = vec_to_arm_action(lock_right) if lock_right is not None else None
    for action in actions[start : end + 1]:
        act = raw_flat_to_dict(action)
        right23 = np.asarray(act["right"], dtype=np.float64).reshape(23)
        left23 = np.asarray(act["left"], dtype=np.float64).reshape(23)
        if active_side == "right" and lock_left is not None:
            left23 = lock_left
            right23 = np.asarray(act["right"], dtype=np.float64).reshape(23)
        elif active_side == "left" and lock_right is not None:
            right23 = lock_right
            left23 = np.asarray(act["left"], dtype=np.float64).reshape(23)
        else:
            if lock_left is not None:
                left23 = lock_left
            if lock_right is not None:
                right23 = lock_right
        settle_bimanual_actions(raw_env, right23=right23, left23=left23, n_substeps=n_sub)
        maybe_capture_frame()
        if active_side == "right" and lock_left is not None:
            enforce_locked_passive(
                raw_env,
                locked_left=lock_left,
                locked_right=vec_to_arm_action(read_arm_action(raw_env, "right")),
                n_substeps=2,
            )
        elif active_side == "left" and lock_right is not None:
            enforce_locked_passive(
                raw_env,
                locked_left=vec_to_arm_action(read_arm_action(raw_env, "left")),
                locked_right=lock_right,
                n_substeps=2,
            )


def replay_demo_passive_follow(
    raw_env,
    entry: dict[str, Any],
    *,
    start_frame: int,
    end_frame: int,
    follow_side: Literal["left", "right"],
    fixed_side: Literal["left", "right"],
    fixed23: np.ndarray,
) -> None:
    """Replay one arm from demo; the other holds ``fixed23`` (L1 tray hold during peg)."""
    actions, _, _ = load_zarr_episode(entry["zarr_path"])
    start = int(np.clip(start_frame, 0, len(actions) - 1))
    end = int(np.clip(end_frame, start, len(actions) - 1))
    fixed23 = vec_to_arm_action(fixed23)
    n_sub = max(int(getattr(raw_env, "_n_substeps", 1)), 1)
    for action in actions[start : end + 1]:
        act = raw_flat_to_dict(action)
        right23 = np.asarray(act["right"], dtype=np.float64).reshape(23)
        left23 = np.asarray(act["left"], dtype=np.float64).reshape(23)
        if follow_side == "left":
            right23 = fixed23 if fixed_side == "right" else right23
            if fixed_side == "left":
                left23 = fixed23
        else:
            left23 = fixed23 if fixed_side == "left" else left23
            if fixed_side == "right":
                right23 = fixed23
        settle_bimanual_actions(raw_env, right23=right23, left23=left23, n_substeps=n_sub)
        maybe_capture_frame()


def replay_demo_privileged_video(
    raw_env,
    entry: dict[str, Any],
    *,
    start_frame: int = 0,
    end_frame: int,
    phase_frames: dict[str, int] | None = None,
    mark_fn: Callable[[str], None] | None = None,
    max_frames: int = 1480,
) -> int:
    """Continuous demo zarr replay (privileged traj): home → tray lift → peg lift.

    One physics step per zarr frame; subsample indices if segment exceeds ``max_frames``.
    """
    actions, _, _ = load_zarr_episode(entry["zarr_path"])
    n_actions = len(actions)
    start = int(np.clip(start_frame, 0, max(n_actions - 1, 0)))
    end = int(np.clip(end_frame, start, n_actions - 1))
    span = end - start + 1
    if span > max_frames:
        idx = np.linspace(start, end, max_frames, dtype=int)
    else:
        idx = np.arange(start, end + 1, dtype=int)
    phase_frames = phase_frames or {}
    phase_by_frame: dict[int, list[str]] = defaultdict(list)
    for name, frame in phase_frames.items():
        phase_by_frame[int(frame)].append(str(name))
    n_sub = max(int(getattr(raw_env, "_n_substeps", 1)), 1)
    for fi in idx:
        act = raw_flat_to_dict(actions[int(fi)])
        right23 = np.asarray(act["right"], dtype=np.float64).reshape(23)
        left23 = np.asarray(act["left"], dtype=np.float64).reshape(23)
        settle_bimanual_actions(raw_env, right23=right23, left23=left23, n_substeps=n_sub)
        maybe_capture_frame()
        for ph in phase_by_frame.get(int(fi), ()):
            if mark_fn is not None:
                mark_fn(ph)
    return int(idx.size)


def demo_video_phase_frames(
    entry: dict[str, Any],
    lift_ref: dict[str, Any],
) -> tuple[dict[str, int], int]:
    """Phase markers + end frame for privileged demo video replay."""
    timing = entry["timing"]
    tray_grasp = int(timing["left_grasp_frame"])
    tray_lift_end = int(lift_ref.get("tray_lift_end_frame", timing.get("tray_lift_start", tray_grasp)))
    peg_grasp = int(timing["right_grasp_frame"])
    peg_lift_end = int(lift_ref.get("peg_lift_end_frame", timing.get("peg_lift_start", peg_grasp)))
    hold = max(tray_lift_end, int(timing.get("tray_lift_start", tray_grasp)))
    phases = {
        "tray_grasp_done": tray_grasp,
        "tray_lift_done": tray_lift_end,
        "tray_hold_done": hold,
        "peg_grasp_done": peg_grasp,
        "peg_lift_done": peg_lift_end,
    }
    return phases, peg_lift_end


def demo_grasp_arm23(
    raw_env,
    entry: dict[str, Any],
    object_name: ObjectName,
) -> np.ndarray:
    """Replay demo zarr on current env to grasp frame, then restore (keeps render context)."""
    side: Literal["left", "right"] = "left" if object_name == "tray" else "right"
    frame_key = _grasp_frame_key(object_name)
    actions, _, _ = load_zarr_episode(entry["zarr_path"])
    frame = int(np.clip(int(entry["timing"][frame_key]), 0, len(actions) - 1))

    snap = snapshot_sim(raw_env)
    try:
        for action in actions[: frame + 1]:
            act = raw_flat_to_dict(action)
            right23 = np.asarray(act["right"], dtype=np.float64).reshape(23)
            left23 = np.asarray(act["left"], dtype=np.float64).reshape(23)
            n_sub = max(int(getattr(raw_env, "_n_substeps", 1)), 1)
            settle_bimanual_actions(
                raw_env, right23=right23, left23=left23, n_substeps=n_sub
            )
        return vec_to_arm_action(read_arm_action(raw_env, side))
    finally:
        restore_sim(raw_env, snap)
