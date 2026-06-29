"""Apply grasp via demo replay or object-frame qpos transform."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from interaction_retarget.grasp.contact_targets import load_contact_targets_from_npz, object_pose_world
from interaction_retarget.grasp.qpos_distill import load_qpos_grasp_npz
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.replay import raw_flat_to_dict
from interaction_retarget.sim.settle import read_arm_action, settle_bimanual_actions, vec_to_arm_action
from interaction_retarget.transforms import mocap_world_from_object_frame

Side = Literal["left", "right"]
ObjectName = Literal["tray", "peg"]

# Pre-grasp mocap offset in object frame (approach from above, avoid table scrape).
_PRE_GRASP_OFFSET_OBJ: dict[ObjectName, np.ndarray] = {
    "tray": np.array([0.0, 0.0, 0.055], dtype=np.float64),
    "peg": np.array([0.0, 0.0, 0.080], dtype=np.float64),
}


def load_qpos_prototype(npz_path) -> dict:
    data = load_qpos_grasp_npz(npz_path)
    ct = load_contact_targets_from_npz(data)
    return {
        "object_name": str(data["object_name"][0]),
        "hand_side": str(data["hand_side"][0]),
        "mocap_pos_obj": np.asarray(data["mocap_pos_obj"], dtype=np.float64).reshape(3),
        "mocap_quat_obj": np.asarray(data["mocap_quat_obj"], dtype=np.float64).reshape(4),
        "hand_joint": np.asarray(data["hand_joint"], dtype=np.float64).reshape(16),
        "passive_action23": np.asarray(data["passive_action23"], dtype=np.float64).reshape(23)
        if "passive_action23" in data
        else None,
        "grasp_action23": np.asarray(data["grasp_action23"], dtype=np.float64).reshape(23)
        if "grasp_action23" in data
        else None,
        "squeeze_action23": np.asarray(data["squeeze_action23"], dtype=np.float64).reshape(23)
        if "squeeze_action23" in data
        else None,
        "contact_targets": ct,
        "representative_episode_index": int(data.get("representative_episode_index", [-1])[0]),
    }


def prototype_to_action23(
    prototype: dict,
    raw_env,
    *,
    object_name: ObjectName,
) -> np.ndarray:
    obj_pos, obj_quat = object_pose_world(raw_env, object_name)
    pos_w, quat_w = mocap_world_from_object_frame(
        prototype["mocap_pos_obj"],
        prototype["mocap_quat_obj"],
        obj_pos,
        obj_quat,
    )
    hand = np.asarray(prototype["hand_joint"], dtype=np.float64).reshape(16)
    return np.concatenate([pos_w, quat_w, hand], axis=0)


def prototype_pre_grasp_action23(
    prototype: dict,
    raw_env,
    *,
    object_name: ObjectName,
    offset_obj: np.ndarray | None = None,
    open_hand23: np.ndarray | None = None,
) -> np.ndarray:
    """Pre-grasp: same wrist as grasp, mocap shifted in object frame, fingers open."""
    grasp = prototype_to_action23(prototype, raw_env, object_name=object_name)
    off = _PRE_GRASP_OFFSET_OBJ[object_name] if offset_obj is None else np.asarray(offset_obj, dtype=np.float64)
    obj_pos, obj_quat = object_pose_world(raw_env, object_name)
    pos_obj = np.asarray(prototype["mocap_pos_obj"], dtype=np.float64).reshape(3) + off.reshape(3)
    quat_obj = np.asarray(prototype["mocap_quat_obj"], dtype=np.float64).reshape(4)
    pos_w, quat_w = mocap_world_from_object_frame(pos_obj, quat_obj, obj_pos, obj_quat)
    side: Side = "left" if object_name == "tray" else "right"
    if open_hand23 is not None:
        hand = vec_to_arm_action(open_hand23)[7:23]
    else:
        hand = vec_to_arm_action(read_arm_action(raw_env, side))[7:23]
    pre = grasp.copy()
    pre[0:3] = pos_w
    pre[3:7] = quat_w
    pre[7:23] = hand
    return pre


def apply_side_prototype(
    raw_env,
    prototype: dict,
    *,
    object_name: ObjectName,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    n_substeps: int | None = None,
) -> np.ndarray:
    side: Side = "left" if object_name == "tray" else "right"
    active = prototype_to_action23(prototype, raw_env, object_name=object_name)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    n = n_substeps if n_substeps is not None else max(int(getattr(raw_env, "_n_substeps", 1)), 1)
    if side == "left":
        settle_bimanual_actions(raw_env, right23=hold_right, left23=active, n_substeps=n)
    else:
        settle_bimanual_actions(raw_env, right23=active, left23=hold_left, n_substeps=n)
    return vec_to_arm_action(read_arm_action(raw_env, side))


def prototype_squeeze_action23(
    prototype: dict,
    raw_env,
    *,
    object_name: ObjectName,
) -> np.ndarray:
    """Object-frame mocap at grasp + demo squeeze finger joints."""
    grasp = prototype_to_action23(prototype, raw_env, object_name=object_name)
    squeeze23 = prototype.get("squeeze_action23")
    if squeeze23 is None:
        return grasp
    squeeze23 = vec_to_arm_action(squeeze23)
    out = grasp.copy()
    out[7:23] = squeeze23[7:23]
    return out


def passive_hold23(prototype: dict, raw_env, *, side: Side) -> np.ndarray:
    """Passive arm qpos from prototype, else current sim pose."""
    passive = prototype.get("passive_action23")
    if passive is not None:
        return vec_to_arm_action(passive)
    return vec_to_arm_action(read_arm_action(raw_env, side))


def replay_demo_to_hold_frame(
    env,
    entry: dict[str, Any],
    *,
    hold_frame: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay unified demo to pre-lift squeeze frame (stable bimanual contacts)."""
    from dexjoco.tasks import CONFIG_MAPPING
    from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

    from interaction_retarget.grasp.qpos_distill import _squeeze_frame

    raw_env = env.unwrapped
    actions, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    timing = entry["timing"]
    if hold_frame is None:
        hold_frame = max(
            _squeeze_frame(timing, "tray", n_actions=len(actions)),
            _squeeze_frame(timing, "peg", n_actions=len(actions)),
        )
    hold_frame = int(np.clip(int(hold_frame), 0, len(actions) - 1))
    config = CONFIG_MAPPING["bimanual_assembly"]()
    env.reset()
    if initial_state is not None and has_restorer("bimanual_assembly"):
        restore_initial_state(env, "bimanual_assembly", config, initial_state)
    for action in actions[: hold_frame + 1]:
        raw_env.step(raw_flat_to_dict(action))
    return (
        vec_to_arm_action(read_arm_action(raw_env, "right")),
        vec_to_arm_action(read_arm_action(raw_env, "left")),
    )


def replay_demo_to_grasp_frame(
    env,
    entry: dict[str, Any],
    *,
    object_name: ObjectName,
) -> tuple[np.ndarray, np.ndarray]:
    """DexGraspBench-style: replay zarr with env.step to grasp frame (establishes contacts)."""
    from interaction_retarget.grasp.ik import warm_start_from_demo

    timing = entry["timing"]
    frame_key = "left_grasp_frame" if object_name == "tray" else "right_grasp_frame"
    frame = int(timing[frame_key])
    return warm_start_from_demo(env, zarr_path=Path(entry["zarr_path"]), grasp_frame=frame)


def replay_demo_peg_segment(
    raw_env,
    entry: dict[str, Any],
    *,
    locked_left23: np.ndarray | None = None,
) -> np.ndarray:
    """Replay left_grasp→right_grasp (full demo bimanual keeps tray contacts)."""
    timing = entry["timing"]
    start = int(timing["left_grasp_frame"])
    end = int(timing["right_grasp_frame"])
    actions, _, _ = load_zarr_episode(Path(entry["zarr_path"]))
    start = int(np.clip(start, 0, len(actions) - 1))
    end = int(np.clip(end, start, len(actions) - 1))
    locked = vec_to_arm_action(locked_left23) if locked_left23 is not None else None
    for action in actions[start : end + 1]:
        act = raw_flat_to_dict(action)
        if locked is not None:
            act["left"] = locked
        raw_env.step(act)
    return vec_to_arm_action(read_arm_action(raw_env, "right"))
