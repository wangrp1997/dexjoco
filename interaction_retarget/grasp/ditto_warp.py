"""DITTO-style demo trajectory warp (object-centric SE(3)).

Refs:
  - https://github.com/robot-learning-freiburg/DITTO — ``tracking_3D.warp_3D_trajectory``
  - https://arxiv.org/abs/2403.15203 — Demonstration Imitation by Trajectory Transformation
  - Paper site: https://ditto.cs.uni-freiburg.de

DITTO warps each demo 4×4 pose with ``T_delta @ T_demo_i`` (RGB-D + vision).
We have sim GT object pose + mocap control → store mocap in object frame, replay with
``mocap_world_from_object_frame`` (same math, no vision stack).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from interaction_retarget.transforms import (
    matrix_to_quat_wxyz,
    mocap_world_from_object_frame,
    quat_wxyz_to_matrix,
    relative_mocap_in_object_frame,
)

# Ported from refs/DITTO/DITTO/geometry.py (MIT/GPL — small SE(3) helpers).
__all__ = [
    "mocap_pose4",
    "pose4_to_mocap",
    "se3_from_body_pose",
    "se3_delta_demo_to_live",
    "warp_pose4_trajectory",
    "warp_mocap_trajectory_object_frame",
    "arm23_from_object_frame_waypoint",
]


def mocap_pose4(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """4×4 wrist pose (MuJoCo mocap pos + wxyz quat)."""
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = quat_wxyz_to_matrix(quat_wxyz)
    out[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return out


def pose4_to_mocap(pose4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose4 = np.asarray(pose4, dtype=np.float64).reshape(4, 4)
    pos = pose4[:3, 3].copy()
    quat = matrix_to_quat_wxyz(pose4[:3, :3])
    return pos, quat


def se3_from_body_pose(obj_pos: np.ndarray, obj_quat_wxyz: np.ndarray) -> np.ndarray:
    return mocap_pose4(obj_pos, obj_quat_wxyz)


def se3_delta_demo_to_live(
    demo_obj_pos: np.ndarray,
    demo_obj_quat_wxyz: np.ndarray,
    live_obj_pos: np.ndarray,
    live_obj_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """``T_live @ inv(T_demo)`` — DITTO demo→live alignment (GT pose, no RGB-D)."""
    t_demo = se3_from_body_pose(demo_obj_pos, demo_obj_quat_wxyz)
    t_live = se3_from_body_pose(live_obj_pos, live_obj_quat_wxyz)
    return t_live @ np.linalg.inv(t_demo)


def warp_pose4_trajectory(
    demo_poses4: np.ndarray,
    delta_se3: np.ndarray,
) -> np.ndarray:
    """DITTO ``warp_3D_trajectory``: ``delta @ pose_i`` for each 4×4 pose."""
    demo_poses4 = np.asarray(demo_poses4, dtype=np.float64)
    assert demo_poses4.ndim == 3 and demo_poses4.shape[1:] == (4, 4)
    delta_se3 = np.asarray(delta_se3, dtype=np.float64).reshape(4, 4)
    return np.array([delta_se3 @ p for p in demo_poses4], dtype=np.float64)


def warp_mocap_trajectory_object_frame(
    mocap_pos_obj: np.ndarray,
    mocap_quat_obj: np.ndarray,
    *,
    live_obj_pos: np.ndarray,
    live_obj_quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay object-frame mocap waypoints at current object pose (DITTO warp, sim GT)."""
    pos_obj = np.asarray(mocap_pos_obj, dtype=np.float64)
    quat_obj = np.asarray(mocap_quat_obj, dtype=np.float64)
    n = int(pos_obj.shape[0])
    pos_w = np.zeros((n, 3), dtype=np.float64)
    quat_w = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        pw, qw = mocap_world_from_object_frame(
            pos_obj[i], quat_obj[i], live_obj_pos, live_obj_quat_wxyz
        )
        pos_w[i] = pw
        quat_w[i] = qw
    return pos_w, quat_w


def arm23_from_object_frame_waypoint(
    pos_obj: np.ndarray,
    quat_obj: np.ndarray,
    hand: np.ndarray,
    *,
    live_obj_pos: np.ndarray,
    live_obj_quat_wxyz: np.ndarray,
) -> np.ndarray:
    pos_w, quat_w = mocap_world_from_object_frame(
        pos_obj, quat_obj, live_obj_pos, live_obj_quat_wxyz
    )
    return np.concatenate([pos_w, quat_w, np.asarray(hand, dtype=np.float64).reshape(16)], axis=0)


def demo_arm_to_object_frame(
    arm23: np.ndarray,
    obj_pos: np.ndarray,
    obj_quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Record one demo waypoint in object frame (offline extract)."""
    arm23 = np.asarray(arm23, dtype=np.float64).reshape(23)
    pos_obj, quat_obj = relative_mocap_in_object_frame(
        arm23[0:3], arm23[3:7], obj_pos, obj_quat_wxyz
    )
    return pos_obj, quat_obj, arm23[7:23].copy()
