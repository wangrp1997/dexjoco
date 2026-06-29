"""Retrieve nearest demo by tabletop object pose."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class ObjectPose:
    pos: np.ndarray
    quat: np.ndarray  # wxyz


@dataclass(frozen=True)
class ScenePose:
    tray: ObjectPose
    peg: ObjectPose


def _yaw_rad(quat_wxyz: np.ndarray) -> float:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return float(R.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")[2])


def pose_distance(
    current: ScenePose,
    demo: ScenePose,
    *,
    xy_weight: float = 1.0,
    yaw_weight: float = 0.08,
) -> float:
    """Weighted xy + yaw distance between two tabletop scenes."""
    d = 0.0
    for cur, ref in ((current.tray, demo.tray), (current.peg, demo.peg)):
        dxy = float(np.linalg.norm(cur.pos[:2] - ref.pos[:2]))
        dyaw = abs(_yaw_rad(cur.quat) - _yaw_rad(ref.quat))
        dyaw = min(dyaw, 2.0 * np.pi - dyaw)
        d += xy_weight * dxy + yaw_weight * dyaw
    return d


def nearest_demo_index(
    current: ScenePose,
    demo_poses: list[tuple[int, ScenePose]],
) -> tuple[int, float]:
    """Return (episode_index, distance) for the closest demo."""
    if not demo_poses:
        raise ValueError("empty demo pose list")
    best_idx = -1
    best_dist = float("inf")
    for ep_idx, pose in demo_poses:
        dist = pose_distance(current, pose)
        if dist < best_dist:
            best_dist = dist
            best_idx = int(ep_idx)
    return best_idx, best_dist
