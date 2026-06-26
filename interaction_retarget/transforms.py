"""Rigid transforms between world and object body frames."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
    return R.from_quat(quat_xyzw).as_matrix()


def world_to_object(points_world: np.ndarray, obj_pos: np.ndarray, obj_quat_wxyz: np.ndarray) -> np.ndarray:
    """Map world points into the object body frame (MuJoCo xpos/xquat)."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    rot = quat_wxyz_to_matrix(obj_quat_wxyz)
    return (points - obj_pos.reshape(1, 3)) @ rot


def object_to_world(points_obj: np.ndarray, obj_pos: np.ndarray, obj_quat_wxyz: np.ndarray) -> np.ndarray:
    points = np.asarray(points_obj, dtype=np.float64).reshape(-1, 3)
    rot = quat_wxyz_to_matrix(obj_quat_wxyz)
    return points @ rot.T + obj_pos.reshape(1, 3)
