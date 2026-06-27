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


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(np.asarray(rot, dtype=np.float64).reshape(3, 3)).as_quat()
    return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def relative_mocap_in_object_frame(
    mocap_pos_world: np.ndarray,
    mocap_quat_wxyz: np.ndarray,
    obj_pos: np.ndarray,
    obj_quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Store mocap pose relative to object (R_rel = R_obj^T R_mocap)."""
    pos_obj = world_to_object(np.asarray(mocap_pos_world, dtype=np.float64).reshape(1, 3), obj_pos, obj_quat_wxyz)[0]
    rot_obj = quat_wxyz_to_matrix(obj_quat_wxyz)
    rot_mocap = quat_wxyz_to_matrix(mocap_quat_wxyz)
    rot_rel = rot_obj.T @ rot_mocap
    return pos_obj, matrix_to_quat_wxyz(rot_rel)


def mocap_world_from_object_frame(
    mocap_pos_obj: np.ndarray,
    mocap_quat_obj: np.ndarray,
    obj_pos: np.ndarray,
    obj_quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply T_world_obj to canonical grasp mocap (TopoRetarget 套构型)."""
    pos_world = object_to_world(np.asarray(mocap_pos_obj, dtype=np.float64).reshape(1, 3), obj_pos, obj_quat_wxyz)[0]
    rot_world = quat_wxyz_to_matrix(obj_quat_wxyz) @ quat_wxyz_to_matrix(mocap_quat_obj)
    return pos_world, matrix_to_quat_wxyz(rot_world)
