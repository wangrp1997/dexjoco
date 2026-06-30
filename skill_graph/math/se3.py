"""Rigid transforms (object frame ↔ world)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
    return R.from_quat(q).as_matrix()


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    q = R.from_matrix(np.asarray(rot, dtype=np.float64).reshape(3, 3)).as_quat()
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64)


def world_to_object(points_world: np.ndarray, obj_pos: np.ndarray, obj_quat_wxyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    rot = quat_wxyz_to_matrix(obj_quat_wxyz)
    return (pts - obj_pos.reshape(1, 3)) @ rot


def object_to_world(points_obj: np.ndarray, obj_pos: np.ndarray, obj_quat_wxyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_obj, dtype=np.float64).reshape(-1, 3)
    rot = quat_wxyz_to_matrix(obj_quat_wxyz)
    return pts @ rot.T + obj_pos.reshape(1, 3)


def relative_mocap_in_object_frame(
    mocap_pos_world: np.ndarray,
    mocap_quat_wxyz: np.ndarray,
    obj_pos: np.ndarray,
    obj_quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pos_obj = world_to_object(np.asarray(mocap_pos_world, dtype=np.float64).reshape(1, 3), obj_pos, obj_quat_wxyz)[0]
    rot_obj = quat_wxyz_to_matrix(obj_quat_wxyz)
    rot_mocap = quat_wxyz_to_matrix(mocap_quat_wxyz)
    return pos_obj, matrix_to_quat_wxyz(rot_obj.T @ rot_mocap)


def mocap_world_from_object_frame(
    mocap_pos_obj: np.ndarray,
    mocap_quat_obj: np.ndarray,
    obj_pos: np.ndarray,
    obj_quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rot_obj = quat_wxyz_to_matrix(obj_quat_wxyz)
    rot_rel = quat_wxyz_to_matrix(mocap_quat_obj)
    rot_world = rot_obj @ rot_rel
    pos_world = object_to_world(mocap_pos_obj.reshape(1, 3), obj_pos, obj_quat_wxyz)[0]
    return pos_world, matrix_to_quat_wxyz(rot_world)


def arm23_from_object_frame(
    pos_obj: np.ndarray,
    quat_obj: np.ndarray,
    hand: np.ndarray,
    *,
    live_obj_pos: np.ndarray,
    live_obj_quat: np.ndarray,
) -> np.ndarray:
    pos_w, quat_w = mocap_world_from_object_frame(pos_obj, quat_obj, live_obj_pos, live_obj_quat)
    return np.concatenate([pos_w, quat_w, np.asarray(hand, dtype=np.float64).reshape(16)], axis=0)


def rotate_mocap_about_object_z(
    pos_obj: np.ndarray,
    quat_obj: np.ndarray,
    yaw_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate grasp pose around peg local Z (cylinder symmetry axis)."""
    if abs(float(yaw_rad)) < 1e-9:
        return np.asarray(pos_obj, dtype=np.float64), np.asarray(quat_obj, dtype=np.float64)
    rot_z = R.from_euler("z", float(yaw_rad)).as_matrix()
    pos = np.asarray(pos_obj, dtype=np.float64).reshape(3)
    rot_rel = quat_wxyz_to_matrix(quat_obj)
    return rot_z @ pos, matrix_to_quat_wxyz(rot_z @ rot_rel)


def rotate_mocap_stack_about_object_z(
    pos_obj: np.ndarray,
    quat_obj: np.ndarray,
    yaw_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(pos_obj, dtype=np.float64)
    quat = np.asarray(quat_obj, dtype=np.float64)
    if pos.ndim == 1:
        p, q = rotate_mocap_about_object_z(pos, quat, yaw_rad)
        return p, q
    out_pos, out_quat = [], []
    for i in range(pos.shape[0]):
        p, q = rotate_mocap_about_object_z(pos[i], quat[i], yaw_rad)
        out_pos.append(p)
        out_quat.append(q)
    return np.stack(out_pos, axis=0), np.stack(out_quat, axis=0)
