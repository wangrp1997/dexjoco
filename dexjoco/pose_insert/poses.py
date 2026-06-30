"""Pose helpers compatible with PoseInsert ``dataset/pose_data.py``."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """Convert (x, y, z, qx, qy, qz, qw) to a 4x4 homogeneous matrix."""
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_quat(pose7[3:7]).as_matrix()
    mat[:3, 3] = pose7[:3]
    return mat


def matrix_to_pose7(matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 matrix to (x, y, z, qx, qy, qz, qw)."""
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    quat = R.from_matrix(matrix[:3, :3]).as_quat()
    return np.concatenate([matrix[:3, 3], quat], dtype=np.float64)


def xpos_xquat_to_pose7(xpos: np.ndarray, xquat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_quat(np.asarray(xquat_wxyz, dtype=np.float64)[[1, 2, 3, 0]]).as_quat()
    return np.concatenate([np.asarray(xpos, dtype=np.float64).reshape(3), quat_xyzw])


def xpos_xmat_to_pose7(xpos: np.ndarray, xmat: np.ndarray) -> np.ndarray:
    rot = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    quat = R.from_matrix(rot).as_quat()
    return np.concatenate([np.asarray(xpos, dtype=np.float64).reshape(3), quat])


def poses7_to_matrices(poses7: np.ndarray) -> np.ndarray:
    poses7 = np.asarray(poses7, dtype=np.float64)
    if poses7.ndim != 2 or poses7.shape[1] != 7:
        raise ValueError(f"Expected (T, 7) poses, got {poses7.shape}")
    mats = np.zeros((poses7.shape[0], 4, 4), dtype=np.float64)
    for i in range(poses7.shape[0]):
        mats[i] = pose7_to_matrix(poses7[i])
    return mats


def source_in_target_poses(source_pose7: np.ndarray, target_pose7: np.ndarray) -> np.ndarray:
    """PoseInsert-style relative pose: source expressed in target frame, shape (T, 7)."""
    source_mats = poses7_to_matrices(source_pose7)
    target_mats = poses7_to_matrices(target_pose7)
    rel = np.zeros_like(source_pose7, dtype=np.float64)
    for i in range(source_pose7.shape[0]):
        t_target_inv = np.linalg.inv(target_mats[i])
        rel[i] = matrix_to_pose7(t_target_inv @ source_mats[i])
    return rel
