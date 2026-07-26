"""Object-centric SE(3) transforms (MimicGen-style) for 23-d arm actions."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_wxyz_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return R.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def mat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    xyzw = R.from_matrix(np.asarray(rot, dtype=np.float64).reshape(3, 3)).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def pose7_to_mat(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = quat_wxyz_to_mat(quat_wxyz)
    t[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return t


def mat_to_pose7(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    return mat[:3, 3].copy(), mat_to_quat_wxyz(mat[:3, :3])


def invert_pose(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = mat[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ mat[:3, 3]
    return out


def transform_eef_poses_by_object(
    src_eef_mats: np.ndarray,
    src_obj_mat: np.ndarray,
    cur_obj_mat: np.ndarray,
) -> np.ndarray:
    """Preserve EE-relative-to-object poses under a new object pose (MimicGen)."""
    src = np.asarray(src_eef_mats, dtype=np.float64).reshape(-1, 4, 4)
    src_obj_inv = invert_pose(src_obj_mat)
    rel = np.matmul(src_obj_inv[None], src)
    return np.matmul(np.asarray(cur_obj_mat, dtype=np.float64).reshape(4, 4)[None], rel)


def action23_pose_mat(action23: np.ndarray) -> np.ndarray:
    a = np.asarray(action23, dtype=np.float64).reshape(23)
    return pose7_to_mat(a[0:3], a[3:7])


def apply_pose_mat_to_action23(action23: np.ndarray, pose_mat: np.ndarray) -> np.ndarray:
    a = np.asarray(action23, dtype=np.float64).reshape(23).copy()
    pos, quat = mat_to_pose7(pose_mat)
    a[0:3] = pos
    a[3:7] = quat
    return a


def transform_action23_segment(
    actions23: np.ndarray,
    src_obj_pos: np.ndarray,
    src_obj_quat: np.ndarray,
    cur_obj_pos: np.ndarray,
    cur_obj_quat: np.ndarray,
) -> np.ndarray:
    """Transform a sequence of 23-d actions (pos+quat+hand); hands copied as-is."""
    acts = np.asarray(actions23, dtype=np.float64).reshape(-1, 23)
    src_obj = pose7_to_mat(src_obj_pos, src_obj_quat)
    cur_obj = pose7_to_mat(cur_obj_pos, cur_obj_quat)
    src_mats = np.stack([action23_pose_mat(a) for a in acts], axis=0)
    new_mats = transform_eef_poses_by_object(src_mats, src_obj, cur_obj)
    out = []
    for a, m in zip(acts, new_mats):
        out.append(apply_pose_mat_to_action23(a, m))
    return np.stack(out, axis=0)
