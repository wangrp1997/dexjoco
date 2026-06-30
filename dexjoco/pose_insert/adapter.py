"""Convert PoseInsert 9d relative poses to DexJoCo right-arm mocap actions."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from interaction_retarget.constants import PEG_BODY
from interaction_retarget.sim.settle import read_arm_action
from pose_insert.dataset_sim import normalize_translation, pose7_sequence_to_pose9
from pose_insert.poses import matrix_to_pose7, pose7_to_matrix, source_in_target_poses, xpos_xmat_to_pose7, xpos_xquat_to_pose7

_SOCKET_SITE = "industreal_tray_insert_round_peg_8mm_socket_site"


def pose9_to_matrix4(pose9: np.ndarray) -> np.ndarray:
    """Rebuild a 4x4 transform from PoseInsert (3,3) columns [x, y, translation]."""
    mat33 = np.asarray(pose9, dtype=np.float64).reshape(3, 3)
    x = mat33[:, 0]
    y = mat33[:, 1]
    z = np.cross(x, y)
    z_norm = float(np.linalg.norm(z))
    if z_norm < 1e-8:
        raise ValueError("degenerate rotation columns in pose9")
    z = z / z_norm
    out = np.eye(4, dtype=np.float64)
    out[:3, 0] = x
    out[:3, 1] = y
    out[:3, 2] = z
    out[:3, 3] = mat33[:, 2]
    return out


def matrix4_to_pose9(matrix4: np.ndarray) -> np.ndarray:
    matrix4 = np.asarray(matrix4, dtype=np.float64).reshape(4, 4)
    return matrix4[:3, [0, 1, 3]]


def workspace_translation_bounds(workspace: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    workspace = np.asarray(workspace, dtype=np.float64)
    trans_max = np.array(
        [workspace[:, 0].max(), workspace[:, 1].max(), workspace[:, 2].max()],
        dtype=np.float64,
    )
    trans_min = np.array(
        [workspace[:, 3].min(), workspace[:, 4].min(), workspace[:, 5].min()],
        dtype=np.float64,
    )
    return trans_max, trans_min


def denormalize_translation(workspace: np.ndarray, matrix4: np.ndarray) -> np.ndarray:
    """Invert ``normalize_translation`` on the translation column."""
    out = np.asarray(matrix4, dtype=np.float64).reshape(4, 4).copy()
    trans_max, trans_min = workspace_translation_bounds(workspace)
    out[:3, 3] = (out[:3, 3] + 1.0) / 2.0 * (trans_max - trans_min) + trans_min
    return out


def relative_pose9_to_world_source(
    pose9: np.ndarray,
    target_pose7: np.ndarray,
    *,
    workspace: np.ndarray | None = None,
) -> np.ndarray:
    """Map normalized relative pose9 to peg pose7 in world frame."""
    rel4 = pose9_to_matrix4(pose9)
    if workspace is not None:
        rel4 = denormalize_translation(workspace, rel4)
    target4 = pose7_to_matrix(target_pose7)
    source4 = target4 @ rel4
    return matrix_to_pose7(source4)


def calibrate_peg_to_wrist(peg_pose7: np.ndarray, wrist_pose7: np.ndarray) -> np.ndarray:
    """Fixed peg-body -> wrist transform captured at handoff."""
    peg4 = pose7_to_matrix(peg_pose7)
    wrist4 = pose7_to_matrix(wrist_pose7)
    return np.linalg.inv(peg4) @ wrist4


def calibrate_socket_in_left_wrist(left_wrist_pose7: np.ndarray, socket_pose7: np.ndarray) -> np.ndarray:
    """Socket frame expressed in left wrist frame at handoff."""
    lw4 = pose7_to_matrix(left_wrist_pose7)
    socket4 = pose7_to_matrix(socket_pose7)
    return np.linalg.inv(lw4) @ socket4


def socket_pose7_to_left_wrist_pose7(socket_pose7: np.ndarray, socket_in_left_wrist4: np.ndarray) -> np.ndarray:
    socket4 = pose7_to_matrix(socket_pose7)
    lw4 = socket4 @ np.linalg.inv(socket_in_left_wrist4)
    return matrix_to_pose7(lw4)


def source_pose7_to_wrist_pose7(source_pose7: np.ndarray, peg_to_wrist4: np.ndarray) -> np.ndarray:
    source4 = pose7_to_matrix(source_pose7)
    wrist4 = source4 @ peg_to_wrist4
    return matrix_to_pose7(wrist4)


def wrist_pose7_to_rotvec_action(wrist_pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wrist_pose7 = np.asarray(wrist_pose7, dtype=np.float64).reshape(7)
    xyz = wrist_pose7[:3]
    rotvec = R.from_quat(wrist_pose7[3:7]).as_rotvec()
    return xyz, rotvec


def mocap23_to_wrist_pose7(action23: np.ndarray) -> np.ndarray:
    action23 = np.asarray(action23, dtype=np.float64).reshape(23)
    quat_wxyz = action23[3:7]
    quat_xyzw = R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_quat()
    return np.concatenate([action23[0:3], quat_xyzw], dtype=np.float64)


def read_sim_poses7(raw_env) -> tuple[np.ndarray, np.ndarray]:
    model = raw_env._model
    data = raw_env._data
    peg_id = int(model.body(PEG_BODY).id)
    socket_id = int(model.site(_SOCKET_SITE).id)
    source_pose7 = xpos_xquat_to_pose7(data.xpos[peg_id], data.xquat[peg_id])
    target_pose7 = xpos_xmat_to_pose7(data.site_xpos[socket_id], data.site_xmat[socket_id])
    return source_pose7, target_pose7


def build_obs_pose9(
    source_pose7: np.ndarray,
    target_pose7: np.ndarray,
    *,
    workspace: np.ndarray | None = None,
) -> np.ndarray:
    rel7 = source_in_target_poses(
        np.asarray(source_pose7, dtype=np.float64).reshape(1, 7),
        np.asarray(target_pose7, dtype=np.float64).reshape(1, 7),
    )
    if workspace is not None:
        rel7 = normalize_translation(workspace, rel7)
    return pose7_sequence_to_pose9(rel7)[0]


def clamp_wrist_target(
    cur_xyz: np.ndarray,
    cur_rotvec: np.ndarray,
    tgt_xyz: np.ndarray,
    tgt_rotvec: np.ndarray,
    *,
    max_step_m: float,
    max_rot_step_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    cur_xyz = np.asarray(cur_xyz, dtype=np.float64)
    cur_rotvec = np.asarray(cur_rotvec, dtype=np.float64)
    tgt_xyz = np.asarray(tgt_xyz, dtype=np.float64)
    tgt_rotvec = np.asarray(tgt_rotvec, dtype=np.float64)

    delta = tgt_xyz - cur_xyz
    norm = float(np.linalg.norm(delta))
    if norm > max_step_m and norm > 1e-9:
        delta = delta * (max_step_m / norm)
    new_xyz = cur_xyz + delta

    r_cur = R.from_rotvec(cur_rotvec)
    r_tgt = R.from_rotvec(tgt_rotvec)
    delta_r = r_cur.inv() * r_tgt
    delta_rotvec = delta_r.as_rotvec()
    angle = float(np.linalg.norm(delta_rotvec))
    if angle > max_rot_step_rad and angle > 1e-9:
        delta_rotvec = delta_rotvec * (max_rot_step_rad / angle)
    new_rotvec = (r_cur * R.from_rotvec(delta_rotvec)).as_rotvec()
    return new_xyz, new_rotvec


def read_right_wrist_state(raw_env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    right23 = read_arm_action(raw_env, "right")
    xyz = right23[0:3].copy()
    rotvec = R.from_quat(right23[3:7], scalar_first=True).as_rotvec()
    hand = right23[7:23].copy()
    return xyz, rotvec, hand
