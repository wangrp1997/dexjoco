"""Convert DexJoCo / LeRobot action layouts for sim replay."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def rotvec_dual_arm_to_policy(action44: np.ndarray) -> np.ndarray:
    """LeRobot 44-d rotvec action -> DualArmPolicyWrapper 46-d quat action."""
    action44 = np.asarray(action44, dtype=np.float64).reshape(-1)
    if action44.shape[0] != 44:
        raise ValueError(f"Expected 44-d rotvec action, got shape {action44.shape}")

    r_xyz = action44[0:3]
    r_rot = action44[3:6]
    r_hand = action44[6:22]
    l_xyz = action44[22:25]
    l_rot = action44[25:28]
    l_hand = action44[28:44]

    r_quat = R.from_rotvec(r_rot).as_quat(scalar_first=True)
    l_quat = R.from_rotvec(l_rot).as_quat(scalar_first=True)
    return np.concatenate([r_xyz, r_quat, l_xyz, l_quat, r_hand, l_hand], axis=0)


def policy_dual_arm_to_raw(action46: np.ndarray) -> dict[str, np.ndarray]:
    """DualArmPolicyWrapper 46-d action -> raw bimanual env dict action."""
    action46 = np.asarray(action46, dtype=np.float64).reshape(-1)
    if action46.shape[0] != 46:
        raise ValueError(f"Expected 46-d policy action, got shape {action46.shape}")

    right_pose = action46[0:7]
    left_pose = action46[7:14]
    right_hand = action46[14:30]
    left_hand = action46[30:46]
    return {
        "right": np.concatenate([right_pose, right_hand], axis=0),
        "left": np.concatenate([left_pose, left_hand], axis=0),
    }
