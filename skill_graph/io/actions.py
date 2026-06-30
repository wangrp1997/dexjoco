"""Zarr flat action → env policy 46d."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def zarr_to_policy46(action_flat: np.ndarray) -> np.ndarray:
    action_flat = np.asarray(action_flat, dtype=np.float64).reshape(-1)
    if action_flat.shape[0] == 46:
        right, left = action_flat[:23], action_flat[23:46]
    elif action_flat.shape[0] == 44:
        r_xyz, r_rot, r_hand = action_flat[0:3], action_flat[3:6], action_flat[6:22]
        l_xyz, l_rot, l_hand = action_flat[22:25], action_flat[25:28], action_flat[28:44]
        r_quat = R.from_rotvec(r_rot).as_quat(scalar_first=True)
        l_quat = R.from_rotvec(l_rot).as_quat(scalar_first=True)
        right = np.concatenate([r_xyz, r_quat, r_hand])
        left = np.concatenate([l_xyz, l_quat, l_hand])
    else:
        raise ValueError(f"Unsupported action dim {action_flat.shape[0]}")
    return np.concatenate([right[:7], left[:7], right[7:], left[7:]], axis=0).astype(np.float32)


def zarr_to_raw_dict(action_flat: np.ndarray) -> dict[str, np.ndarray]:
    action_flat = np.asarray(action_flat, dtype=np.float64).reshape(-1)
    if action_flat.shape[0] == 46:
        return {"right": action_flat[:23], "left": action_flat[23:46]}
    if action_flat.shape[0] == 44:
        policy = zarr_to_policy46(action_flat)
        return {
            "right": np.concatenate([policy[:7], policy[14:30]]),
            "left": np.concatenate([policy[7:14], policy[30:46]]),
        }
    raise ValueError(f"Unsupported action dim {action_flat.shape[0]}")
