"""Port of GraspTTA ``Contact_loss`` for Allegro fingertips.

Source: refs/GraspTTA/utils/loss.py
"""

from __future__ import annotations

import numpy as np

from interaction_retarget.constants import FINGERTIP_KEYPOINT_INDICES
from interaction_retarget.GraspTTA.utils.utils_loss import nearest_neighbor_distances


def contact_loss_object_cmap(
    obj_xyz: np.ndarray,
    hand_xyz: np.ndarray,
    cmap: np.ndarray,
    *,
    scale: float = 3000.0,
    hand_prior_indices: tuple[int, ...] | None = None,
) -> float:
    """Hand-centric contact: object points in cmap region should be close to hand prior verts.

    ``cmap``: bool or float mask on object points (GraspTTA dynamic contact region).
    """
    obj_xyz = np.asarray(obj_xyz, dtype=np.float64).reshape(-1, 3)
    hand_xyz = np.asarray(hand_xyz, dtype=np.float64).reshape(-1, 3)
    cmap = np.asarray(cmap, dtype=np.float64).reshape(-1)
    if obj_xyz.shape[0] != cmap.shape[0]:
        raise ValueError(f"cmap length {cmap.shape[0]} != obj points {obj_xyz.shape[0]}")

    prior = hand_prior_indices if hand_prior_indices is not None else FINGERTIP_KEYPOINT_INDICES
    hand_prior = hand_xyz[list(prior), :]
    obj_cd, _ = nearest_neighbor_distances(obj_xyz, hand_prior)
    mask = cmap > 0.5
    n_points = int(np.sum(mask))
    if n_points <= 0:
        return 0.0
    return float(scale * np.sum(obj_cd[mask]) / n_points)
