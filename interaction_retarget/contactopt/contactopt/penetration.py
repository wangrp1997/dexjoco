"""Port of ContactOpt ``calculate_penetration_cost`` (numpy).

Source: refs/contactopt/contactopt/diffcontact.py
"""

from __future__ import annotations

import numpy as np

from interaction_retarget.contactopt.contactopt.diffcontact import _knn_query_to_target


def penetration_cost_along_normal(
    hand_verts: np.ndarray,
    hand_normals: np.ndarray,
    object_verts: np.ndarray,
    object_normals: np.ndarray,
    *,
    allowable_pen: float = 0.002,
    contact_norm_method: int = 0,
) -> np.ndarray:
    """Per-hand-vertex penetration score (non-negative)."""
    hand_verts = np.asarray(hand_verts, dtype=np.float64).reshape(-1, 3)
    hand_normals = np.asarray(hand_normals, dtype=np.float64).reshape(-1, 3)
    object_verts = np.asarray(object_verts, dtype=np.float64).reshape(-1, 3)
    object_normals = np.asarray(object_normals, dtype=np.float64).reshape(-1, 3)

    if contact_norm_method == 5:
        hand_verts = hand_verts + hand_normals * -0.004

    _, nn_idx, _ = _knn_query_to_target(hand_verts, object_verts)
    closest_obj = object_verts[np.clip(nn_idx, 0, max(object_verts.shape[0] - 1, 0))]
    closest_normals = object_normals[np.clip(nn_idx, 0, max(object_normals.shape[0] - 1, 0))]
    delta = hand_verts - closest_obj
    dist_along_normal = np.sum(delta * closest_normals, axis=1)
    return np.maximum(0.0, -dist_along_normal - float(allowable_pen))
