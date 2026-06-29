"""Port of GraspTTA ``utils_loss.get_NN`` (numpy / scipy).

Source: refs/GraspTTA/utils/utils_loss.py
"""

from __future__ import annotations

import numpy as np


def nearest_neighbor_distances(
    src_xyz: np.ndarray,
    trg_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For each src point, NN on trg. Returns (squared_dist, index), shapes (N1,)."""
    src = np.asarray(src_xyz, dtype=np.float64).reshape(-1, 3)
    trg = np.asarray(trg_xyz, dtype=np.float64).reshape(-1, 3)
    if src.size == 0 or trg.size == 0:
        n = src.shape[0]
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.int64)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(trg)
        dist, idx = tree.query(src, k=1)
        return np.square(np.asarray(dist, dtype=np.float64)), np.asarray(idx, dtype=np.int64)
    except ImportError:
        diff = src[:, None, :] - trg[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        idx = np.argmin(d2, axis=1)
        return d2[np.arange(src.shape[0]), idx], idx
