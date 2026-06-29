"""Contact map matching loss from ContactOpt ``optimize_pose.py``.

Source: refs/contactopt/contactopt/optimize_pose.py (L71–L79)
"""

from __future__ import annotations

import numpy as np


def contact_map_match_loss(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    w_cont_asym: float = 2.0,
) -> float:
    """L1 mean loss with asymmetric penalty on missing contact (under-contact)."""
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if target.size == 0:
        return 0.0
    sub = target - predicted
    weighted = sub + np.maximum(sub, 0.0) * float(w_cont_asym)
    return float(np.mean(np.abs(weighted)))
