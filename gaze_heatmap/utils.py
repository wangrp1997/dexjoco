"""Heatmap helpers (from peg-in-hole-visual-servoing-model/utils.py)."""

from __future__ import annotations

import cv2
import numpy as np


def draw_points(img, points, color=(0, 0, 255)):
    for p in points:
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.drawMarker(img, (x, y), color, cv2.MARKER_TILTED_CROSS, 10, 1, cv2.LINE_AA)


def heatmap_peak(hm: np.ndarray) -> tuple[float, float]:
    y, x = np.unravel_index(int(np.argmax(hm)), hm.shape)
    return float(x), float(y)


def heatmap(sigma: float, w: int, h: int, points, d: int = 3) -> np.ndarray:
    s = int(sigma * d)
    hm = np.zeros((h, w), dtype=np.float32)
    for x, y in points:
        _x, _y = int(round(x)), int(round(y))
        xmi, xma = max(0, _x - s), min(w, _x + s)
        ymi, yma = max(0, _y - s), min(h, _y + s)
        _h, _w = yma - ymi, xma - xmi
        if _h > 0 and _w > 0:
            xs = np.arange(_w, dtype=np.float32).reshape(1, _w)
            ys = np.arange(_h, dtype=np.float32).reshape(_h, 1)
            patch = np.exp(-((x - xmi - xs) ** 2 + (y - ymi - ys) ** 2) / (2 * sigma**2))
            hm[ymi:yma, xmi:xma] = np.maximum(hm[ymi:yma, xmi:xma], patch)
    return hm
