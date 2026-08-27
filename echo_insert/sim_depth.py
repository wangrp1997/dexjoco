"""Fail-closed access to the fixed ego metric-depth sensor."""

from __future__ import annotations

import numpy as np


_DEPTH_SHAPE = (640, 640)


def read_ego_depth_m(raw) -> np.ndarray:
    """Return the fixed ego camera's linear z-depth in metres."""
    viewer = getattr(raw, "_viewer", None)
    camera_id = getattr(raw, "_front_camera_id", None)
    camera_ids = tuple(getattr(raw, "camera_id", ()))
    if viewer is None or camera_id is None:
        raise ValueError("ego depth renderer is unavailable")
    if len(camera_ids) != 3 or int(camera_id) != int(camera_ids[0]):
        raise ValueError("ego camera ordering does not match the audited sensor contract")

    zbuf = np.asarray(
        viewer.render(render_mode="depth_array", camera_id=int(camera_id)),
        dtype=np.float32,
    )
    if zbuf.shape != _DEPTH_SHAPE:
        raise ValueError(f"ego depth must have shape {_DEPTH_SHAPE}, got {zbuf.shape}")
    if not np.isfinite(zbuf).all() or np.any((zbuf < 0.0) | (zbuf > 1.0)):
        raise ValueError("ego depth z-buffer must be finite and within [0, 1]")

    extent = float(raw._model.stat.extent)
    znear = float(raw._model.vis.map.znear) * extent
    zfar = float(raw._model.vis.map.zfar) * extent
    denominator = zfar - zbuf * (zfar - znear)
    depth = np.full_like(zbuf, np.inf, dtype=np.float32)
    valid = (zbuf < 1.0) & (denominator > 0.0)
    depth[valid] = (znear * zfar / denominator)[valid]
    if not np.isfinite(depth).any():
        raise ValueError("ego depth contains no surface returns")
    return depth.copy()
