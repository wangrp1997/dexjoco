"""Cross-attention heatmap overlays for DexQuery eval visualization."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import zoom


def _to_uint8_hwc(image: Any) -> np.ndarray:
    if isinstance(image, np.ndarray):
        frame = np.ascontiguousarray(image)
    else:
        frame = np.ascontiguousarray(np.array(image))
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255.0).clip(0, 255)
        frame = frame.astype(np.uint8)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def _jet_colormap(normalized: np.ndarray) -> np.ndarray:
    x = np.clip(normalized, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def split_attn_by_camera(
    attn_weights: np.ndarray,
    *,
    num_cameras: int,
) -> list[np.ndarray]:
    """Split flat patch attention into one 2D grid per camera."""
    weights = np.asarray(attn_weights, dtype=np.float32).reshape(-1)
    if num_cameras < 1:
        raise ValueError(f"num_cameras must be >= 1, got {num_cameras}")
    if weights.size % num_cameras != 0:
        raise ValueError(
            f"Attention length {weights.size} is not divisible by num_cameras={num_cameras}"
        )
    patches_per_camera = weights.size // num_cameras
    side = int(round(np.sqrt(patches_per_camera)))
    if side * side != patches_per_camera:
        raise ValueError(
            f"Expected square patch grid per camera, got {patches_per_camera} patches"
        )
    grids: list[np.ndarray] = []
    for cam_idx in range(num_cameras):
        start = cam_idx * patches_per_camera
        end = start + patches_per_camera
        grids.append(weights[start:end].reshape(side, side))
    return grids


def overlay_heatmap_on_image(
    image: Any,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a 2D attention grid onto an RGB frame."""
    frame = _to_uint8_hwc(image)
    heat = np.asarray(heatmap, dtype=np.float32)
    if heat.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {heat.shape}")

    height, width = frame.shape[:2]
    if heat.shape != (height, width):
        zoom_y = height / heat.shape[0]
        zoom_x = width / heat.shape[1]
        heat = zoom(heat, (zoom_y, zoom_x), order=1)

    heat = heat - heat.min()
    denom = float(heat.max()) if float(heat.max()) > 0 else 1.0
    heat_norm = heat / denom
    color = _jet_colormap(heat_norm)
    blended = (alpha * color + (1.0 - alpha) * frame).clip(0, 255).astype(np.uint8)
    return blended


def build_attn_overlay_frames(
    observation: dict[str, Any],
    attn_weights: np.ndarray,
    *,
    camera_names: tuple[str, ...],
    num_cameras: int,
    alpha: float = 0.45,
) -> dict[str, np.ndarray]:
    """Return per-camera RGB frames with cross-attention heatmaps overlaid."""
    grids = split_attn_by_camera(attn_weights, num_cameras=num_cameras)
    overlays: dict[str, np.ndarray] = {}
    for camera_name, grid in zip(camera_names, grids, strict=True):
        if camera_name not in observation:
            continue
        overlays[camera_name] = overlay_heatmap_on_image(
            observation[camera_name],
            grid,
            alpha=alpha,
        )
    return overlays
