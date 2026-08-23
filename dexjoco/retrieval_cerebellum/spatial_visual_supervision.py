"""Offline spatial supervision for V2 multi-camera visual estimation.

This module converts MuJoCo replay state into image-plane labels.  The labels
are teacher-only training artifacts; no function here is used by the online V2
controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CAMERA_KEYS = ("ego", "wrist_left", "wrist_right")
KEYPOINT_NAMES = (
    "peg_tip",
    "peg_axis_point",
    "hole_entry",
    "hole_axis_point",
)
BACKGROUND_CLASS = 0
PEG_CLASS = 1
SOCKET_CLASS = 2
GEOM_OBJECT_TYPE = 5


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole calibration using MuJoCo's camera frame convention.

    ``rotation_world_from_camera`` maps camera-frame vectors to world vectors.
    MuJoCo cameras look along local ``-z`` with local ``+y`` pointing upward.
    """

    width: int
    height: int
    focal_x: float
    focal_y: float
    center_x: float
    center_y: float
    position_world: np.ndarray
    rotation_world_from_camera: np.ndarray

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [self.focal_x, 0.0, self.center_x],
                [0.0, self.focal_y, self.center_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def rescaled(self, *, width: int, height: int) -> "CameraCalibration":
        """Scale pixel coordinates to a resized image with unchanged camera pose."""
        if width <= 0 or height <= 0:
            raise ValueError("camera width and height must be positive")
        scale_x = float(width) / float(self.width)
        scale_y = float(height) / float(self.height)
        return CameraCalibration(
            width=int(width),
            height=int(height),
            focal_x=self.focal_x * scale_x,
            focal_y=self.focal_y * scale_y,
            center_x=self.center_x * scale_x,
            center_y=self.center_y * scale_y,
            position_world=self.position_world.copy(),
            rotation_world_from_camera=self.rotation_world_from_camera.copy(),
        )

    @classmethod
    def from_vertical_fov(
        cls,
        *,
        width: int,
        height: int,
        vertical_fov_degrees: float,
        position_world: np.ndarray,
        rotation_world_from_camera: np.ndarray,
    ) -> "CameraCalibration":
        if width <= 0 or height <= 0:
            raise ValueError("camera width and height must be positive")
        if not 0.0 < vertical_fov_degrees < 180.0:
            raise ValueError("vertical_fov_degrees must be in (0, 180)")
        focal = 0.5 * float(height) / np.tan(
            0.5 * np.deg2rad(float(vertical_fov_degrees))
        )
        return cls(
            width=int(width),
            height=int(height),
            focal_x=float(focal),
            focal_y=float(focal),
            center_x=0.5 * float(width),
            center_y=0.5 * float(height),
            position_world=np.asarray(position_world, dtype=np.float64).reshape(3),
            rotation_world_from_camera=np.asarray(
                rotation_world_from_camera,
                dtype=np.float64,
            ).reshape(3, 3),
        )

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project world points to pixels, returning ``uv``, depth, and in-frame."""
        points = np.asarray(points_world, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError(f"points_world must end in dimension 3, got {points.shape}")
        flattened = points.reshape(-1, 3)
        camera = (
            self.rotation_world_from_camera.T
            @ (flattened - self.position_world).T
        ).T
        depth = -camera[:, 2]
        safe_depth = np.where(depth > 1e-9, depth, 1.0)
        horizontal = self.focal_x * camera[:, 0] / safe_depth + self.center_x
        vertical = self.center_y - self.focal_y * camera[:, 1] / safe_depth
        uv = np.column_stack([horizontal, vertical])
        in_frame = (
            (depth > 1e-9)
            & (horizontal >= 0.0)
            & (horizontal < self.width)
            & (vertical >= 0.0)
            & (vertical < self.height)
        )
        leading_shape = points.shape[:-1]
        return (
            uv.reshape(*leading_shape, 2),
            depth.reshape(leading_shape),
            in_frame.reshape(leading_shape),
        )


def assembly_keypoints_world(
    *,
    peg_tip_world: np.ndarray,
    peg_axis_world: np.ndarray,
    hole_entry_world: np.ndarray,
    hole_axis_world: np.ndarray,
    axis_length_m: float,
) -> np.ndarray:
    """Return peg/hole origins and short axis points as four 3D keypoints."""
    if axis_length_m <= 0.0:
        raise ValueError("axis_length_m must be positive")

    def normalized(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(value))
        if norm <= 1e-9:
            raise ValueError("assembly axis must be non-zero")
        return value / norm

    peg_tip = np.asarray(peg_tip_world, dtype=np.float64).reshape(3)
    hole_entry = np.asarray(hole_entry_world, dtype=np.float64).reshape(3)
    peg_axis = normalized(peg_axis_world)
    hole_axis = normalized(hole_axis_world)
    return np.stack(
        [
            peg_tip,
            peg_tip + peg_axis * axis_length_m,
            hole_entry,
            hole_entry + hole_axis * axis_length_m,
        ]
    )


def semantic_mask_from_segmentation(
    segmentation: np.ndarray,
    *,
    peg_geom_ids: set[int] | frozenset[int],
    socket_geom_ids: set[int] | frozenset[int],
    geom_object_type: int = GEOM_OBJECT_TYPE,
) -> np.ndarray:
    """Map MuJoCo ``(object_id, object_type)`` pixels to three semantic classes."""
    values = np.asarray(segmentation)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(f"segmentation must have shape (H, W, 2), got {values.shape}")
    object_ids = values[..., 0]
    object_types = values[..., 1]
    is_geom = object_types == int(geom_object_type)
    mask = np.zeros(values.shape[:2], dtype=np.uint8)
    if peg_geom_ids:
        mask[is_geom & np.isin(object_ids, tuple(peg_geom_ids))] = PEG_CLASS
    if socket_geom_ids:
        mask[is_geom & np.isin(object_ids, tuple(socket_geom_ids))] = SOCKET_CLASS
    return mask


def downsample_semantic_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbor downsample without adding an image-library dependency."""
    values = np.asarray(mask, dtype=np.uint8)
    if values.ndim != 2:
        raise ValueError(f"mask must have shape (H, W), got {values.shape}")
    if factor <= 0:
        raise ValueError("factor must be positive")
    if factor == 1:
        return values.copy()
    return values[::factor, ::factor].copy()


def resize_semantic_mask(
    mask: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Resize semantic labels with deterministic nearest-neighbor sampling."""
    values = np.asarray(mask, dtype=np.uint8)
    if values.ndim != 2:
        raise ValueError(f"mask must have shape (H, W), got {values.shape}")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    source_height, source_width = values.shape
    rows = np.minimum(
        (np.arange(height, dtype=np.float64) * source_height / height).astype(np.int64),
        source_height - 1,
    )
    columns = np.minimum(
        (np.arange(width, dtype=np.float64) * source_width / width).astype(np.int64),
        source_width - 1,
    )
    return values[rows[:, None], columns[None, :]].copy()


def keypoint_visibility(
    semantic_mask: np.ndarray,
    keypoints_uv: np.ndarray,
    in_frame: np.ndarray,
    *,
    support_classes: tuple[int, ...] = (
        PEG_CLASS,
        PEG_CLASS,
        SOCKET_CLASS,
        SOCKET_CLASS,
    ),
    support_radius_px: int = 8,
) -> np.ndarray:
    """Check whether each projected keypoint has nearby visible object pixels."""
    mask = np.asarray(semantic_mask, dtype=np.uint8)
    points = np.asarray(keypoints_uv, dtype=np.float64)
    inside = np.asarray(in_frame, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"semantic_mask must have shape (H, W), got {mask.shape}")
    if points.shape != (len(support_classes), 2):
        raise ValueError(
            f"keypoints_uv must have shape ({len(support_classes)}, 2), got {points.shape}"
        )
    if inside.shape != (len(support_classes),):
        raise ValueError(f"in_frame has wrong shape {inside.shape}")
    if support_radius_px < 0:
        raise ValueError("support_radius_px must be non-negative")
    height, width = mask.shape
    visible = np.zeros(len(support_classes), dtype=bool)
    for index, semantic_class in enumerate(support_classes):
        if not inside[index] or not np.all(np.isfinite(points[index])):
            continue
        column = int(np.rint(points[index, 0]))
        row = int(np.rint(points[index, 1]))
        row_min = max(0, row - support_radius_px)
        row_max = min(height, row + support_radius_px + 1)
        col_min = max(0, column - support_radius_px)
        col_max = min(width, column + support_radius_px + 1)
        visible[index] = bool(
            np.any(mask[row_min:row_max, col_min:col_max] == semantic_class)
        )
    return visible


def gaussian_keypoint_heatmaps(
    keypoints_uv: np.ndarray,
    visible: np.ndarray,
    *,
    height: int,
    width: int,
    sigma_px: float,
) -> np.ndarray:
    """Rasterize keypoints into one Gaussian heatmap per point."""
    points = np.asarray(keypoints_uv, dtype=np.float64)
    valid = np.asarray(visible, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"keypoints_uv must have shape (K, 2), got {points.shape}")
    if valid.shape != (points.shape[0],):
        raise ValueError(f"visible must have shape ({points.shape[0]},), got {valid.shape}")
    if height <= 0 or width <= 0 or sigma_px <= 0.0:
        raise ValueError("height, width, and sigma_px must be positive")
    rows, columns = np.mgrid[:height, :width]
    heatmaps = np.zeros((points.shape[0], height, width), dtype=np.float32)
    denominator = 2.0 * float(sigma_px) ** 2
    for index, point in enumerate(points):
        if not valid[index] or not np.all(np.isfinite(point)):
            continue
        squared_distance = (columns - point[0]) ** 2 + (rows - point[1]) ** 2
        heatmaps[index] = np.exp(-squared_distance / denominator).astype(np.float32)
    return heatmaps
