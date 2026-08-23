"""RGB-only fixed-CAD features for search and alignment prototypes."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RGBAssemblyFeature:
    peg_tip_uv: np.ndarray
    hole_uv: np.ndarray
    peg_axis_angle_rad: float
    confidence: float

    @property
    def error3(self) -> np.ndarray:
        return np.asarray(
            [
                self.peg_tip_uv[0] - self.hole_uv[0],
                self.peg_tip_uv[1] - self.hole_uv[1],
                self.peg_axis_angle_rad,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class RGBSearchAlignConfig:
    yellow_hsv_lower: tuple[int, int, int] = (15, 80, 80)
    yellow_hsv_upper: tuple[int, int, int] = (45, 255, 255)
    blue_hsv_lower: tuple[int, int, int] = (85, 40, 25)
    blue_hsv_upper: tuple[int, int, int] = (135, 255, 230)
    minimum_peg_pixels: int = 300
    minimum_socket_pixels: int = 1500
    hole_vertical_fraction: float = 0.42


class RGBSearchAlignEstimator:
    """Detect the yellow peg axis and blue socket opening from ego RGB."""

    def __init__(self, config: RGBSearchAlignConfig | None = None) -> None:
        self.config = config or RGBSearchAlignConfig()

    def estimate(self, image_rgb: np.ndarray) -> RGBAssemblyFeature | None:
        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image_rgb must have shape (H, W, 3)")
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        peg_mask = cv2.inRange(
            hsv,
            self.config.yellow_hsv_lower,
            self.config.yellow_hsv_upper,
        )
        socket_mask = cv2.inRange(
            hsv,
            self.config.blue_hsv_lower,
            self.config.blue_hsv_upper,
        )
        peg_points = np.column_stack(np.nonzero(peg_mask))[:, ::-1].astype(np.float64)
        if peg_points.shape[0] < self.config.minimum_peg_pixels:
            return None
        count, _, stats, centroids = cv2.connectedComponentsWithStats(socket_mask)
        if count <= 1:
            return None
        socket_index = max(
            range(1, count),
            key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
        )
        socket_area = int(stats[socket_index, cv2.CC_STAT_AREA])
        if socket_area < self.config.minimum_socket_pixels:
            return None

        peg_center = np.mean(peg_points, axis=0)
        _, _, axes = np.linalg.svd(peg_points - peg_center, full_matrices=False)
        axis = axes[0]
        if axis[1] < 0.0:
            axis = -axis
        projection = (peg_points - peg_center) @ axis
        insertion_tip = peg_center + axis * np.percentile(projection, 98.0)
        peg_axis_angle = float(np.arctan2(axis[0], axis[1]))

        socket_height = float(stats[socket_index, cv2.CC_STAT_HEIGHT])
        socket_centroid = np.asarray(centroids[socket_index], dtype=np.float64)
        hole = socket_centroid.copy()
        hole[1] -= self.config.hole_vertical_fraction * socket_height

        peg_support = min(1.0, peg_points.shape[0] / 1200.0)
        socket_support = min(1.0, socket_area / 6000.0)
        confidence = float(min(peg_support, socket_support))
        return RGBAssemblyFeature(
            peg_tip_uv=insertion_tip,
            hole_uv=hole,
            peg_axis_angle_rad=peg_axis_angle,
            confidence=confidence,
        )


def damped_visual_command(
    jacobian: np.ndarray,
    error3: np.ndarray,
    *,
    damping: float = 4.0,
) -> np.ndarray:
    matrix = np.asarray(jacobian, dtype=np.float64)
    error = np.asarray(error3, dtype=np.float64).reshape(3)
    if matrix.shape != (3, 6):
        raise ValueError(f"jacobian must have shape (3, 6), got {matrix.shape}")
    regularized = matrix @ matrix.T + damping**2 * np.eye(3)
    return -matrix.T @ np.linalg.solve(regularized, error)
