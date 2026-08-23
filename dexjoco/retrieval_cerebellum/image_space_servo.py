"""RGB-only image-space observations and locally identified visual servoing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .spatial_visual_learning import build_spatial_visual_model
from .spatial_visual_supervision import CAMERA_KEYS, KEYPOINT_NAMES


FEATURE_DIM = 4


def _vector(value: np.ndarray, size: int, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


@dataclass(frozen=True)
class ImageSpaceObservation:
    """Deployable ego-camera assembly observation."""

    feature4: np.ndarray
    keypoints_uv: np.ndarray
    visibility_probability: np.ndarray
    heatmap_peak: np.ndarray
    reliability: float

    def __post_init__(self) -> None:
        feature = _vector(self.feature4, FEATURE_DIM, name="feature4")
        keypoints = np.asarray(self.keypoints_uv, dtype=np.float64)
        visibility = np.asarray(self.visibility_probability, dtype=np.float64).reshape(-1)
        peak = np.asarray(self.heatmap_peak, dtype=np.float64).reshape(-1)
        expected = len(KEYPOINT_NAMES)
        if keypoints.shape != (expected, 2):
            raise ValueError(f"keypoints_uv must have shape ({expected}, 2)")
        if visibility.shape != (expected,) or peak.shape != (expected,):
            raise ValueError("visibility_probability and heatmap_peak have invalid shape")
        reliability = float(self.reliability)
        if not 0.0 <= reliability <= 1.0:
            raise ValueError("reliability must be in [0, 1]")
        for name, value in (
            ("feature4", feature),
            ("keypoints_uv", keypoints.copy()),
            ("visibility_probability", visibility.copy()),
            ("heatmap_peak", peak.copy()),
        ):
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "reliability", reliability)


def image_alignment_feature(keypoints_uv: np.ndarray, *, image_size: int) -> np.ndarray:
    """Return tip and projected-axis mismatch in normalized image coordinates."""
    points = np.asarray(keypoints_uv, dtype=np.float64)
    if points.shape != (len(KEYPOINT_NAMES), 2):
        raise ValueError("keypoints_uv has invalid shape")
    if image_size <= 1:
        raise ValueError("image_size must be greater than one")
    peg_tip, peg_axis, hole_entry, hole_axis = points
    scale = float(image_size - 1)
    return np.concatenate(
        [
            (peg_tip - hole_entry) / scale,
            ((peg_axis - peg_tip) - (hole_axis - hole_entry)) / scale,
        ]
    )


class EgoSpatialVisualEstimator:
    """Frozen RGB-only keypoint estimator used by the visual-servo prototype."""

    def __init__(self, checkpoint_path: Path, *, device: str = "cpu") -> None:
        import torch

        self._torch = torch
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.input_size = int(checkpoint["input_size"])
        self.output_size = int(checkpoint["output_size"])
        self.model = build_spatial_visual_model(
            base_channels=int(checkpoint["base_channels"])
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.camera_index = CAMERA_KEYS.index("ego")

    def predict(self, image: np.ndarray) -> ImageSpaceObservation:
        torch = self._torch
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("image must have shape (H, W, 3)")
        resized = cv2.resize(
            rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0
        normalized = (resized - 0.5) / 0.5
        tensor = torch.from_numpy(np.transpose(normalized, (2, 0, 1))[None]).to(
            self.device
        )
        camera = torch.tensor([self.camera_index], device=self.device)
        with torch.no_grad():
            output = self.model(tensor, camera)
            heatmap_probability = torch.softmax(
                output["heatmap_logits"].flatten(start_dim=2), dim=-1
            )[0]
            rows, columns = torch.meshgrid(
                torch.arange(self.output_size, device=self.device),
                torch.arange(self.output_size, device=self.device),
                indexing="ij",
            )
            keypoints = torch.stack(
                [
                    torch.sum(heatmap_probability * columns.flatten(), dim=-1),
                    torch.sum(heatmap_probability * rows.flatten(), dim=-1),
                ],
                dim=-1,
            )
            peak = heatmap_probability.max(dim=-1).values
            entropy = -torch.sum(
                heatmap_probability
                * torch.log(torch.clamp(heatmap_probability, min=1e-12)),
                dim=-1,
            )
            visibility = torch.sigmoid(output["visibility_logits"])[0]
        keypoints_np = keypoints.cpu().numpy()
        visibility_np = visibility.cpu().numpy()
        peak_np = peak.cpu().numpy()
        concentration_np = (
            1.0 - entropy.cpu().numpy() / np.log(float(self.output_size**2))
        )
        reliability = float(
            np.clip(
                np.sqrt(np.min(visibility_np) * np.min(concentration_np)),
                0.0,
                1.0,
            )
        )
        return ImageSpaceObservation(
            feature4=image_alignment_feature(keypoints_np, image_size=self.output_size),
            keypoints_uv=keypoints_np,
            visibility_probability=visibility_np,
            heatmap_peak=peak_np,
            reliability=reliability,
        )


@dataclass(frozen=True)
class LocalImageJacobian:
    matrix: np.ndarray
    singular_values: np.ndarray
    rank: int
    condition_number: float


def identify_central_difference_jacobian(
    positive_features: np.ndarray,
    negative_features: np.ndarray,
    probe_amplitudes: np.ndarray,
    *,
    minimum_singular_value: float = 1e-5,
) -> LocalImageJacobian:
    """Identify feature change per control unit from symmetric safe probes."""
    positive = np.asarray(positive_features, dtype=np.float64)
    negative = np.asarray(negative_features, dtype=np.float64)
    amplitudes = np.asarray(probe_amplitudes, dtype=np.float64).reshape(-1)
    if positive.shape != negative.shape or positive.ndim != 2:
        raise ValueError("positive_features and negative_features must match")
    if positive.shape[1] != FEATURE_DIM or positive.shape[0] != len(amplitudes):
        raise ValueError("probe feature arrays have invalid shape")
    if np.any(amplitudes <= 0.0):
        raise ValueError("probe amplitudes must be positive")
    matrix = ((positive - negative) / (2.0 * amplitudes[:, None])).T
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular_values >= minimum_singular_value))
    condition = float(
        np.inf
        if singular_values[-1] < minimum_singular_value
        else singular_values[0] / singular_values[-1]
    )
    return LocalImageJacobian(matrix, singular_values, rank, condition)


def damped_servo_command(
    feature4: np.ndarray,
    jacobian: LocalImageJacobian,
    *,
    damping: float = 0.02,
    gain: float = 0.7,
    limits: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a bounded control that locally reduces the image-space error."""
    feature = _vector(feature4, FEATURE_DIM, name="feature4")
    matrix = np.asarray(jacobian.matrix, dtype=np.float64)
    if matrix.shape[0] != FEATURE_DIM:
        raise ValueError("jacobian output dimension must be four")
    if jacobian.rank < FEATURE_DIM:
        raise ValueError("image jacobian is not full row rank")
    if damping <= 0.0 or not 0.0 < gain <= 1.0:
        raise ValueError("damping must be positive and gain must be in (0, 1]")
    gram = matrix @ matrix.T + damping**2 * np.eye(FEATURE_DIM)
    command = -gain * matrix.T @ np.linalg.solve(gram, feature)
    if limits is not None:
        bounds = np.asarray(limits, dtype=np.float64).reshape(-1)
        if bounds.shape != command.shape or np.any(bounds <= 0.0):
            raise ValueError("limits must be positive and match the command")
        command = np.clip(command, -bounds, bounds)
    return command
