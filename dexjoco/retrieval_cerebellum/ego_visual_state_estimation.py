"""Deployable ego-camera spatial features for V2 state estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .spatial_visual_learning import IMAGE_MEAN, IMAGE_STD, build_spatial_visual_model


EGO_SPATIAL_FEATURE_DIM = 38
EGO_VISUAL_PROPRIO_FEATURE_DIM = EGO_SPATIAL_FEATURE_DIM + 46 + 44


def preprocess_rgb_batch(
    images: list[np.ndarray],
    *,
    input_size: int,
) -> np.ndarray:
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    batch = np.empty((len(images), 3, input_size, input_size), dtype=np.float32)
    for row, image in enumerate(images):
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected RGB image, got {rgb.shape}")
        resized = cv2.resize(
            rgb,
            (input_size, input_size),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0
        batch[row] = np.transpose((resized - IMAGE_MEAN) / IMAGE_STD, (2, 0, 1))
    return batch


def spatial_features_from_logits(
    segmentation_logits,
    heatmap_logits,
    visibility_logits,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Convert frozen ego-network outputs into geometry-aware numeric features."""
    import torch

    segmentation_probability = torch.softmax(segmentation_logits, dim=1)
    batch, _, height, width = segmentation_probability.shape
    heatmap_probability = torch.softmax(
        heatmap_logits.reshape(batch, heatmap_logits.shape[1], -1),
        dim=-1,
    )
    indices = heatmap_probability.argmax(dim=-1)
    rows = torch.div(indices, width, rounding_mode="floor")
    columns = indices % width
    keypoints_uv = torch.stack([columns, rows], dim=-1).to(heatmap_logits.dtype)
    normalized_uv = keypoints_uv.clone()
    normalized_uv[..., 0] /= max(width - 1, 1)
    normalized_uv[..., 1] /= max(height - 1, 1)
    heatmap_peak = heatmap_probability.max(dim=-1).values
    entropy = -(heatmap_probability * heatmap_probability.clamp_min(1e-12).log()).sum(
        dim=-1
    )
    normalized_entropy = entropy / np.log(float(height * width))
    concentration = (1.0 - normalized_entropy).clamp(0.0, 1.0)
    visibility_probability = torch.sigmoid(visibility_logits)

    grid_rows = torch.linspace(
        0.0,
        1.0,
        height,
        device=segmentation_logits.device,
        dtype=segmentation_logits.dtype,
    )
    grid_columns = torch.linspace(
        0.0,
        1.0,
        width,
        device=segmentation_logits.device,
        dtype=segmentation_logits.dtype,
    )
    row_grid, column_grid = torch.meshgrid(grid_rows, grid_columns, indexing="ij")
    segmentation_features = []
    for semantic_class in (1, 2):
        probability = segmentation_probability[:, semantic_class]
        mass = probability.sum(dim=(-2, -1)).clamp_min(1e-8)
        area = probability.mean(dim=(-2, -1))
        center_column = (probability * column_grid).sum(dim=(-2, -1)) / mass
        center_row = (probability * row_grid).sum(dim=(-2, -1)) / mass
        std_column = torch.sqrt(
            (probability * (column_grid - center_column[:, None, None]).square()).sum(
                dim=(-2, -1)
            )
            / mass
        )
        std_row = torch.sqrt(
            (probability * (row_grid - center_row[:, None, None]).square()).sum(
                dim=(-2, -1)
            )
            / mass
        )
        segmentation_features.extend(
            [area, center_column, center_row, std_column, std_row]
        )

    sample_rows = torch.arange(batch, device=segmentation_logits.device)[:, None]
    support_classes = torch.tensor(
        [1, 1, 2, 2],
        device=segmentation_logits.device,
    )[None, :]
    keypoint_support = segmentation_probability[
        sample_rows,
        support_classes,
        rows,
        columns,
    ]
    peg_axis = normalized_uv[:, 1] - normalized_uv[:, 0]
    hole_axis = normalized_uv[:, 3] - normalized_uv[:, 2]
    origin_delta = normalized_uv[:, 0] - normalized_uv[:, 2]
    axis_lengths = torch.stack(
        [torch.linalg.norm(peg_axis, dim=-1), torch.linalg.norm(hole_axis, dim=-1)],
        dim=-1,
    )
    feature_tensor = torch.cat(
        [
            normalized_uv.reshape(batch, -1),
            heatmap_peak,
            concentration,
            visibility_probability,
            torch.stack(segmentation_features, dim=-1),
            peg_axis,
            hole_axis,
            origin_delta,
            axis_lengths,
        ],
        dim=-1,
    )
    if feature_tensor.shape[1] != EGO_SPATIAL_FEATURE_DIM:
        raise RuntimeError(f"unexpected ego feature shape {feature_tensor.shape}")
    keypoint_quality = visibility_probability * concentration * torch.sqrt(
        keypoint_support.clamp(0.0, 1.0)
    )
    reliability = torch.exp(
        torch.mean(torch.log(keypoint_quality.clamp_min(1e-6)), dim=-1)
    ).clamp(0.0, 1.0)
    diagnostics = {
        "keypoints_uv": keypoints_uv.detach().cpu().numpy().astype(np.float32),
        "heatmap_peak": heatmap_peak.detach().cpu().numpy().astype(np.float32),
        "heatmap_concentration": concentration.detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "visibility_probability": visibility_probability.detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "segmentation_support": keypoint_support.detach()
        .cpu()
        .numpy()
        .astype(np.float32),
    }
    return (
        feature_tensor.detach().cpu().numpy().astype(np.float32),
        reliability.detach().cpu().numpy().astype(np.float32),
        diagnostics,
    )


def causal_feature_history(features: np.ndarray, history: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"features must have shape (T, D), got {values.shape}")
    if history <= 0:
        raise ValueError("history must be positive")
    output = np.empty((len(values), values.shape[1] * history), dtype=np.float32)
    for row in range(len(values)):
        indices = np.maximum(np.arange(row - history + 1, row + 1), 0)
        output[row] = values[indices].reshape(-1)
    return output


def deployable_visual_proprio_features(
    visual_features: np.ndarray,
    state46: np.ndarray,
    previous_action44: np.ndarray,
) -> np.ndarray:
    visual = np.asarray(visual_features, dtype=np.float32)
    state = np.asarray(state46, dtype=np.float32)
    action = np.asarray(previous_action44, dtype=np.float32)
    if visual.ndim != 2 or visual.shape[1] != EGO_SPATIAL_FEATURE_DIM:
        raise ValueError(
            f"visual_features must have shape (T, {EGO_SPATIAL_FEATURE_DIM}), "
            f"got {visual.shape}"
        )
    if state.shape != (len(visual), 46):
        raise ValueError(f"state46 must have shape ({len(visual)}, 46), got {state.shape}")
    if action.shape != (len(visual), 44):
        raise ValueError(
            f"previous_action44 must have shape ({len(visual)}, 44), got {action.shape}"
        )
    combined = np.concatenate([visual, state, action], axis=1)
    if combined.shape[1] != EGO_VISUAL_PROPRIO_FEATURE_DIM:
        raise RuntimeError(f"unexpected visual-proprio shape {combined.shape}")
    if not np.isfinite(combined).all():
        raise ValueError("visual-proprio features must be finite")
    return combined


@dataclass
class EgoSpatialPredictor:
    model: object
    input_size: int
    device: object

    @classmethod
    def load(cls, checkpoint_path: Path, *, device: str) -> "EgoSpatialPredictor":
        import torch

        resolved_device = torch.device(device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=resolved_device,
            weights_only=True,
        )
        model = build_spatial_visual_model(
            base_channels=int(checkpoint["base_channels"])
        ).to(resolved_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return cls(
            model=model,
            input_size=int(checkpoint["input_size"]),
            device=resolved_device,
        )

    def encode(self, images: list[np.ndarray], *, batch_size: int = 32):
        import torch

        features = []
        reliabilities = []
        diagnostics: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(images), batch_size):
            selected = images[start : start + batch_size]
            tensor = torch.from_numpy(
                preprocess_rgb_batch(selected, input_size=self.input_size)
            ).to(self.device)
            camera_index = torch.zeros(len(selected), dtype=torch.long, device=self.device)
            with torch.no_grad():
                output = self.model(tensor, camera_index)
            batch_features, batch_reliability, batch_diagnostics = (
                spatial_features_from_logits(
                    output["segmentation_logits"],
                    output["heatmap_logits"],
                    output["visibility_logits"],
                )
            )
            features.append(batch_features)
            reliabilities.append(batch_reliability)
            for name, values in batch_diagnostics.items():
                diagnostics.setdefault(name, []).append(values)
        return (
            np.concatenate(features, axis=0),
            np.concatenate(reliabilities, axis=0),
            {name: np.concatenate(values, axis=0) for name, values in diagnostics.items()},
        )
