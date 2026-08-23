"""Causal handoff selection before high-speed attachment drift dominates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class StableHandoffConfig:
    minimum_approach_height_m: float = 0.02
    maximum_approach_height_m: float = 0.08
    maximum_lateral_error_m: float = 0.005
    maximum_tilt_error_rad: float = 0.12
    minimum_bilateral_grasp_rows: int = 3

    def __post_init__(self) -> None:
        if self.minimum_approach_height_m < 0.0:
            raise ValueError("minimum_approach_height_m must be non-negative")
        if self.maximum_approach_height_m <= self.minimum_approach_height_m:
            raise ValueError("maximum approach height must exceed minimum height")
        if self.maximum_lateral_error_m <= 0.0:
            raise ValueError("maximum_lateral_error_m must be positive")
        if self.maximum_tilt_error_rad <= 0.0:
            raise ValueError("maximum_tilt_error_rad must be positive")
        if self.minimum_bilateral_grasp_rows <= 0:
            raise ValueError("minimum_bilateral_grasp_rows must be positive")


@dataclass(frozen=True)
class HandoffSelection:
    row: int
    entry_row: int
    metrics: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "entry_row": self.entry_row,
            "metrics": dict(self.metrics),
        }


def _rotation_step(before: np.ndarray, after: np.ndarray) -> float:
    delta = Rotation.from_rotvec(after) * Rotation.from_rotvec(before).inv()
    return float(delta.magnitude())


def final_entry_row(features: np.ndarray) -> int:
    values = np.asarray(features, dtype=np.float64)
    signed_progress = -values[:, 20]
    crossings = np.flatnonzero(
        (signed_progress[1:] >= 0.0) & (signed_progress[:-1] < 0.0)
    ) + 1
    if crossings.size == 0:
        raise ValueError("no final hole-entry crossing found")
    return int(crossings[-1])


def select_stable_handoff_row(
    features: np.ndarray,
    *,
    maximum_depth_advance_m: float,
    right_attachment_translation_step_m: float,
    left_attachment_translation_step_m: float,
    right_attachment_rotation_step_rad: float,
    left_attachment_rotation_step_rad: float,
    config: StableHandoffConfig | None = None,
) -> HandoffSelection:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 26:
        raise ValueError(f"features must have shape (T, >=26), got {values.shape}")
    cfg = config or StableHandoffConfig()
    entry_row = final_entry_row(values)
    candidates = []
    for row in range(1, entry_row):
        grasp_start = row - cfg.minimum_bilateral_grasp_rows + 1
        if grasp_start < 0:
            continue
        bilateral_grasp = values[grasp_start : row + 1, 24:26] > 0.5
        if not bool(np.all(bilateral_grasp)):
            continue
        approach_height = float(values[row, 20])
        if not (
            cfg.minimum_approach_height_m
            <= approach_height
            <= cfg.maximum_approach_height_m
        ):
            continue
        lateral_error = float(np.linalg.norm(values[row, :2]))
        tilt_error = float(np.linalg.norm(values[row, 3:5]))
        axial_advance = max(0.0, float(values[row - 1, 20] - approach_height))
        right_translation = float(
            np.linalg.norm(values[row, 6:9] - values[row - 1, 6:9])
        )
        left_translation = float(
            np.linalg.norm(values[row, 12:15] - values[row - 1, 12:15])
        )
        right_rotation = _rotation_step(values[row - 1, 9:12], values[row, 9:12])
        left_rotation = _rotation_step(values[row - 1, 15:18], values[row, 15:18])
        metrics = {
            "approach_height_m": approach_height,
            "lateral_error_m": lateral_error,
            "tilt_error_rad": tilt_error,
            "axial_advance_m": axial_advance,
            "right_attachment_translation_step_m": right_translation,
            "left_attachment_translation_step_m": left_translation,
            "right_attachment_rotation_step_rad": right_rotation,
            "left_attachment_rotation_step_rad": left_rotation,
            "bilateral_grasp_rows": float(cfg.minimum_bilateral_grasp_rows),
        }
        if lateral_error > cfg.maximum_lateral_error_m:
            continue
        if tilt_error > cfg.maximum_tilt_error_rad:
            continue
        if axial_advance > maximum_depth_advance_m:
            continue
        if right_translation > right_attachment_translation_step_m:
            continue
        if left_translation > left_attachment_translation_step_m:
            continue
        if right_rotation > right_attachment_rotation_step_rad:
            continue
        if left_rotation > left_attachment_rotation_step_rad:
            continue
        candidates.append(HandoffSelection(row=row, entry_row=entry_row, metrics=metrics))
    if not candidates:
        raise ValueError("no causally stable pre-entry handoff row satisfies the limits")
    return candidates[-1]
