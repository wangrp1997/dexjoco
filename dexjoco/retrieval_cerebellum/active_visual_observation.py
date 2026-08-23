"""Deployable local model of how wrist motion changes ego visual reliability."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


BIMANUAL_CONTROL_DIM = 12


def _matrix(value: np.ndarray, *, columns: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _reliability(value: np.ndarray, *, rows: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (rows,):
        raise ValueError(f"{name} must have shape ({rows},), got {array.shape}")
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain values in [0, 1]")
    return array


@dataclass(frozen=True)
class ActiveVisualReliabilityModel:
    """Bilinear ridge model for deployable action-conditioned visual reliability."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    context_weights: np.ndarray
    interaction_weights: np.ndarray
    residual_variance: float
    alpha: float

    @property
    def feature_dim(self) -> int:
        return int(self.feature_mean.shape[0])

    @property
    def context_dim(self) -> int:
        return self.feature_dim + 2

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        controls12: np.ndarray,
        current_reliability: np.ndarray,
        next_reliability: np.ndarray,
        *,
        alpha: float,
    ) -> ActiveVisualReliabilityModel:
        feature_values = np.asarray(features, dtype=np.float64)
        if feature_values.ndim != 2 or not np.isfinite(feature_values).all():
            raise ValueError("features must have finite shape (N, D)")
        rows, feature_dim = feature_values.shape
        controls = _matrix(controls12, columns=BIMANUAL_CONTROL_DIM, name="controls12")
        if controls.shape[0] != rows:
            raise ValueError("features and controls12 row counts must match")
        current = _reliability(
            current_reliability,
            rows=rows,
            name="current_reliability",
        )
        target = _reliability(next_reliability, rows=rows, name="next_reliability")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")

        feature_mean = feature_values.mean(axis=0)
        feature_scale = np.where(feature_values.std(axis=0) > 1e-8, feature_values.std(axis=0), 1.0)
        normalized = (feature_values - feature_mean) / feature_scale
        context = np.column_stack([np.ones(rows), normalized, current])
        interaction = np.einsum("ni,nj->nij", controls, context).reshape(rows, -1)
        design = np.concatenate([context, interaction], axis=1)
        regularizer = np.eye(design.shape[1], dtype=np.float64) * alpha
        regularizer[0, 0] = 0.0
        target_delta = target - current
        weights = np.linalg.solve(
            design.T @ design + regularizer,
            design.T @ target_delta,
        )
        context_dim = feature_dim + 2
        prediction = current + design @ weights
        residual_variance = float(np.mean((prediction - target) ** 2))
        return cls(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            context_weights=weights[:context_dim],
            interaction_weights=weights[context_dim:].reshape(
                BIMANUAL_CONTROL_DIM,
                context_dim,
            ),
            residual_variance=residual_variance,
            alpha=float(alpha),
        )

    def _context(self, feature: np.ndarray, current_reliability: float) -> np.ndarray:
        feature_value = np.asarray(feature, dtype=np.float64).reshape(-1)
        if feature_value.shape != (self.feature_dim,):
            raise ValueError(
                f"feature must have shape ({self.feature_dim},), got {feature_value.shape}"
            )
        if not np.isfinite(feature_value).all():
            raise ValueError("feature must contain finite values")
        reliability = float(current_reliability)
        if not 0.0 <= reliability <= 1.0:
            raise ValueError("current_reliability must be in [0, 1]")
        normalized = (feature_value - self.feature_mean) / self.feature_scale
        return np.concatenate([[1.0], normalized, [reliability]])

    def predict_raw(
        self,
        feature: np.ndarray,
        current_reliability: float,
        control12: np.ndarray,
    ) -> float:
        context = self._context(feature, current_reliability)
        control = np.asarray(control12, dtype=np.float64).reshape(-1)
        if control.shape != (BIMANUAL_CONTROL_DIM,):
            raise ValueError(
                f"control12 must have shape ({BIMANUAL_CONTROL_DIM},), got {control.shape}"
            )
        if not np.isfinite(control).all():
            raise ValueError("control12 must contain finite values")
        return float(
            current_reliability
            + self.context_weights @ context
            + control @ self.interaction_weights @ context
        )

    def predict(
        self,
        feature: np.ndarray,
        current_reliability: float,
        control12: np.ndarray,
    ) -> float:
        return float(
            np.clip(
                self.predict_raw(feature, current_reliability, control12),
                0.0,
                1.0,
            )
        )

    def control_jacobian(
        self,
        feature: np.ndarray,
        current_reliability: float,
        control12: np.ndarray,
    ) -> np.ndarray:
        context = self._context(feature, current_reliability)
        raw = self.predict_raw(feature, current_reliability, control12)
        if raw <= 0.0 or raw >= 1.0:
            return np.zeros(BIMANUAL_CONTROL_DIM, dtype=np.float64)
        return self.interaction_weights @ context

    def as_sqp_observation_model(
        self,
        estimate: object,
        feature: np.ndarray,
        current_perceptual_reliability: float,
        reference_right_control: np.ndarray,
        reference_left_control: np.ndarray,
    ):
        from .belief_space_sqp import LinearizedVisualObservationModel

        reference_control = np.concatenate(
            [
                np.asarray(reference_right_control, dtype=np.float64).reshape(6),
                np.asarray(reference_left_control, dtype=np.float64).reshape(6),
            ]
        )
        predicted_reliability = self.predict(
            feature,
            current_perceptual_reliability,
            reference_control,
        )
        reliability_jacobian = self.control_jacobian(
            feature,
            current_perceptual_reliability,
            reference_control,
        )
        return LinearizedVisualObservationModel(
            observation_covariance=np.asarray(estimate.covariance5, dtype=np.float64),
            reliability=predicted_reliability,
            control_reliability_jacobian=reliability_jacobian,
            reference_control=reference_control,
        )

    def save(self, path: Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            context_weights=self.context_weights,
            interaction_weights=self.interaction_weights,
            residual_variance=np.asarray(self.residual_variance),
            alpha=np.asarray(self.alpha),
        )

    @classmethod
    def load(cls, path: Path) -> ActiveVisualReliabilityModel:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                feature_mean=np.asarray(data["feature_mean"], dtype=np.float64),
                feature_scale=np.asarray(data["feature_scale"], dtype=np.float64),
                context_weights=np.asarray(data["context_weights"], dtype=np.float64),
                interaction_weights=np.asarray(
                    data["interaction_weights"], dtype=np.float64
                ),
                residual_variance=float(data["residual_variance"]),
                alpha=float(data["alpha"]),
            )

    @classmethod
    def load_approved(cls, output_dir: Path) -> ActiveVisualReliabilityModel:
        directory = Path(output_dir)
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing active visual summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not bool(summary.get("approved_for_active_control", False)):
            raise RuntimeError(
                "active visual model is not approved on held-out sensor-only data"
            )
        model_path = Path(summary["model"])
        if not model_path.is_absolute() and not model_path.is_file():
            model_path = directory / model_path.name
        return cls.load(model_path)
