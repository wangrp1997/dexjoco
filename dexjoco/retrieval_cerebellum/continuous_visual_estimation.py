"""Continuous multi-camera visual features and deployable assembly estimates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .v2_control import V2AssemblyEstimate


VISUAL_TARGET_DIM = 11


@dataclass(frozen=True)
class ContinuousVisualFeatureStore:
    episode_index: np.ndarray
    frame_index: np.ndarray
    split: np.ndarray
    features: np.ndarray
    camera_keys: tuple[str, ...]
    model_name: str

    @classmethod
    def load(cls, path: Path) -> ContinuousVisualFeatureStore:
        with np.load(path, allow_pickle=False) as data:
            store = cls(
                episode_index=np.asarray(data["episode_index"], dtype=np.int64),
                frame_index=np.asarray(data["frame_index"], dtype=np.int64),
                split=np.asarray(data["split"], dtype=str),
                features=np.asarray(data["projected_features"], dtype=np.float32),
                camera_keys=tuple(str(value) for value in data["camera_keys"]),
                model_name=str(data["model_name"]),
            )
        store.validate()
        return store

    def validate(self) -> None:
        count = self.episode_index.shape[0]
        if self.frame_index.shape != (count,) or self.split.shape != (count,):
            raise ValueError("continuous visual metadata lengths differ")
        if self.features.ndim != 2 or self.features.shape[0] != count:
            raise ValueError("projected_features must have shape (N, D)")
        if not np.isfinite(self.features).all():
            raise ValueError("continuous visual features contain non-finite values")
        pairs = np.stack([self.episode_index, self.frame_index], axis=1)
        if np.unique(pairs, axis=0).shape[0] != count:
            raise ValueError("continuous visual cache contains duplicate frame keys")

    def episode(self, episode_index: int) -> tuple[np.ndarray, np.ndarray, str]:
        rows = np.flatnonzero(self.episode_index == int(episode_index))
        if rows.size == 0:
            raise KeyError(f"episode {episode_index} missing from visual cache")
        splits = np.unique(self.split[rows])
        if splits.size != 1:
            raise ValueError("episode has inconsistent split labels")
        order = np.argsort(self.frame_index[rows])
        selected = rows[order]
        return self.frame_index[selected].copy(), self.features[selected].copy(), str(splits[0])


@dataclass(frozen=True)
class ContinuousVisualStateModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    weights: np.ndarray
    information_inverse: np.ndarray
    residual_covariance5: np.ndarray
    reliability_scale: float
    alpha: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        targets11: np.ndarray,
        *,
        alpha: float,
        calibration_features: np.ndarray,
        calibration_targets11: np.ndarray,
    ) -> ContinuousVisualStateModel:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets11, dtype=np.float64)
        calibration_x = np.asarray(calibration_features, dtype=np.float64)
        calibration_y = np.asarray(calibration_targets11, dtype=np.float64)
        if x.ndim != 2 or y.shape != (x.shape[0], VISUAL_TARGET_DIM):
            raise ValueError("features and visual targets have incompatible shapes")
        if calibration_x.ndim != 2 or calibration_y.shape != (
            calibration_x.shape[0],
            VISUAL_TARGET_DIM,
        ):
            raise ValueError("calibration arrays have incompatible shapes")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        feature_mean = x.mean(axis=0)
        feature_scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
        target_mean = y.mean(axis=0)
        target_scale = np.where(y.std(axis=0) > 1e-8, y.std(axis=0), 1.0)
        normalized_x = (x - feature_mean) / feature_scale
        normalized_y = (y - target_mean) / target_scale
        information = normalized_x.T @ normalized_x
        information += np.eye(information.shape[0]) * alpha
        information_inverse = np.linalg.inv(information)
        weights = information_inverse @ normalized_x.T @ normalized_y
        calibration_prediction = cls._predict_array(
            calibration_x,
            feature_mean,
            feature_scale,
            target_mean,
            target_scale,
            weights,
        )
        residual5 = calibration_prediction[:, :5] - calibration_y[:, :5]
        residual_covariance5 = np.cov(residual5, rowvar=False)
        residual_covariance5 += np.eye(5) * 1e-12
        normalized_calibration = (calibration_x - feature_mean) / feature_scale
        leverage = 1.0 + np.einsum(
            "bi,ij,bj->b",
            normalized_calibration,
            information_inverse,
            normalized_calibration,
        )
        reliability_scale = float(max(np.quantile(leverage, 0.9), 1.0))
        return cls(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            weights=weights,
            information_inverse=information_inverse,
            residual_covariance5=residual_covariance5,
            reliability_scale=reliability_scale,
            alpha=float(alpha),
        )

    @staticmethod
    def _predict_array(
        features: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        target_mean: np.ndarray,
        target_scale: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        normalized = (features - feature_mean) / feature_scale
        return (normalized @ weights) * target_scale + target_mean

    def predict_arrays(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"features must have shape (T, {self.weights.shape[0]})"
            )
        normalized = (x - self.feature_mean) / self.feature_scale
        prediction = (normalized @ self.weights) * self.target_scale + self.target_mean
        leverage = 1.0 + np.einsum(
            "bi,ij,bj->b",
            normalized,
            self.information_inverse,
            normalized,
        )
        covariance5 = leverage[:, None, None] * self.residual_covariance5[None, :, :]
        standard_deviation = np.sqrt(
            np.maximum(np.diagonal(covariance5, axis1=1, axis2=2), 0.0)
        )
        precision_budget = np.asarray([0.00085, 0.00085, 0.021, 0.021, 0.003])
        precision_ratio = standard_deviation / precision_budget[None, :]
        precision_reliability = np.exp(-0.5 * np.mean(precision_ratio**2, axis=1))
        reliability = np.clip(
            precision_reliability * self.reliability_scale / leverage,
            0.0,
            1.0,
        )
        rotations = np.stack([_rotation_from_sixd(value[5:]) for value in prediction])
        return prediction[:, :5], covariance5, rotations, reliability

    def estimate(
        self,
        feature: np.ndarray,
        *,
        timestamp_s: float,
    ) -> V2AssemblyEstimate:
        mean5, covariance5, rotations, reliability = self.predict_arrays(
            np.asarray(feature, dtype=np.float64).reshape(1, -1)
        )
        return V2AssemblyEstimate(
            timestamp_s=timestamp_s,
            mean5=mean5[0],
            covariance5=covariance5[0],
            hole_rotation_world=rotations[0],
            visual_reliability=float(reliability[0]),
        )

    def save(self, path: Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            target_mean=self.target_mean,
            target_scale=self.target_scale,
            weights=self.weights,
            information_inverse=self.information_inverse,
            residual_covariance5=self.residual_covariance5,
            reliability_scale=np.asarray(self.reliability_scale),
            alpha=np.asarray(self.alpha),
        )

    @classmethod
    def load(cls, path: Path) -> ContinuousVisualStateModel:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                feature_mean=data["feature_mean"],
                feature_scale=data["feature_scale"],
                target_mean=data["target_mean"],
                target_scale=data["target_scale"],
                weights=data["weights"],
                information_inverse=data["information_inverse"],
                residual_covariance5=data["residual_covariance5"],
                reliability_scale=float(data["reliability_scale"]),
                alpha=float(data["alpha"]),
            )


def rotation_to_sixd(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    return matrix[:, :2].reshape(-1)


def _rotation_from_sixd(value: np.ndarray) -> np.ndarray:
    columns = np.asarray(value, dtype=np.float64).reshape(3, 2)
    first = columns[:, 0]
    first /= np.linalg.norm(first) + 1e-12
    second = columns[:, 1] - first * np.dot(first, columns[:, 1])
    second /= np.linalg.norm(second) + 1e-12
    third = np.cross(first, second)
    matrix = np.column_stack([first, second, third])
    return Rotation.from_matrix(matrix).as_matrix()
