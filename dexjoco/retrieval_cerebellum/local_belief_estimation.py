"""Local deployable belief model for precision-handoff states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


BELIEF18_DIM = 18
ASSEMBLY_STATE_DIM = 5
LOCAL_TARGET_DIM = BELIEF18_DIM + ASSEMBLY_STATE_DIM


def state5_from_belief18(belief18: np.ndarray) -> np.ndarray:
    belief = np.asarray(belief18, dtype=np.float64)
    if belief.shape[-1] != BELIEF18_DIM:
        raise ValueError("belief18 must end in 18 values")
    return np.stack(
        [belief[..., 0], belief[..., 1], belief[..., 3], belief[..., 4], -belief[..., 2]],
        axis=-1,
    )


def local_target(belief18: np.ndarray) -> np.ndarray:
    belief = np.asarray(belief18, dtype=np.float64)
    return np.concatenate([belief, state5_from_belief18(belief)], axis=-1)


@dataclass(frozen=True)
class LocalAssemblyBeliefModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    weights: np.ndarray
    information_inverse: np.ndarray
    residual_covariance: np.ndarray
    history_size: int
    alpha: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        history_size: int,
        alpha: float,
        calibration_features: np.ndarray | None = None,
        calibration_targets: np.ndarray | None = None,
    ) -> LocalAssemblyBeliefModel:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.shape != (x.shape[0], LOCAL_TARGET_DIM):
            raise ValueError("features and targets have incompatible shapes")
        if x.shape[0] < 2:
            raise ValueError("local belief fit requires at least two rows")
        if alpha <= 0.0 or history_size <= 0:
            raise ValueError("alpha and history_size must be positive")
        feature_mean = x.mean(axis=0)
        feature_scale = x.std(axis=0)
        feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
        target_mean = y.mean(axis=0)
        target_scale = y.std(axis=0)
        target_scale = np.where(target_scale > 1e-8, target_scale, 1.0)
        normalized_x = (x - feature_mean) / feature_scale
        normalized_y = (y - target_mean) / target_scale
        information = normalized_x.T @ normalized_x
        information += np.eye(information.shape[0], dtype=np.float64) * alpha
        information_inverse = np.linalg.inv(information)
        weights = information_inverse @ normalized_x.T @ normalized_y

        if calibration_features is None or calibration_targets is None:
            residual = cls._predict_arrays(
                x,
                feature_mean,
                feature_scale,
                target_mean,
                target_scale,
                weights,
            ) - y
        else:
            calibration_x = np.asarray(calibration_features, dtype=np.float64)
            calibration_y = np.asarray(calibration_targets, dtype=np.float64)
            if calibration_x.ndim != 2 or calibration_y.shape != (
                calibration_x.shape[0],
                LOCAL_TARGET_DIM,
            ):
                raise ValueError("calibration arrays have incompatible shapes")
            residual = cls._predict_arrays(
                calibration_x,
                feature_mean,
                feature_scale,
                target_mean,
                target_scale,
                weights,
            ) - calibration_y
        residual_covariance = np.cov(residual, rowvar=False)
        residual_covariance += np.eye(LOCAL_TARGET_DIM, dtype=np.float64) * 1e-12
        return cls(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            weights=weights,
            information_inverse=information_inverse,
            residual_covariance=residual_covariance,
            history_size=int(history_size),
            alpha=float(alpha),
        )

    @staticmethod
    def _predict_arrays(
        features: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        target_mean: np.ndarray,
        target_scale: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        normalized = (features - feature_mean) / feature_scale
        return (normalized @ weights) * target_scale + target_mean

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"features must have shape (T, {self.weights.shape[0]}), got {x.shape}"
            )
        normalized = (x - self.feature_mean) / self.feature_scale
        mean = (normalized @ self.weights) * self.target_scale + self.target_mean
        leverage = 1.0 + np.einsum(
            "bi,ij,bj->b",
            normalized,
            self.information_inverse,
            normalized,
        )
        covariance = leverage[:, None, None] * self.residual_covariance[None, :, :]
        return mean, covariance

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
            residual_covariance=self.residual_covariance,
            history_size=np.asarray(self.history_size, dtype=np.int64),
            alpha=np.asarray(self.alpha, dtype=np.float64),
        )

    @classmethod
    def load(cls, path: Path) -> LocalAssemblyBeliefModel:
        with np.load(path) as data:
            return cls(
                feature_mean=data["feature_mean"],
                feature_scale=data["feature_scale"],
                target_mean=data["target_mean"],
                target_scale=data["target_scale"],
                weights=data["weights"],
                information_inverse=data["information_inverse"],
                residual_covariance=data["residual_covariance"],
                history_size=int(data["history_size"]),
                alpha=float(data["alpha"]),
            )

    @staticmethod
    def belief18(mean23: np.ndarray) -> np.ndarray:
        return np.asarray(mean23, dtype=np.float64)[..., :BELIEF18_DIM]

    @staticmethod
    def state5(mean23: np.ndarray) -> np.ndarray:
        return np.asarray(mean23, dtype=np.float64)[..., BELIEF18_DIM:]

    @staticmethod
    def state5_covariance(covariance23: np.ndarray) -> np.ndarray:
        covariance = np.asarray(covariance23, dtype=np.float64)
        return covariance[..., BELIEF18_DIM:, BELIEF18_DIM:]
