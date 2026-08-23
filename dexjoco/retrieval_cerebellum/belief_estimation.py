"""Truth-distilled observation baseline and finite-window MHE smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .real_sensor_model import RealisticCerebellumObservation


BELIEF_BLOCKS = (
    "peg_in_hole",
    "peg_in_right_palm",
    "tray_in_left_palm",
)
BELIEF_DIM = 18
BASE_SENSOR_FEATURE_DIM = 143
FINGERTIP_KINEMATIC_DIM = 24


def _state46_to_pose_hand_features(state46: np.ndarray) -> np.ndarray:
    state = np.asarray(state46, dtype=np.float64).reshape(46)
    right_rotvec = Rotation.from_quat(
        state[3:7], scalar_first=True
    ).as_rotvec()
    left_rotvec = Rotation.from_quat(
        state[10:14], scalar_first=True
    ).as_rotvec()
    return np.concatenate(
        [
            state[:3],
            right_rotvec,
            state[7:10],
            left_rotvec,
            state[14:46],
        ]
    )


def sensor_feature(
    observation: RealisticCerebellumObservation,
    *,
    fingertip_position_in_palm: np.ndarray | None = None,
) -> np.ndarray:
    """Build one deployable feature vector without privileged geometry."""
    previous_action = (
        np.zeros(44, dtype=np.float64)
        if observation.previous_action44 is None
        else np.asarray(observation.previous_action44, dtype=np.float64)
    )
    parts = [
            _state46_to_pose_hand_features(observation.state46),
            previous_action,
            observation.arm_joint_torque.reshape(-1),
            observation.fingertip_force_magnitude.reshape(-1),
            observation.fingertip_contact.astype(np.float64).reshape(-1),
            observation.wrist_wrench_local.reshape(-1),
            np.asarray([observation.proprio_valid], dtype=np.float64),
            observation.arm_torque_valid.astype(np.float64).reshape(-1),
            observation.fingertip_valid.astype(np.float64).reshape(-1),
            observation.wrist_wrench_valid.astype(np.float64).reshape(-1),
        ]
    if fingertip_position_in_palm is not None:
        positions = np.asarray(fingertip_position_in_palm, dtype=np.float64)
        if positions.shape != (2, 4, 3):
            raise ValueError(
                "fingertip_position_in_palm must have shape (2, 4, 3)"
            )
        parts.append(positions.reshape(-1))
    feature = np.concatenate(parts).astype(np.float32)
    expected_dim = BASE_SENSOR_FEATURE_DIM + (
        FINGERTIP_KINEMATIC_DIM if fingertip_position_in_palm is not None else 0
    )
    if feature.shape != (expected_dim,):
        raise RuntimeError(f"unexpected sensor feature shape {feature.shape}")
    if not np.isfinite(feature).all():
        raise ValueError("sensor feature contains non-finite values")
    return feature


def stack_causal_history(features: np.ndarray, history_size: int) -> np.ndarray:
    """Concatenate current and past features with left-edge replication."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"features must have shape (T, D), got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError("features must be non-empty")
    if history_size <= 0:
        raise ValueError("history_size must be positive")
    padded = np.concatenate(
        [np.repeat(values[:1], history_size - 1, axis=0), values],
        axis=0,
    )
    return np.stack(
        [
            padded[row : row + history_size].reshape(-1)
            for row in range(values.shape[0])
        ]
    ).astype(np.float32)


def belief_target(columns: dict[str, np.ndarray]) -> np.ndarray:
    """Build the 18D privileged target used only for training/evaluation."""
    blocks = []
    for name in BELIEF_BLOCKS:
        position = np.asarray(columns[f"teacher_{name}_position"], dtype=np.float32)
        rotvec = np.asarray(columns[f"teacher_{name}_rotvec"], dtype=np.float32)
        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError(f"teacher_{name}_position must have shape (T, 3)")
        if rotvec.shape != position.shape:
            raise ValueError(f"teacher_{name}_rotvec must match position shape")
        blocks.extend([position, rotvec])
    target = np.concatenate(blocks, axis=1)
    if target.shape[1] != BELIEF_DIM:
        raise RuntimeError(f"unexpected belief target shape {target.shape}")
    return target


@dataclass(frozen=True)
class RidgeObservationModel:
    """Standardized multi-output ridge observation model."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    weights: np.ndarray
    residual_variance: np.ndarray
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
    ) -> "RidgeObservationModel":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.shape != (x.shape[0], BELIEF_DIM):
            raise ValueError("features and targets have incompatible shapes")
        if x.shape[0] < 2:
            raise ValueError("ridge fit requires at least two rows")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        feature_mean = x.mean(axis=0)
        feature_scale = x.std(axis=0)
        feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
        target_mean = y.mean(axis=0)
        target_scale = y.std(axis=0)
        target_scale = np.where(target_scale > 1e-8, target_scale, 1.0)
        normalized_x = (x - feature_mean) / feature_scale
        normalized_y = (y - target_mean) / target_scale
        gram = normalized_x.T @ normalized_x
        regularizer = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
        weights = np.linalg.solve(
            gram + regularizer,
            normalized_x.T @ normalized_y,
        )
        prediction = (normalized_x @ weights) * target_scale + target_mean
        residual_variance = np.var(y - prediction, axis=0, ddof=1)
        residual_variance = np.maximum(residual_variance, 1e-12)
        return cls(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            weights=weights,
            residual_variance=residual_variance,
            history_size=int(history_size),
            alpha=float(alpha),
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"features must have shape (T, {self.weights.shape[0]}), got {x.shape}"
            )
        normalized = (x - self.feature_mean) / self.feature_scale
        return (normalized @ self.weights) * self.target_scale + self.target_mean

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            target_mean=self.target_mean,
            target_scale=self.target_scale,
            weights=self.weights,
            residual_variance=self.residual_variance,
            history_size=np.asarray(self.history_size, dtype=np.int64),
            alpha=np.asarray(self.alpha, dtype=np.float64),
        )

    @classmethod
    def load(cls, path: Path) -> "RidgeObservationModel":
        with np.load(path) as data:
            return cls(
                feature_mean=data["feature_mean"],
                feature_scale=data["feature_scale"],
                target_mean=data["target_mean"],
                target_scale=data["target_scale"],
                weights=data["weights"],
                residual_variance=data["residual_variance"],
                history_size=int(data["history_size"]),
                alpha=float(data["alpha"]),
            )


@dataclass(frozen=True)
class MHEResult:
    mean: np.ndarray
    variance: np.ndarray


class SlidingWindowMHE:
    """Causal finite-window MAP estimate with a random-walk process prior."""

    def __init__(
        self,
        measurement_variance: np.ndarray,
        process_variance: np.ndarray,
        *,
        window_size: int,
    ) -> None:
        measurement = np.asarray(measurement_variance, dtype=np.float64).reshape(-1)
        process = np.asarray(process_variance, dtype=np.float64).reshape(-1)
        if measurement.shape != (BELIEF_DIM,) or process.shape != (BELIEF_DIM,):
            raise ValueError(f"variance vectors must have shape ({BELIEF_DIM},)")
        if np.any(measurement <= 0.0) or np.any(process <= 0.0):
            raise ValueError("variances must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.measurement_variance = measurement
        self.process_variance = process
        self.window_size = int(window_size)
        self._inverse_cache: dict[tuple[int, bool, int], np.ndarray] = {}

    def smooth(self, measurements: np.ndarray) -> MHEResult:
        values = np.asarray(measurements, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != BELIEF_DIM:
            raise ValueError(f"measurements must have shape (T, {BELIEF_DIM})")
        if values.shape[0] == 0:
            raise ValueError("measurements must be non-empty")
        estimates = np.empty_like(values)
        variances = np.empty_like(values)
        for end in range(values.shape[0]):
            start = max(0, end - self.window_size + 1)
            window = values[start : end + 1]
            previous = None if start == 0 else estimates[start - 1]
            mean, variance = self._solve_window(window, previous)
            estimates[end] = mean[-1]
            variances[end] = variance
        return MHEResult(mean=estimates, variance=variances)

    def _solve_window(
        self,
        measurements: np.ndarray,
        previous: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        length = measurements.shape[0]
        solution = np.empty_like(measurements)
        final_variance = np.empty(BELIEF_DIM, dtype=np.float64)
        for dimension in range(BELIEF_DIM):
            measurement_precision = 1.0 / self.measurement_variance[dimension]
            process_precision = 1.0 / self.process_variance[dimension]
            rhs = measurements[:, dimension] * measurement_precision
            if previous is not None:
                rhs[0] += previous[dimension] * process_precision
            inverse = self._precision_inverse(
                length,
                anchored=previous is not None,
                dimension=dimension,
            )
            solution[:, dimension] = inverse @ rhs
            final_variance[dimension] = inverse[-1, -1]
        return solution, final_variance

    def _precision_inverse(
        self,
        length: int,
        *,
        anchored: bool,
        dimension: int,
    ) -> np.ndarray:
        key = (length, anchored, dimension)
        cached = self._inverse_cache.get(key)
        if cached is not None:
            return cached
        measurement_precision = 1.0 / self.measurement_variance[dimension]
        process_precision = 1.0 / self.process_variance[dimension]
        matrix = np.eye(length, dtype=np.float64) * measurement_precision
        for row in range(1, length):
            matrix[row - 1, row - 1] += process_precision
            matrix[row, row] += process_precision
            matrix[row - 1, row] -= process_precision
            matrix[row, row - 1] -= process_precision
        if anchored:
            matrix[0, 0] += process_precision
        inverse = np.linalg.inv(matrix)
        self._inverse_cache[key] = inverse
        return inverse


def default_process_variance(
    *,
    position_std_m: float,
    rotation_std_rad: float,
) -> np.ndarray:
    if position_std_m <= 0.0 or rotation_std_rad <= 0.0:
        raise ValueError("process standard deviations must be positive")
    result = np.empty(BELIEF_DIM, dtype=np.float64)
    for offset in range(0, BELIEF_DIM, 6):
        result[offset : offset + 3] = position_std_m**2
        result[offset + 3 : offset + 6] = rotation_std_rad**2
    return result


def belief_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.shape[1] != BELIEF_DIM:
        raise ValueError("prediction and target must have equal shape (T, 18)")
    metrics: dict[str, object] = {}
    aggregate_position = []
    aggregate_rotation = []
    for block_index, name in enumerate(BELIEF_BLOCKS):
        offset = block_index * 6
        position_error = np.linalg.norm(
            predicted[:, offset : offset + 3] - truth[:, offset : offset + 3],
            axis=1,
        )
        rotation_error = np.linalg.norm(
            predicted[:, offset + 3 : offset + 6]
            - truth[:, offset + 3 : offset + 6],
            axis=1,
        )
        aggregate_position.append(position_error)
        aggregate_rotation.append(rotation_error)
        metrics[name] = {
            "position_mean_m": float(position_error.mean()),
            "position_median_m": float(np.median(position_error)),
            "position_p90_m": float(np.quantile(position_error, 0.9)),
            "rotation_mean_rad": float(rotation_error.mean()),
            "rotation_median_rad": float(np.median(rotation_error)),
            "rotation_p90_rad": float(np.quantile(rotation_error, 0.9)),
        }
    metrics["aggregate"] = {
        "position_mean_m": float(np.concatenate(aggregate_position).mean()),
        "rotation_mean_rad": float(np.concatenate(aggregate_rotation).mean()),
    }
    return metrics


def covariance_calibration(
    prediction: np.ndarray,
    target: np.ndarray,
    variance: np.ndarray,
) -> dict[str, float]:
    error = np.abs(np.asarray(prediction) - np.asarray(target))
    sigma = np.sqrt(np.asarray(variance))
    if error.shape != sigma.shape:
        raise ValueError("error and variance shapes must match")
    return {
        "within_1sigma": float(np.mean(error <= sigma)),
        "within_2sigma": float(np.mean(error <= 2.0 * sigma)),
        "mean_normalized_squared_error": float(np.mean(error**2 / variance)),
    }
