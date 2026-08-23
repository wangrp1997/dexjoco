"""Hardware-facing sensor contract and configurable sim-to-real degradation."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .sensor_observation import CerebellumSensorObservation, _readonly_array


@dataclass(frozen=True)
class SensorModelConfig:
    """Explicit assumptions for one sensor degradation profile."""

    name: str
    hardware_verified: bool
    sample_rate_hz: float
    latency_frames: int
    wrist_position_noise_std_m: float
    wrist_orientation_noise_std_rad: float
    finger_joint_noise_std_rad: float
    arm_torque_noise_std_nm: float
    arm_torque_bias_walk_std_nm: float
    arm_torque_resolution_nm: float
    arm_torque_limit_nm: float
    fingertip_force_noise_std_n: float
    fingertip_force_bias_walk_std_n: float
    fingertip_force_resolution_n: float
    fingertip_force_limit_n: float
    fingertip_contact_threshold_n: float
    wrist_force_noise_std_n: float
    wrist_torque_noise_std_nm: float
    wrist_force_resolution_n: float
    wrist_torque_resolution_nm: float
    wrist_force_limit_n: float
    wrist_torque_limit_nm: float
    proprio_dropout_probability: float
    arm_torque_dropout_probability: float
    fingertip_dropout_probability: float
    wrist_wrench_dropout_probability: float
    random_seed: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("sensor profile name must be non-empty")
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be finite and positive")
        if self.latency_frames < 0:
            raise ValueError("latency_frames must be non-negative")
        nonnegative = (
            "wrist_position_noise_std_m",
            "wrist_orientation_noise_std_rad",
            "finger_joint_noise_std_rad",
            "arm_torque_noise_std_nm",
            "arm_torque_bias_walk_std_nm",
            "arm_torque_resolution_nm",
            "fingertip_force_noise_std_n",
            "fingertip_force_bias_walk_std_n",
            "fingertip_force_resolution_n",
            "fingertip_contact_threshold_n",
            "wrist_force_noise_std_n",
            "wrist_torque_noise_std_nm",
            "wrist_force_resolution_n",
            "wrist_torque_resolution_nm",
        )
        positive = (
            "arm_torque_limit_nm",
            "fingertip_force_limit_n",
            "wrist_force_limit_n",
            "wrist_torque_limit_nm",
        )
        probabilities = (
            "proprio_dropout_probability",
            "arm_torque_dropout_probability",
            "fingertip_dropout_probability",
            "wrist_wrench_dropout_probability",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in probabilities:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    @property
    def latency_s(self) -> float:
        return self.latency_frames / self.sample_rate_hz

    def require_hardware_verified(self) -> None:
        if not self.hardware_verified:
            raise RuntimeError(
                f"sensor profile {self.name!r} is not hardware verified and cannot "
                "be used to claim real-robot readiness"
            )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["latency_s"] = self.latency_s
        return payload

    @classmethod
    def from_json(cls, path: Path) -> "SensorModelConfig":
        payload = json.loads(Path(path).read_text())
        payload.pop("latency_s", None)
        return cls(**payload)


@dataclass(frozen=True)
class RealisticCerebellumObservation:
    """Common-denominator signals expected from a realizable robot interface."""

    timestamp_s: float
    state46: np.ndarray
    arm_joint_torque: np.ndarray
    fingertip_force_magnitude: np.ndarray
    fingertip_contact: np.ndarray
    wrist_wrench_local: np.ndarray
    previous_action44: np.ndarray | None
    images: Mapping[str, np.ndarray]
    proprio_valid: bool
    arm_torque_valid: np.ndarray
    fingertip_valid: np.ndarray
    wrist_wrench_valid: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state46",
            _readonly_array(self.state46, (46,), name="state46"),
        )
        object.__setattr__(
            self,
            "arm_joint_torque",
            _readonly_array(
                self.arm_joint_torque,
                (2, 7),
                name="arm_joint_torque",
            ),
        )
        object.__setattr__(
            self,
            "fingertip_force_magnitude",
            _readonly_array(
                self.fingertip_force_magnitude,
                (2, 4),
                name="fingertip_force_magnitude",
            ),
        )
        object.__setattr__(
            self,
            "wrist_wrench_local",
            _readonly_array(
                self.wrist_wrench_local,
                (2, 6),
                name="wrist_wrench_local",
            ),
        )
        if self.previous_action44 is not None:
            object.__setattr__(
                self,
                "previous_action44",
                _readonly_array(
                    self.previous_action44,
                    (44,),
                    name="previous_action44",
                ),
            )
        fingertip_contact = np.array(self.fingertip_contact, dtype=bool, copy=True)
        arm_valid = np.array(self.arm_torque_valid, dtype=bool, copy=True)
        fingertip_valid = np.array(self.fingertip_valid, dtype=bool, copy=True)
        wrist_valid = np.array(self.wrist_wrench_valid, dtype=bool, copy=True)
        for name, value, shape in (
            ("fingertip_contact", fingertip_contact, (2, 4)),
            ("arm_torque_valid", arm_valid, (2,)),
            ("fingertip_valid", fingertip_valid, (2, 4)),
            ("wrist_wrench_valid", wrist_valid, (2,)),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        image_copy: dict[str, np.ndarray] = {}
        for name, value in self.images.items():
            image = np.array(value, copy=True)
            image.flags.writeable = False
            image_copy[str(name)] = image
        object.__setattr__(self, "images", MappingProxyType(image_copy))
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        object.__setattr__(self, "proprio_valid", bool(self.proprio_valid))


def _quantize(values: np.ndarray, resolution: float) -> np.ndarray:
    if resolution == 0.0:
        return values
    return np.round(values / resolution) * resolution


def _clip(values: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(values, -limit, limit)


class SensorDegrader:
    """Stateful latency, noise, bias, quantization, saturation and dropout."""

    def __init__(self, config: SensorModelConfig, *, random_seed: int | None = None) -> None:
        self.config = config
        self._random_seed = config.random_seed if random_seed is None else int(random_seed)
        self.reset()

    def reset(self) -> None:
        self._queue: deque[CerebellumSensorObservation] = deque()
        self._rng = np.random.default_rng(self._random_seed)
        self._arm_bias = np.zeros((2, 7), dtype=np.float64)
        self._fingertip_bias = np.zeros((2, 4), dtype=np.float64)
        self._last_state = np.zeros(46, dtype=np.float64)
        self._last_arm = np.zeros((2, 7), dtype=np.float64)
        self._last_fingertip = np.zeros((2, 4), dtype=np.float64)
        self._last_wrist = np.zeros((2, 6), dtype=np.float64)
        self._has_last = False

    def transform(
        self,
        observation: CerebellumSensorObservation,
    ) -> RealisticCerebellumObservation | None:
        self._queue.append(observation)
        if len(self._queue) <= self.config.latency_frames:
            return None
        delayed = self._queue.popleft()

        state = self._degrade_state(delayed.state46)
        arm = self._degrade_arm(delayed.arm_joint_torque)
        fingertip = self._degrade_fingertip(delayed.fingertip_force_world)
        wrist = self._degrade_wrist(delayed.wrist_wrench_world, delayed.state46)

        proprio_valid = self._rng.random() >= self.config.proprio_dropout_probability
        arm_valid = self._rng.random(2) >= self.config.arm_torque_dropout_probability
        fingertip_valid = (
            self._rng.random((2, 4)) >= self.config.fingertip_dropout_probability
        )
        wrist_valid = (
            self._rng.random(2) >= self.config.wrist_wrench_dropout_probability
        )
        if not proprio_valid and self._has_last:
            state = self._last_state.copy()
        if self._has_last:
            arm = np.where(arm_valid[:, None], arm, self._last_arm)
            fingertip = np.where(fingertip_valid, fingertip, self._last_fingertip)
            wrist = np.where(wrist_valid[:, None], wrist, self._last_wrist)

        self._last_state = state.copy()
        self._last_arm = arm.copy()
        self._last_fingertip = fingertip.copy()
        self._last_wrist = wrist.copy()
        self._has_last = True
        return RealisticCerebellumObservation(
            timestamp_s=delayed.timestamp_s,
            state46=state,
            arm_joint_torque=arm,
            fingertip_force_magnitude=fingertip,
            fingertip_contact=fingertip >= self.config.fingertip_contact_threshold_n,
            wrist_wrench_local=wrist,
            previous_action44=delayed.previous_action44,
            images=delayed.images,
            proprio_valid=proprio_valid,
            arm_torque_valid=arm_valid,
            fingertip_valid=fingertip_valid,
            wrist_wrench_valid=wrist_valid,
        )

    def _degrade_state(self, state46: np.ndarray) -> np.ndarray:
        state = np.asarray(state46, dtype=np.float64).copy()
        for position_slice in (slice(0, 3), slice(7, 10)):
            state[position_slice] += self._rng.normal(
                0.0,
                self.config.wrist_position_noise_std_m,
                3,
            )
        for quaternion_slice in (slice(3, 7), slice(10, 14)):
            rotation = Rotation.from_quat(state[quaternion_slice], scalar_first=True)
            perturbation = Rotation.from_rotvec(
                self._rng.normal(
                    0.0,
                    self.config.wrist_orientation_noise_std_rad,
                    3,
                )
            )
            state[quaternion_slice] = (perturbation * rotation).as_quat(
                scalar_first=True
            )
        state[14:46] += self._rng.normal(
            0.0,
            self.config.finger_joint_noise_std_rad,
            32,
        )
        return state

    def _degrade_arm(self, values: np.ndarray) -> np.ndarray:
        self._arm_bias += self._rng.normal(
            0.0,
            self.config.arm_torque_bias_walk_std_nm,
            (2, 7),
        )
        result = np.asarray(values, dtype=np.float64) + self._arm_bias
        result += self._rng.normal(
            0.0,
            self.config.arm_torque_noise_std_nm,
            (2, 7),
        )
        result = _quantize(result, self.config.arm_torque_resolution_nm)
        return _clip(result, self.config.arm_torque_limit_nm)

    def _degrade_fingertip(self, force_world: np.ndarray) -> np.ndarray:
        magnitude = np.linalg.norm(np.asarray(force_world, dtype=np.float64), axis=-1)
        self._fingertip_bias += self._rng.normal(
            0.0,
            self.config.fingertip_force_bias_walk_std_n,
            (2, 4),
        )
        magnitude += self._fingertip_bias
        magnitude += self._rng.normal(
            0.0,
            self.config.fingertip_force_noise_std_n,
            (2, 4),
        )
        magnitude = _quantize(magnitude, self.config.fingertip_force_resolution_n)
        return np.clip(magnitude, 0.0, self.config.fingertip_force_limit_n)

    def _degrade_wrist(self, wrench_world: np.ndarray, state46: np.ndarray) -> np.ndarray:
        world = np.asarray(wrench_world, dtype=np.float64).reshape(2, 6)
        quaternions = (state46[3:7], state46[10:14])
        local = np.empty((2, 6), dtype=np.float64)
        for arm, quaternion in enumerate(quaternions):
            rotation = Rotation.from_quat(quaternion, scalar_first=True)
            local[arm, :3] = rotation.inv().apply(world[arm, :3])
            local[arm, 3:] = rotation.inv().apply(world[arm, 3:])
        local[:, :3] += self._rng.normal(
            0.0,
            self.config.wrist_force_noise_std_n,
            (2, 3),
        )
        local[:, 3:] += self._rng.normal(
            0.0,
            self.config.wrist_torque_noise_std_nm,
            (2, 3),
        )
        local[:, :3] = _quantize(
            local[:, :3], self.config.wrist_force_resolution_n
        )
        local[:, 3:] = _quantize(
            local[:, 3:], self.config.wrist_torque_resolution_nm
        )
        local[:, :3] = _clip(local[:, :3], self.config.wrist_force_limit_n)
        local[:, 3:] = _clip(local[:, 3:], self.config.wrist_torque_limit_nm)
        return local
