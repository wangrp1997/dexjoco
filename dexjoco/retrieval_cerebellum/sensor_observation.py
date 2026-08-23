"""Deployable sensor-only observations for the dexterous cerebellum."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np


STATE_DIM = 46
ACTION_DIM = 44
NUM_ARMS = 2
ARM_JOINTS = 7
NUM_FINGERS = 4


def _readonly_array(
    value: np.ndarray,
    shape: tuple[int, ...],
    *,
    name: str,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class CerebellumSensorObservation:
    """Signals that a real robot adapter can provide without object truth."""

    timestamp_s: float
    state46: np.ndarray
    arm_joint_torque: np.ndarray
    fingertip_force_world: np.ndarray
    wrist_wrench_world: np.ndarray
    images: Mapping[str, np.ndarray]
    previous_action44: np.ndarray | None = None

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp_s)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(
            self,
            "state46",
            _readonly_array(self.state46, (STATE_DIM,), name="state46"),
        )
        object.__setattr__(
            self,
            "arm_joint_torque",
            _readonly_array(
                self.arm_joint_torque,
                (NUM_ARMS, ARM_JOINTS),
                name="arm_joint_torque",
            ),
        )
        object.__setattr__(
            self,
            "fingertip_force_world",
            _readonly_array(
                self.fingertip_force_world,
                (NUM_ARMS, NUM_FINGERS, 3),
                name="fingertip_force_world",
            ),
        )
        object.__setattr__(
            self,
            "wrist_wrench_world",
            _readonly_array(
                self.wrist_wrench_world,
                (NUM_ARMS, 6),
                name="wrist_wrench_world",
            ),
        )
        if self.previous_action44 is not None:
            object.__setattr__(
                self,
                "previous_action44",
                _readonly_array(
                    self.previous_action44,
                    (ACTION_DIM,),
                    name="previous_action44",
                ),
            )

        image_copy: dict[str, np.ndarray] = {}
        for name, value in self.images.items():
            image = np.array(value, copy=True)
            if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
                raise ValueError(f"image {name!r} must have shape (H, W, C)")
            image.flags.writeable = False
            image_copy[str(name)] = image
        object.__setattr__(self, "images", MappingProxyType(image_copy))

    def trace_record(self, *, timestamp: int) -> dict[str, object]:
        """Serialize numeric sensors while video files store image pixels."""
        return {
            "timestamp": int(timestamp),
            "timestamp_s": self.timestamp_s,
            "state46": self.state46.tolist(),
            "arm_joint_torque": self.arm_joint_torque.tolist(),
            "fingertip_force_world": self.fingertip_force_world.tolist(),
            "wrist_wrench_world": self.wrist_wrench_world.tolist(),
            "previous_action44": (
                None
                if self.previous_action44 is None
                else self.previous_action44.tolist()
            ),
            "images": {
                name: {"shape": list(image.shape), "dtype": str(image.dtype)}
                for name, image in self.images.items()
            },
        }


class SensorTraceRecorder:
    """Episode trace containing no simulator object truth."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.records: list[dict[str, object]] = []

    def append(self, observation: CerebellumSensorObservation, *, timestamp: int) -> None:
        self.records.append(observation.trace_record(timestamp=timestamp))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for record in self.records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
