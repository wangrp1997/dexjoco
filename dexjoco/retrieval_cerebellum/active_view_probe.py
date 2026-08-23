"""Sensor-only safe excitation primitives for active ego-view data collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .assembly_kinematics import apply_bimanual_wrist_twists
from .sensor_observation import CerebellumSensorObservation


@dataclass(frozen=True)
class ActiveViewProbeConfig:
    translation_step_m: float = 0.002
    rotation_step_rad: float = 0.02
    right_motion_fraction: float = 0.8
    maximum_wrist_force_n: float = 8.0
    maximum_wrist_torque_nm: float = 1.0
    minimum_stable_fingertips_per_hand: int = 2
    fingertip_force_threshold_n: float = 0.5

    def __post_init__(self) -> None:
        if self.translation_step_m <= 0.0 or self.rotation_step_rad <= 0.0:
            raise ValueError("probe steps must be positive")
        if not 0.0 <= self.right_motion_fraction <= 1.0:
            raise ValueError("right_motion_fraction must be in [0, 1]")
        if self.maximum_wrist_force_n <= 0.0 or self.maximum_wrist_torque_nm <= 0.0:
            raise ValueError("wrist safety limits must be positive")
        if not 1 <= self.minimum_stable_fingertips_per_hand <= 4:
            raise ValueError("minimum_stable_fingertips_per_hand must be in [1, 4]")
        if self.fingertip_force_threshold_n <= 0.0:
            raise ValueError("fingertip_force_threshold_n must be positive")


@dataclass(frozen=True)
class ActiveViewTransition:
    feature_before: np.ndarray
    feature_after: np.ndarray
    reliability_before: float
    reliability_after: float
    control12: np.ndarray
    wrist_wrench_before: np.ndarray
    wrist_wrench_after: np.ndarray

    def __post_init__(self) -> None:
        before = np.asarray(self.feature_before, dtype=np.float32).reshape(-1)
        after = np.asarray(self.feature_after, dtype=np.float32).reshape(-1)
        if before.shape != after.shape or before.size == 0:
            raise ValueError("visual features must have the same non-empty shape")
        control = np.asarray(self.control12, dtype=np.float32).reshape(-1)
        if control.shape != (12,):
            raise ValueError("control12 must have shape (12,)")
        wrench_before = np.asarray(self.wrist_wrench_before, dtype=np.float32)
        wrench_after = np.asarray(self.wrist_wrench_after, dtype=np.float32)
        if wrench_before.shape != (2, 6) or wrench_after.shape != (2, 6):
            raise ValueError("wrist wrench arrays must have shape (2, 6)")
        for name, value in (
            ("reliability_before", self.reliability_before),
            ("reliability_after", self.reliability_after),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not all(
            np.isfinite(value).all()
            for value in (before, after, control, wrench_before, wrench_after)
        ):
            raise ValueError("active view transition must contain finite values")
        object.__setattr__(self, "feature_before", before.copy())
        object.__setattr__(self, "feature_after", after.copy())
        object.__setattr__(self, "control12", control.copy())
        object.__setattr__(self, "wrist_wrench_before", wrench_before.copy())
        object.__setattr__(self, "wrist_wrench_after", wrench_after.copy())


class SensorOnlyActiveViewProbe:
    """Generate transverse view probes only while grasp and wrench gates are safe."""

    def __init__(self, config: ActiveViewProbeConfig | None = None) -> None:
        self.config = config or ActiveViewProbeConfig()

    def sensor_gate(self, observation: CerebellumSensorObservation) -> bool:
        wrench = np.asarray(observation.wrist_wrench_world, dtype=np.float64)
        force_safe = np.all(
            np.linalg.norm(wrench[:, :3], axis=1)
            <= self.config.maximum_wrist_force_n
        )
        torque_safe = np.all(
            np.linalg.norm(wrench[:, 3:], axis=1)
            <= self.config.maximum_wrist_torque_nm
        )
        fingertip_force = np.linalg.norm(
            np.asarray(observation.fingertip_force_world, dtype=np.float64),
            axis=-1,
        )
        stable_counts = np.sum(
            fingertip_force >= self.config.fingertip_force_threshold_n,
            axis=1,
        )
        grasp_safe = np.all(
            stable_counts >= self.config.minimum_stable_fingertips_per_hand
        )
        return bool(force_safe and torque_safe and grasp_safe)

    def relative_probe_twists(self) -> tuple[np.ndarray, ...]:
        translation = self.config.translation_step_m
        rotation = self.config.rotation_step_rad
        return (
            np.asarray([translation, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.asarray([-translation, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.asarray([0.0, translation, 0.0, 0.0, 0.0, 0.0]),
            np.asarray([0.0, -translation, 0.0, 0.0, 0.0, 0.0]),
            np.asarray([0.0, 0.0, 0.0, rotation, 0.0, 0.0]),
            np.asarray([0.0, 0.0, 0.0, -rotation, 0.0, 0.0]),
            np.asarray([0.0, 0.0, 0.0, 0.0, rotation, 0.0]),
            np.asarray([0.0, 0.0, 0.0, 0.0, -rotation, 0.0]),
        )

    def bimanual_controls(
        self,
        observation: CerebellumSensorObservation,
    ) -> tuple[np.ndarray, ...]:
        if not self.sensor_gate(observation):
            return ()
        right_fraction = self.config.right_motion_fraction
        controls = []
        for relative in self.relative_probe_twists():
            controls.append(
                np.concatenate(
                    [
                        right_fraction * relative,
                        -(1.0 - right_fraction) * relative,
                    ]
                )
            )
        return tuple(controls)

    def action_candidates(
        self,
        observation: CerebellumSensorObservation,
        current_action44: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        actions = []
        for control in self.bimanual_controls(observation):
            actions.append(
                apply_bimanual_wrist_twists(
                    current_action44,
                    control[:6],
                    control[6:],
                    finger_reference44=current_action44,
                )
            )
        return tuple(actions)


def save_active_view_transitions(
    path: Path,
    transitions: tuple[ActiveViewTransition, ...],
    *,
    episode_index: int,
    frame_index: int,
) -> None:
    if not transitions:
        raise ValueError("at least one active view transition is required")
    feature_dim = transitions[0].feature_before.shape[0]
    if any(item.feature_before.shape != (feature_dim,) for item in transitions):
        raise ValueError("all active view transitions must share one feature dimension")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        episode_index=np.full(len(transitions), int(episode_index), dtype=np.int64),
        frame_index=np.full(len(transitions), int(frame_index), dtype=np.int64),
        feature_before=np.stack([item.feature_before for item in transitions]),
        feature_after=np.stack([item.feature_after for item in transitions]),
        reliability_before=np.asarray(
            [item.reliability_before for item in transitions], dtype=np.float32
        ),
        reliability_after=np.asarray(
            [item.reliability_after for item in transitions], dtype=np.float32
        ),
        control12=np.stack([item.control12 for item in transitions]),
        wrist_wrench_before=np.stack(
            [item.wrist_wrench_before for item in transitions]
        ),
        wrist_wrench_after=np.stack(
            [item.wrist_wrench_after for item in transitions]
        ),
    )


def load_active_view_transitions(path: Path) -> tuple[ActiveViewTransition, ...]:
    with np.load(path, allow_pickle=False) as data:
        feature_before = np.asarray(data["feature_before"], dtype=np.float32)
        feature_after = np.asarray(data["feature_after"], dtype=np.float32)
        reliability_before = np.asarray(data["reliability_before"], dtype=np.float32)
        reliability_after = np.asarray(data["reliability_after"], dtype=np.float32)
        control12 = np.asarray(data["control12"], dtype=np.float32)
        wrist_wrench_before = np.asarray(
            data["wrist_wrench_before"], dtype=np.float32
        )
        wrist_wrench_after = np.asarray(data["wrist_wrench_after"], dtype=np.float32)
    return tuple(
        ActiveViewTransition(
            feature_before=before,
            feature_after=after,
            reliability_before=float(before_reliability),
            reliability_after=float(after_reliability),
            control12=control,
            wrist_wrench_before=before_wrench,
            wrist_wrench_after=after_wrench,
        )
        for before, after, before_reliability, after_reliability, control, before_wrench, after_wrench in zip(
            feature_before,
            feature_after,
            reliability_before,
            reliability_after,
            control12,
            wrist_wrench_before,
            wrist_wrench_after,
            strict=True,
        )
    )
