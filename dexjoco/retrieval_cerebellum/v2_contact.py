"""Sensor-only contact event interpretation for the V2 controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sensor_observation import CerebellumSensorObservation
from .v2_control import V2AssemblyEstimate, V2ContactSignals


@dataclass(frozen=True)
class V2ContactInterpreterConfig:
    fingertip_contact_threshold_n: float = 0.5
    stable_grasp_contact_count: int = 3
    baseline_update_rate: float = 0.05
    rim_lateral_force_threshold_n: float = 2.5
    rim_axial_force_threshold_n: float = 4.0
    bilateral_lateral_force_threshold_n: float = 10.0
    jam_axial_force_threshold_n: float = 18.0
    bottom_axial_force_threshold_n: float = 12.0
    bottom_depth_margin_m: float = 0.002
    slip_contact_drop_count: int = 2
    slip_force_ratio: float = 0.45

    def __post_init__(self) -> None:
        if self.fingertip_contact_threshold_n <= 0.0:
            raise ValueError("fingertip_contact_threshold_n must be positive")
        if self.stable_grasp_contact_count <= 0:
            raise ValueError("stable_grasp_contact_count must be positive")
        if not 0.0 < self.baseline_update_rate <= 1.0:
            raise ValueError("baseline_update_rate must be in (0, 1]")
        if not 0.0 < self.slip_force_ratio < 1.0:
            raise ValueError("slip_force_ratio must be in (0, 1)")


class V2ContactInterpreter:
    """Convert deployable force channels into conservative contact events."""

    def __init__(self, config: V2ContactInterpreterConfig | None = None) -> None:
        self.config = config or V2ContactInterpreterConfig()
        self.reset()

    def reset(self) -> None:
        self._wrench_baseline = np.zeros((2, 6), dtype=np.float64)
        self._baseline_initialized = False
        self._previous_contact_count: np.ndarray | None = None
        self._previous_total_force: np.ndarray | None = None

    def update(
        self,
        observation: CerebellumSensorObservation,
        estimate: V2AssemblyEstimate,
        *,
        allow_baseline_update: bool,
    ) -> V2ContactSignals:
        fingertip_magnitude = np.linalg.norm(
            np.asarray(observation.fingertip_force_world, dtype=np.float64),
            axis=-1,
        )
        contact_count = np.sum(
            fingertip_magnitude >= self.config.fingertip_contact_threshold_n,
            axis=1,
        )
        total_force = np.sum(fingertip_magnitude, axis=1)
        wrist = np.asarray(observation.wrist_wrench_world, dtype=np.float64)
        if not self._baseline_initialized:
            self._wrench_baseline = wrist.copy()
            self._baseline_initialized = True
        elif allow_baseline_update:
            rate = self.config.baseline_update_rate
            self._wrench_baseline = (1.0 - rate) * self._wrench_baseline + rate * wrist
        residual = wrist - self._wrench_baseline
        right_force_hole = estimate.hole_rotation_world.T @ residual[0, :3]
        lateral_force = float(np.linalg.norm(right_force_hole[:2]))
        axial_force = float(abs(right_force_hole[2]))
        right_slip, left_slip = self._slip_events(contact_count, total_force)
        stability = np.clip(
            contact_count / float(self.config.stable_grasp_contact_count),
            0.0,
            1.0,
        )
        rim_contact = bool(
            lateral_force >= self.config.rim_lateral_force_threshold_n
            or axial_force >= self.config.rim_axial_force_threshold_n
        )
        bilateral_contact = bool(
            lateral_force >= self.config.bilateral_lateral_force_threshold_n
        )
        jammed = bool(axial_force >= self.config.jam_axial_force_threshold_n)
        bottom_contact = bool(
            axial_force >= self.config.bottom_axial_force_threshold_n
            and estimate.mean5[4]
            >= self.config.bottom_depth_margin_m
        )
        self._previous_contact_count = contact_count.copy()
        self._previous_total_force = total_force.copy()
        return V2ContactSignals(
            right_wrist_wrench_world=residual[0],
            left_wrist_wrench_world=residual[1],
            right_grasp_stability=float(stability[0]),
            left_grasp_stability=float(stability[1]),
            rim_contact=rim_contact,
            bilateral_contact=bilateral_contact,
            jammed=jammed,
            right_slip=right_slip,
            left_slip=left_slip,
            bottom_contact=bottom_contact,
        )

    def _slip_events(
        self,
        contact_count: np.ndarray,
        total_force: np.ndarray,
    ) -> tuple[bool, bool]:
        if self._previous_contact_count is None or self._previous_total_force is None:
            return False, False
        contact_drop = self._previous_contact_count - contact_count
        force_ratio = total_force / np.maximum(self._previous_total_force, 1e-6)
        slip = (
            (contact_drop >= self.config.slip_contact_drop_count)
            | (
                (self._previous_contact_count >= self.config.stable_grasp_contact_count)
                & (force_ratio <= self.config.slip_force_ratio)
            )
        )
        return bool(slip[0]), bool(slip[1])
