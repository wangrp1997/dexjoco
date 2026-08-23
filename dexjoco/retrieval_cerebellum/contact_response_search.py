"""Sensor-only bounded contact search for bimanual insertion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ContactSearchStrategy(str, Enum):
    FIXED_SPIRAL = "fixed_spiral"
    ADAPTIVE_SINGLE = "adaptive_single"
    ADAPTIVE_BIMANUAL = "adaptive_bimanual"


@dataclass(frozen=True)
class ContactSearchConfig:
    descent_step_m: float = 0.0005
    probe_step_m: float = 0.00035
    correction_step_m: float = 0.00045
    maximum_search_radius_m: float = 0.004
    contact_force_threshold_n: float = 1.0
    contact_release_threshold_n: float = 0.5
    contact_release_frames: int = 3
    hard_force_limit_n: float = 60.0
    spiral_angle_step_rad: float = 0.8
    spiral_radius_step_m: float = 0.00008


@dataclass(frozen=True)
class ContactSearchObservation:
    wrist_wrench_residual_world: np.ndarray
    grasp_stability: np.ndarray

    def __post_init__(self) -> None:
        wrench = np.asarray(self.wrist_wrench_residual_world, dtype=np.float64)
        stability = np.asarray(self.grasp_stability, dtype=np.float64).reshape(-1)
        if wrench.shape != (2, 6):
            raise ValueError("wrist_wrench_residual_world must have shape (2, 6)")
        if stability.shape != (2,) or np.any((stability < 0.0) | (stability > 1.0)):
            raise ValueError("grasp_stability must have shape (2,) in [0, 1]")


@dataclass(frozen=True)
class ContactSearchCommand:
    relative_translation_world: np.ndarray
    right_motion_fraction: float
    phase: str
    retreat: bool = False


class BoundedContactResponseSearch:
    """Guarded descent with fixed or response-identified tangential search."""

    def __init__(
        self,
        approach_axis_world: np.ndarray,
        strategy: ContactSearchStrategy,
        config: ContactSearchConfig | None = None,
    ) -> None:
        self.config = config or ContactSearchConfig()
        self.strategy = ContactSearchStrategy(strategy)
        axis = np.asarray(approach_axis_world, dtype=np.float64).reshape(3)
        axis /= np.linalg.norm(axis) + 1e-12
        hint = np.asarray([1.0, 0.0, 0.0])
        if abs(float(axis @ hint)) > 0.9:
            hint = np.asarray([0.0, 1.0, 0.0])
        tangent1 = hint - axis * float(axis @ hint)
        tangent1 /= np.linalg.norm(tangent1) + 1e-12
        tangent2 = np.cross(axis, tangent1)
        self.axis = axis
        self.tangents = np.stack([tangent1, tangent2])
        self._contact = False
        self._release_count = 0
        self._spiral_angle = 0.0
        self._spiral_radius = 0.0
        self._offset = np.zeros(2, dtype=np.float64)
        self._probe_index = 0
        self._pending_probe: tuple[int, int] | None = None
        self._probe_costs: dict[tuple[int, int], float] = {}

    def step(self, observation: ContactSearchObservation) -> ContactSearchCommand:
        force_norms = np.linalg.norm(observation.wrist_wrench_residual_world[:, :3], axis=1)
        if float(np.max(force_norms)) >= self.config.hard_force_limit_n:
            return ContactSearchCommand(
                relative_translation_world=-self.axis * self.config.descent_step_m,
                right_motion_fraction=self._motion_fraction(observation),
                phase="hard_force_retreat",
                retreat=True,
            )
        contact_cost = float(np.max(force_norms))
        if contact_cost >= self.config.contact_force_threshold_n:
            self._contact = True
            self._release_count = 0
        elif self._contact and contact_cost <= self.config.contact_release_threshold_n:
            self._release_count += 1
            if self._release_count >= self.config.contact_release_frames:
                self._reset_contact_search()
        elif self._contact:
            self._release_count = 0
        if not self._contact:
            return self._command(self.axis * self.config.descent_step_m, observation, "descent")
        if self.strategy == ContactSearchStrategy.FIXED_SPIRAL:
            return self._fixed_spiral(observation)
        return self._adaptive_step(observation, contact_cost)

    def _fixed_spiral(self, observation: ContactSearchObservation) -> ContactSearchCommand:
        self._spiral_angle += self.config.spiral_angle_step_rad
        self._spiral_radius = min(
            self.config.maximum_search_radius_m,
            self._spiral_radius + self.config.spiral_radius_step_m,
        )
        target = self._spiral_radius * np.asarray(
            [np.cos(self._spiral_angle), np.sin(self._spiral_angle)]
        )
        delta = target - self._offset
        self._offset = target
        translation = delta @ self.tangents + 0.35 * self.config.descent_step_m * self.axis
        return self._command(translation, observation, "fixed_spiral")

    def _adaptive_step(
        self,
        observation: ContactSearchObservation,
        contact_cost: float,
    ) -> ContactSearchCommand:
        if self._pending_probe is not None:
            self._probe_costs[self._pending_probe] = contact_cost
            self._pending_probe = None
        sequence = (
            (0, 1, 1.0),
            (0, -1, -2.0),
            (0, 0, 1.0),
            (1, 1, 1.0),
            (1, -1, -2.0),
            (1, 0, 1.0),
        )
        if self._probe_index < len(sequence):
            axis_index, sign, multiplier = sequence[self._probe_index]
            self._probe_index += 1
            if sign:
                self._pending_probe = (axis_index, sign)
            translation = (
                multiplier * self.config.probe_step_m * self.tangents[axis_index]
            )
            return self._command(translation, observation, f"probe_{axis_index}_{sign}")
        gradient = np.asarray(
            [
                self._probe_costs.get((axis_index, 1), contact_cost)
                - self._probe_costs.get((axis_index, -1), contact_cost)
                for axis_index in range(2)
            ]
        )
        if np.linalg.norm(gradient) > 1e-9:
            direction = -gradient / np.linalg.norm(gradient)
        else:
            direction = np.zeros(2)
        candidate = self._offset + self.config.correction_step_m * direction
        norm = float(np.linalg.norm(candidate))
        if norm > self.config.maximum_search_radius_m:
            candidate *= self.config.maximum_search_radius_m / norm
        delta = candidate - self._offset
        self._offset = candidate
        self._probe_index = 0
        self._probe_costs.clear()
        translation = delta @ self.tangents + 0.35 * self.config.descent_step_m * self.axis
        return self._command(translation, observation, "adaptive_correction")

    def _reset_contact_search(self) -> None:
        self._contact = False
        self._release_count = 0
        self._probe_index = 0
        self._pending_probe = None
        self._probe_costs.clear()

    def _command(
        self,
        translation: np.ndarray,
        observation: ContactSearchObservation,
        phase: str,
    ) -> ContactSearchCommand:
        return ContactSearchCommand(
            relative_translation_world=np.asarray(translation, dtype=np.float64),
            right_motion_fraction=self._motion_fraction(observation),
            phase=phase,
        )

    def _motion_fraction(self, observation: ContactSearchObservation) -> float:
        if self.strategy != ContactSearchStrategy.ADAPTIVE_BIMANUAL:
            return 1.0 if self.strategy == ContactSearchStrategy.ADAPTIVE_SINGLE else 0.8
        force = np.linalg.norm(observation.wrist_wrench_residual_world[:, :3], axis=1)
        score = observation.grasp_stability / (1.0 + force)
        if float(np.sum(score)) <= 1e-9:
            return 0.5
        return float(np.clip(score[0] / np.sum(score), 0.2, 0.8))
