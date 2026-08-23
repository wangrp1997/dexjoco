"""Safe independent execution of a π0.5 handoff action chunk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assembly_kinematics import (
    apply_bimanual_wrist_twists,
    pose_from_action44,
    world_wrist_twist,
)
from .contact_response_search import (
    BoundedContactResponseSearch,
    ContactSearchConfig,
    ContactSearchObservation,
    ContactSearchStrategy,
)
from .sensor_observation import CerebellumSensorObservation


@dataclass(frozen=True)
class IntentChunkExecutionConfig:
    soft_force_limit_n: float = 8.0
    hard_force_limit_n: float = 20.0
    fingertip_contact_threshold_n: float = 0.5
    soft_grasp_retention: float = 0.7
    hard_grasp_retention: float = 0.35
    enforce_grasp_retention: bool = False
    soft_tracking_translation_m: float = 0.003
    hard_tracking_translation_m: float = 0.015
    soft_tracking_rotation_rad: float = 0.03
    hard_tracking_rotation_rad: float = 0.15
    grasp_allocation_gain: float = 0.25
    force_allocation_gain: float = 0.2
    minimum_right_motion_fraction: float = 0.2
    maximum_right_motion_fraction: float = 0.8
    minimum_time_scale: float = 0.1
    maximum_translation_step_m: float = 0.0015
    maximum_rotation_step_rad: float = 0.015
    retreat_step_m: float = 0.001
    contact_response_enabled: bool = False
    contact_force_threshold_n: float = 4.0
    contact_probe_step_m: float = 0.0002
    contact_correction_step_m: float = 0.0003
    maximum_contact_search_radius_m: float = 0.003
    contact_rotation_compliance_rad_per_nm: float = 0.0
    maximum_contact_rotation_correction_rad: float = 0.0015

    def __post_init__(self) -> None:
        if not 0.0 < self.soft_force_limit_n < self.hard_force_limit_n:
            raise ValueError("force limits must satisfy 0 < soft < hard")
        if self.fingertip_contact_threshold_n < 0.0:
            raise ValueError("fingertip_contact_threshold_n must be non-negative")
        if not 0.0 <= self.hard_grasp_retention < self.soft_grasp_retention <= 1.0:
            raise ValueError("grasp retention must satisfy 0 <= hard < soft <= 1")
        if not (
            0.0
            < self.soft_tracking_translation_m
            < self.hard_tracking_translation_m
        ):
            raise ValueError("translation tracking limits must satisfy 0 < soft < hard")
        if not 0.0 < self.soft_tracking_rotation_rad < self.hard_tracking_rotation_rad:
            raise ValueError("rotation tracking limits must satisfy 0 < soft < hard")
        if self.grasp_allocation_gain < 0.0 or self.force_allocation_gain < 0.0:
            raise ValueError("allocation gains must be non-negative")
        if not (
            0.0
            <= self.minimum_right_motion_fraction
            <= 0.5
            <= self.maximum_right_motion_fraction
            <= 1.0
        ):
            raise ValueError("right motion fraction limits must contain 0.5")
        if not 0.0 < self.minimum_time_scale <= 1.0:
            raise ValueError("minimum_time_scale must be in (0, 1]")
        if self.maximum_translation_step_m <= 0.0:
            raise ValueError("maximum_translation_step_m must be positive")
        if self.maximum_rotation_step_rad <= 0.0:
            raise ValueError("maximum_rotation_step_rad must be positive")
        if self.retreat_step_m <= 0.0:
            raise ValueError("retreat_step_m must be positive")
        if self.contact_force_threshold_n <= 0.0:
            raise ValueError("contact_force_threshold_n must be positive")
        if self.contact_probe_step_m <= 0.0:
            raise ValueError("contact_probe_step_m must be positive")
        if self.contact_correction_step_m <= 0.0:
            raise ValueError("contact_correction_step_m must be positive")
        if self.maximum_contact_search_radius_m <= 0.0:
            raise ValueError("maximum_contact_search_radius_m must be positive")
        if self.contact_rotation_compliance_rad_per_nm < 0.0:
            raise ValueError("contact_rotation_compliance_rad_per_nm must be non-negative")
        if self.maximum_contact_rotation_correction_rad <= 0.0:
            raise ValueError("maximum_contact_rotation_correction_rad must be positive")


@dataclass(frozen=True)
class IntentChunkStep:
    action44: np.ndarray
    active: bool
    outcome: str
    phase: float
    time_scale: float
    peak_force_n: float
    force_scale: float = 1.0
    grasp_scale: float = 1.0
    tracking_scale: float = 1.0
    minimum_grasp_retention: float = 1.0
    right_motion_fraction: float = 0.5
    grasp_observable: bool = True
    contact_phase: str = "inactive"
    contact_correction_m: float = 0.0
    contact_rotation_correction_rad: float = 0.0


class OnlineIntentChunkExecutor:
    """Retimes a frozen policy chunk using only proprioception and force sensing."""

    def __init__(self, config: IntentChunkExecutionConfig | None = None) -> None:
        self.config = config or IntentChunkExecutionConfig()
        self.reset()

    def reset(self) -> None:
        self._chunk: np.ndarray | None = None
        self._baseline_wrench: np.ndarray | None = None
        self._baseline_fingertip_force: np.ndarray | None = None
        self._grasp_observable_sides = np.zeros(2, dtype=bool)
        self._contact_search: BoundedContactResponseSearch | None = None
        self._phase = 0.0
        self._peak_force_n = 0.0
        self._outcome = "idle"

    @property
    def active(self) -> bool:
        return self._chunk is not None and self._outcome == "executing"

    @property
    def outcome(self) -> str:
        return self._outcome

    def start(
        self,
        action_chunk44: np.ndarray,
        observation: CerebellumSensorObservation,
    ) -> None:
        chunk = np.asarray(action_chunk44, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != 44:
            raise ValueError(f"action_chunk44 must have shape (H, 44), got {chunk.shape}")
        if chunk.shape[0] < 2:
            raise ValueError("action_chunk44 must contain at least two actions")
        if not np.isfinite(chunk).all():
            raise ValueError("action_chunk44 must contain finite values")
        self._chunk = chunk.copy()
        self._baseline_wrench = np.asarray(
            observation.wrist_wrench_world,
            dtype=np.float64,
        ).copy()
        self._baseline_fingertip_force = np.linalg.norm(
            np.asarray(observation.fingertip_force_world, dtype=np.float64),
            axis=2,
        )
        self._grasp_observable_sides = np.any(
            self._baseline_fingertip_force
            >= self.config.fingertip_contact_threshold_n,
            axis=1,
        )
        self._contact_search = self._build_contact_search(chunk)
        self._phase = 0.0
        self._peak_force_n = 0.0
        self._outcome = "executing"

    def step(
        self,
        observation: CerebellumSensorObservation,
        current_action44: np.ndarray,
    ) -> IntentChunkStep:
        if (
            not self.active
            or self._chunk is None
            or self._baseline_wrench is None
            or self._baseline_fingertip_force is None
        ):
            raise RuntimeError("intent chunk executor is not active")
        residual = np.asarray(observation.wrist_wrench_world) - self._baseline_wrench
        side_force_n = np.linalg.norm(residual[:, :3], axis=1)
        force_n = float(np.max(side_force_n))
        self._peak_force_n = max(self._peak_force_n, force_n)
        if force_n >= self.config.hard_force_limit_n:
            action = self._retreat(current_action44)
            self._outcome = "hard_force_retreat"
            return self._result(
                action,
                active=False,
                time_scale=0.0,
                force_scale=0.0,
            )

        grasp_retention = self._grasp_retention(observation)
        grasp_observable = bool(np.all(self._grasp_observable_sides))
        minimum_grasp_retention = float(np.min(grasp_retention))
        if (
            self.config.enforce_grasp_retention
            and minimum_grasp_retention <= self.config.hard_grasp_retention
        ):
            self._outcome = "grasp_unstable_stop"
            return self._result(
                np.asarray(current_action44, dtype=np.float32),
                active=False,
                time_scale=0.0,
                grasp_scale=0.0,
                minimum_grasp_retention=minimum_grasp_retention,
                grasp_observable=grasp_observable,
            )

        force_scale = self._force_scale(force_n)
        grasp_scale = (
            self._grasp_scale(minimum_grasp_retention)
            if self.config.enforce_grasp_retention
            else 1.0
        )
        tracking_scale = self._tracking_scale(observation, current_action44)
        if tracking_scale <= 0.0:
            self._outcome = "tracking_error_stop"
            return self._result(
                np.asarray(current_action44, dtype=np.float32),
                active=False,
                time_scale=0.0,
                force_scale=force_scale,
                grasp_scale=grasp_scale,
                tracking_scale=0.0,
                minimum_grasp_retention=minimum_grasp_retention,
                grasp_observable=grasp_observable,
            )
        time_scale = force_scale * grasp_scale * tracking_scale
        if time_scale > 0.0:
            time_scale = max(time_scale, self.config.minimum_time_scale)
        segment = min(int(self._phase), self._chunk.shape[0] - 2)
        segment_fraction = self._phase - segment
        remaining = 1.0 - segment_fraction
        right_segment_twist = world_wrist_twist(
            pose_from_action44(self._chunk[segment], side="right"),
            pose_from_action44(self._chunk[segment + 1], side="right"),
        )
        left_segment_twist = world_wrist_twist(
            pose_from_action44(self._chunk[segment], side="left"),
            pose_from_action44(self._chunk[segment + 1], side="left"),
        )
        right_motion_fraction = self._right_motion_fraction(
            side_force_n,
            (
                grasp_retention
                if self.config.enforce_grasp_retention
                else np.ones(2, dtype=np.float64)
            ),
        )
        right_segment_twist, left_segment_twist = self._allocate_relative_motion(
            right_segment_twist,
            left_segment_twist,
            right_motion_fraction,
        )
        phase_step = min(
            time_scale,
            remaining,
            self._safe_phase_step(right_segment_twist),
            self._safe_phase_step(left_segment_twist),
        )
        right_twist = right_segment_twist * phase_step
        left_twist = left_segment_twist * phase_step
        contact_phase = "inactive"
        contact_correction_m = 0.0
        contact_rotation_correction_rad = 0.0
        if self._contact_search is not None:
            contact_command = self._contact_search.step(
                ContactSearchObservation(
                    wrist_wrench_residual_world=residual,
                    grasp_stability=grasp_retention,
                )
            )
            contact_phase = contact_command.phase
            if contact_phase != "descent" and not contact_command.retreat:
                correction = np.asarray(
                    contact_command.relative_translation_world,
                    dtype=np.float64,
                )
                axis = self._contact_search.axis
                correction -= axis * float(axis @ correction)
                contact_correction_m = float(np.linalg.norm(correction))
                correction_twist = np.concatenate([correction, np.zeros(3)])
                fraction = contact_command.right_motion_fraction
                right_twist += fraction * correction_twist
                left_twist -= (1.0 - fraction) * correction_twist
                torque_error = residual[0, 3:] - residual[1, 3:]
                torque_error -= axis * float(axis @ torque_error)
                rotation_correction = (
                    -self.config.contact_rotation_compliance_rad_per_nm
                    * torque_error
                )
                rotation_norm = float(np.linalg.norm(rotation_correction))
                if rotation_norm > self.config.maximum_contact_rotation_correction_rad:
                    rotation_correction *= (
                        self.config.maximum_contact_rotation_correction_rad
                        / rotation_norm
                    )
                contact_rotation_correction_rad = float(
                    np.linalg.norm(rotation_correction)
                )
                right_twist[3:] += fraction * rotation_correction
                left_twist[3:] -= (1.0 - fraction) * rotation_correction
        right_twist = self._bounded_twist(right_twist)
        left_twist = self._bounded_twist(left_twist)
        target_fraction = segment_fraction + phase_step
        finger_reference = (
            (1.0 - target_fraction) * self._chunk[segment]
            + target_fraction * self._chunk[segment + 1]
        )
        action = apply_bimanual_wrist_twists(
            current_action44,
            right_twist,
            left_twist,
            finger_reference44=finger_reference,
        )
        self._phase += phase_step
        if self._phase >= self._chunk.shape[0] - 1 - 1e-9:
            self._outcome = "chunk_complete"
            return self._result(
                action,
                active=False,
                time_scale=time_scale,
                force_scale=force_scale,
                grasp_scale=grasp_scale,
                tracking_scale=tracking_scale,
                minimum_grasp_retention=minimum_grasp_retention,
                right_motion_fraction=right_motion_fraction,
                grasp_observable=grasp_observable,
                contact_phase=contact_phase,
                contact_correction_m=contact_correction_m,
                contact_rotation_correction_rad=contact_rotation_correction_rad,
            )
        return self._result(
            action,
            active=True,
            time_scale=time_scale,
            force_scale=force_scale,
            grasp_scale=grasp_scale,
            tracking_scale=tracking_scale,
            minimum_grasp_retention=minimum_grasp_retention,
            right_motion_fraction=right_motion_fraction,
            grasp_observable=grasp_observable,
            contact_phase=contact_phase,
            contact_correction_m=contact_correction_m,
            contact_rotation_correction_rad=contact_rotation_correction_rad,
        )

    def _build_contact_search(
        self,
        chunk: np.ndarray,
    ) -> BoundedContactResponseSearch | None:
        if not self.config.contact_response_enabled:
            return None
        relative_start = (
            pose_from_action44(chunk[0], side="right")[:3, 3]
            - pose_from_action44(chunk[0], side="left")[:3, 3]
        )
        relative_end = (
            pose_from_action44(chunk[-1], side="right")[:3, 3]
            - pose_from_action44(chunk[-1], side="left")[:3, 3]
        )
        approach = relative_end - relative_start
        if float(np.linalg.norm(approach)) <= 1e-6:
            return None
        return BoundedContactResponseSearch(
            approach,
            ContactSearchStrategy.ADAPTIVE_BIMANUAL,
            ContactSearchConfig(
                descent_step_m=self.config.maximum_translation_step_m,
                probe_step_m=self.config.contact_probe_step_m,
                correction_step_m=self.config.contact_correction_step_m,
                maximum_search_radius_m=self.config.maximum_contact_search_radius_m,
                contact_force_threshold_n=self.config.contact_force_threshold_n,
                contact_release_threshold_n=(
                    0.5 * self.config.contact_force_threshold_n
                ),
                hard_force_limit_n=self.config.hard_force_limit_n,
            ),
        )

    def _force_scale(self, force_n: float) -> float:
        if force_n <= self.config.soft_force_limit_n:
            return 1.0
        span = self.config.hard_force_limit_n - self.config.soft_force_limit_n
        fraction = (self.config.hard_force_limit_n - force_n) / span
        return float(np.clip(fraction, 0.0, 1.0))

    def _grasp_retention(
        self,
        observation: CerebellumSensorObservation,
    ) -> np.ndarray:
        assert self._baseline_fingertip_force is not None
        current = np.linalg.norm(
            np.asarray(observation.fingertip_force_world, dtype=np.float64),
            axis=2,
        )
        retention = np.ones(2, dtype=np.float64)
        for side in range(2):
            active = (
                self._baseline_fingertip_force[side]
                >= self.config.fingertip_contact_threshold_n
            )
            if np.any(active):
                ratios = current[side, active] / np.maximum(
                    self._baseline_fingertip_force[side, active],
                    self.config.fingertip_contact_threshold_n,
                )
                retention[side] = float(np.clip(np.mean(ratios), 0.0, 1.0))
        return retention

    def _grasp_scale(self, retention: float) -> float:
        if retention >= self.config.soft_grasp_retention:
            return 1.0
        span = self.config.soft_grasp_retention - self.config.hard_grasp_retention
        return float(
            np.clip(
                (retention - self.config.hard_grasp_retention) / span,
                0.0,
                1.0,
            )
        )

    def _tracking_scale(
        self,
        observation: CerebellumSensorObservation,
        current_action44: np.ndarray,
    ) -> float:
        if observation.previous_action44 is None:
            return 1.0
        translation_error = 0.0
        rotation_error = 0.0
        for side in ("right", "left"):
            error = world_wrist_twist(
                pose_from_action44(current_action44, side=side),
                pose_from_action44(observation.previous_action44, side=side),
            )
            translation_error = max(translation_error, float(np.linalg.norm(error[:3])))
            rotation_error = max(rotation_error, float(np.linalg.norm(error[3:])))
        translation_scale = self._decreasing_scale(
            translation_error,
            self.config.soft_tracking_translation_m,
            self.config.hard_tracking_translation_m,
        )
        rotation_scale = self._decreasing_scale(
            rotation_error,
            self.config.soft_tracking_rotation_rad,
            self.config.hard_tracking_rotation_rad,
        )
        return min(translation_scale, rotation_scale)

    @staticmethod
    def _decreasing_scale(value: float, soft: float, hard: float) -> float:
        if value <= soft:
            return 1.0
        return float(np.clip((hard - value) / (hard - soft), 0.0, 1.0))

    def _right_motion_fraction(
        self,
        side_force_n: np.ndarray,
        grasp_retention: np.ndarray,
    ) -> float:
        force_normalizer = max(self.config.soft_force_limit_n, 1e-9)
        force_imbalance = float((side_force_n[1] - side_force_n[0]) / force_normalizer)
        grasp_imbalance = float(grasp_retention[0] - grasp_retention[1])
        fraction = (
            0.5
            + self.config.force_allocation_gain * force_imbalance
            + self.config.grasp_allocation_gain * grasp_imbalance
        )
        return float(
            np.clip(
                fraction,
                self.config.minimum_right_motion_fraction,
                self.config.maximum_right_motion_fraction,
            )
        )

    @staticmethod
    def _allocate_relative_motion(
        right_twist: np.ndarray,
        left_twist: np.ndarray,
        right_motion_fraction: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        right = np.asarray(right_twist, dtype=np.float64)
        left = np.asarray(left_twist, dtype=np.float64)
        common = 0.5 * (right + left)
        relative = right - left
        return (
            common + right_motion_fraction * relative,
            common - (1.0 - right_motion_fraction) * relative,
        )

    def _bounded_twist(self, twist: np.ndarray) -> np.ndarray:
        bounded = np.asarray(twist, dtype=np.float64).copy()
        translation_norm = float(np.linalg.norm(bounded[:3]))
        if translation_norm > self.config.maximum_translation_step_m:
            bounded[:3] *= self.config.maximum_translation_step_m / translation_norm
        rotation_norm = float(np.linalg.norm(bounded[3:]))
        if rotation_norm > self.config.maximum_rotation_step_rad:
            bounded[3:] *= self.config.maximum_rotation_step_rad / rotation_norm
        return bounded

    def _safe_phase_step(self, twist: np.ndarray) -> float:
        limits = [1.0]
        translation_norm = float(np.linalg.norm(twist[:3]))
        if translation_norm > 1e-12:
            limits.append(self.config.maximum_translation_step_m / translation_norm)
        rotation_norm = float(np.linalg.norm(twist[3:]))
        if rotation_norm > 1e-12:
            limits.append(self.config.maximum_rotation_step_rad / rotation_norm)
        return float(min(limits))

    def _retreat(self, current_action44: np.ndarray) -> np.ndarray:
        assert self._chunk is not None
        right = world_wrist_twist(
            pose_from_action44(self._chunk[0], side="right"),
            pose_from_action44(self._chunk[-1], side="right"),
        )
        left = world_wrist_twist(
            pose_from_action44(self._chunk[0], side="left"),
            pose_from_action44(self._chunk[-1], side="left"),
        )
        for twist in (right, left):
            norm = float(np.linalg.norm(twist[:3]))
            direction = np.zeros(3) if norm <= 1e-9 else -twist[:3] / norm
            twist[:] = 0.0
            twist[:3] = direction * self.config.retreat_step_m
        return apply_bimanual_wrist_twists(
            current_action44,
            self._bounded_twist(right),
            self._bounded_twist(left),
            finger_reference44=current_action44,
        )

    def _result(
        self,
        action44: np.ndarray,
        *,
        active: bool,
        time_scale: float,
        force_scale: float = 1.0,
        grasp_scale: float = 1.0,
        tracking_scale: float = 1.0,
        minimum_grasp_retention: float = 1.0,
        right_motion_fraction: float = 0.5,
        grasp_observable: bool = True,
        contact_phase: str = "inactive",
        contact_correction_m: float = 0.0,
        contact_rotation_correction_rad: float = 0.0,
    ) -> IntentChunkStep:
        return IntentChunkStep(
            action44=np.asarray(action44, dtype=np.float32),
            active=active,
            outcome=self._outcome,
            phase=float(self._phase),
            time_scale=float(time_scale),
            peak_force_n=float(self._peak_force_n),
            force_scale=float(force_scale),
            grasp_scale=float(grasp_scale),
            tracking_scale=float(tracking_scale),
            minimum_grasp_retention=float(minimum_grasp_retention),
            right_motion_fraction=float(right_motion_fraction),
            grasp_observable=bool(grasp_observable),
            contact_phase=str(contact_phase),
            contact_correction_m=float(contact_correction_m),
            contact_rotation_correction_rad=float(
                contact_rotation_correction_rad
            ),
        )
