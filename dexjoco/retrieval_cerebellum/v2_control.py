"""Mode-gated compliant controller for the V2 insertion cerebellum."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


STATE_DIM = 5
TWIST_DIM = 6
WRENCH_DIM = 6


def _readonly_vector(value: np.ndarray, size: int, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    result = array.copy()
    result.flags.writeable = False
    return result


def _readonly_matrix(
    value: np.ndarray,
    shape: tuple[int, int],
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    result = array.copy()
    result.flags.writeable = False
    return result


class V2Mode(str, Enum):
    ALIGN = "align"
    GUARDED_DESCENT = "guarded_descent"
    CONTACT_CORRECTION = "contact_correction"
    INSERT = "insert"
    RETREAT = "retreat"
    REGRASP_REQUEST = "regrasp_request"
    SUCCESS = "success"


@dataclass(frozen=True)
class V2AssemblyEstimate:
    """Deployable visual-fusion estimate consumed by the V2 controller."""

    timestamp_s: float
    mean5: np.ndarray
    covariance5: np.ndarray
    hole_rotation_world: np.ndarray
    visual_reliability: float

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp_s)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        reliability = float(self.visual_reliability)
        if not 0.0 <= reliability <= 1.0:
            raise ValueError("visual_reliability must be in [0, 1]")
        covariance = _readonly_matrix(
            self.covariance5,
            (STATE_DIM, STATE_DIM),
            name="covariance5",
        )
        if not np.allclose(covariance, covariance.T, atol=1e-9, rtol=0.0):
            raise ValueError("covariance5 must be symmetric")
        if float(np.linalg.eigvalsh(covariance).min()) < -1e-9:
            raise ValueError("covariance5 must be positive semidefinite")
        rotation = _readonly_matrix(
            self.hole_rotation_world,
            (3, 3),
            name="hole_rotation_world",
        )
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("hole_rotation_world must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("hole_rotation_world must be right-handed")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "mean5", _readonly_vector(self.mean5, 5, name="mean5"))
        object.__setattr__(self, "covariance5", covariance)
        object.__setattr__(self, "hole_rotation_world", rotation)
        object.__setattr__(self, "visual_reliability", reliability)


@dataclass(frozen=True)
class V2ContactSignals:
    """Sensor-derived events without object or contact ground truth."""

    right_wrist_wrench_world: np.ndarray
    left_wrist_wrench_world: np.ndarray
    right_grasp_stability: float
    left_grasp_stability: float
    rim_contact: bool = False
    bilateral_contact: bool = False
    jammed: bool = False
    right_slip: bool = False
    left_slip: bool = False
    bottom_contact: bool = False

    def __post_init__(self) -> None:
        for name in ("right_grasp_stability", "left_grasp_stability"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "right_wrist_wrench_world",
            _readonly_vector(
                self.right_wrist_wrench_world,
                WRENCH_DIM,
                name="right_wrist_wrench_world",
            ),
        )
        object.__setattr__(
            self,
            "left_wrist_wrench_world",
            _readonly_vector(
                self.left_wrist_wrench_world,
                WRENCH_DIM,
                name="left_wrist_wrench_world",
            ),
        )


@dataclass(frozen=True)
class V2ControllerConfig:
    minimum_visual_reliability: float = 0.55
    minimum_grasp_stability: float = 0.45
    align_lateral_threshold_m: float = 0.003
    align_tilt_threshold_rad: float = 0.08
    insert_lateral_threshold_m: float = 0.0012
    insert_tilt_threshold_rad: float = 0.03
    target_depth_m: float = 0.0125
    align_translation_gain: float = 0.35
    align_rotation_gain: float = 0.35
    contact_translation_gain: float = 0.18
    contact_rotation_gain: float = 0.18
    guarded_descent_step_m: float = 0.00035
    insert_step_m: float = 0.00025
    retreat_step_m: float = 0.0005
    maximum_translation_step_m: float = 0.0006
    maximum_rotation_step_rad: float = 0.006
    hard_force_limit_n: float = 80.0
    hard_torque_limit_nm: float = 8.0
    contact_force_deadband_n: float = 1.0
    contact_force_gain_m_per_n: float = 0.00003
    right_motion_fraction: float = 0.8
    transition_confirmation_steps: int = 2
    maximum_stagnant_insert_steps: int = 5
    minimum_depth_progress_m: float = 0.00002

    def __post_init__(self) -> None:
        positive = (
            "align_lateral_threshold_m",
            "align_tilt_threshold_rad",
            "insert_lateral_threshold_m",
            "insert_tilt_threshold_rad",
            "target_depth_m",
            "guarded_descent_step_m",
            "insert_step_m",
            "retreat_step_m",
            "maximum_translation_step_m",
            "maximum_rotation_step_rad",
            "hard_force_limit_n",
            "hard_torque_limit_nm",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "minimum_visual_reliability",
            "minimum_grasp_stability",
            "right_motion_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.transition_confirmation_steps <= 0:
            raise ValueError("transition_confirmation_steps must be positive")
        if self.maximum_stagnant_insert_steps <= 0:
            raise ValueError("maximum_stagnant_insert_steps must be positive")


@dataclass(frozen=True)
class V2ControllerCommand:
    mode: V2Mode
    right_twist_world: np.ndarray
    left_twist_world: np.ndarray
    right_stiffness_scale: np.ndarray
    left_stiffness_scale: np.ndarray
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")
        object.__setattr__(
            self,
            "right_twist_world",
            _readonly_vector(self.right_twist_world, TWIST_DIM, name="right_twist_world"),
        )
        object.__setattr__(
            self,
            "left_twist_world",
            _readonly_vector(self.left_twist_world, TWIST_DIM, name="left_twist_world"),
        )
        object.__setattr__(
            self,
            "right_stiffness_scale",
            _readonly_vector(
                self.right_stiffness_scale,
                TWIST_DIM,
                name="right_stiffness_scale",
            ),
        )
        object.__setattr__(
            self,
            "left_stiffness_scale",
            _readonly_vector(
                self.left_stiffness_scale,
                TWIST_DIM,
                name="left_stiffness_scale",
            ),
        )


class ModeGatedCompliantController:
    """Four-state controller with explicit retreat and regrasp exits."""

    def __init__(self, config: V2ControllerConfig | None = None) -> None:
        self.config = config or V2ControllerConfig()
        self.reset()

    def reset(self) -> None:
        self.mode = V2Mode.ALIGN
        self._transition_count = 0
        self._stagnant_insert_steps = 0
        self._previous_depth: float | None = None

    def step(
        self,
        estimate: V2AssemblyEstimate,
        contact: V2ContactSignals,
    ) -> V2ControllerCommand:
        terminal = self._terminal_transition(estimate, contact)
        if terminal is not None:
            mode, reason = terminal
            self.mode = mode
            if mode == V2Mode.RETREAT:
                return self._retreat_command(estimate, reason)
            return self._stationary_command(mode, reason)

        if self.mode == V2Mode.ALIGN:
            return self._step_align(estimate, contact)
        if self.mode == V2Mode.GUARDED_DESCENT:
            return self._step_guarded_descent(estimate, contact)
        if self.mode == V2Mode.CONTACT_CORRECTION:
            return self._step_contact_correction(estimate, contact)
        if self.mode == V2Mode.INSERT:
            return self._step_insert(estimate, contact)
        if self.mode == V2Mode.RETREAT:
            return self._retreat_command(estimate, "retreat in progress")
        return self._stationary_command(self.mode, f"terminal mode {self.mode.value}")

    def _terminal_transition(
        self,
        estimate: V2AssemblyEstimate,
        contact: V2ContactSignals,
    ) -> tuple[V2Mode, str] | None:
        if contact.right_slip or contact.right_grasp_stability < self.config.minimum_grasp_stability:
            return V2Mode.REGRASP_REQUEST, "right grasp is unstable"
        if contact.left_slip or contact.left_grasp_stability < self.config.minimum_grasp_stability:
            return V2Mode.REGRASP_REQUEST, "left grasp is unstable"
        if self._hard_wrench_limit_exceeded(contact):
            return V2Mode.RETREAT, "hard wrist wrench limit exceeded"
        if contact.jammed or contact.bilateral_contact:
            return V2Mode.RETREAT, "jammed or bilateral rim contact detected"
        if contact.bottom_contact or estimate.mean5[4] >= self.config.target_depth_m:
            return V2Mode.SUCCESS, "target insertion depth reached"
        if estimate.visual_reliability < self.config.minimum_visual_reliability:
            return V2Mode.RETREAT, "visual estimate is not reliable enough"
        return None

    def _step_align(
        self,
        estimate: V2AssemblyEstimate,
        contact: V2ContactSignals,
    ) -> V2ControllerCommand:
        if contact.rim_contact:
            self.mode = V2Mode.CONTACT_CORRECTION
            self._transition_count = 0
            return self._step_contact_correction(estimate, contact)
        if self._inside_align_funnel(estimate.mean5):
            self._transition_count += 1
            if self._transition_count >= self.config.transition_confirmation_steps:
                self.mode = V2Mode.GUARDED_DESCENT
                self._transition_count = 0
                return self._step_guarded_descent(estimate, contact)
        else:
            self._transition_count = 0
        relative = self._visual_correction(
            estimate,
            translation_gain=self.config.align_translation_gain,
            rotation_gain=self.config.align_rotation_gain,
            include_axial=False,
        )
        return self._split_command(
            V2Mode.ALIGN,
            relative,
            right_stiffness=np.asarray([0.8, 0.8, 0.5, 0.7, 0.7, 0.4]),
            left_stiffness=np.asarray([0.6, 0.6, 0.8, 0.5, 0.5, 0.7]),
            reason="visual lateral and tilt alignment",
        )

    def _step_guarded_descent(
        self,
        estimate: V2AssemblyEstimate,
        contact: V2ContactSignals,
    ) -> V2ControllerCommand:
        if contact.rim_contact:
            self.mode = V2Mode.CONTACT_CORRECTION
            return self._step_contact_correction(estimate, contact)
        if not self._inside_align_funnel(estimate.mean5):
            self.mode = V2Mode.ALIGN
            return self._step_align(estimate, contact)
        relative = self._visual_correction(
            estimate,
            translation_gain=self.config.contact_translation_gain,
            rotation_gain=self.config.contact_rotation_gain,
            include_axial=False,
        )
        relative[:3] -= (
            estimate.hole_rotation_world[:, 2] * self.config.guarded_descent_step_m
        )
        return self._split_command(
            V2Mode.GUARDED_DESCENT,
            relative,
            right_stiffness=np.asarray([0.45, 0.45, 0.75, 0.35, 0.35, 0.5]),
            left_stiffness=np.asarray([0.55, 0.55, 0.8, 0.45, 0.45, 0.6]),
            reason="low-speed descent with lateral compliance",
        )

    def _step_contact_correction(
        self,
        estimate: V2AssemblyEstimate,
        contact: V2ContactSignals,
    ) -> V2ControllerCommand:
        if not contact.rim_contact and self._inside_insert_funnel(estimate.mean5):
            self._transition_count += 1
            if self._transition_count >= self.config.transition_confirmation_steps:
                self.mode = V2Mode.INSERT
                self._transition_count = 0
                return self._step_insert(estimate, contact)
        else:
            self._transition_count = 0
        relative = self._visual_correction(
            estimate,
            translation_gain=self.config.contact_translation_gain,
            rotation_gain=self.config.contact_rotation_gain,
            include_axial=False,
        )
        force_hole = estimate.hole_rotation_world.T @ contact.right_wrist_wrench_world[:3]
        lateral_force = force_hole[:2]
        magnitude = float(np.linalg.norm(lateral_force))
        if magnitude > self.config.contact_force_deadband_n:
            force_correction_hole = np.asarray(
                [-lateral_force[0], -lateral_force[1], 0.0],
                dtype=np.float64,
            )
            force_correction_hole *= self.config.contact_force_gain_m_per_n
            relative[:3] += estimate.hole_rotation_world @ force_correction_hole
        return self._split_command(
            V2Mode.CONTACT_CORRECTION,
            relative,
            right_stiffness=np.asarray([0.25, 0.25, 0.45, 0.2, 0.2, 0.35]),
            left_stiffness=np.asarray([0.4, 0.4, 0.65, 0.3, 0.3, 0.5]),
            reason="rim-contact correction with force relief",
        )

    def _step_insert(
        self,
        estimate: V2AssemblyEstimate,
        contact: V2ContactSignals,
    ) -> V2ControllerCommand:
        if contact.rim_contact and not self._inside_insert_funnel(estimate.mean5):
            self.mode = V2Mode.CONTACT_CORRECTION
            return self._step_contact_correction(estimate, contact)
        depth = float(estimate.mean5[4])
        if self._previous_depth is not None:
            progress = depth - self._previous_depth
            if progress < self.config.minimum_depth_progress_m:
                self._stagnant_insert_steps += 1
            else:
                self._stagnant_insert_steps = 0
        self._previous_depth = depth
        if self._stagnant_insert_steps >= self.config.maximum_stagnant_insert_steps:
            self.mode = V2Mode.RETREAT
            return self._retreat_command(estimate, "insertion depth is stagnant")
        relative = self._visual_correction(
            estimate,
            translation_gain=self.config.contact_translation_gain,
            rotation_gain=self.config.contact_rotation_gain,
            include_axial=False,
        )
        relative[:3] -= estimate.hole_rotation_world[:, 2] * self.config.insert_step_m
        return self._split_command(
            V2Mode.INSERT,
            relative,
            right_stiffness=np.asarray([0.2, 0.2, 0.6, 0.18, 0.18, 0.35]),
            left_stiffness=np.asarray([0.35, 0.35, 0.7, 0.25, 0.25, 0.45]),
            reason="axial insertion with lateral and rotational compliance",
        )

    def _visual_correction(
        self,
        estimate: V2AssemblyEstimate,
        *,
        translation_gain: float,
        rotation_gain: float,
        include_axial: bool,
    ) -> np.ndarray:
        state = estimate.mean5
        translation_hole = np.asarray(
            [-translation_gain * state[0], -translation_gain * state[1], 0.0],
            dtype=np.float64,
        )
        rotation_hole = np.asarray(
            [-rotation_gain * state[2], -rotation_gain * state[3], 0.0],
            dtype=np.float64,
        )
        if include_axial:
            translation_hole[2] = translation_gain * state[4]
        relative = np.concatenate(
            [
                estimate.hole_rotation_world @ translation_hole,
                estimate.hole_rotation_world @ rotation_hole,
            ]
        )
        relative[:3] = np.clip(
            relative[:3],
            -self.config.maximum_translation_step_m,
            self.config.maximum_translation_step_m,
        )
        relative[3:] = np.clip(
            relative[3:],
            -self.config.maximum_rotation_step_rad,
            self.config.maximum_rotation_step_rad,
        )
        return relative

    def _split_command(
        self,
        mode: V2Mode,
        relative_twist: np.ndarray,
        *,
        right_stiffness: np.ndarray,
        left_stiffness: np.ndarray,
        reason: str,
    ) -> V2ControllerCommand:
        fraction = self.config.right_motion_fraction
        right = np.asarray(relative_twist, dtype=np.float64) * fraction
        left = -np.asarray(relative_twist, dtype=np.float64) * (1.0 - fraction)
        return V2ControllerCommand(
            mode=mode,
            right_twist_world=right,
            left_twist_world=left,
            right_stiffness_scale=right_stiffness,
            left_stiffness_scale=left_stiffness,
            reason=reason,
        )

    def _retreat_command(
        self,
        estimate: V2AssemblyEstimate,
        reason: str,
    ) -> V2ControllerCommand:
        relative = np.zeros(TWIST_DIM, dtype=np.float64)
        relative[:3] = estimate.hole_rotation_world[:, 2] * self.config.retreat_step_m
        return self._split_command(
            V2Mode.RETREAT,
            relative,
            right_stiffness=np.asarray([0.35, 0.35, 0.55, 0.25, 0.25, 0.4]),
            left_stiffness=np.asarray([0.45, 0.45, 0.65, 0.35, 0.35, 0.5]),
            reason=reason,
        )

    @staticmethod
    def _stationary_command(mode: V2Mode, reason: str) -> V2ControllerCommand:
        return V2ControllerCommand(
            mode=mode,
            right_twist_world=np.zeros(TWIST_DIM),
            left_twist_world=np.zeros(TWIST_DIM),
            right_stiffness_scale=np.zeros(TWIST_DIM),
            left_stiffness_scale=np.zeros(TWIST_DIM),
            reason=reason,
        )

    def _hard_wrench_limit_exceeded(self, contact: V2ContactSignals) -> bool:
        for wrench in (
            contact.right_wrist_wrench_world,
            contact.left_wrist_wrench_world,
        ):
            if np.linalg.norm(wrench[:3]) > self.config.hard_force_limit_n:
                return True
            if np.linalg.norm(wrench[3:]) > self.config.hard_torque_limit_nm:
                return True
        return False

    def _inside_align_funnel(self, state5: np.ndarray) -> bool:
        return bool(
            np.linalg.norm(state5[:2]) <= self.config.align_lateral_threshold_m
            and np.linalg.norm(state5[2:4]) <= self.config.align_tilt_threshold_rad
        )

    def _inside_insert_funnel(self, state5: np.ndarray) -> bool:
        return bool(
            np.linalg.norm(state5[:2]) <= self.config.insert_lateral_threshold_m
            and np.linalg.norm(state5[2:4]) <= self.config.insert_tilt_threshold_rad
        )
