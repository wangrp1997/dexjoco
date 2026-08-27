"""Public-input-only ECHO insertion controller."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from echo_insert.optimizer import EnergyInformationOptimizer, OptimizerConfig
from echo_insert.public_io import (
    PublicObservation,
    apply_right_micro_action,
    apply_right_tip_pivot_action,
    checked_task_basis,
    state46_to_action44,
    wrist_wrench_task,
)


_INTERACTION_CONFIRM_STEPS = 3
_INTERACTION_EXIT_RATIO = 0.5


@dataclass(frozen=True, slots=True)
class EchoConfig:
    baseline_steps: int = 9
    positive_work_window_steps: int = 30
    optimize_action_repeat_steps: int = 6
    interaction_force_n: float = 0.75
    interaction_torque_nm: float = 0.05
    alignment_tolerance_rad: float = 0.05
    alignment_step_rad: float = 0.08
    maximum_alignment_translation_step_m: float = 0.010
    centering_tolerance_m: float = 0.0015
    centering_step_m: float = 0.012
    precontact_approach_step_m: float = 0.008
    precontact_abort_force_n: float = 4.0
    precontact_abort_torque_nm: float = 0.35
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self) -> None:
        if self.baseline_steps <= 0:
            raise ValueError("baseline_steps must be positive")
        if self.optimize_action_repeat_steps <= 0:
            raise ValueError("optimize_action_repeat_steps must be positive")
        if self.positive_work_window_steps <= 0:
            raise ValueError("positive_work_window_steps must be positive")
        thresholds = (
            self.interaction_force_n,
            self.interaction_torque_nm,
            self.alignment_tolerance_rad,
            self.alignment_step_rad,
            self.maximum_alignment_translation_step_m,
            self.centering_tolerance_m,
            self.centering_step_m,
            self.precontact_approach_step_m,
            self.precontact_abort_force_n,
            self.precontact_abort_torque_nm,
        )
        if not np.isfinite(thresholds).all() or min(thresholds) <= 0.0:
            raise ValueError("pre-contact scales and load thresholds must be positive")


@dataclass(frozen=True, slots=True)
class ControllerDiagnostics:
    step_index: int
    status: str
    baseline_remaining: int
    precontact_stage: str
    axis_error_rad: float
    lateral_error_m: float
    selected_candidate: str
    selected_u5: tuple[float, float, float, float, float]
    selected_score: float | None
    wrench5: tuple[float, float, float, float, float]
    wrench_bias6: tuple[float, float, float, float, float, float]
    cumulative_positive_work_j: float
    measured_positive_work_j: float
    tactile_delta: float
    information_gain: float
    rls_updates: int
    search_cells: int
    frontier_cells_remaining: int
    axial_probe_cells: int
    entry_mode: bool
    recovering_interaction: bool
    command_offset5: tuple[float, float, float, float, float]
    safety_reason: str


def _wrench5(wrench6: np.ndarray) -> np.ndarray:
    wrench = np.asarray(wrench6, dtype=np.float64)
    if wrench.shape != (6,):
        raise ValueError("wrench6 must have shape (6,)")
    return wrench[:5].copy()


def _checked_vector3(value: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite with shape (3,)")
    return vector.copy()


def _rotation_step_world(
    source_axis: np.ndarray,
    target_axis: np.ndarray,
    maximum_angle: float,
) -> np.ndarray:
    source = _checked_vector3(source_axis, name="source_axis")
    target = _checked_vector3(target_axis, name="target_axis")
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    angle = float(np.arccos(dot))
    if angle <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    cross = np.cross(source, target)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm <= 1e-12:
        helper = np.eye(3)[int(np.argmin(np.abs(source)))]
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
    else:
        axis = cross / cross_norm
    return axis * min(angle, maximum_angle)


def _relative_motion5(
    previous_state46: np.ndarray,
    state46: np.ndarray,
    basis_world: np.ndarray,
) -> np.ndarray:
    previous = np.asarray(previous_state46, dtype=np.float64)
    current = np.asarray(state46, dtype=np.float64)
    basis = np.asarray(basis_world, dtype=np.float64)
    translation_world = (current[:3] - previous[:3]) - (
        current[7:10] - previous[7:10]
    )
    translation_task = basis.T @ translation_world

    right_previous = Rotation.from_quat(previous[3:7], scalar_first=True)
    right_current = Rotation.from_quat(current[3:7], scalar_first=True)
    left_previous = Rotation.from_quat(previous[10:14], scalar_first=True)
    left_current = Rotation.from_quat(current[10:14], scalar_first=True)
    right_delta = (right_current * right_previous.inv()).as_rotvec()
    left_delta = (left_current * left_previous.inv()).as_rotvec()
    rotation_task = basis.T @ (right_delta - left_delta)
    return np.asarray(
        [
            translation_task[0],
            translation_task[1],
            translation_task[2],
            rotation_task[0],
            rotation_task[1],
        ],
        dtype=np.float64,
    )


class EchoController:
    def __init__(
        self,
        task_basis_world: np.ndarray,
        peg_axis_world: np.ndarray,
        peg_insert_end_world: np.ndarray,
        tray_entry_center_world: np.ndarray,
        config: EchoConfig | None = None,
    ) -> None:
        self.task_basis_world = checked_task_basis(task_basis_world)
        peg_axis = _checked_vector3(peg_axis_world, name="peg_axis_world")
        peg_axis_norm = float(np.linalg.norm(peg_axis))
        if peg_axis_norm <= 1e-12:
            raise ValueError("peg_axis_world must be non-zero")
        self.peg_axis_world = peg_axis / peg_axis_norm
        self.peg_insert_end_world = _checked_vector3(
            peg_insert_end_world,
            name="peg_insert_end_world",
        )
        self.tray_entry_center_world = _checked_vector3(
            tray_entry_center_world,
            name="tray_entry_center_world",
        )
        self._initial_surface_distance_m = float(
            (self.tray_entry_center_world - self.peg_insert_end_world)
            @ self.task_basis_world[:, 2]
        )
        if self._initial_surface_distance_m <= 0.0:
            raise ValueError("peg insert end must start outside the tray plane")
        self.config = config or EchoConfig()
        self.optimizer = EnergyInformationOptimizer(self.config.optimizer)
        self._frozen_action44: np.ndarray | None = None
        self._baseline: list[np.ndarray] = []
        self._wrench_bias6: np.ndarray | None = None
        self._last_state46: np.ndarray | None = None
        self._last_wrench5: np.ndarray | None = None
        self._last_u5 = np.zeros(5, dtype=np.float64)
        self._last_tactile: np.ndarray | None = None
        self._search_origin_state46: np.ndarray | None = None
        self._command_offset5 = np.zeros(5, dtype=np.float64)
        self._positive_work_window: deque[float] = deque(
            maxlen=self.config.positive_work_window_steps
        )
        self._cumulative_positive_work_j = 0.0
        self._interaction_active = False
        self._recovering_interaction = False
        self._repeat_candidate = ""
        self._repeat_steps_remaining = 0
        self._spiral_theta = 0.0
        self._spiral_target_xy = np.zeros(2, dtype=np.float64)
        self._spiral_target_steps_remaining = 0
        self._interaction_confirmations = 0
        self._onset_sample: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._peg_axis_right_tcp: np.ndarray | None = None
        self._peg_tip_right_tcp: np.ndarray | None = None
        self._task_basis_left_tcp: np.ndarray | None = None
        self._tray_entry_left_tcp: np.ndarray | None = None
        self._precontact_stage = "baseline"
        self._alignment_commanded = False
        self._step_index = 0

    def reset(self, observation: PublicObservation) -> None:
        if not isinstance(observation, PublicObservation):
            raise TypeError("observation must be a PublicObservation")
        frozen_action44 = state46_to_action44(observation.state46).copy()
        frozen_action44[6:22] = observation.previous_action44[6:22]
        frozen_action44[28:44] = observation.previous_action44[28:44]
        self._frozen_action44 = frozen_action44
        quaternion = observation.state46[3:7]
        right_rotation = Rotation.from_quat(
            quaternion / np.linalg.norm(quaternion),
            scalar_first=True,
        )
        self._peg_axis_right_tcp = right_rotation.inv().apply(self.peg_axis_world)
        self._peg_tip_right_tcp = right_rotation.inv().apply(
            self.peg_insert_end_world - observation.state46[0:3]
        )
        left_quaternion = observation.state46[10:14]
        left_rotation = Rotation.from_quat(
            left_quaternion / np.linalg.norm(left_quaternion),
            scalar_first=True,
        )
        self._task_basis_left_tcp = (
            left_rotation.inv().as_matrix() @ self.task_basis_world
        )
        self._tray_entry_left_tcp = left_rotation.inv().apply(
            self.tray_entry_center_world - observation.state46[7:10]
        )
        self._baseline.clear()
        self._wrench_bias6 = None
        self._last_state46 = None
        self._last_wrench5 = None
        self._last_u5 = np.zeros(5, dtype=np.float64)
        self._last_tactile = None
        self._search_origin_state46 = None
        self._command_offset5 = np.zeros(5, dtype=np.float64)
        self._positive_work_window.clear()
        self._cumulative_positive_work_j = 0.0
        self._interaction_active = False
        self._recovering_interaction = False
        self._repeat_candidate = ""
        self._repeat_steps_remaining = 0
        self._spiral_theta = 0.0
        self._spiral_target_xy = np.zeros(2, dtype=np.float64)
        self._spiral_target_steps_remaining = 0
        self._interaction_confirmations = 0
        self._onset_sample = None
        self._precontact_stage = "baseline"
        self._alignment_commanded = False
        self._step_index = 0
        self.optimizer.reset()

    def _spiral_micro_action(self, wrench5: np.ndarray) -> np.ndarray:
        config = self.config.optimizer
        u5 = np.zeros(5, dtype=np.float64)
        axial_force = abs(float(wrench5[2]))
        axial_step = 0.5 * config.advance_step_m
        if axial_force > config.axial_probe_force_limit_n:
            u5[2] = -axial_step
            return u5
        if axial_force < config.axial_contact_loss_n:
            u5[2] = axial_step
        elif axial_force < config.axial_preload_n:
            u5[2] = min(
                axial_step,
                config.advance_step_m
                * (config.axial_preload_n - axial_force)
                / config.axial_preload_n,
            )

        torque = np.asarray(wrench5[3:5], dtype=np.float64)
        torque_norm = float(np.linalg.norm(torque))
        if torque_norm > 1e-12:
            angular_step = config.rotation_step_rad * min(
                torque_norm / config.lateral_torque_soft_nm,
                1.0,
            )
            u5[3:] = -angular_step * torque / torque_norm
        current_tilt = self._command_offset5[3:].copy()
        if np.linalg.norm(current_tilt + u5[3:]) > config.maximum_tilt_offset_rad:
            current_tilt_norm = float(np.linalg.norm(current_tilt))
            u5[3:] = (
                -min(config.rotation_step_rad, current_tilt_norm)
                * current_tilt
                / current_tilt_norm
                if current_tilt_norm > 1e-12
                else np.zeros(2, dtype=np.float64)
            )

        step = config.tangent_step_m
        previous_target_xy = self._spiral_target_xy.copy()
        if self._spiral_target_steps_remaining <= 0:
            pitch_scale = config.spiral_pitch_m / (2.0 * np.pi)
            radius = pitch_scale * self._spiral_theta
            self._spiral_theta += min(
                step / max(np.hypot(radius, pitch_scale), 1e-12),
                np.pi / 4.0,
            )
            radius = min(
                pitch_scale * self._spiral_theta,
                config.maximum_tangent_offset_m,
            )
            self._spiral_target_xy[:] = radius * np.asarray(
                [np.cos(self._spiral_theta), np.sin(self._spiral_theta)]
            )
            self._spiral_target_steps_remaining = max(
                1, self.config.optimize_action_repeat_steps // 3
            ) - 1
        else:
            self._spiral_target_steps_remaining -= 1
        delta = self._spiral_target_xy - previous_target_xy
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > step:
            delta *= step / delta_norm
        u5[:2] = delta
        return u5

    def _precontact_geometry(
        self,
        state46: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float]:
        assert self._peg_axis_right_tcp is not None
        assert self._peg_tip_right_tcp is not None
        assert self._task_basis_left_tcp is not None
        assert self._tray_entry_left_tcp is not None
        right_quaternion = state46[3:7]
        right_rotation = Rotation.from_quat(
            right_quaternion / np.linalg.norm(right_quaternion),
            scalar_first=True,
        )
        left_quaternion = state46[10:14]
        left_rotation = Rotation.from_quat(
            left_quaternion / np.linalg.norm(left_quaternion),
            scalar_first=True,
        )
        basis = left_rotation.as_matrix() @ self._task_basis_left_tcp
        tray_entry = state46[7:10] + left_rotation.apply(
            self._tray_entry_left_tcp
        )
        peg_axis = right_rotation.apply(self._peg_axis_right_tcp)
        peg_axis /= np.linalg.norm(peg_axis)
        peg_tip = state46[0:3] + right_rotation.apply(self._peg_tip_right_tcp)
        axis_error = float(
            np.arccos(np.clip(peg_axis @ basis[:, 2], -1.0, 1.0))
        )
        target_delta_task = basis.T @ (tray_entry - peg_tip)
        lateral_error = float(np.linalg.norm(target_delta_task[:2]))
        return basis, peg_axis, peg_tip, axis_error, target_delta_task, lateral_error

    def step(
        self,
        observation: PublicObservation,
    ) -> tuple[np.ndarray, ControllerDiagnostics]:
        if not isinstance(observation, PublicObservation):
            raise TypeError("observation must be a PublicObservation")
        if self._frozen_action44 is None:
            raise RuntimeError("reset must be called before step")

        (
            basis,
            peg_axis,
            peg_tip,
            axis_error,
            target_delta_task,
            lateral_error,
        ) = self._precontact_geometry(observation.state46)
        right_wrench6 = wrist_wrench_task(observation, basis)[0].copy()
        tactile = (
            None
            if observation.fingertip_load is None
            else np.asarray(observation.fingertip_load[0], dtype=np.float64).copy()
        )

        if self._precontact_stage in {"baseline", "aligned_baseline"}:
            baseline_stage = self._precontact_stage
            self._baseline.append(right_wrench6)
            if len(self._baseline) == self.config.baseline_steps:
                self._wrench_bias6 = np.median(
                    np.asarray(self._baseline, dtype=np.float64), axis=0
                )
                self._last_wrench5 = _wrench5(right_wrench6 - self._wrench_bias6)
                self._precontact_stage = (
                    "align" if baseline_stage == "baseline" else "center"
                )
                self._positive_work_window.clear()
                self._cumulative_positive_work_j = 0.0
            current_bias = np.median(np.asarray(self._baseline), axis=0)
            self._last_state46 = observation.state46.copy()
            self._last_tactile = tactile
            remaining = self.config.baseline_steps - len(self._baseline)
            diagnostics = self._diagnostics(
                status=baseline_stage,
                baseline_remaining=remaining,
                precontact_stage=baseline_stage,
                axis_error_rad=axis_error,
                lateral_error_m=lateral_error,
                selected_candidate="hold",
                selected_u5=np.zeros(5),
                selected_score=None,
                wrench5=np.zeros(5),
                wrench_bias6=current_bias,
                measured_positive_work_j=0.0,
                tactile_delta=0.0,
                information_gain=0.0,
                safety_reason="",
            )
            self._step_index += 1
            action44 = apply_right_micro_action(
                observation.previous_action44,
                self._frozen_action44,
                np.zeros(5),
                basis,
            )
            return action44, diagnostics

        assert self._wrench_bias6 is not None
        centered_wrench6 = right_wrench6 - self._wrench_bias6
        wrench5 = _wrench5(centered_wrench6)
        measured_motion5 = np.zeros(5, dtype=np.float64)
        measured_work = 0.0
        causal_sample: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if self._last_state46 is not None and self._last_wrench5 is not None:
            measured_motion5 = _relative_motion5(
                self._last_state46,
                observation.state46,
                basis,
            )
            wrench_delta5 = wrench5 - self._last_wrench5
            causal_sample = (
                self._last_u5.copy(),
                wrench_delta5.copy(),
                measured_motion5.copy(),
            )
            measured_work = max(
                0.0,
                self.config.optimizer.power_sign
                * float(0.5 * (self._last_wrench5 + wrench5) @ measured_motion5),
            )
        if self._search_origin_state46 is not None:
            self._command_offset5[:] = _relative_motion5(
                self._search_origin_state46,
                observation.state46,
                basis,
            )
        self._positive_work_window.append(measured_work)
        self._cumulative_positive_work_j = float(
            sum(self._positive_work_window)
        )

        tactile_delta = 0.0
        if tactile is not None and self._last_tactile is not None:
            tactile_delta = float(np.linalg.norm(tactile - self._last_tactile))

        force_norm = float(np.linalg.norm(centered_wrench6[:3]))
        torque_norm = float(np.linalg.norm(centered_wrench6[3:]))
        hard_load = (
            force_norm > self.config.optimizer.hard_force_n
            or torque_norm > self.config.optimizer.hard_torque_nm
        )
        hard_work = (
            self._cumulative_positive_work_j
            > self.config.optimizer.hard_positive_work_j
        )
        above_entry = (
            force_norm >= self.config.interaction_force_n
            or torque_norm >= self.config.interaction_torque_nm
        )
        below_exit = (
            force_norm
            < _INTERACTION_EXIT_RATIO * self.config.interaction_force_n
            and torque_norm
            < _INTERACTION_EXIT_RATIO * self.config.interaction_torque_nm
        )
        stage_before = self._precontact_stage
        unexpected_precontact_load = (
            stage_before in {"align", "center"}
            and (
                force_norm >= self.config.precontact_abort_force_n
                or torque_norm >= self.config.precontact_abort_torque_nm
            )
        )
        alignment_settled = False
        if (
            stage_before == "align"
            and axis_error <= self.config.alignment_tolerance_rad
        ):
            if self._alignment_commanded:
                self._precontact_stage = "aligned_baseline"
                self._baseline.clear()
                alignment_settled = True
            else:
                self._precontact_stage = "center"
        if (
            self._precontact_stage == "center"
            and lateral_error <= self.config.centering_tolerance_m
        ):
            self._precontact_stage = "approach"

        entered_interaction = False
        confirming_interaction = False
        if self._precontact_stage != "approach":
            self._interaction_active = False
            self._recovering_interaction = False
            self._interaction_confirmations = 0
            self._onset_sample = None
        elif self._interaction_active:
            if below_exit and self._recovering_interaction:
                self._interaction_active = False
                self._repeat_steps_remaining = 0
                self._recovering_interaction = False
                self._interaction_confirmations = 0
                self._onset_sample = None
                self._positive_work_window.clear()
                self._cumulative_positive_work_j = 0.0
                self.optimizer.reset_interaction()
                if axis_error > self.config.alignment_tolerance_rad:
                    self._precontact_stage = "align"
                elif lateral_error > self.config.centering_tolerance_m:
                    self._precontact_stage = "center"
        elif above_entry:
            if self._interaction_confirmations == 0:
                self._onset_sample = causal_sample
            self._interaction_confirmations += 1
            if self._interaction_confirmations >= _INTERACTION_CONFIRM_STEPS:
                self._interaction_active = True
                self._interaction_confirmations = 0
                self.optimizer.reset_interaction()
                if self._onset_sample is not None:
                    self.optimizer.update(*self._onset_sample)
                self._onset_sample = None
                entered_interaction = True
                self._repeat_steps_remaining = 0
                if self._search_origin_state46 is None:
                    self._search_origin_state46 = observation.state46.copy()
            else:
                confirming_interaction = True
        else:
            self._interaction_confirmations = 0
            self._onset_sample = None

        if (
            self._interaction_active
            and not entered_interaction
            and not hard_load
            and not hard_work
            and causal_sample is not None
        ):
            self.optimizer.update(*causal_sample)

        if (
            hard_load
            or hard_work
            or unexpected_precontact_load
            or not self._interaction_active
        ):
            self._repeat_steps_remaining = 0
        selected_score: float | None = None
        information_gain = 0.0
        safety_reason = ""
        action_override: np.ndarray | None = None
        action_base44 = observation.previous_action44
        forced_candidate: str | None = None
        entered_entry = False
        tracked_action44 = state46_to_action44(observation.state46).copy()
        tracked_action44[3:6] = observation.previous_action44[3:6]
        if hard_load:
            unload = max(
                -self.config.optimizer.unload_step_m,
                -self.config.optimizer.maximum_retreat_offset_m
                - self._command_offset5[2],
            )
            u5 = np.asarray([0.0, 0.0, unload, 0.0, 0.0])
            selected_candidate = "safety_unload" if unload < 0.0 else "safety_hold"
            status = "safety"
            safety_reason = "force_or_torque_limit"
        elif hard_work:
            u5 = np.zeros(5, dtype=np.float64)
            selected_candidate = "safety_hold"
            status = "safety"
            safety_reason = "positive_work_limit"
        elif unexpected_precontact_load:
            unload = max(
                -self.config.optimizer.unload_step_m,
                -self.config.optimizer.maximum_retreat_offset_m
                - self._command_offset5[2],
            )
            u5 = np.asarray([0.0, 0.0, unload, 0.0, 0.0])
            selected_candidate = "precontact_unload" if unload < 0.0 else "safety_hold"
            status = "safety"
            safety_reason = "unexpected_precontact_load"
        elif alignment_settled:
            u5 = np.zeros(5, dtype=np.float64)
            selected_candidate = "hold"
            status = "alignment_settle"
        elif self._precontact_stage == "align":
            rotation_step_world = _rotation_step_world(
                peg_axis,
                basis[:, 2],
                self.config.alignment_step_rad,
            )
            candidate_action44 = apply_right_tip_pivot_action(
                observation.state46,
                self._frozen_action44,
                rotation_step_world,
                peg_tip,
                target_pivot_world=self.peg_insert_end_world,
            )
            translation_delta_world = candidate_action44[:3] - observation.state46[:3]
            translation_norm = float(np.linalg.norm(translation_delta_world))
            if translation_norm > self.config.maximum_alignment_translation_step_m:
                candidate_action44 = candidate_action44.copy()
                translation_delta_world *= (
                    self.config.maximum_alignment_translation_step_m
                    / translation_norm
                )
                candidate_action44[:3] = (
                    observation.state46[:3] + translation_delta_world
                )
            u5 = np.concatenate(
                [
                    basis.T @ translation_delta_world,
                    (basis.T @ rotation_step_world)[:2],
                ]
            )
            action_override = candidate_action44
            self._alignment_commanded = True
            selected_candidate = "align_axis"
            status = "align"
        elif self._precontact_stage == "center":
            scale = min(self.config.centering_step_m / lateral_error, 1.0)
            u5 = np.zeros(5, dtype=np.float64)
            u5[:2] = target_delta_task[:2] * scale
            action_base44 = tracked_action44
            selected_candidate = "coarse_center"
            status = "center"
        elif confirming_interaction:
            u5 = np.zeros(5, dtype=np.float64)
            selected_candidate = "hold"
            status = "interaction_confirm"
            action_base44 = tracked_action44
        elif not self._interaction_active:
            measured_advance = (
                self._initial_surface_distance_m - target_delta_task[2]
            )
            remaining = (
                self.config.optimizer.maximum_advance_offset_m
                - measured_advance
            )
            advance = min(
                self.config.precontact_approach_step_m,
                max(0.0, remaining),
            )
            u5 = np.asarray([0.0, 0.0, advance, 0.0, 0.0])
            action_base44 = tracked_action44
            if advance > 0.0:
                selected_candidate = "advance"
                status = "approach"
            else:
                selected_candidate = "safety_hold"
                status = "safety"
                safety_reason = "advance_workspace_limit"
        else:
            action_base44 = tracked_action44
            if (
                self._repeat_candidate == "advance"
                and abs(float(wrench5[2]))
                >= (
                    self.config.optimizer.entry_force_limit_n
                    if self.optimizer.entry_mode
                    else self.config.optimizer.axial_probe_force_limit_n
                )
            ):
                self._repeat_steps_remaining = 0
            forced_candidate = (
                self._repeat_candidate
                if self._repeat_steps_remaining > 0
                and not self._recovering_interaction
                else None
            )
            extra_candidate = None
            entry_mode_before = self.optimizer.entry_mode
            optimizer_forced_name = forced_candidate
            if (
                entry_mode_before
                and abs(float(wrench5[2])) < self.config.optimizer.entry_force_limit_n
            ):
                compliant_u5 = np.zeros(5, dtype=np.float64)
                lateral_force_norm = float(np.linalg.norm(wrench5[:2]))
                if lateral_force_norm > 1e-12:
                    compliant_u5[:2] = (
                        0.25
                        * self.config.optimizer.tangent_step_m
                        * -wrench5[:2]
                        / lateral_force_norm
                    )
                compliant_u5[2] = self.config.optimizer.advance_step_m
                extra_candidate = ("compliant_advance", compliant_u5)
                optimizer_forced_name = "compliant_advance"
            elif entry_mode_before:
                optimizer_forced_name = None
            elif not self._recovering_interaction:
                spiral_name = "spiral"
                extra_candidate = (spiral_name, self._spiral_micro_action(wrench5))
                optimizer_forced_name = spiral_name
            selection = self.optimizer.select(
                wrench5,
                self._last_u5,
                cumulative_positive_work_j=self._cumulative_positive_work_j,
                tactile_delta=tactile_delta,
                command_offset5=self._command_offset5,
                forced_name=optimizer_forced_name,
                extra_candidate=extra_candidate,
            )
            selected = selection.best
            entered_entry = not entry_mode_before and self.optimizer.entry_mode
            if (
                self.optimizer.entry_mode
                and selected.name not in {"compliant_advance", "advance"}
            ):
                advance_candidate = next(
                    (
                        candidate
                        for candidate in selection.candidates
                        if candidate.safe and candidate.name == "advance"
                    ),
                    None,
                )
                if advance_candidate is not None:
                    selected = advance_candidate
            if forced_candidate is not None:
                if selected.name == forced_candidate:
                    self._repeat_steps_remaining -= 1
                else:
                    self._repeat_steps_remaining = 0
            if not self._recovering_interaction and selected.name == "hold":
                self._recovering_interaction = True
                self._repeat_steps_remaining = 0
            if self._recovering_interaction:
                recovery_axes = [0, 1, 3, 4]
                action_scale = self.config.optimizer.action_scale
                current_recovery_norm = float(
                    np.linalg.norm(
                        self._command_offset5[recovery_axes]
                        / action_scale[recovery_axes]
                    )
                )
                recenter_candidate = min(
                    (
                        candidate
                        for candidate in selection.candidates
                        if candidate.safe and candidate.u5[2] == 0.0
                    ),
                    default=None,
                    key=lambda candidate: np.linalg.norm(
                        (
                            self._command_offset5[recovery_axes]
                            + candidate.u5[recovery_axes]
                        )
                        / action_scale[recovery_axes]
                    ),
                )
                unload_candidate = next(
                    candidate
                    for candidate in selection.candidates
                    if candidate.name == "unload"
                )
                recentered_norm = (
                    float(
                        np.linalg.norm(
                            (
                                self._command_offset5[recovery_axes]
                                + recenter_candidate.u5[recovery_axes]
                            )
                            / action_scale[recovery_axes]
                        )
                    )
                    if recenter_candidate is not None
                    else float("inf")
                )
                if (recenter_candidate is not None and recentered_norm < current_recovery_norm):
                    selected = recenter_candidate
                elif unload_candidate.safe:
                    selected = unload_candidate
            if not selected.safe:
                u5 = np.zeros(5, dtype=np.float64)
                selected_candidate = "safety_hold"
                status = "safety"
                safety_reason = "predicted_limit"
            else:
                u5 = selected.u5.copy()
                selected_candidate = selected.name
                status = "optimize"
                selected_score = selected.score
                information_gain = selected.information_gain
                if (
                    forced_candidate is None
                    and not self._recovering_interaction
                    and not self.optimizer.entry_mode
                    and selected.name not in {
                        "spiral", "spiral_probe", "compliant_advance"
                    }
                ):
                    self._repeat_candidate = selected.name
                    self._repeat_steps_remaining = (
                        self.config.optimize_action_repeat_steps - 1
                    )

        if status == "optimize" and entered_entry:
            action_base44 = tracked_action44
        elif status == "optimize" and selected_candidate in {"spiral", "spiral_probe"}:
            action_base44 = observation.previous_action44
        elif status == "optimize" and selected_candidate == "compliant_advance":
            action_base44 = tracked_action44.copy()
            command_error_world = (
                observation.previous_action44[:3] - observation.state46[:3]
            )
            action_base44[:3] += basis[:, 2] * float(
                basis[:, 2] @ command_error_world
            )
        elif (
            status == "optimize"
            and (
                self._recovering_interaction
                or forced_candidate is not None
                or (
                    self.optimizer.entry_mode
                    and selected_candidate == "advance"
                )
            )
            and np.any(u5[:3])
        ):
            action_base44 = observation.previous_action44
        if (
            status == "optimize"
            and np.any(u5[3:])
            and not self._recovering_interaction
            and selected_candidate not in {"spiral", "spiral_probe"}
        ):
            action_base44 = state46_to_action44(observation.state46)
        if action_override is None:
            action44 = apply_right_micro_action(
                action_base44,
                self._frozen_action44,
                u5,
                basis,
            )
        else:
            action44 = action_override
        diagnostics = self._diagnostics(
            status=status,
            baseline_remaining=0,
            precontact_stage=self._precontact_stage,
            axis_error_rad=axis_error,
            lateral_error_m=lateral_error,
            selected_candidate=selected_candidate,
            selected_u5=u5,
            selected_score=selected_score,
            wrench5=wrench5,
            wrench_bias6=self._wrench_bias6,
            measured_positive_work_j=measured_work,
            tactile_delta=tactile_delta,
            information_gain=information_gain,
            safety_reason=safety_reason,
        )
        self._last_state46 = observation.state46.copy()
        self._last_wrench5 = wrench5
        self._last_u5 = u5.copy()
        self._last_tactile = tactile
        self._step_index += 1
        return action44, diagnostics

    def _diagnostics(
        self,
        *,
        status: str,
        baseline_remaining: int,
        precontact_stage: str,
        axis_error_rad: float,
        lateral_error_m: float,
        selected_candidate: str,
        selected_u5: np.ndarray,
        selected_score: float | None,
        wrench5: np.ndarray,
        wrench_bias6: np.ndarray,
        measured_positive_work_j: float,
        tactile_delta: float,
        information_gain: float,
        safety_reason: str,
    ) -> ControllerDiagnostics:
        return ControllerDiagnostics(
            step_index=self._step_index,
            status=status,
            baseline_remaining=int(baseline_remaining),
            precontact_stage=precontact_stage,
            axis_error_rad=float(axis_error_rad),
            lateral_error_m=float(lateral_error_m),
            selected_candidate=selected_candidate,
            selected_u5=tuple(float(value) for value in selected_u5),
            selected_score=(
                None if selected_score is None else float(selected_score)
            ),
            wrench5=tuple(float(value) for value in wrench5),
            wrench_bias6=tuple(float(value) for value in wrench_bias6),
            cumulative_positive_work_j=float(self._cumulative_positive_work_j),
            measured_positive_work_j=float(measured_positive_work_j),
            tactile_delta=float(tactile_delta),
            information_gain=float(information_gain),
            rls_updates=int(self.optimizer.model.updates),
            search_cells=self.optimizer.search_cells,
            frontier_cells_remaining=self.optimizer.frontier_cells_remaining,
            axial_probe_cells=self.optimizer.axial_probe_cells,
            entry_mode=self.optimizer.entry_mode,
            recovering_interaction=bool(self._recovering_interaction),
            command_offset5=tuple(float(value) for value in self._command_offset5),
            safety_reason=safety_reason,
        )
