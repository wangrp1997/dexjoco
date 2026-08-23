"""Retrieval-conditioned chance-constrained SQP for bimanual insertion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


ASSEMBLY_STATE_DIM = 5
LOCAL_CONTROL_DIM = 6
GRASP_WRENCH_DIM = 6


def _vector(value: np.ndarray, size: int, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _matrix(value: np.ndarray, shape: tuple[int, int], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _positive_semidefinite(value: np.ndarray, size: int, *, name: str) -> np.ndarray:
    matrix = _matrix(value, (size, size), name=name)
    if not np.allclose(matrix, matrix.T, atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    if float(np.linalg.eigvalsh(matrix).min()) < -1e-9:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


@dataclass(frozen=True)
class BimanualInsertionBelief:
    """Five-dimensional assembly belief with bilateral 6D attachment noise."""

    mean: np.ndarray
    covariance: np.ndarray
    attachment_process_covariance: np.ndarray
    attachment_to_state: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mean",
            _vector(self.mean, ASSEMBLY_STATE_DIM, name="mean"),
        )
        object.__setattr__(
            self,
            "covariance",
            _positive_semidefinite(
                self.covariance,
                ASSEMBLY_STATE_DIM,
                name="covariance",
            ),
        )

    @classmethod
    def from_visual_estimate(
        cls,
        estimate: object,
        *,
        attachment_process_covariance: np.ndarray,
        attachment_to_state: np.ndarray,
    ) -> BimanualInsertionBelief:
        """Build the current assembly belief from a deployable visual estimate."""
        return cls(
            mean=np.asarray(estimate.mean5, dtype=np.float64),
            covariance=np.asarray(estimate.covariance5, dtype=np.float64),
            attachment_process_covariance=attachment_process_covariance,
            attachment_to_state=attachment_to_state,
        )
        object.__setattr__(
            self,
            "attachment_process_covariance",
            _positive_semidefinite(
                self.attachment_process_covariance,
                2 * LOCAL_CONTROL_DIM,
                name="attachment_process_covariance",
            ),
        )
        object.__setattr__(
            self,
            "attachment_to_state",
            _matrix(
                self.attachment_to_state,
                (ASSEMBLY_STATE_DIM, 2 * LOCAL_CONTROL_DIM),
                name="attachment_to_state",
            ),
        )

    def state_process_covariance(self) -> np.ndarray:
        return (
            self.attachment_to_state
            @ self.attachment_process_covariance
            @ self.attachment_to_state.T
        )

    def propagated_covariances(self, horizon: int) -> np.ndarray:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        process = self.state_process_covariance()
        return np.asarray(
            [self.covariance + (step + 1) * process for step in range(horizon)],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class LinearizedVisualObservationModel:
    """Local visual posterior model with optional control-dependent reliability."""

    observation_covariance: np.ndarray
    reliability: float
    state_jacobian: np.ndarray = field(
        default_factory=lambda: np.eye(ASSEMBLY_STATE_DIM, dtype=np.float64)
    )
    control_reliability_jacobian: np.ndarray = field(
        default_factory=lambda: np.zeros(2 * LOCAL_CONTROL_DIM, dtype=np.float64)
    )
    reference_control: np.ndarray = field(
        default_factory=lambda: np.zeros(2 * LOCAL_CONTROL_DIM, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        covariance = _positive_semidefinite(
            self.observation_covariance,
            ASSEMBLY_STATE_DIM,
            name="observation_covariance",
        )
        if float(np.linalg.eigvalsh(covariance).min()) <= 0.0:
            raise ValueError("observation_covariance must be positive definite")
        reliability = float(self.reliability)
        if not 0.0 <= reliability <= 1.0:
            raise ValueError("reliability must be in [0, 1]")
        object.__setattr__(self, "observation_covariance", covariance)
        object.__setattr__(self, "reliability", reliability)
        object.__setattr__(
            self,
            "state_jacobian",
            _matrix(
                self.state_jacobian,
                (ASSEMBLY_STATE_DIM, ASSEMBLY_STATE_DIM),
                name="state_jacobian",
            ),
        )
        object.__setattr__(
            self,
            "control_reliability_jacobian",
            _vector(
                self.control_reliability_jacobian,
                2 * LOCAL_CONTROL_DIM,
                name="control_reliability_jacobian",
            ),
        )
        object.__setattr__(
            self,
            "reference_control",
            _vector(
                self.reference_control,
                2 * LOCAL_CONTROL_DIM,
                name="reference_control",
            ),
        )

    @classmethod
    def from_visual_estimate(
        cls,
        estimate: object,
        *,
        control_reliability_jacobian: np.ndarray | None = None,
        reference_control: np.ndarray | None = None,
    ) -> LinearizedVisualObservationModel:
        """Build from an object exposing covariance5 and visual_reliability."""
        return cls(
            observation_covariance=np.asarray(estimate.covariance5, dtype=np.float64),
            reliability=float(estimate.visual_reliability),
            control_reliability_jacobian=(
                np.zeros(2 * LOCAL_CONTROL_DIM, dtype=np.float64)
                if control_reliability_jacobian is None
                else control_reliability_jacobian
            ),
            reference_control=(
                np.zeros(2 * LOCAL_CONTROL_DIM, dtype=np.float64)
                if reference_control is None
                else reference_control
            ),
        )

    def predicted_reliability(
        self,
        right_control: np.ndarray,
        left_control: np.ndarray,
    ) -> float:
        control = np.concatenate(
            [
                _vector(right_control, LOCAL_CONTROL_DIM, name="right_control"),
                _vector(left_control, LOCAL_CONTROL_DIM, name="left_control"),
            ]
        )
        reliability = self.reliability + float(
            self.control_reliability_jacobian @ (control - self.reference_control)
        )
        return float(np.clip(reliability, 0.0, 1.0))

    def posterior_covariance(
        self,
        predicted_covariance: np.ndarray,
        right_control: np.ndarray,
        left_control: np.ndarray,
    ) -> np.ndarray:
        predicted = _positive_semidefinite(
            predicted_covariance,
            ASSEMBLY_STATE_DIM,
            name="predicted_covariance",
        )
        if float(np.linalg.eigvalsh(predicted).min()) <= 0.0:
            raise ValueError(
                "predicted_covariance must be positive definite for visual update"
            )
        reliability = self.predicted_reliability(right_control, left_control)
        if reliability <= 0.0:
            return predicted
        information = np.linalg.inv(predicted)
        information += reliability * (
            self.state_jacobian.T
            @ np.linalg.solve(self.observation_covariance, self.state_jacobian)
        )
        posterior = np.linalg.inv(information)
        return 0.5 * (posterior + posterior.T)


@dataclass(frozen=True)
class LocalBimanualInsertionModel:
    """Local maps from bilateral keypoint corrections to assembly state and wrench."""

    right_state_jacobian: np.ndarray
    left_state_jacobian: np.ndarray
    right_wrench_jacobian: np.ndarray
    left_wrench_jacobian: np.ndarray
    state_drift: np.ndarray = field(
        default_factory=lambda: np.zeros(ASSEMBLY_STATE_DIM, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "right_state_jacobian",
            _matrix(
                self.right_state_jacobian,
                (ASSEMBLY_STATE_DIM, LOCAL_CONTROL_DIM),
                name="right_state_jacobian",
            ),
        )
        object.__setattr__(
            self,
            "left_state_jacobian",
            _matrix(
                self.left_state_jacobian,
                (ASSEMBLY_STATE_DIM, LOCAL_CONTROL_DIM),
                name="left_state_jacobian",
            ),
        )
        object.__setattr__(
            self,
            "right_wrench_jacobian",
            _matrix(
                self.right_wrench_jacobian,
                (GRASP_WRENCH_DIM, LOCAL_CONTROL_DIM),
                name="right_wrench_jacobian",
            ),
        )
        object.__setattr__(
            self,
            "left_wrench_jacobian",
            _matrix(
                self.left_wrench_jacobian,
                (GRASP_WRENCH_DIM, LOCAL_CONTROL_DIM),
                name="left_wrench_jacobian",
            ),
        )
        object.__setattr__(
            self,
            "state_drift",
            _vector(self.state_drift, ASSEMBLY_STATE_DIM, name="state_drift"),
        )


@dataclass(frozen=True)
class InsertionGeometry:
    radial_clearance_m: float
    target_depth_m: float
    terminal_depth_m: float | None = None

    def __post_init__(self) -> None:
        if self.radial_clearance_m <= 0.0:
            raise ValueError("radial_clearance_m must be positive")
        if self.target_depth_m <= 0.0:
            raise ValueError("target_depth_m must be positive")
        if (
            self.terminal_depth_m is not None
            and self.terminal_depth_m > self.target_depth_m
        ):
            raise ValueError("terminal_depth_m cannot exceed target_depth_m")

    @property
    def planning_terminal_depth_m(self) -> float:
        if self.terminal_depth_m is None:
            return self.target_depth_m
        return self.terminal_depth_m


@dataclass(frozen=True)
class RetrievedInsertionCandidate:
    """One retrieved contact-mode prior for the continuous SQP subproblem."""

    skill_id: str
    retrieval_distance: float
    nominal_states: np.ndarray
    nominal_actions44: np.ndarray
    nominal_right_controls: np.ndarray
    nominal_left_controls: np.ndarray
    nominal_right_wrenches: np.ndarray
    nominal_left_wrenches: np.ndarray
    right_wrench_capacity: np.ndarray
    left_wrench_capacity: np.ndarray
    right_capacity_std: np.ndarray
    left_capacity_std: np.ndarray
    mean_state_disturbance: np.ndarray
    contact_modes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.skill_id.strip():
            raise ValueError("skill_id must be non-empty")
        if self.retrieval_distance < 0.0:
            raise ValueError("retrieval_distance must be non-negative")
        states = np.asarray(self.nominal_states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != ASSEMBLY_STATE_DIM:
            raise ValueError(
                "nominal_states must have shape (H + 1, 5), "
                f"got {states.shape}"
            )
        horizon = states.shape[0] - 1
        if horizon <= 0:
            raise ValueError("candidate horizon must be positive")
        arrays = {
            "nominal_right_controls": (
                self.nominal_right_controls,
                (horizon, LOCAL_CONTROL_DIM),
            ),
            "nominal_left_controls": (
                self.nominal_left_controls,
                (horizon, LOCAL_CONTROL_DIM),
            ),
            "nominal_right_wrenches": (
                self.nominal_right_wrenches,
                (horizon, GRASP_WRENCH_DIM),
            ),
            "nominal_left_wrenches": (
                self.nominal_left_wrenches,
                (horizon, GRASP_WRENCH_DIM),
            ),
            "mean_state_disturbance": (
                self.mean_state_disturbance,
                (horizon, ASSEMBLY_STATE_DIM),
            ),
        }
        object.__setattr__(self, "nominal_states", states.copy())
        actions = np.asarray(self.nominal_actions44, dtype=np.float64)
        if actions.shape != (horizon + 1, 44):
            raise ValueError(
                "nominal_actions44 must have shape "
                f"({horizon + 1}, 44), got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("nominal_actions44 must contain finite values")
        object.__setattr__(self, "nominal_actions44", actions.copy())
        for name, (value, shape) in arrays.items():
            object.__setattr__(self, name, _matrix(value, shape, name=name))
        for name in (
            "right_wrench_capacity",
            "left_wrench_capacity",
            "right_capacity_std",
            "left_capacity_std",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if values.shape != (horizon,):
                raise ValueError(f"{name} must have shape ({horizon},)")
            if not np.isfinite(values).all() or np.any(values < 0.0):
                raise ValueError(f"{name} must contain finite non-negative values")
            object.__setattr__(self, name, values.copy())
        if len(self.contact_modes) != horizon:
            raise ValueError(f"contact_modes must contain {horizon} entries")
        object.__setattr__(self, "contact_modes", tuple(self.contact_modes))

    @property
    def horizon(self) -> int:
        return self.nominal_states.shape[0] - 1


@dataclass(frozen=True)
class BeliefSpaceSQPConfig:
    confidence_multiplier: float = 2.326347874
    terminal_depth_tolerance_m: float = 5e-4
    max_correction: tuple[float, ...] = (0.004, 0.004, 0.004, 0.04, 0.04, 0.04)
    state_scale: tuple[float, ...] = (0.001, 0.001, 0.02, 0.02, 0.005)
    prior_state_weight: float = 1.0
    correction_weight: float = 0.2
    smoothness_weight: float = 0.1
    terminal_weight: float = 20.0
    retrieval_distance_weight: float = 0.01
    covariance_weight: float = 0.0
    covariance_state_weight: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    feasibility_tolerance: float = 1e-6
    max_iterations: int = 300
    ftol: float = 1e-9
    preentry_clearance_multiplier: float = 2.0
    maximum_preentry_tilt_rad: float = 0.12
    max_total_control: tuple[float, ...] | None = None
    use_candidate_transition_residual: bool = True

    def __post_init__(self) -> None:
        if self.confidence_multiplier < 0.0:
            raise ValueError("confidence_multiplier must be non-negative")
        if self.terminal_depth_tolerance_m < 0.0:
            raise ValueError("terminal_depth_tolerance_m must be non-negative")
        if len(self.max_correction) != LOCAL_CONTROL_DIM:
            raise ValueError("max_correction must contain six values")
        if len(self.state_scale) != ASSEMBLY_STATE_DIM:
            raise ValueError("state_scale must contain five values")
        if len(self.covariance_state_weight) != ASSEMBLY_STATE_DIM:
            raise ValueError("covariance_state_weight must contain five values")
        if min(self.max_correction) <= 0.0 or min(self.state_scale) <= 0.0:
            raise ValueError("max_correction and state_scale must be positive")
        if min(self.covariance_state_weight) < 0.0:
            raise ValueError("covariance_state_weight must be non-negative")
        if min(
            self.prior_state_weight,
            self.correction_weight,
            self.smoothness_weight,
            self.terminal_weight,
            self.retrieval_distance_weight,
            self.covariance_weight,
        ) < 0.0:
            raise ValueError("objective weights must be non-negative")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.ftol <= 0.0:
            raise ValueError("ftol must be positive")
        if self.preentry_clearance_multiplier < 1.0:
            raise ValueError("preentry_clearance_multiplier must be at least one")
        if self.maximum_preentry_tilt_rad <= 0.0:
            raise ValueError("maximum_preentry_tilt_rad must be positive")
        if self.max_total_control is not None:
            if len(self.max_total_control) != LOCAL_CONTROL_DIM:
                raise ValueError("max_total_control must contain six values")
            if min(self.max_total_control) <= 0.0:
                raise ValueError("max_total_control must be positive")


@dataclass(frozen=True)
class BeliefSpaceSQPResult:
    skill_id: str
    success: bool
    feasible: bool
    message: str
    objective: float
    min_constraint_margin: float
    iterations: int
    states: np.ndarray
    right_controls: np.ndarray
    left_controls: np.ndarray
    right_corrections: np.ndarray
    left_corrections: np.ndarray
    covariances: np.ndarray


class RetrievalConditionedBeliefSpaceSQP:
    """Solve and rank mode-fixed chance-constrained insertion subproblems."""

    def __init__(
        self,
        model: LocalBimanualInsertionModel,
        geometry: InsertionGeometry,
        config: BeliefSpaceSQPConfig | None = None,
        visual_observation_model: LinearizedVisualObservationModel | None = None,
    ) -> None:
        self.model = model
        self.geometry = geometry
        self.config = config or BeliefSpaceSQPConfig()
        self.visual_observation_model = visual_observation_model

    def solve_candidate(
        self,
        belief: BimanualInsertionBelief,
        candidate: RetrievedInsertionCandidate,
    ) -> BeliefSpaceSQPResult:
        horizon = candidate.horizon
        process_covariance = belief.state_process_covariance()
        bounds = []
        for _ in range(2 * horizon):
            bounds.extend(
                [(-limit, limit) for limit in self.config.max_correction]
            )

        def unpack(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            corrections = np.asarray(vector, dtype=np.float64).reshape(
                2, horizon, LOCAL_CONTROL_DIM
            )
            return corrections[0], corrections[1]

        terminal_target = np.zeros(ASSEMBLY_STATE_DIM, dtype=np.float64)
        terminal_target[4] = self.geometry.planning_terminal_depth_m
        phase = np.linspace(0.0, 1.0, horizon + 1)[:, None]
        start_offset = belief.mean - candidate.nominal_states[0]
        terminal_offset = terminal_target - candidate.nominal_states[-1]
        reference_states = (
            candidate.nominal_states
            + (1.0 - phase) * start_offset
            + phase * terminal_offset
        )
        combined_jacobian = np.concatenate(
            [
                self.model.right_state_jacobian,
                self.model.left_state_jacobian,
            ],
            axis=1,
        )
        inverse_combined = np.linalg.pinv(combined_jacobian)
        if self.config.use_candidate_transition_residual:
            nominal_state_disturbance = (
                np.diff(candidate.nominal_states, axis=0)
                - candidate.nominal_right_controls @ self.model.right_state_jacobian.T
                - candidate.nominal_left_controls @ self.model.left_state_jacobian.T
                + candidate.mean_state_disturbance
            )
        else:
            nominal_state_disturbance = candidate.mean_state_disturbance.copy()
        correction_limits = np.asarray(self.config.max_correction, dtype=np.float64)
        initial = np.zeros((2, horizon, LOCAL_CONTROL_DIM), dtype=np.float64)
        warm_state = belief.mean.copy()
        for step in range(horizon):
            nominal_effect = (
                self.model.state_drift
                + self.model.right_state_jacobian
                @ candidate.nominal_right_controls[step]
                + self.model.left_state_jacobian
                @ candidate.nominal_left_controls[step]
                + nominal_state_disturbance[step]
            )
            residual = reference_states[step + 1] - warm_state - nominal_effect
            correction = inverse_combined @ residual
            initial[0, step] = np.clip(
                correction[:LOCAL_CONTROL_DIM],
                -correction_limits,
                correction_limits,
            )
            initial[1, step] = np.clip(
                correction[LOCAL_CONTROL_DIM:],
                -correction_limits,
                correction_limits,
            )
            warm_state = (
                warm_state
                + nominal_effect
                + self.model.right_state_jacobian @ initial[0, step]
                + self.model.left_state_jacobian @ initial[1, step]
            )

        def rollout(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            right_correction, left_correction = unpack(vector)
            states = np.empty((horizon + 1, ASSEMBLY_STATE_DIM), dtype=np.float64)
            states[0] = belief.mean
            for step in range(horizon):
                states[step + 1] = (
                    states[step]
                    + self.model.state_drift
                    + self.model.right_state_jacobian
                    @ (
                        candidate.nominal_right_controls[step]
                        + right_correction[step]
                    )
                    + self.model.left_state_jacobian
                    @ (
                        candidate.nominal_left_controls[step]
                        + left_correction[step]
                    )
                    + nominal_state_disturbance[step]
                )
            return states, right_correction, left_correction

        def covariance_rollout(
            right_correction: np.ndarray,
            left_correction: np.ndarray,
        ) -> np.ndarray:
            covariances = np.empty(
                (horizon, ASSEMBLY_STATE_DIM, ASSEMBLY_STATE_DIM),
                dtype=np.float64,
            )
            covariance = belief.covariance.copy()
            for step in range(horizon):
                covariance = covariance + process_covariance
                if self.visual_observation_model is not None:
                    covariance = self.visual_observation_model.posterior_covariance(
                        covariance,
                        candidate.nominal_right_controls[step]
                        + right_correction[step],
                        candidate.nominal_left_controls[step]
                        + left_correction[step],
                    )
                covariances[step] = covariance
            return covariances

        state_scale = np.asarray(self.config.state_scale, dtype=np.float64)
        covariance_state_weight = np.asarray(
            self.config.covariance_state_weight,
            dtype=np.float64,
        )

        def objective(vector: np.ndarray) -> float:
            states, right_correction, left_correction = rollout(vector)
            covariances = covariance_rollout(right_correction, left_correction)
            state_error = (states[1:] - reference_states[1:]) / state_scale
            terminal_error = (states[-1] - terminal_target) / state_scale
            correction_cost = np.sum(right_correction**2) + np.sum(left_correction**2)
            smoothness_cost = 0.0
            if horizon > 1:
                smoothness_cost = np.sum(np.diff(right_correction, axis=0) ** 2)
                smoothness_cost += np.sum(np.diff(left_correction, axis=0) ** 2)
            return float(
                self.config.prior_state_weight * np.sum(state_error**2)
                + self.config.correction_weight * correction_cost
                + self.config.smoothness_weight * smoothness_cost
                + self.config.terminal_weight * np.sum(terminal_error**2)
                + self.config.retrieval_distance_weight
                * candidate.retrieval_distance**2
                + self.config.covariance_weight
                * np.sum(
                    np.einsum(
                        "i,tii->t",
                        covariance_state_weight,
                        covariances,
                    )
                )
            )

        def constraints(vector: np.ndarray) -> np.ndarray:
            states, right_correction, left_correction = rollout(vector)
            covariances = covariance_rollout(right_correction, left_correction)
            margins: list[float] = []
            previous_depth = float(states[0, 4])
            for step, (state, covariance) in enumerate(
                zip(states[1:], covariances, strict=True)
            ):
                lateral = float(np.linalg.norm(state[:2]))
                tilt = float(np.linalg.norm(state[2:4]))
                depth = float(state[4])
                embedded_depth = float(
                    np.clip(depth, 0.0, self.geometry.target_depth_m)
                )
                lateral_sigma = float(
                    np.sqrt(max(0.0, np.linalg.eigvalsh(covariance[:2, :2]).max()))
                )
                tilt_sigma = float(
                    np.sqrt(max(0.0, np.linalg.eigvalsh(covariance[2:4, 2:4]).max()))
                )
                if depth >= 0.0:
                    robust_radius = (
                        lateral
                        + embedded_depth * tilt
                        + self.config.confidence_multiplier
                        * (lateral_sigma + embedded_depth * tilt_sigma)
                    )
                    margins.append(self.geometry.radial_clearance_m - robust_radius)
                    margins.append(
                        self.config.maximum_preentry_tilt_rad
                        - (
                            tilt
                            + self.config.confidence_multiplier * tilt_sigma
                        )
                    )
                else:
                    robust_lateral = (
                        lateral
                        + self.config.confidence_multiplier * lateral_sigma
                    )
                    robust_tilt = tilt + self.config.confidence_multiplier * tilt_sigma
                    margins.append(
                        self.config.preentry_clearance_multiplier
                        * self.geometry.radial_clearance_m
                        - robust_lateral
                    )
                    margins.append(
                        self.config.maximum_preentry_tilt_rad - robust_tilt
                    )
                margins.append(depth - float(states[0, 4]))
                margins.append(self.geometry.target_depth_m - depth)
                margins.append(depth - previous_depth)
                previous_depth = depth

                right_wrench = (
                    candidate.nominal_right_wrenches[step]
                    + self.model.right_wrench_jacobian @ right_correction[step]
                )
                left_wrench = (
                    candidate.nominal_left_wrenches[step]
                    + self.model.left_wrench_jacobian @ left_correction[step]
                )
                right_capacity = (
                    candidate.right_wrench_capacity[step]
                    - self.config.confidence_multiplier
                    * candidate.right_capacity_std[step]
                )
                left_capacity = (
                    candidate.left_wrench_capacity[step]
                    - self.config.confidence_multiplier
                    * candidate.left_capacity_std[step]
                )
                margins.append(right_capacity - float(np.linalg.norm(right_wrench)))
                margins.append(left_capacity - float(np.linalg.norm(left_wrench)))
                if self.config.max_total_control is not None:
                    total_limit = np.asarray(
                        self.config.max_total_control, dtype=np.float64
                    )
                    margins.extend(
                        (
                            total_limit
                            - np.abs(
                                candidate.nominal_right_controls[step]
                                + right_correction[step]
                            )
                        ).tolist()
                    )
                    margins.extend(
                        (
                            total_limit
                            - np.abs(
                                candidate.nominal_left_controls[step]
                                + left_correction[step]
                            )
                        ).tolist()
                    )

            terminal_minimum = (
                self.geometry.planning_terminal_depth_m
                - self.config.terminal_depth_tolerance_m
            )
            margins.append(float(states[-1, 4]) - terminal_minimum)
            return np.asarray(margins, dtype=np.float64)

        optimization = minimize(
            objective,
            initial.reshape(-1),
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": constraints},
            options={
                "maxiter": self.config.max_iterations,
                "ftol": self.config.ftol,
                "disp": False,
            },
        )
        states, right_correction, left_correction = rollout(optimization.x)
        covariances = covariance_rollout(right_correction, left_correction)
        margins = constraints(optimization.x)
        min_margin = float(margins.min())
        feasible = bool(min_margin >= -self.config.feasibility_tolerance)
        return BeliefSpaceSQPResult(
            skill_id=candidate.skill_id,
            success=bool(optimization.success and feasible),
            feasible=feasible,
            message=str(optimization.message),
            objective=float(objective(optimization.x)),
            min_constraint_margin=min_margin,
            iterations=int(optimization.nit),
            states=states.astype(np.float32),
            right_controls=(
                candidate.nominal_right_controls + right_correction
            ).astype(np.float32),
            left_controls=(
                candidate.nominal_left_controls + left_correction
            ).astype(np.float32),
            right_corrections=right_correction.astype(np.float32),
            left_corrections=left_correction.astype(np.float32),
            covariances=covariances.astype(np.float32),
        )

    def solve(
        self,
        belief: BimanualInsertionBelief,
        candidates: Sequence[RetrievedInsertionCandidate],
    ) -> tuple[BeliefSpaceSQPResult, tuple[BeliefSpaceSQPResult, ...]]:
        if not candidates:
            raise ValueError("at least one retrieved candidate is required")
        results = tuple(self.solve_candidate(belief, item) for item in candidates)
        feasible = [item for item in results if item.feasible]
        if not feasible:
            best = max(results, key=lambda item: item.min_constraint_margin)
            raise RuntimeError(
                "no retrieved contact mode produced a feasible SQP solution; "
                f"best={best.skill_id} margin={best.min_constraint_margin:.6g}"
            )
        selected = min(feasible, key=lambda item: (item.objective, item.skill_id))
        return selected, results
