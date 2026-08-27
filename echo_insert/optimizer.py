"""Online energy-information optimizer for five-dimensional micro-actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    tangent_step_m: float = 2e-4
    spiral_pitch_m: float = 1e-3
    advance_step_m: float = 2e-4
    rotation_step_rad: float = 5e-4
    unload_step_m: float = 1.0e-3
    maximum_tangent_offset_m: float = 12e-3
    maximum_advance_offset_m: float = 4e-2
    maximum_retreat_offset_m: float = 1e-2
    maximum_tilt_offset_rad: float = 8e-2
    wrench_scale: tuple[float, float, float, float, float] = (
        8.0,
        8.0,
        12.0,
        0.8,
        0.8,
    )
    initial_covariance: float = 25.0
    forgetting: float = 0.995
    covariance_floor: float = 1e-9
    lateral_force_soft_n: float = 4.0
    lateral_torque_soft_nm: float = 0.35
    positive_work_soft_j: float = 3e-3
    axial_preload_n: float = 1.5
    axial_probe_force_limit_n: float = 3.0
    entry_force_limit_n: float = 8.0
    entry_progress_m: float = 2e-3
    entry_stall_steps: int = 100
    hard_force_n: float = 18.0
    hard_torque_nm: float = 1.5
    hard_positive_work_j: float = 4e-2
    power_sign: float = -1.0
    axial_progress_weight: float = 1.0
    entry_progress_weight: float = 0.5
    lateral_weight: float = 2.0
    positive_work_weight: float = 1.0
    effort_weight: float = 0.02
    slew_weight: float = 0.02
    tactile_weight: float = 0.15
    information_weight: float = 0.05
    frontier_weight: float = 0.25
    spatial_energy_weight: float = 0.25
    search_novelty_weight: float = 0.045
    axial_preload_weight: float = 1.0
    axial_probe_weight: float = 0.25
    tilt_offset_weight: float = 0.25
    revisit_weight: float = 0.02

    def __post_init__(self) -> None:
        positive = (
            self.tangent_step_m,
            self.spiral_pitch_m,
            self.advance_step_m,
            self.rotation_step_rad,
            self.unload_step_m,
            self.maximum_tangent_offset_m,
            self.maximum_advance_offset_m,
            self.maximum_retreat_offset_m,
            self.maximum_tilt_offset_rad,
            *self.wrench_scale,
            self.initial_covariance,
            self.forgetting,
            self.covariance_floor,
            self.lateral_force_soft_n,
            self.lateral_torque_soft_nm,
            self.positive_work_soft_j,
            self.axial_preload_n,
            self.axial_probe_force_limit_n,
            self.entry_force_limit_n,
            self.entry_progress_m,
            self.hard_force_n,
            self.hard_torque_nm,
            self.hard_positive_work_j,
        )
        if not np.isfinite(positive).all() or min(positive) <= 0.0:
            raise ValueError("optimizer scales and limits must be finite and positive")
        if self.entry_stall_steps <= 0:
            raise ValueError("entry stall steps must be positive")
        if self.entry_force_limit_n < self.axial_probe_force_limit_n:
            raise ValueError("entry force limit must cover the probe force limit")
        if self.forgetting > 1.0:
            raise ValueError("forgetting must be in (0, 1]")
        if self.power_sign not in (-1.0, 1.0):
            raise ValueError("power_sign must be -1 or +1")
        weights = (
            self.axial_progress_weight,
            self.entry_progress_weight,
            self.lateral_weight,
            self.positive_work_weight,
            self.effort_weight,
            self.slew_weight,
            self.tactile_weight,
            self.information_weight,
            self.frontier_weight,
            self.spatial_energy_weight,
            self.search_novelty_weight,
            self.axial_preload_weight,
            self.axial_probe_weight,
            self.tilt_offset_weight,
            self.revisit_weight,
        )
        if not np.isfinite(weights).all() or min(weights) < 0.0:
            raise ValueError("optimizer weights must be finite and non-negative")

    @property
    def axial_contact_loss_n(self) -> float:
        return 0.75 * self.axial_preload_n

    @property
    def action_scale(self) -> np.ndarray:
        return np.asarray(
            [
                self.tangent_step_m,
                self.tangent_step_m,
                self.advance_step_m,
                self.rotation_step_rad,
                self.rotation_step_rad,
            ],
            dtype=np.float64,
        )

    @property
    def wrench_scale_array(self) -> np.ndarray:
        return np.asarray(self.wrench_scale, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class InteractionPrediction:
    wrench_delta5: np.ndarray
    motion5: np.ndarray
    information_gain: float


class MultiOutputRidgeRLS:
    """Shared-covariance RLS for wrench delta and measured relative motion."""

    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self.config = config or OptimizerConfig()
        self.reset()

    def reset(self) -> None:
        self.weights = np.zeros((10, 5), dtype=np.float64)
        self.weights[5:, :] = np.eye(5, dtype=np.float64)
        self.covariance = (
            np.eye(5, dtype=np.float64) * self.config.initial_covariance
        )
        self.updates = 0

    def _phi(self, u5: np.ndarray) -> np.ndarray:
        u = np.asarray(u5, dtype=np.float64)
        if u.shape != (5,) or not np.isfinite(u).all():
            raise ValueError("u5 must be a finite vector with shape (5,)")
        return u / self.config.action_scale


    def update(
        self,
        u5: np.ndarray,
        wrench_delta5: np.ndarray,
        motion5: np.ndarray,
    ) -> bool:
        phi = self._phi(u5)
        if float(phi @ phi) <= 1e-12:
            return False
        wrench_delta = np.asarray(wrench_delta5, dtype=np.float64)
        motion = np.asarray(motion5, dtype=np.float64)
        if wrench_delta.shape != (5,) or motion.shape != (5,):
            raise ValueError("RLS outputs must have shape (5,)")
        if not np.isfinite(wrench_delta).all() or not np.isfinite(motion).all():
            raise ValueError("RLS outputs must be finite")
        target = np.concatenate(
            [
                wrench_delta / self.config.wrench_scale_array,
                motion / self.config.action_scale,
            ]
        )
        projected = self.covariance @ phi
        denominator = self.config.forgetting + float(phi @ projected)
        gain = projected / max(denominator, self.config.covariance_floor)
        residual = target - self.weights @ phi
        self.weights += np.outer(residual, gain)
        self.covariance = (
            self.covariance - np.outer(gain, phi) @ self.covariance
        ) / self.config.forgetting
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.covariance += np.eye(5) * self.config.covariance_floor
        if not np.isfinite(self.weights).all() or not np.isfinite(self.covariance).all():
            raise FloatingPointError("non-finite RLS state")
        self.updates += 1
        return True

    def predict(self, u5: np.ndarray) -> InteractionPrediction:
        phi = self._phi(u5)
        output = self.weights @ phi
        variance = max(float(phi @ self.covariance @ phi), 0.0)
        return InteractionPrediction(
            wrench_delta5=(output[:5] * self.config.wrench_scale_array).copy(),
            motion5=(output[5:] * self.config.action_scale).copy(),
            information_gain=float(np.log1p(variance)),
        )


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    name: str
    u5: np.ndarray
    score: float
    safe: bool
    predicted_wrench5: np.ndarray
    predicted_motion5: np.ndarray
    predicted_positive_work_j: float
    information_gain: float


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    best: CandidateEvaluation
    candidates: tuple[CandidateEvaluation, ...]


def candidate_micro_actions(config: OptimizerConfig) -> tuple[tuple[str, np.ndarray], ...]:
    tx = config.tangent_step_m
    tz = config.advance_step_m
    rr = config.rotation_step_rad
    return (
        ("hold", np.zeros(5, dtype=np.float64)),
        ("advance", np.asarray([0.0, 0.0, tz, 0.0, 0.0])),
        ("unload", np.asarray([0.0, 0.0, -config.unload_step_m, 0.0, 0.0])),
        ("tangent_x_pos", np.asarray([tx, 0.0, 0.0, 0.0, 0.0])),
        ("tangent_x_neg", np.asarray([-tx, 0.0, 0.0, 0.0, 0.0])),
        ("tangent_y_pos", np.asarray([0.0, tx, 0.0, 0.0, 0.0])),
        ("tangent_y_neg", np.asarray([0.0, -tx, 0.0, 0.0, 0.0])),
        ("roll_pos", np.asarray([0.0, 0.0, 0.0, rr, 0.0])),
        ("roll_neg", np.asarray([0.0, 0.0, 0.0, -rr, 0.0])),
        ("pitch_pos", np.asarray([0.0, 0.0, 0.0, 0.0, rr])),
        ("pitch_neg", np.asarray([0.0, 0.0, 0.0, 0.0, -rr])),
    )


class EnergyInformationOptimizer:
    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self.config = config or OptimizerConfig()
        self.model = MultiOutputRidgeRLS(self.config)
        search_radius = int(
            np.ceil(
                self.config.maximum_tangent_offset_m
                / self.config.tangent_step_m
            )
        )
        self._search_grid = tuple(
            (x, y)
            for x in range(-search_radius, search_radius + 1)
            for y in range(-search_radius, search_radius + 1)
            if np.hypot(x, y)
            <= self.config.maximum_tangent_offset_m
            / self.config.tangent_step_m
        )
        self._spatial_energy: dict[tuple[int, int], float] = {}
        self._spatial_visits: dict[tuple[int, int], int] = {}
        self._axial_probed_cells: set[tuple[int, int]] = set()
        self._failed_entry_cells: set[tuple[int, int]] = set()
        self._pending_probe_cell: tuple[int, int] | None = None
        self._pending_probe_start_z = 0.0
        self._entry_xy: np.ndarray | None = None
        self._entry_cell: tuple[int, int] | None = None
        self._entry_best_z = 0.0
        self._entry_stall_steps = 0
        self._entry_mode = False

    def reset(self) -> None:
        self.model.reset()
        self._spatial_energy.clear()
        self._spatial_visits.clear()
        self._axial_probed_cells.clear()
        self._failed_entry_cells.clear()
        self._pending_probe_cell = None
        self._pending_probe_start_z = 0.0
        self._entry_xy = None
        self._entry_cell = None
        self._entry_best_z = 0.0
        self._entry_stall_steps = 0
        self._entry_mode = False

    @property
    def axial_probe_cells(self) -> int:
        return len(self._axial_probed_cells)

    @property
    def entry_mode(self) -> bool:
        return self._entry_mode

    @property
    def entry_xy(self) -> np.ndarray | None:
        return None if self._entry_xy is None else self._entry_xy.copy()

    @property
    def probe_pending(self) -> bool:
        return self._pending_probe_cell is not None

    @property
    def search_cells(self) -> int:
        return len(self._spatial_energy)

    @property
    def frontier_cells_remaining(self) -> int:
        return sum(cell not in self._spatial_energy for cell in self._search_grid)

    def reset_interaction(self) -> None:
        self.model.reset()
        self._pending_probe_cell = None
        self._entry_xy = None
        self._entry_cell = None
        self._entry_best_z = 0.0
        self._entry_stall_steps = 0
        self._entry_mode = False

    def update(
        self,
        previous_u5: np.ndarray,
        wrench_delta5: np.ndarray,
        measured_motion5: np.ndarray,
    ) -> bool:
        return self.model.update(previous_u5, wrench_delta5, measured_motion5)

    def select(
        self,
        wrench5: np.ndarray,
        previous_u5: np.ndarray,
        *,
        cumulative_positive_work_j: float,
        tactile_delta: float = 0.0,
        command_offset5: np.ndarray | None = None,
        forced_name: str | None = None,
        extra_candidate: tuple[str, np.ndarray] | None = None,
    ) -> CandidateSelection:
        wrench = np.asarray(wrench5, dtype=np.float64)
        previous = np.asarray(previous_u5, dtype=np.float64)
        if wrench.shape != (5,) or previous.shape != (5,):
            raise ValueError("wrench5 and previous_u5 must have shape (5,)")
        if not np.isfinite(wrench).all() or not np.isfinite(previous).all():
            raise ValueError("optimizer inputs must be finite")
        if not np.isfinite(cumulative_positive_work_j) or cumulative_positive_work_j < 0:
            raise ValueError("cumulative_positive_work_j must be finite and non-negative")
        if not np.isfinite(tactile_delta) or tactile_delta < 0:
            raise ValueError("tactile_delta must be finite and non-negative")
        offset = (
            np.zeros(5, dtype=np.float64)
            if command_offset5 is None
            else np.asarray(command_offset5, dtype=np.float64)
        )
        if offset.shape != (5,) or not np.isfinite(offset).all():
            raise ValueError("command_offset5 must be a finite vector with shape (5,)")
        actions = candidate_micro_actions(self.config)
        if extra_candidate is not None:
            extra_name, extra_u5 = extra_candidate
            extra_u5 = np.asarray(extra_u5, dtype=np.float64)
            if not extra_name or any(name == extra_name for name, _ in actions):
                raise ValueError("extra candidate name must be non-empty and unique")
            if extra_u5.shape != (5,) or not np.isfinite(extra_u5).all():
                raise ValueError("extra candidate action must be finite with shape (5,)")
            actions += ((extra_name, extra_u5),)

        scale = self.config.action_scale
        def workspace_violation(value: np.ndarray) -> float:
            return (
                max(
                    float(np.linalg.norm(value[:2]))
                    - self.config.maximum_tangent_offset_m,
                    0.0,
                )
                / self.config.tangent_step_m
                + max(value[2] - self.config.maximum_advance_offset_m, 0.0)
                / self.config.advance_step_m
                + max(-self.config.maximum_retreat_offset_m - value[2], 0.0)
                / self.config.unload_step_m
                + max(
                    float(np.linalg.norm(value[3:]))
                    - self.config.maximum_tilt_offset_rad,
                    0.0,
                )
                / self.config.rotation_step_rad
            )

        current_workspace_violation = workspace_violation(offset)
        search_axes = np.asarray([0, 1])
        search_key = tuple(
            np.rint(offset[search_axes] / scale[search_axes]).astype(int)
        )
        observed_spatial_energy = (
            self.config.lateral_weight
            * (
                float(wrench[0] ** 2 + wrench[1] ** 2)
                / self.config.lateral_force_soft_n**2
                + float(wrench[3] ** 2 + wrench[4] ** 2)
                / self.config.lateral_torque_soft_nm**2
            )
            + self.config.positive_work_weight
            * cumulative_positive_work_j
            / self.config.positive_work_soft_j
        )
        self._spatial_visits[search_key] = self._spatial_visits.get(search_key, 0) + 1
        self._spatial_energy[search_key] = min(
            observed_spatial_energy,
            self._spatial_energy.get(search_key, float("inf")),
        )
        previous_phi = previous / scale
        if (
            self._entry_mode
            and abs(float(wrench[2])) > self.config.entry_force_limit_n
        ):
            failed_cell = self._entry_cell or search_key
            self._entry_mode = False
            self._entry_xy = None
            self._entry_cell = None
            self._entry_stall_steps = 0
            self._failed_entry_cells.add(failed_cell)
        if self._pending_probe_cell is not None:
            probe_progress = float(offset[2]) - self._pending_probe_start_z
            probe_supported = (
                abs(float(wrench[2])) >= self.config.axial_preload_n
            )
            if probe_supported or probe_progress >= self.config.entry_progress_m:
                probed_cell = self._pending_probe_cell
                self._axial_probed_cells.add(probed_cell)
                self._pending_probe_cell = None
                if probe_supported:
                    self._failed_entry_cells.add(probed_cell)
                else:
                    self._entry_mode = True
                    self._entry_xy = offset[:2].copy()
                    self._entry_cell = probed_cell
                    self._entry_best_z = float(offset[2])
                    self._entry_stall_steps = 0
        if self._entry_mode:
            if (
                float(offset[2])
                >= self._entry_best_z + 0.5 * self.config.advance_step_m
            ):
                self._entry_best_z = float(offset[2])
                self._entry_stall_steps = 0
            else:
                self._entry_stall_steps += 1
            if self._entry_stall_steps >= self.config.entry_stall_steps:
                assert self._entry_cell is not None
                self._failed_entry_cells.add(self._entry_cell)
                self._entry_mode = False
                self._entry_xy = None
                self._entry_cell = None
                self._entry_stall_steps = 0
        productive_entry = self._entry_mode
        unvisited_cells = tuple(
            cell for cell in self._search_grid if cell not in self._spatial_energy
        )
        evaluations: list[CandidateEvaluation] = []
        for name, u5 in actions:
            prediction = self.model.predict(u5)
            predicted_wrench = wrench + prediction.wrench_delta5
            average_wrench = 0.5 * (wrench + predicted_wrench)
            positive_work = max(
                0.0,
                self.config.power_sign
                * float(average_wrench @ prediction.motion5),
            )
            force_norm = float(np.linalg.norm(predicted_wrench[:3]))
            torque_norm = float(np.linalg.norm(predicted_wrench[3:]))
            proposed_offset = offset + u5
            candidate_key = tuple(
                np.rint(
                    proposed_offset[search_axes] / scale[search_axes]
                ).astype(int)
            )
            spatial_energy = self._spatial_energy.get(
                candidate_key, observed_spatial_energy
            )
            spatial_novelty = float(candidate_key not in self._spatial_energy)
            revisit_count = (
                0
                if name == "advance" and productive_entry
                else self._spatial_visits.get(candidate_key, 0)
            )
            axial_probe_novelty = float(
                u5[2] > 0.0 and search_key not in self._axial_probed_cells
            )
            entry_progress = float(u5[2] > 0.0 and productive_entry)
            frontier_distance = (
                min(
                    float(np.hypot(candidate_key[0] - cell[0], candidate_key[1] - cell[1]))
                    for cell in unvisited_cells
                )
                if search_key in self._axial_probed_cells and unvisited_cells
                else 0.0
            )
            proposed_workspace_violation = workspace_violation(proposed_offset)
            inside_workspace = bool(
                proposed_workspace_violation <= 1e-12
                or proposed_workspace_violation
                < current_workspace_violation - 1e-12
            )
            safe = bool(
                inside_workspace
                and (
                    not productive_entry
                    or not np.any(u5[:2])
                    or (
                        self._entry_xy is not None
                        and (
                            np.linalg.norm(proposed_offset[:2] - self._entry_xy)
                            <= 4.0 * self.config.tangent_step_m + 1e-12
                            or np.linalg.norm(proposed_offset[:2] - self._entry_xy)
                            < np.linalg.norm(offset[:2] - self._entry_xy) - 1e-12
                        )
                    )
                )
                and force_norm <= self.config.hard_force_n
                and torque_norm <= self.config.hard_torque_nm
                and cumulative_positive_work_j + positive_work
                <= self.config.hard_positive_work_j
            )

            phi = u5 / scale
            motion_phi = prediction.motion5 / scale
            lateral_load = (
                float(predicted_wrench[0] ** 2 + predicted_wrench[1] ** 2)
                / self.config.lateral_force_soft_n**2
                + float(predicted_wrench[3] ** 2 + predicted_wrench[4] ** 2)
                / self.config.lateral_torque_soft_nm**2
            )
            axial_preload_error = (
                (abs(float(predicted_wrench[2])) - self.config.axial_preload_n)
                / self.config.axial_preload_n
            ) ** 2
            effort = float(phi @ phi)
            slew = float(np.sum((phi - previous_phi) ** 2))
            tactile_risk = tactile_delta * float(
                max(phi[2], 0.0) ** 2
                + phi[0] ** 2
                + phi[1] ** 2
                + phi[3] ** 2
                + phi[4] ** 2
            )
            score = (
                self.config.lateral_weight * lateral_load
                + self.config.positive_work_weight
                * positive_work
                / self.config.positive_work_soft_j
                + self.config.effort_weight * effort
                + self.config.slew_weight * slew
                + self.config.tactile_weight * tactile_risk
                - self.config.axial_progress_weight * float(motion_phi[2])
                - self.config.entry_progress_weight * entry_progress
                - self.config.information_weight * prediction.information_gain
                + self.config.axial_preload_weight * axial_preload_error
                + self.config.tilt_offset_weight
                * float(proposed_offset[3:] @ proposed_offset[3:])
                / self.config.maximum_tilt_offset_rad**2
                + self.config.spatial_energy_weight * spatial_energy
                + self.config.frontier_weight * frontier_distance
                + self.config.revisit_weight * revisit_count
                - self.config.search_novelty_weight * spatial_novelty
                - self.config.axial_probe_weight * axial_probe_novelty
            )
            evaluations.append(
                CandidateEvaluation(
                    name=name,
                    u5=u5.copy(),
                    score=float(score) if safe else float("inf"),
                    safe=safe,
                    predicted_wrench5=predicted_wrench.copy(),
                    predicted_motion5=prediction.motion5.copy(),
                    predicted_positive_work_j=float(positive_work),
                    information_gain=prediction.information_gain,
                )
            )

        safe_candidates = [candidate for candidate in evaluations if candidate.safe]
        effective_forced_name = (
            forced_name
            if extra_candidate is not None and forced_name == extra_candidate[0]
            else "advance" if self._pending_probe_cell is not None else forced_name
        )
        forced = next(
            (
                candidate
                for candidate in evaluations
                if candidate.name == effective_forced_name
            ),
            None,
        )
        if effective_forced_name is not None and forced is None:
            raise ValueError(f"unknown forced candidate: {effective_forced_name}")
        if forced is not None and forced.safe:
            best = forced
        elif safe_candidates:
            best = min(safe_candidates, key=lambda candidate: candidate.score)
        else:
            best = next(candidate for candidate in evaluations if candidate.name == "hold")
        if (
            best.safe
            and best.u5[2] > 0.0
            and (
                best.name == "spiral_probe"
                or (best.name != "spiral" and not np.any(best.u5[:2]))
            )
            and not productive_entry
            and self._pending_probe_cell is None
            and search_key not in self._axial_probed_cells
        ):
            self._pending_probe_cell = search_key
            self._pending_probe_start_z = float(offset[2])
        return CandidateSelection(best=best, candidates=tuple(evaluations))
