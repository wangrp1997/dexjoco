"""Temporal deployable grasp evidence for controller handoff."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from .sensor_observation import CerebellumSensorObservation


@dataclass(frozen=True)
class HandoffGateConfig:
    window_frames: int = 100
    minimum_contact_fingers: int = 2
    contact_force_threshold_n: float = 0.5
    minimum_total_fingertip_force_n: float = 1.0
    minimum_contact_fraction: float = 0.6
    maximum_contact_gap_frames: int = 10
    minimum_transport_span_m: float = 0.015

    def __post_init__(self) -> None:
        if self.window_frames <= 1:
            raise ValueError("window_frames must be greater than one")
        if not 1 <= self.minimum_contact_fingers <= 4:
            raise ValueError("minimum_contact_fingers must be in [1, 4]")
        if self.contact_force_threshold_n < 0.0:
            raise ValueError("contact_force_threshold_n must be non-negative")
        if self.minimum_total_fingertip_force_n < 0.0:
            raise ValueError("minimum_total_fingertip_force_n must be non-negative")
        if not 0.0 < self.minimum_contact_fraction <= 1.0:
            raise ValueError("minimum_contact_fraction must be in (0, 1]")
        if not 0 <= self.maximum_contact_gap_frames < self.window_frames:
            raise ValueError("maximum_contact_gap_frames must be within the window")
        if self.minimum_transport_span_m <= 0.0:
            raise ValueError("minimum_transport_span_m must be positive")


@dataclass(frozen=True)
class SideHandoffEvidence:
    contact_fraction: float
    trailing_contact_gap_frames: int
    transport_span_m: float
    current_contact_fingers: int
    current_total_force_n: float
    ready: bool


@dataclass(frozen=True)
class HandoffGateDecision:
    ready: bool
    frames_observed: int
    right: SideHandoffEvidence
    left: SideHandoffEvidence
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "frames_observed": self.frames_observed,
            "right": asdict(self.right),
            "left": asdict(self.left),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CoarseAlignmentGateConfig:
    persistence_frames: int = 10
    midpoint_lower_world: tuple[float, float, float] = (-0.50, -0.08, 1.30)
    midpoint_upper_world: tuple[float, float, float] = (-0.34, 0.08, 1.56)
    minimum_relative_wrist_distance_m: float = 0.24
    maximum_relative_wrist_distance_m: float = 0.36
    minimum_hand_joint_norm: float = 3.0
    hard_wrist_force_limit_n: float = 20.0

    def __post_init__(self) -> None:
        if self.persistence_frames <= 0:
            raise ValueError("persistence_frames must be positive")
        lower = np.asarray(self.midpoint_lower_world, dtype=np.float64)
        upper = np.asarray(self.midpoint_upper_world, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
            raise ValueError("midpoint workspace bounds must be ordered 3-vectors")
        if not (
            0.0
            < self.minimum_relative_wrist_distance_m
            < self.maximum_relative_wrist_distance_m
        ):
            raise ValueError("relative wrist distance bounds must be ordered")
        if self.minimum_hand_joint_norm <= 0.0:
            raise ValueError("minimum_hand_joint_norm must be positive")
        if self.hard_wrist_force_limit_n <= 0.0:
            raise ValueError("hard_wrist_force_limit_n must be positive")


@dataclass(frozen=True)
class CoarseAlignmentGateDecision:
    ready: bool
    ready_streak: int
    midpoint_world: tuple[float, float, float]
    relative_wrist_distance_m: float
    right_hand_joint_norm: float
    left_hand_joint_norm: float
    peak_wrist_force_n: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DeployableCoarseAlignmentGate:
    """Detect the fixed pre-insert workspace using robot-side signals only."""

    def __init__(self, config: CoarseAlignmentGateConfig | None = None) -> None:
        self.config = config or CoarseAlignmentGateConfig()
        self.reset()

    def reset(self) -> None:
        self._ready_streak = 0

    def update(
        self,
        observation: CerebellumSensorObservation,
    ) -> CoarseAlignmentGateDecision:
        state = np.asarray(observation.state46, dtype=np.float64)
        right_position = state[:3]
        left_position = state[7:10]
        midpoint = 0.5 * (right_position + left_position)
        relative_distance = float(np.linalg.norm(right_position - left_position))
        right_hand_norm = float(np.linalg.norm(state[14:30]))
        left_hand_norm = float(np.linalg.norm(state[30:46]))
        peak_force = float(
            np.max(
                np.linalg.norm(
                    np.asarray(observation.wrist_wrench_world, dtype=np.float64)[:, :3],
                    axis=1,
                )
            )
        )
        lower = np.asarray(self.config.midpoint_lower_world, dtype=np.float64)
        upper = np.asarray(self.config.midpoint_upper_world, dtype=np.float64)
        workspace_ready = bool(np.all(midpoint >= lower) and np.all(midpoint <= upper))
        separation_ready = bool(
            self.config.minimum_relative_wrist_distance_m
            <= relative_distance
            <= self.config.maximum_relative_wrist_distance_m
            and right_position[1] < left_position[1]
        )
        hands_ready = bool(
            right_hand_norm >= self.config.minimum_hand_joint_norm
            and left_hand_norm >= self.config.minimum_hand_joint_norm
        )
        force_safe = peak_force < self.config.hard_wrist_force_limit_n
        current_ready = workspace_ready and separation_ready and hands_ready and force_safe
        self._ready_streak = self._ready_streak + 1 if current_ready else 0
        ready = self._ready_streak >= self.config.persistence_frames
        return CoarseAlignmentGateDecision(
            ready=ready,
            ready_streak=self._ready_streak,
            midpoint_world=tuple(float(value) for value in midpoint),
            relative_wrist_distance_m=relative_distance,
            right_hand_joint_norm=right_hand_norm,
            left_hand_joint_norm=left_hand_norm,
            peak_wrist_force_n=peak_force,
            reason=(
                "persistent robot-side coarse-alignment contract satisfied"
                if ready
                else "waiting for persistent closed-hand pre-insert workspace state"
            ),
        )


class DeployableGraspHandoffGate:
    """Require persistent multi-finger contact through a transport interval."""

    def __init__(self, config: HandoffGateConfig | None = None) -> None:
        self.config = config or HandoffGateConfig()
        self.reset()

    def reset(self) -> None:
        self._contact_valid: deque[np.ndarray] = deque(
            maxlen=self.config.window_frames
        )
        self._contact_count: deque[np.ndarray] = deque(
            maxlen=self.config.window_frames
        )
        self._total_force: deque[np.ndarray] = deque(
            maxlen=self.config.window_frames
        )
        self._wrist_position: deque[np.ndarray] = deque(
            maxlen=self.config.window_frames
        )

    def update(self, observation: CerebellumSensorObservation) -> HandoffGateDecision:
        force_norm = np.linalg.norm(observation.fingertip_force_world, axis=2)
        contact_count = np.sum(
            force_norm >= self.config.contact_force_threshold_n,
            axis=1,
        )
        total_force = np.sum(force_norm, axis=1)
        contact_valid = (
            contact_count >= self.config.minimum_contact_fingers
        ) & (total_force >= self.config.minimum_total_fingertip_force_n)
        wrist_position = np.stack(
            [observation.state46[:3], observation.state46[7:10]],
            axis=0,
        )
        self._contact_valid.append(contact_valid.astype(bool))
        self._contact_count.append(contact_count.astype(np.int64))
        self._total_force.append(total_force.astype(np.float64))
        self._wrist_position.append(wrist_position.astype(np.float64))

        frames = len(self._contact_valid)
        if frames < self.config.window_frames:
            right = SideHandoffEvidence(
                contact_fraction=0.0,
                trailing_contact_gap_frames=frames,
                transport_span_m=0.0,
                current_contact_fingers=int(contact_count[0]),
                current_total_force_n=float(total_force[0]),
                ready=False,
            )
            left = SideHandoffEvidence(
                contact_fraction=0.0,
                trailing_contact_gap_frames=frames,
                transport_span_m=0.0,
                current_contact_fingers=int(contact_count[1]),
                current_total_force_n=float(total_force[1]),
                ready=False,
            )
            return HandoffGateDecision(
                ready=False,
                frames_observed=frames,
                right=right,
                left=left,
                reason="insufficient temporal grasp history",
            )

        valid = np.stack(self._contact_valid)
        counts = np.stack(self._contact_count)
        totals = np.stack(self._total_force)
        positions = np.stack(self._wrist_position)
        right = self._side_evidence(
            valid[:, 0], positions[:, 0], counts[-1, 0], totals[-1, 0]
        )
        left = self._side_evidence(
            valid[:, 1], positions[:, 1], counts[-1, 1], totals[-1, 1]
        )
        ready = right.ready and left.ready
        return HandoffGateDecision(
            ready=ready,
            frames_observed=frames,
            right=right,
            left=left,
            reason=(
                "bilateral grasp survived contact persistence and transport checks"
                if ready
                else "bilateral temporal grasp evidence is incomplete"
            ),
        )

    def _side_evidence(
        self,
        valid: np.ndarray,
        positions: np.ndarray,
        current_count: int,
        current_total: float,
    ) -> SideHandoffEvidence:
        contact_fraction = float(np.mean(valid))
        trailing_gap = 0
        for is_valid in valid[::-1]:
            if is_valid:
                break
            trailing_gap += 1
        active_positions = positions[valid]
        if active_positions.shape[0] < 2:
            transport_span = 0.0
        else:
            pairwise = active_positions[:, None, :] - active_positions[None, :, :]
            transport_span = float(np.max(np.linalg.norm(pairwise, axis=2)))
        ready = (
            contact_fraction >= self.config.minimum_contact_fraction
            and trailing_gap <= self.config.maximum_contact_gap_frames
            and transport_span >= self.config.minimum_transport_span_m
            and current_count >= self.config.minimum_contact_fingers
            and current_total >= self.config.minimum_total_fingertip_force_n
        )
        return SideHandoffEvidence(
            contact_fraction=contact_fraction,
            trailing_contact_gap_frames=trailing_gap,
            transport_span_m=transport_span,
            current_contact_fingers=int(current_count),
            current_total_force_n=float(current_total),
            ready=ready,
        )
