"""Conservative asymmetric grasp assistance for bimanual assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .control import CerebellumObservation


_RIGHT_HAND = slice(6, 22)
_LEFT_HAND = slice(28, 44)
_ASSISTED_HAND_JOINTS = np.asarray([1, 2, 5, 6, 9, 10, 12, 13, 14, 15])
_HAND_LOWER = np.asarray(
    [-0.47, -0.196, -0.174, -0.227] * 3
    + [0.263, -0.105, -0.189, -0.162],
    dtype=np.float64,
)
_HAND_UPPER = np.asarray(
    [0.47, 1.61, 1.709, 1.618] * 3
    + [1.396, 1.163, 1.644, 1.719],
    dtype=np.float64,
)


@dataclass(frozen=True)
class AsymmetricGraspAssistConfig:
    closure_step_rad: float = 0.025
    max_extra_closure_rad: float = 0.35
    max_active_frames: int = 180
    reactivation_cooldown_frames: int = 60

    def __post_init__(self) -> None:
        if self.closure_step_rad <= 0.0:
            raise ValueError("closure_step_rad must be positive")
        if self.max_extra_closure_rad <= 0.0:
            raise ValueError("max_extra_closure_rad must be positive")
        if self.max_active_frames <= 0:
            raise ValueError("max_active_frames must be positive")
        if self.reactivation_cooldown_frames < 0:
            raise ValueError("reactivation_cooldown_frames must be non-negative")


@dataclass(frozen=True)
class GraspAssistDiagnostics:
    active: bool
    held_side: str | None
    assisted_side: str | None
    assisting_fingers: bool
    active_frames: int
    outcome: str


class AsymmetricGraspAssist:
    """Hold one stable grasp while the VLA positions and closes the other hand."""

    def __init__(self, config: AsymmetricGraspAssistConfig | None = None) -> None:
        self.config = config or AsymmetricGraspAssistConfig()
        self.reset()

    def reset(self) -> None:
        self._held_side: str | None = None
        self._held_hand16: np.ndarray | None = None
        self._assist_target16: np.ndarray | None = None
        self._assist_limit16: np.ndarray | None = None
        self._active_frames = 0
        self._cooldown_frames = 0
        self._activation_count = 0
        self._completion_count = 0
        self._abort_count = 0
        self._completed_once = False
        self._outcome = "idle"
        self.last_diagnostics = GraspAssistDiagnostics(
            active=False,
            held_side=None,
            assisted_side=None,
            assisting_fingers=False,
            active_frames=0,
            outcome=self._outcome,
        )

    @property
    def active(self) -> bool:
        return self._held_side is not None

    def step(
        self,
        observation: CerebellumObservation,
        vla_action44: np.ndarray,
    ) -> np.ndarray:
        action = np.asarray(vla_action44, dtype=np.float64).reshape(-1)
        if action.shape != (44,):
            raise ValueError(f"vla_action44 must have shape (44,), got {action.shape}")

        if not self.active:
            self._cooldown_frames = max(0, self._cooldown_frames - 1)
            self._maybe_activate(observation)

        if not self.active:
            self._set_diagnostics(assisting_fingers=False)
            return action.astype(np.float32)

        if self._should_finish(observation):
            self._deactivate("completed")
            return action.astype(np.float32)
        if self._held_grasp_lost(observation):
            self._deactivate("held_grasp_lost")
            return action.astype(np.float32)
        if self._active_frames >= self.config.max_active_frames:
            self._deactivate("timeout")
            return action.astype(np.float32)

        assisted_side = self._assisted_side
        assisting_fingers = self._pregrasp_ready(observation, assisted_side) or self._grasped(
            observation, assisted_side
        )
        merged = action.copy()
        self._hold_stable_side(merged)
        if assisting_fingers:
            self._close_assisted_hand(merged, assisted_side)

        self._active_frames += 1
        self._outcome = "active"
        self._set_diagnostics(assisting_fingers=assisting_fingers)
        return merged.astype(np.float32)

    @property
    def _assisted_side(self) -> str:
        if self._held_side == "right":
            return "left"
        if self._held_side == "left":
            return "right"
        raise RuntimeError("grasp assist is not active")

    def _maybe_activate(self, observation: CerebellumObservation) -> None:
        if self._completed_once or self._cooldown_frames > 0:
            return
        if observation.peg_grasp_stable and not observation.tray_grasped:
            self._activate(observation, held_side="right")
        elif observation.tray_grasp_stable and not observation.peg_grasped:
            self._activate(observation, held_side="left")

    def _activate(self, observation: CerebellumObservation, *, held_side: str) -> None:
        state = observation.state44.astype(np.float64, copy=True)
        if held_side == "right":
            self._held_hand16 = state[_RIGHT_HAND].copy()
            assisted_hand = state[_LEFT_HAND]
        else:
            self._held_hand16 = state[_LEFT_HAND].copy()
            assisted_hand = state[_RIGHT_HAND]
        self._held_side = held_side
        self._assist_target16 = assisted_hand.copy()
        self._assist_limit16 = np.clip(
            assisted_hand + self.config.max_extra_closure_rad,
            _HAND_LOWER,
            _HAND_UPPER,
        )
        self._active_frames = 0
        self._activation_count += 1
        self._outcome = "activated"

    def _should_finish(self, observation: CerebellumObservation) -> bool:
        return observation.peg_grasp_stable and observation.tray_grasp_stable

    def _held_grasp_lost(self, observation: CerebellumObservation) -> bool:
        if self._held_side == "right":
            return not observation.peg_grasped
        return not observation.tray_grasped

    def _hold_stable_side(self, action: np.ndarray) -> None:
        assert self._held_hand16 is not None
        if self._held_side == "right":
            action[_RIGHT_HAND] = self._held_hand16
        else:
            action[_LEFT_HAND] = self._held_hand16

    def _close_assisted_hand(
        self,
        action: np.ndarray,
        assisted_side: str,
    ) -> None:
        assert self._assist_target16 is not None
        assert self._assist_limit16 is not None
        hand_slice = _RIGHT_HAND if assisted_side == "right" else _LEFT_HAND
        policy_hand = action[hand_slice].copy()

        next_target = self._assist_target16.copy()
        next_target[_ASSISTED_HAND_JOINTS] += self.config.closure_step_rad
        next_target = np.minimum(next_target, self._assist_limit16)
        next_target = np.clip(next_target, _HAND_LOWER, _HAND_UPPER)
        self._assist_target16 = next_target

        assisted_hand = policy_hand.copy()
        assisted_hand[_ASSISTED_HAND_JOINTS] = np.maximum(
            policy_hand[_ASSISTED_HAND_JOINTS],
            next_target[_ASSISTED_HAND_JOINTS],
        )
        action[hand_slice] = np.clip(assisted_hand, _HAND_LOWER, _HAND_UPPER)

    @staticmethod
    def _pregrasp_ready(observation: CerebellumObservation, side: str) -> bool:
        return (
            observation.peg_pregrasp_ready
            if side == "right"
            else observation.tray_pregrasp_ready
        )

    @staticmethod
    def _grasped(observation: CerebellumObservation, side: str) -> bool:
        return observation.peg_grasped if side == "right" else observation.tray_grasped

    def _deactivate(self, outcome: str) -> None:
        held_side = self._held_side
        active_frames = self._active_frames
        assisted_side = self._assisted_side if held_side is not None else None
        self._held_side = None
        self._held_hand16 = None
        self._assist_target16 = None
        self._assist_limit16 = None
        self._active_frames = 0
        self._outcome = outcome
        if outcome == "completed":
            self._completion_count += 1
            self._completed_once = True
        else:
            self._abort_count += 1
            self._cooldown_frames = self.config.reactivation_cooldown_frames
        self.last_diagnostics = GraspAssistDiagnostics(
            active=False,
            held_side=held_side,
            assisted_side=assisted_side,
            assisting_fingers=False,
            active_frames=active_frames,
            outcome=outcome,
        )

    def _set_diagnostics(self, *, assisting_fingers: bool) -> None:
        self.last_diagnostics = GraspAssistDiagnostics(
            active=self.active,
            held_side=self._held_side,
            assisted_side=self._assisted_side if self.active else None,
            assisting_fingers=assisting_fingers,
            active_frames=self._active_frames,
            outcome=self._outcome,
        )

    def episode_summary(self) -> str:
        return (
            f"active={int(self.active)} activations={self._activation_count} "
            f"completed={self._completion_count} aborted={self._abort_count} "
            f"locked_out={int(self._completed_once)} last={self._outcome}"
        )
