"""Privileged read-only observation adapter for the P0 hierarchy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybrid_insert.assembly_contacts import AssemblyContactLabeler

from .control import CerebellumObservation
from .privileged import PrivilegedAssemblyPrimitiveProvider


def _rotation_matrix(xmat: np.ndarray) -> np.ndarray:
    return np.asarray(xmat, dtype=np.float64).reshape(3, 3)


@dataclass(frozen=True)
class SlipEstimate:
    translation_mps: float
    rotation_radps: float


class RelativePoseSlipTracker:
    """Measure object motion relative to the holding palm."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._previous_position: np.ndarray | None = None
        self._previous_rotation: np.ndarray | None = None
        self._previous_time: float | None = None

    def update(
        self,
        *,
        hand_position_world: np.ndarray,
        hand_rotation_world: np.ndarray,
        object_position_world: np.ndarray,
        object_rotation_world: np.ndarray,
        timestamp_s: float,
        active: bool,
    ) -> SlipEstimate | None:
        if not active:
            self.reset()
            return None

        hand_position = np.asarray(hand_position_world, dtype=np.float64)
        hand_rotation = _rotation_matrix(hand_rotation_world)
        object_position = np.asarray(object_position_world, dtype=np.float64)
        object_rotation = _rotation_matrix(object_rotation_world)

        relative_position = hand_rotation.T @ (object_position - hand_position)
        relative_rotation = hand_rotation.T @ object_rotation
        current_time = float(timestamp_s)

        if self._previous_time is None:
            estimate = SlipEstimate(translation_mps=0.0, rotation_radps=0.0)
        else:
            delta_time = current_time - self._previous_time
            if delta_time <= 1e-8:
                estimate = SlipEstimate(translation_mps=0.0, rotation_radps=0.0)
            else:
                translation = float(
                    np.linalg.norm(relative_position - self._previous_position) / delta_time
                )
                relative_delta = self._previous_rotation.T @ relative_rotation
                cosine = float(np.clip((np.trace(relative_delta) - 1.0) * 0.5, -1.0, 1.0))
                rotation = float(np.arccos(cosine) / delta_time)
                estimate = SlipEstimate(
                    translation_mps=translation,
                    rotation_radps=rotation,
                )

        self._previous_position = relative_position.copy()
        self._previous_rotation = relative_rotation.copy()
        self._previous_time = current_time
        return estimate


@dataclass(frozen=True)
class PrivilegedObserverConfig:
    pregrasp_distance_m: float = 0.12
    stable_translation_mps: float = 0.020
    unstable_translation_mps: float = 0.040
    stable_rotation_radps: float = 0.30
    unstable_rotation_radps: float = 0.60
    stable_confirm_frames: int = 5
    unstable_confirm_frames: int = 5

    def __post_init__(self) -> None:
        if self.pregrasp_distance_m <= 0.0:
            raise ValueError("pregrasp_distance_m must be positive")
        if self.stable_translation_mps <= 0.0:
            raise ValueError("stable_translation_mps must be positive")
        if self.unstable_translation_mps <= self.stable_translation_mps:
            raise ValueError(
                "unstable_translation_mps must exceed stable_translation_mps"
            )
        if self.stable_rotation_radps <= 0.0:
            raise ValueError("stable_rotation_radps must be positive")
        if self.unstable_rotation_radps <= self.stable_rotation_radps:
            raise ValueError("unstable_rotation_radps must exceed stable_rotation_radps")
        if self.stable_confirm_frames <= 0:
            raise ValueError("stable_confirm_frames must be positive")
        if self.unstable_confirm_frames <= 0:
            raise ValueError("unstable_confirm_frames must be positive")


@dataclass(frozen=True)
class ObserverDiagnostics:
    peg_palm_distance_m: float
    tray_palm_distance_m: float
    peg_translation_slip_mps: float | None
    tray_translation_slip_mps: float | None
    peg_rotation_slip_radps: float | None
    tray_rotation_slip_radps: float | None


class PrivilegedCerebellumObserver:
    """Build a CerebellumObservation from simulator state without changing actions."""

    _RIGHT_PALM = "allegro_palm_right"
    _LEFT_PALM = "allegro_palm_left"

    def __init__(
        self,
        raw_env,
        *,
        config: PrivilegedObserverConfig | None = None,
        labeler: AssemblyContactLabeler | None = None,
        primitive_provider: PrivilegedAssemblyPrimitiveProvider | None = None,
    ) -> None:
        self.config = config or PrivilegedObserverConfig()
        self._provider = primitive_provider or PrivilegedAssemblyPrimitiveProvider(raw_env)
        self._labeler = labeler or AssemblyContactLabeler(raw_env)
        model = raw_env._model
        names = self._provider.names
        self._peg_body_id = int(model.body(names.peg_body).id)
        self._tray_body_id = int(model.body(names.socket_body).id)
        self._right_palm_id = int(model.body(self._RIGHT_PALM).id)
        self._left_palm_id = int(model.body(self._LEFT_PALM).id)
        self._peg_tracker = RelativePoseSlipTracker()
        self._tray_tracker = RelativePoseSlipTracker()
        self._peg_stable_streak = 0
        self._tray_stable_streak = 0
        self._peg_unstable_streak = 0
        self._tray_unstable_streak = 0
        self._peg_stable = False
        self._tray_stable = False
        self.last_diagnostics: ObserverDiagnostics | None = None

    def reset(self, raw_env) -> None:
        self._labeler.reset_reference(raw_env)
        self._peg_tracker.reset()
        self._tray_tracker.reset()
        self._peg_stable_streak = 0
        self._tray_stable_streak = 0
        self._peg_unstable_streak = 0
        self._tray_unstable_streak = 0
        self._peg_stable = False
        self._tray_stable = False
        self.last_diagnostics = None

    def observe(self, raw_env, state44: np.ndarray) -> CerebellumObservation:
        data = raw_env._data
        outcome = self._labeler.compute(raw_env)
        primitives = self._provider.snapshot(raw_env)
        current_time = float(getattr(data, "time", 0.0))

        peg_slip = self._peg_tracker.update(
            hand_position_world=data.xpos[self._right_palm_id],
            hand_rotation_world=data.xmat[self._right_palm_id],
            object_position_world=data.xpos[self._peg_body_id],
            object_rotation_world=data.xmat[self._peg_body_id],
            timestamp_s=current_time,
            active=outcome.peg_ok,
        )
        tray_slip = self._tray_tracker.update(
            hand_position_world=data.xpos[self._left_palm_id],
            hand_rotation_world=data.xmat[self._left_palm_id],
            object_position_world=data.xpos[self._tray_body_id],
            object_rotation_world=data.xmat[self._tray_body_id],
            timestamp_s=current_time,
            active=outcome.tray_ok,
        )

        peg_stable = self._update_stability(outcome.peg_ok, peg_slip, object_name="peg")
        tray_stable = self._update_stability(outcome.tray_ok, tray_slip, object_name="tray")
        peg_distance = float(
            np.linalg.norm(data.xpos[self._right_palm_id] - data.xpos[self._peg_body_id])
        )
        tray_distance = float(
            np.linalg.norm(data.xpos[self._left_palm_id] - data.xpos[self._tray_body_id])
        )

        translation_slips = [
            estimate.translation_mps
            for estimate in (peg_slip, tray_slip)
            if estimate is not None
        ]
        rotation_slips = [
            estimate.rotation_radps
            for estimate in (peg_slip, tray_slip)
            if estimate is not None
        ]
        self.last_diagnostics = ObserverDiagnostics(
            peg_palm_distance_m=peg_distance,
            tray_palm_distance_m=tray_distance,
            peg_translation_slip_mps=None if peg_slip is None else peg_slip.translation_mps,
            tray_translation_slip_mps=None if tray_slip is None else tray_slip.translation_mps,
            peg_rotation_slip_radps=None if peg_slip is None else peg_slip.rotation_radps,
            tray_rotation_slip_radps=None if tray_slip is None else tray_slip.rotation_radps,
        )
        return CerebellumObservation(
            state44=state44,
            primitives=primitives,
            peg_grasped=outcome.peg_ok,
            tray_grasped=outcome.tray_ok,
            peg_pregrasp_ready=(
                not outcome.peg_ok and peg_distance <= self.config.pregrasp_distance_m
            ),
            tray_pregrasp_ready=(
                not outcome.tray_ok and tray_distance <= self.config.pregrasp_distance_m
            ),
            peg_grasp_stable=peg_stable,
            tray_grasp_stable=tray_stable,
            peg_contact_count=outcome.peg_contact_count,
            tray_contact_count=outcome.tray_contact_count,
            insert_contact=outcome.insert_ok,
            slip_speed_mps=max(translation_slips, default=None),
            rotation_slip_radps=max(rotation_slips, default=None),
        )

    def _update_stability(
        self,
        grasped: bool,
        slip: SlipEstimate | None,
        *,
        object_name: str,
    ) -> bool:
        stable_name = f"_{object_name}_stable"
        stable_streak_name = f"_{object_name}_stable_streak"
        unstable_streak_name = f"_{object_name}_unstable_streak"
        if not grasped or slip is None:
            setattr(self, stable_name, False)
            setattr(self, stable_streak_name, 0)
            setattr(self, unstable_streak_name, 0)
            return False

        is_stable = bool(getattr(self, stable_name))
        if is_stable:
            unstable_sample = bool(
                slip.translation_mps >= self.config.unstable_translation_mps
                or slip.rotation_radps >= self.config.unstable_rotation_radps
            )
            unstable_streak = int(getattr(self, unstable_streak_name))
            unstable_streak = unstable_streak + 1 if unstable_sample else 0
            setattr(self, unstable_streak_name, unstable_streak)
            if unstable_streak >= self.config.unstable_confirm_frames:
                setattr(self, stable_name, False)
                setattr(self, stable_streak_name, 0)
                return False
            return True

        stable_sample = bool(
            slip.translation_mps <= self.config.stable_translation_mps
            and slip.rotation_radps <= self.config.stable_rotation_radps
        )
        stable_streak = int(getattr(self, stable_streak_name))
        stable_streak = stable_streak + 1 if stable_sample else 0
        setattr(self, stable_streak_name, stable_streak)
        if stable_streak >= self.config.stable_confirm_frames:
            setattr(self, stable_name, True)
            setattr(self, unstable_streak_name, 0)
            return True
        return False
