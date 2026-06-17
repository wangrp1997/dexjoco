"""Subtask phase selection from predicted outcome probabilities."""

from __future__ import annotations

from dataclasses import dataclass

from dexquery.data.subtask_prompts import infer_subtask_phase


@dataclass
class PhaseControllerConfig:
    """Hysteresis thresholds for stable tray/peg completion flags."""

    threshold_high: float = 0.8
    threshold_low: float = 0.6
    confirm_frames: int = 8
    insert_min_prob: float = 0.8
    use_sim_guard: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold_low < self.threshold_high < 1.0:
            raise ValueError(
                "Expected 0 < threshold_low < threshold_high < 1, "
                f"got low={self.threshold_low}, high={self.threshold_high}"
            )
        if self.confirm_frames < 1:
            raise ValueError(f"confirm_frames must be >= 1, got {self.confirm_frames}")
        if not 0.0 < self.insert_min_prob <= 1.0:
            raise ValueError(f"insert_min_prob must be in (0, 1], got {self.insert_min_prob}")


@dataclass(frozen=True)
class PhaseControllerState:
    tray_ok: bool
    peg_ok: bool
    subtask_phase: int
    tray_prob: float
    peg_prob: float


class PhaseController:
    """Track subtask completion from outcome probabilities with debounced switching."""

    def __init__(self, config: PhaseControllerConfig | None = None) -> None:
        self.config = config or PhaseControllerConfig()
        self.reset()

    def reset(self) -> None:
        self.tray_ok = False
        self.peg_ok = False
        self._tray_streak = 0
        self._peg_streak = 0
        self._tray_target: bool | None = None
        self._peg_target: bool | None = None

    def update(
        self,
        tray_prob: float,
        peg_prob: float,
        *,
        tray_ok_sim: bool | None = None,
        peg_ok_sim: bool | None = None,
    ) -> PhaseControllerState:
        """Update debounced completion flags and return the active subtask phase."""
        tray_prob = float(tray_prob)
        peg_prob = float(peg_prob)

        self.tray_ok = self._update_flag(
            current=self.tray_ok,
            prob=tray_prob,
            streak_attr="_tray_streak",
            target_attr="_tray_target",
        )
        self.peg_ok = self._update_flag(
            current=self.peg_ok,
            prob=peg_prob,
            streak_attr="_peg_streak",
            target_attr="_peg_target",
        )

        if self.config.use_sim_guard:
            if tray_ok_sim is not None and not tray_ok_sim:
                self._force_tray_lost()
            elif peg_ok_sim is not None and not peg_ok_sim:
                self._force_peg_lost()

        phase = infer_subtask_phase(
            self.tray_ok,
            self.peg_ok,
            tray_prob=tray_prob,
            peg_prob=peg_prob,
            insert_min_prob=self.config.insert_min_prob,
        )
        return PhaseControllerState(
            tray_ok=self.tray_ok,
            peg_ok=self.peg_ok,
            subtask_phase=phase,
            tray_prob=tray_prob,
            peg_prob=peg_prob,
        )

    def _force_tray_lost(self) -> None:
        self.tray_ok = False
        self.peg_ok = False
        self._tray_streak = 0
        self._peg_streak = 0
        self._tray_target = None
        self._peg_target = None

    def _force_peg_lost(self) -> None:
        self.peg_ok = False
        self._peg_streak = 0
        self._peg_target = None

    def _update_flag(
        self,
        *,
        current: bool,
        prob: float,
        streak_attr: str,
        target_attr: str,
    ) -> bool:
        cfg = self.config
        if current:
            candidate = prob >= cfg.threshold_low
        else:
            candidate = prob >= cfg.threshold_high

        target = getattr(self, target_attr)
        streak = getattr(self, streak_attr)

        if candidate == current:
            setattr(self, target_attr, None)
            setattr(self, streak_attr, 0)
            return current

        if target != candidate:
            setattr(self, target_attr, candidate)
            setattr(self, streak_attr, 1)
        else:
            setattr(self, streak_attr, streak + 1)

        if getattr(self, streak_attr) >= cfg.confirm_frames:
            setattr(self, target_attr, None)
            setattr(self, streak_attr, 0)
            return candidate
        return current
