"""Privileged truth evaluator for OpenPI rollout diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .control import CerebellumMode, CerebellumObservation, HandoffDecision
from .handoff import RuleBasedHandoffPolicy
from .observer import PrivilegedCerebellumObserver


@dataclass(frozen=True)
class MonitorConfig:
    log_interval: int = 30

    def __post_init__(self) -> None:
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")


class PrivilegedCerebellumEvaluator:
    """Classify phases from simulator truth without participating in control."""

    def __init__(
        self,
        raw_env,
        *,
        config: MonitorConfig | None = None,
        observer: PrivilegedCerebellumObserver | None = None,
        handoff_policy: RuleBasedHandoffPolicy | None = None,
    ) -> None:
        self.config = config or MonitorConfig()
        self.observer = observer or PrivilegedCerebellumObserver(raw_env)
        self.handoff_policy = handoff_policy or RuleBasedHandoffPolicy()
        self.reset(raw_env)

    def reset(self, raw_env) -> None:
        self.observer.reset(raw_env)
        self.handoff_policy.reset()
        self._last_mode: CerebellumMode | None = None
        self._mode_counts: Counter[str] = Counter()
        self._transitions: list[tuple[str, str]] = []
        self._trace: list[dict] = []
        self.last_observation: CerebellumObservation | None = None

    def observe(self, raw_env, state44: np.ndarray, *, timestamp: int) -> HandoffDecision:
        observation = self.observer.observe(raw_env, state44)
        self.last_observation = observation
        decision = self.handoff_policy.decide(observation)
        mode = decision.mode
        self._mode_counts[mode.value] += 1
        changed = mode is not self._last_mode
        if changed and self._last_mode is not None:
            self._transitions.append((self._last_mode.value, mode.value))

        primitives = observation.primitives
        diagnostics = self.observer.last_diagnostics
        row = {
            "timestamp": int(timestamp),
            "mode": mode.value,
            "reason": decision.reason,
            "confidence": float(decision.confidence),
            "peg_grasped": bool(observation.peg_grasped),
            "tray_grasped": bool(observation.tray_grasped),
            "peg_grasp_stable": bool(observation.peg_grasp_stable),
            "tray_grasp_stable": bool(observation.tray_grasp_stable),
            "peg_contact_count": int(observation.peg_contact_count),
            "tray_contact_count": int(observation.tray_contact_count),
            "insert_contact": bool(observation.insert_contact),
            "slip_speed_mps": observation.slip_speed_mps,
            "rotation_slip_radps": observation.rotation_slip_radps,
            "lateral_error_m": primitives.lateral_error_m,
            "axis_error_rad": primitives.axis_error_rad,
            "approach_height_m": primitives.approach_height_m,
            "insertion_depth_m": primitives.insertion_depth_m,
        }
        if diagnostics is not None:
            row["diagnostics"] = asdict(diagnostics)
        self._trace.append(row)

        if changed or timestamp % self.config.log_interval == 0:
            print(
                "  retrieval_cerebellum: "
                f"[t={timestamp:04d}] mode={mode.value} "
                f"peg={int(observation.peg_grasped)}/{int(observation.peg_grasp_stable)} "
                f"tray={int(observation.tray_grasped)}/{int(observation.tray_grasp_stable)} "
                f"slip={self._format_optional(observation.slip_speed_mps, scale=1000.0, suffix='mm/s')} "
                f"rot_slip={self._format_optional(observation.rotation_slip_radps, suffix='rad/s')} "
                f"lat={primitives.lateral_error_m * 1000.0:.1f}mm "
                f"axis={np.degrees(primitives.axis_error_rad):.1f}deg | {decision.reason}",
                flush=True,
            )
        self._last_mode = mode
        return decision

    def annotate_last(self, **values) -> None:
        if self._trace:
            self._trace[-1].update(values)

    def save_trace(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for row in self._trace:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def episode_summary(self) -> str:
        counts = ",".join(
            f"{mode}={count}" for mode, count in sorted(self._mode_counts.items())
        )
        final_mode = "none" if self._last_mode is None else self._last_mode.value
        return f"final={final_mode} transitions={len(self._transitions)} counts=[{counts}]"

    @staticmethod
    def _format_optional(value: float | None, *, scale: float = 1.0, suffix: str) -> str:
        if value is None:
            return "n/a"
        return f"{value * scale:.2f}{suffix}"


ReadOnlyCerebellumMonitor = PrivilegedCerebellumEvaluator
