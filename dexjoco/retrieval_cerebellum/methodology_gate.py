"""Irreversible project-route decision from held-out deployable perception gates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping


P1_METRICS = (
    "lateral_p90_m",
    "tilt_p90_rad",
    "depth_p90_m",
)
CONTINUE_ROUTE = "iekf_pbvs_candidate"
STOP_ROUTE = "intent_preserving_sensor_compliance"


@dataclass(frozen=True)
class MethodologyDecision:
    route: str
    stopped_candidate: bool
    metrics: Mapping[str, float]
    thresholds: Mapping[str, float]
    failed_metrics: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "stopped_candidate": self.stopped_candidate,
            "metrics": dict(self.metrics),
            "thresholds": dict(self.thresholds),
            "failed_metrics": list(self.failed_metrics),
            "reason": self.reason,
        }


def decide_methodology_route(summary: Mapping[str, object]) -> MethodologyDecision:
    try:
        local = summary["test"]["local"]
        thresholds = summary["p1_thresholds"]
    except (KeyError, TypeError) as error:
        raise ValueError("summary lacks held-out P1 local metrics or thresholds") from error
    if local is None:
        raise ValueError("summary has no held-out local P1 subset")
    metrics = {name: float(local[name]) for name in P1_METRICS}
    limits = {name: float(thresholds[name]) for name in P1_METRICS}
    if any(value < 0.0 for value in (*metrics.values(), *limits.values())):
        raise ValueError("P1 metrics and thresholds must be non-negative")
    failed = tuple(name for name in P1_METRICS if metrics[name] > limits[name])
    if failed:
        details = ", ".join(
            f"{name}={metrics[name]:.6g}>{limits[name]:.6g}" for name in failed
        )
        return MethodologyDecision(
            route=STOP_ROUTE,
            stopped_candidate=True,
            metrics=metrics,
            thresholds=limits,
            failed_metrics=failed,
            reason=(
                "held-out deployable perception failed the hard P1 gate; "
                f"stop IEKF-PBVS and switch route ({details})"
            ),
        )
    return MethodologyDecision(
        route=CONTINUE_ROUTE,
        stopped_candidate=False,
        metrics=metrics,
        thresholds=limits,
        failed_metrics=(),
        reason="all held-out deployable P1 precision gates passed",
    )


def decide_methodology_route_from_file(path: Path) -> MethodologyDecision:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    return decide_methodology_route(summary)


def write_methodology_decision(path: Path, decision: MethodologyDecision) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(decision.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
