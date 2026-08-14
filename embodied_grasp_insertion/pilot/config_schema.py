"""Strict config schema for micro-demo pilot v0 (YAML cannot loosen code caps)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from embodied_grasp_insertion.pilot import (
    MAX_EPISODES_PER_FAMILY,
    MAX_FAMILIES,
    MAX_HORIZON_STEPS,
    MAX_TOTAL_TRAJECTORIES,
    MAX_TRAJECTORIES_PER_EPISODE,
    MIN_HORIZON_STEPS,
)

ALLOWED_TOP_KEYS = frozenset(
    {
        "dry_run",
        "training_forbidden",
        "stop_on_first_gate_failure",
        "max_families",
        "families",
        "max_episodes_per_family",
        "max_trajectories_per_episode",
        "max_total_trajectories",
        "max_horizon_steps",
        "out_root",
        "gates",
    }
)

REQUIRED_GATES = (
    "physical_grasp",
    "target_hole_semantics",
    "insert_label_consistency",
    "require_matched_snapshot",
    "require_snap_after_establish_eq_0",
    "require_transport_lateral_fields",
)

ALLOWED_GATE_KEYS = frozenset(REQUIRED_GATES)


class PilotConfigError(ValueError):
    """Invalid pilot YAML / runtime config."""


def _require_bool(d: dict[str, Any], key: str) -> bool:
    if key not in d:
        raise PilotConfigError(f"missing required key: {key}")
    v = d[key]
    if type(v) is not bool:
        raise PilotConfigError(f"{key} must be bool, got {type(v).__name__}")
    return v


def _require_pos_int(
    d: dict[str, Any],
    key: str,
    *,
    cap: int,
    minimum: int = 1,
) -> int:
    if key not in d:
        raise PilotConfigError(f"missing required key: {key}")
    v = d[key]
    if type(v) is not int or isinstance(v, bool):
        raise PilotConfigError(f"{key} must be int, got {type(v).__name__}")
    if v < minimum:
        raise PilotConfigError(f"{key} must be >= {minimum}, got {v}")
    if v > cap:
        raise PilotConfigError(f"{key}={v} exceeds code hard cap {cap}")
    return v


@dataclass(frozen=True)
class ValidatedPilotConfig:
    dry_run: bool
    training_forbidden: bool
    stop_on_first_gate_failure: bool
    max_families: int
    families: tuple[str, ...]
    max_episodes_per_family: int
    max_trajectories_per_episode: int
    max_total_trajectories: int
    max_horizon_steps: int
    out_root: str
    gates: dict[str, bool]


def validate_pilot_config(raw: Any) -> ValidatedPilotConfig:
    """Validate YAML before any env/MuJoCo work. Raises PilotConfigError."""
    if not isinstance(raw, dict):
        raise PilotConfigError(f"config root must be mapping, got {type(raw).__name__}")

    unknown = set(raw.keys()) - ALLOWED_TOP_KEYS
    if unknown:
        raise PilotConfigError(f"unknown config keys: {sorted(unknown)}")

    dry_run = _require_bool(raw, "dry_run")
    training_forbidden = _require_bool(raw, "training_forbidden")
    if training_forbidden is not True:
        raise PilotConfigError("training_forbidden must be true in v0")
    stop = _require_bool(raw, "stop_on_first_gate_failure")
    if stop is not True:
        raise PilotConfigError("stop_on_first_gate_failure must be true in v0")

    max_families = _require_pos_int(raw, "max_families", cap=MAX_FAMILIES)
    families = raw.get("families")
    if not isinstance(families, list) or not families:
        raise PilotConfigError("families must be a non-empty list")
    if not all(isinstance(x, str) and x for x in families):
        raise PilotConfigError("families entries must be non-empty strings")
    if len(families) > max_families:
        raise PilotConfigError(
            f"len(families)={len(families)} exceeds config max_families={max_families}"
        )
    if len(families) > MAX_FAMILIES:
        raise PilotConfigError(
            f"len(families)={len(families)} exceeds code MAX_FAMILIES={MAX_FAMILIES}"
        )

    max_ep = _require_pos_int(raw, "max_episodes_per_family", cap=MAX_EPISODES_PER_FAMILY)
    max_tr_ep = _require_pos_int(
        raw, "max_trajectories_per_episode", cap=MAX_TRAJECTORIES_PER_EPISODE
    )
    max_total = _require_pos_int(raw, "max_total_trajectories", cap=MAX_TOTAL_TRAJECTORIES)
    max_horizon = _require_pos_int(
        raw, "max_horizon_steps", cap=MAX_HORIZON_STEPS, minimum=MIN_HORIZON_STEPS
    )

    out_root = raw.get("out_root")
    if not isinstance(out_root, str) or not out_root.strip():
        raise PilotConfigError("out_root must be a non-empty string")

    gates = raw.get("gates")
    if not isinstance(gates, dict):
        raise PilotConfigError("gates must be a mapping")
    unknown_g = set(gates.keys()) - ALLOWED_GATE_KEYS
    if unknown_g:
        raise PilotConfigError(f"unknown gate keys: {sorted(unknown_g)}")
    for g in REQUIRED_GATES:
        if g not in gates:
            raise PilotConfigError(f"missing required gate: {g}")
        if type(gates[g]) is not bool:
            raise PilotConfigError(f"gates.{g} must be bool")
        if gates[g] is not True:
            raise PilotConfigError(f"gates.{g} must be true in v0 (got false)")

    return ValidatedPilotConfig(
        dry_run=dry_run,
        training_forbidden=training_forbidden,
        stop_on_first_gate_failure=stop,
        max_families=max_families,
        families=tuple(families),
        max_episodes_per_family=max_ep,
        max_trajectories_per_episode=max_tr_ep,
        max_total_trajectories=max_total,
        max_horizon_steps=max_horizon,
        out_root=str(out_root).strip(),
        gates={k: True for k in REQUIRED_GATES},
    )


def plan_physical_horizon(max_horizon_steps: int) -> dict[str, int]:
    """Allocate hold/lift/transport/neg steps totaling <= max_horizon_steps."""
    if max_horizon_steps < MIN_HORIZON_STEPS:
        raise PilotConfigError(
            f"max_horizon_steps must be >= {MIN_HORIZON_STEPS}, got {max_horizon_steps}"
        )
    pos_budget = max(3, (max_horizon_steps * 3) // 5)
    neg_budget = max_horizon_steps - pos_budget
    hold = max(1, pos_budget // 5)
    lift = max(1, (pos_budget * 2) // 5)
    transport = max(1, pos_budget - hold - lift)
    neg_steps = max(1, neg_budget // 2)
    total = hold + lift + transport + neg_steps + neg_steps
    while total > max_horizon_steps and transport > 1:
        transport -= 1
        total -= 1
    while total > max_horizon_steps and lift > 1:
        lift -= 1
        total -= 1
    while total > max_horizon_steps and hold > 1:
        hold -= 1
        total -= 1
    while total > max_horizon_steps and neg_steps > 1:
        neg_steps -= 1
        total = hold + lift + transport + neg_steps + neg_steps
    if total > max_horizon_steps:
        raise PilotConfigError(
            f"cannot fit physical phases into max_horizon_steps={max_horizon_steps}"
        )
    return {
        "hold_steps": hold,
        "lift_steps": lift,
        "transport_steps": transport,
        "neg_steps": neg_steps,
        "planned_total_steps": hold + lift + transport + 2 * neg_steps,
    }
