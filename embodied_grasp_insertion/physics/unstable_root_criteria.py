"""Absolute unstable/stable labeling for grasp roots (P0-C1.1).

Quantile-only rules are forbidden when they can collapse (e.g. retention_q=1.0).
"""

from __future__ import annotations

from typing import Any


DEFAULTS = {
    "retention_eps": 0.02,
    "retention_abs_max_for_unstable": 0.98,
    "load_drift_increase_m": 0.002,
    "load_slip_ratio": 1.25,
    "load_slip_increase_mps": 0.01,
}


def _f(m: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = m.get(key, default)
    return float(default if v is None else v)


def collect_unstable_reasons(
    hold: dict[str, Any],
    load: dict[str, Any],
    *,
    retention_eps: float = DEFAULTS["retention_eps"],
    retention_abs_max_for_unstable: float = DEFAULTS["retention_abs_max_for_unstable"],
    load_drift_increase_m: float = DEFAULTS["load_drift_increase_m"],
    load_slip_ratio: float = DEFAULTS["load_slip_ratio"],
    load_slip_increase_mps: float = DEFAULTS["load_slip_increase_mps"],
    # Optional secondary info only; never sole reason.
    hold_drift_threshold: float | None = None,
) -> list[str]:
    """Return absolute unstable reason codes. Empty => not unstable by absolute rules."""
    reasons: list[str] = []

    for branch_name, m in (("hold", hold), ("load", load)):
        ret = _f(m, "contact_retention_vs_root_mean", 1.0)
        # Forbid retention <= 1.0 style; require strict reduction below 1-eps and abs gate.
        if ret < (1.0 - float(retention_eps)) and ret < float(retention_abs_max_for_unstable):
            reasons.append(f"{branch_name}_reduced_retention")
        if int(m.get("peg_contact_absent_steps", 0) or 0) > 0:
            reasons.append(f"{branch_name}_peg_contact_absent_steps")

    hold_drift = _f(hold, "trans_drift_max_m")
    load_drift = _f(load, "trans_drift_max_m")
    # Strict > absolute load-vs-hold increase.
    if load_drift - hold_drift > float(load_drift_increase_m):
        reasons.append("load_drift_increase_vs_hold")

    hold_slip = _f(hold, "slip_proxy_tangential_rel_vel_mean_mps")
    load_slip = _f(load, "slip_proxy_tangential_rel_vel_mean_mps")
    if hold_slip > 1e-9 and load_slip > hold_slip * float(load_slip_ratio):
        reasons.append("load_slip_ratio_vs_hold")
    elif load_slip - hold_slip > float(load_slip_increase_mps):
        reasons.append("load_slip_increase_vs_hold")

    hold_peg = bool(hold.get("terminal_peg_ok", True))
    load_peg = bool(load.get("terminal_peg_ok", True))
    hold_drop = bool(hold.get("object_dropped_proxy", False))
    load_drop = bool(load.get("object_dropped_proxy", False))
    hold_contact_end = bool(hold.get("peg_contact_present_end", True))
    load_contact_end = bool(load.get("peg_contact_present_end", True))
    if hold_peg and not load_peg:
        reasons.append("load_terminal_peg_worse")
    if (not hold_drop) and load_drop:
        reasons.append("load_object_dropped_proxy")
    if hold_contact_end and not load_contact_end:
        reasons.append("load_peg_contact_lost_end")

    # Document elevated hold drift for audit, but do NOT treat as unstable alone.
    if hold_drift_threshold is not None and hold_drift > float(hold_drift_threshold):
        # Informative only — not appended to reasons.
        pass

    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def label_root(
    hold: dict[str, Any],
    load: dict[str, Any],
    *,
    root_contact_total: int,
    criteria: dict[str, Any] | None = None,
    hold_drift_threshold: float | None = None,
) -> dict[str, Any]:
    cfg = {**DEFAULTS, **(criteria or {})}
    reasons = collect_unstable_reasons(
        hold,
        load,
        retention_eps=float(cfg["retention_eps"]),
        retention_abs_max_for_unstable=float(cfg["retention_abs_max_for_unstable"]),
        load_drift_increase_m=float(cfg["load_drift_increase_m"]),
        load_slip_ratio=float(cfg["load_slip_ratio"]),
        load_slip_increase_mps=float(cfg["load_slip_increase_mps"]),
        hold_drift_threshold=hold_drift_threshold,
    )
    hold_ok = bool(hold.get("terminal_peg_ok", False)) and not bool(
        hold.get("object_dropped_proxy", False)
    )
    intervenable = int(root_contact_total) > 0 and hold_ok
    elevated_hold_drift_only = bool(
        hold_drift_threshold is not None
        and _f(hold, "trans_drift_max_m") > float(hold_drift_threshold)
        and not reasons
    )
    unstable = bool(reasons) and intervenable
    stable = intervenable and not reasons
    return {
        "unstable_reasons": reasons,
        "unstable_flag": unstable,
        "stable_control_flag": stable,
        "intervenable": intervenable,
        "elevated_hold_drift_only": elevated_hold_drift_only,
        "note": (
            "elevated hold drift alone is not unstable"
            if elevated_hold_drift_only
            else None
        ),
    }


def screening_gate(
    unstable: list[dict[str, Any]],
    stable: list[dict[str, Any]],
    *,
    min_unstable: int = 4,
    min_stable: int = 3,
    min_unstable_episodes: int = 3,
) -> dict[str, Any]:
    n_u = len(unstable)
    n_s = len(stable)
    eps = {int(r["episode_index"]) for r in unstable}
    missing_reasons = [r for r in unstable if not r.get("unstable_reasons")]
    ok = (
        n_u >= int(min_unstable)
        and n_s >= int(min_stable)
        and len(eps) >= int(min_unstable_episodes)
        and len(missing_reasons) == 0
    )
    return {
        "passed": ok,
        "label": "ok" if ok else "screening_fail",
        "n_unstable": n_u,
        "n_stable": n_s,
        "n_unstable_episodes": len(eps),
        "missing_reasons": len(missing_reasons),
        "checks": {
            "unstable_ge": n_u >= int(min_unstable),
            "stable_ge": n_s >= int(min_stable),
            "episodes_ge": len(eps) >= int(min_unstable_episodes),
            "all_have_reasons": len(missing_reasons) == 0,
        },
    }
