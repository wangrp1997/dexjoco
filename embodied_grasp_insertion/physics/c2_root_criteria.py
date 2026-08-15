"""P0-C2 Stage-1 frozen root eligibility (no MuJoCo).

Selection must be decided before reading intervention outcomes.
Do not edit thresholds after seeing finger-branch results.
"""

from __future__ import annotations

from typing import Any

# Frozen before Stage-1 run. Changing these requires a new protocol id.
PROTOCOL = "P0-C2-S1"
CRITERIA_VERSION = "c2_root_v1"

# Root must already hold the peg.
MIN_ROOT_CONTACT_TOTAL = 1

# Hold-finger + shared-wrist screen: instability / excitation signs.
MIN_HOLD_TRANS_DRIFT_M = 0.002
MAX_HOLD_RETENTION_FOR_EXCITED = 0.95
MIN_PEG_CONTACT_ABSENT_STEPS = 1

# Still intervenable after hold screen (peg not fully lost under hold).
REQUIRE_HOLD_TERMINAL_PEG_OK = True
FORBID_HOLD_OBJECT_DROPPED = True

# Coverage caps (programmatic; not episode frame hardcodes).
MAX_ROOTS_TOTAL = 8
MAX_ROOTS_PER_EPISODE = 2
MIN_EPISODES_REQUIRED = 3


def root_excited_by_hold_screen(hold_metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (eligible_excitation, reason_codes) from hold-screen metrics only."""
    reasons: list[str] = []
    drift = float(hold_metrics.get("trans_drift_max_m", 0.0) or 0.0)
    ret = float(hold_metrics.get("contact_retention_vs_root_mean", 1.0) or 1.0)
    absent = int(hold_metrics.get("peg_contact_absent_steps", 0) or 0)
    if drift >= MIN_HOLD_TRANS_DRIFT_M:
        reasons.append("hold_trans_drift")
    if ret <= MAX_HOLD_RETENTION_FOR_EXCITED:
        reasons.append("hold_reduced_retention")
    if absent >= MIN_PEG_CONTACT_ABSENT_STEPS:
        reasons.append("hold_peg_contact_absent")
    return bool(reasons), reasons


def root_intervenable_after_hold(hold_metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    fails: list[str] = []
    if REQUIRE_HOLD_TERMINAL_PEG_OK and not bool(hold_metrics.get("terminal_peg_ok", False)):
        fails.append("hold_terminal_peg_lost")
    if FORBID_HOLD_OBJECT_DROPPED and bool(hold_metrics.get("object_dropped_proxy", False)):
        fails.append("hold_object_dropped")
    return (len(fails) == 0), fails


def accept_screened_root(
    *,
    root_contact_total: int,
    root_peg_ok: bool,
    root_insert_ok: bool,
    hold_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Pre-registered accept/reject for one candidate after hold screen only."""
    if not root_peg_ok or root_insert_ok:
        return {
            "accepted": False,
            "reasons": ["root_peg_or_insert_gate"],
            "excited_reasons": [],
        }
    if int(root_contact_total) < MIN_ROOT_CONTACT_TOTAL:
        return {
            "accepted": False,
            "reasons": ["root_no_contact"],
            "excited_reasons": [],
        }
    excited, excited_reasons = root_excited_by_hold_screen(hold_metrics)
    ok_hold, hold_fails = root_intervenable_after_hold(hold_metrics)
    if not excited:
        return {
            "accepted": False,
            "reasons": ["not_excited_under_hold_screen"],
            "excited_reasons": [],
        }
    if not ok_hold:
        return {
            "accepted": False,
            "reasons": hold_fails,
            "excited_reasons": excited_reasons,
        }
    return {
        "accepted": True,
        "reasons": ["accepted"],
        "excited_reasons": excited_reasons,
    }


def select_ranked_roots(
    candidates: list[dict[str, Any]],
    *,
    max_total: int = MAX_ROOTS_TOTAL,
    max_per_episode: int = MAX_ROOTS_PER_EPISODE,
) -> list[dict[str, Any]]:
    """Rank accepted roots by hold trans drift (desc); cap per-episode and total.

    `candidates` items must already include accept decision + hold_metrics.
    """
    accepted = [c for c in candidates if c.get("accepted")]
    accepted.sort(
        key=lambda c: float(c["hold_metrics"].get("trans_drift_max_m", 0.0) or 0.0),
        reverse=True,
    )
    picked: list[dict[str, Any]] = []
    per_ep: dict[int, int] = {}
    for c in accepted:
        ep = int(c["episode_index"])
        if per_ep.get(ep, 0) >= max_per_episode:
            continue
        picked.append(c)
        per_ep[ep] = per_ep.get(ep, 0) + 1
        if len(picked) >= max_total:
            break
    return picked
