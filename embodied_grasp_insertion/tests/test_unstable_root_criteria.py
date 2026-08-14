"""Unit checks for P0-C1.1 unstable root labeling (no MuJoCo)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEX = _ROOT.parent
for _p in (str(_DEX), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embodied_grasp_insertion.physics.unstable_root_criteria import (
    collect_unstable_reasons,
    label_root,
    screening_gate,
)


def _metrics(**kw):
    base = {
        "contact_retention_vs_root_mean": 1.0,
        "peg_contact_absent_steps": 0,
        "trans_drift_max_m": 0.005,
        "slip_proxy_tangential_rel_vel_mean_mps": 0.02,
        "terminal_peg_ok": True,
        "object_dropped_proxy": False,
        "peg_contact_present_end": True,
    }
    base.update(kw)
    return base


def test_retention_1_not_unstable_by_retention():
    hold = _metrics(contact_retention_vs_root_mean=1.0)
    load = _metrics(contact_retention_vs_root_mean=1.0, trans_drift_max_m=0.0051)
    reasons = collect_unstable_reasons(hold, load)
    assert "hold_reduced_retention" not in reasons
    assert "load_reduced_retention" not in reasons


def test_retention_097_triggers():
    hold = _metrics(contact_retention_vs_root_mean=0.97)
    load = _metrics()
    reasons = collect_unstable_reasons(hold, load)
    assert "hold_reduced_retention" in reasons


def test_drift_equal_threshold_not_load_increase():
    hold = _metrics(trans_drift_max_m=0.010)
    load = _metrics(trans_drift_max_m=0.012)  # +0.002 exactly, need strict >
    reasons = collect_unstable_reasons(hold, load, load_drift_increase_m=0.002)
    assert "load_drift_increase_vs_hold" not in reasons
    load2 = _metrics(trans_drift_max_m=0.012001)
    reasons2 = collect_unstable_reasons(hold, load2, load_drift_increase_m=0.002)
    assert "load_drift_increase_vs_hold" in reasons2


def test_elevated_hold_drift_alone_not_unstable():
    hold = _metrics(trans_drift_max_m=0.02)
    load = _metrics(trans_drift_max_m=0.02)
    lab = label_root(hold, load, root_contact_total=4, hold_drift_threshold=0.009)
    assert lab["unstable_flag"] is False
    assert lab["stable_control_flag"] is True
    assert lab["elevated_hold_drift_only"] is True
    assert lab["unstable_reasons"] == []


def test_reasons_array_required_for_unstable():
    hold = _metrics()
    load = _metrics(peg_contact_absent_steps=2)
    lab = label_root(hold, load, root_contact_total=3)
    assert lab["unstable_flag"] is True
    assert lab["unstable_reasons"]
    assert "load_peg_contact_absent_steps" in lab["unstable_reasons"]


def test_screening_gate_requires_stable():
    unstable = [
        {"episode_index": 0, "unstable_reasons": ["a"]},
        {"episode_index": 1, "unstable_reasons": ["a"]},
        {"episode_index": 2, "unstable_reasons": ["a"]},
        {"episode_index": 3, "unstable_reasons": ["a"]},
    ]
    g = screening_gate(unstable, [], min_unstable=4, min_stable=3, min_unstable_episodes=3)
    assert g["passed"] is False
    assert g["label"] == "screening_fail"
    stable = [{"episode_index": i} for i in range(3)]
    g2 = screening_gate(unstable, stable, min_unstable=4, min_stable=3, min_unstable_episodes=3)
    assert g2["passed"] is True


if __name__ == "__main__":
    test_retention_1_not_unstable_by_retention()
    test_retention_097_triggers()
    test_drift_equal_threshold_not_load_increase()
    test_elevated_hold_drift_alone_not_unstable()
    test_reasons_array_required_for_unstable()
    test_screening_gate_requires_stable()
    print("ok")
