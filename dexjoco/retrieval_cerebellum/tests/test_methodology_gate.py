import json
from pathlib import Path

from retrieval_cerebellum.methodology_gate import (
    CONTINUE_ROUTE,
    STOP_ROUTE,
    decide_methodology_route,
    decide_methodology_route_from_file,
    write_methodology_decision,
)


def _summary(lateral: float, tilt: float, depth: float):
    return {
        "test": {
            "local": {
                "lateral_p90_m": lateral,
                "tilt_p90_rad": tilt,
                "depth_p90_m": depth,
            }
        },
        "p1_thresholds": {
            "lateral_p90_m": 0.0012,
            "tilt_p90_rad": 0.03,
            "depth_p90_m": 0.003,
        },
    }


def test_any_failed_p1_metric_stops_iekf_pbvs_route():
    decision = decide_methodology_route(_summary(0.001, 0.031, 0.002))

    assert decision.stopped_candidate
    assert decision.route == STOP_ROUTE
    assert decision.failed_metrics == ("tilt_p90_rad",)


def test_all_p1_metrics_must_pass_to_continue_candidate():
    decision = decide_methodology_route(_summary(0.001, 0.02, 0.002))

    assert not decision.stopped_candidate
    assert decision.route == CONTINUE_ROUTE
    assert decision.failed_metrics == ()


def test_route_decision_round_trip_has_no_override(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(_summary(0.021, 0.2, 0.03)),
        encoding="utf-8",
    )
    decision = decide_methodology_route_from_file(summary_path)
    output = tmp_path / "decision.json"
    write_methodology_decision(output, decision)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["route"] == STOP_ROUTE
    assert payload["stopped_candidate"] is True
    assert set(payload["failed_metrics"]) == {
        "lateral_p90_m",
        "tilt_p90_rad",
        "depth_p90_m",
    }
    assert "override" not in payload
