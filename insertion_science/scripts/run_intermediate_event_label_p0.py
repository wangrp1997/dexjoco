#!/usr/bin/env python3
"""Read-only audit for statistically usable intermediate physical event labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "IntermediateEventLabelP0"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def event_value(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    value = True
    if spec.get("tip_min_max_m") is not None:
        value = value and float(row["tip_min_m"]) <= float(spec["tip_min_max_m"])
    if bool(spec.get("require_retained")):
        lost = bool(row.get("peg_lost_abort")) or str(row.get("term_reason")) == "peg_lost_abort"
        value = value and not lost
    return bool(value)


def stats(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    flags = [event_value(row, spec) for row in rows]
    event_rows = [row for row, flag in zip(rows, flags) if flag]
    other_rows = [row for row, flag in zip(rows, flags) if not flag]
    n = len(rows)
    prevalence = len(event_rows) / n if n else 0.0
    event_rate = (
        sum(bool(row["insert_ok"]) for row in event_rows) / len(event_rows)
        if event_rows
        else 0.0
    )
    other_rate = (
        sum(bool(row["insert_ok"]) for row in other_rows) / len(other_rows)
        if other_rows
        else 0.0
    )
    return {
        "n": n,
        "n_event": len(event_rows),
        "n_not_event": len(other_rows),
        "prevalence": prevalence,
        "insert_rate_event": event_rate,
        "insert_rate_not_event": other_rate,
        "insert_lift": event_rate - other_rate,
        "false_positive": sum(not bool(row["insert_ok"]) for row in event_rows),
        "false_negative": sum(bool(row["insert_ok"]) for row in other_rows),
    }


def split_pass(stat: dict[str, Any], cfg: dict[str, Any]) -> dict[str, bool]:
    return {
        "prevalence_ok": float(cfg["prevalence_min"])
        <= stat["prevalence"]
        <= float(cfg["prevalence_max"]),
        "groups_ok": stat["n_event"] >= int(cfg["min_group_n"])
        and stat["n_not_event"] >= int(cfg["min_group_n"]),
        "lift_ok": stat["insert_lift"] >= float(cfg["min_insert_lift_primary"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "intermediate_event_label_p0.yaml",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    primary_payload = json.loads(
        (PROJECT_ROOT / cfg["primary_results"]).read_text(encoding="utf-8")
    )
    primary_rows = [
        row
        for row in primary_payload["rows"]
        if str(row.get("kind")) not in ("none", "identity")
    ]
    by_split = {
        split: [row for row in primary_rows if str(row.get("split")) == split]
        for split in ("discovery", "held_out")
    }

    external_payload = json.loads(
        (PROJECT_ROOT / cfg["external_results"]).read_text(encoding="utf-8")
    )
    external_rows = list(external_payload["in_rows"]) + list(external_payload["out_rows"])

    reports = []
    for spec in cfg["events"]:
        split_stats = {split: stats(rows, spec) for split, rows in by_split.items()}
        split_checks = {
            split: split_pass(stat, cfg) for split, stat in split_stats.items()
        }
        pooled = stats(primary_rows, spec)
        external = stats(external_rows, spec)
        pooled_checks = {
            "false_positive_ok": (not bool(cfg["require_false_positive"]))
            or pooled["false_positive"] > 0,
            "false_negative_ok": (not bool(cfg["require_false_negative"]))
            or pooled["false_negative"] > 0,
            "external_lift_ok": external["insert_lift"]
            >= float(cfg["min_insert_lift_external"]),
            "external_groups_ok": external["n_event"] >= int(cfg["min_group_n"])
            and external["n_not_event"] >= int(cfg["min_group_n"]),
        }
        passed = all(all(checks.values()) for checks in split_checks.values()) and all(
            pooled_checks.values()
        )
        reports.append(
            {
                "name": spec["name"],
                "spec": spec,
                "split_stats": split_stats,
                "split_checks": split_checks,
                "pooled_primary": pooled,
                "external": external,
                "pooled_checks": pooled_checks,
                "pass": passed,
            }
        )

    passed_events = [report["name"] for report in reports if report["pass"]]
    if passed_events:
        verdict = "pass_intermediate_event_candidate_exists"
        decision = "allow_timestamp_causal_order_p0"
        reason = f"passed_events={passed_events}"
    else:
        verdict = "fail_no_stable_intermediate_event_label"
        decision = "stop_dense_intermediate_label_direction"
        reason = "no preregistered event passed split, non-copy, and external gates"
    judgment = {
        "verdict": verdict,
        "decision": decision,
        "reason": reason,
        "passed_events": passed_events,
        "n_primary": len(primary_rows),
        "n_discovery": len(by_split["discovery"]),
        "n_held_out": len(by_split["held_out"]),
        "n_external": len(external_rows),
    }

    output_dir = PROJECT_ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / cfg["report_path"]
    manifest_path = PROJECT_ROOT / cfg["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "utc": _utc(),
                "config": cfg,
                "events": reports,
                "judgment": judgment,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Intermediate Physical Event Label P0 — Result",
        "",
        f"- UTC: `{_utc()}`",
        f"- Protocol: `{PROTOCOL}`",
        f"- Verdict: `{verdict}`",
        f"- Decision: `{decision}`",
        f"- Reason: {reason}",
        f"- Data: primary=`{len(primary_rows)}` "
        f"(discovery={len(by_split['discovery'])}, held_out={len(by_split['held_out'])}), "
        f"external=`{len(external_rows)}`",
        "",
        "## Events",
        "",
    ]
    for report in reports:
        discovery = report["split_stats"]["discovery"]
        held_out = report["split_stats"]["held_out"]
        external = report["external"]
        lines.append(
            f"- `{report['name']}` pass=`{report['pass']}`; "
            f"discovery prev/lift=`{discovery['prevalence']:.3f}/{discovery['insert_lift']:.3f}`; "
            f"held_out=`{held_out['prevalence']:.3f}/{held_out['insert_lift']:.3f}`; "
            f"external lift=`{external['insert_lift']:.3f}`; "
            f"FP/FN=`{report['pooled_primary']['false_positive']}/"
            f"{report['pooled_primary']['false_negative']}`"
        )
    lines.extend(
        [
            "",
            "## Note",
            "",
            "本结果不复活 handoff 方向，不运行任何旧项目，只审计既有 insertion_science 输出。",
            "通过仅允许检查事件时间顺序与因果干预，不允许策略训练。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "utc": _utc(),
                "verdict": verdict,
                "decision": decision,
                "passed_events": passed_events,
                "results": str(results_path),
                "report": str(report_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(judgment, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
