#!/usr/bin/env python3
"""Run P0-Obs-B0 minimal Ridge diagnostic on the D1 eval pack (no policy training)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from embodied_grasp_insertion.io_paths import path_for_manifest
from embodied_grasp_insertion.observability.ridge_diagnostic_b0 import (
    PROTOCOL,
    run_b0_diagnostic,
)

DEFAULT_PACK = Path(
    "/mnt/hdd/dexjoco/datasets/embodied_grasp_insertion/observability_eval_v1"
)
PROJECT = Path(__file__).resolve().parents[1]


def _fmt_m(m: dict) -> str:
    return (
        f"tMAE={m['translation_mae_m']:.4f}m "
        f"tRMSE={m['translation_rmse_m']:.4f}m "
        f"rMAE={m['rotation_geodesic_mae_deg']:.2f}deg"
    )


def write_report(result: dict, report_path: Path) -> None:
    v = result["verdict"]
    lines = [
        f"# Observability Minimal Ridge Diagnostic ({PROTOCOL})",
        "",
        f"- 日期：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- overall_verdict：**{v['overall_verdict']}**",
        f"- research_decision：**{v['research_decision']}**",
        f"- pack：`{result['pack_root']}`",
        f"- samples：{result['n_samples']}；target = primary H={result['target']['primary_h']} 窗末帧 o2h",
        f"- 模型：Ridge only；alpha 网格 {result['alpha_grid']}；val 选 alpha，test 评一次",
        f"- episode 等权；bootstrap n={result['bootstrap']['n']}",
        f"- claims_observability_p0_pass={v['claims_observability_p0_pass']}",
        f"- allow_policy_training={v['allow_policy_training']}",
        f"- 稳定优于 train-mean：{v['stable_better_than_train_mean'] or '无'}",
        f"- FT 有效声称：{v['ft_helps_claim']}",
        "",
        "## Val / Test（episode-equal）",
        "",
        "| condition | val | test | alpha_t / alpha_r |",
        "|---|---|---|---|",
    ]
    for name, cond in result["conditions"].items():
        vm = cond["splits"]["val"]["metrics"]
        tm = cond["splits"]["test"]["metrics"]
        at = cond.get("alpha_t")
        ar = cond.get("alpha_r")
        lines.append(
            f"| {name} | {_fmt_m(vm)} | {_fmt_m(tm)} | {at} / {ar} |"
        )
    lines.extend(
        [
            "",
            "## Bootstrap CI (test, 95%)",
            "",
        ]
    )
    for name, cond in result["conditions"].items():
        ci = cond["splits"]["test"]["bootstrap_ci95"]
        lines.append(
            f"- **{name}**: "
            f"tMAE [{ci['translation_mae_m']['ci95_lo']:.4f}, {ci['translation_mae_m']['ci95_hi']:.4f}]；"
            f"rMAE [{ci['rotation_geodesic_mae_deg']['ci95_lo']:.2f}, "
            f"{ci['rotation_geodesic_mae_deg']['ci95_hi']:.2f}] deg"
        )
    lines.extend(
        [
            "",
            "## Guards",
            "",
            f"- {json.dumps(result['guards'], ensure_ascii=False)}",
            "",
            "## Next",
            "",
            "- 本轮结束；不自动进入下一模型。",
            "- 按 research_decision 由人决定继续或停止。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-root", type=Path, default=DEFAULT_PACK)
    ap.add_argument(
        "--metrics-out",
        type=Path,
        default=PROJECT / "data" / "manifests" / "observability_ridge_b0_metrics.json",
    )
    ap.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT / "docs" / "OBSERVABILITY_RIDGE_B0.md",
    )
    args = ap.parse_args()

    result = run_b0_diagnostic(args.pack_root)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    slim = {
        **result,
        "metrics_path": path_for_manifest(args.metrics_out, project_root=PROJECT),
        "report_path": path_for_manifest(args.report_out, project_root=PROJECT),
    }
    # Drop bulky nothing — conditions already compact.
    args.metrics_out.write_text(json.dumps(slim, indent=2, ensure_ascii=False) + "\n")
    write_report(result, args.report_out)
    print(
        json.dumps(
            {
                "overall_verdict": result["verdict"]["overall_verdict"],
                "research_decision": result["verdict"]["research_decision"],
                "stable_better_than_train_mean": result["verdict"]["stable_better_than_train_mean"],
                "ft_helps_claim": result["verdict"]["ft_helps_claim"],
                "claims_observability_p0_pass": False,
                "allow_policy_training": False,
                "metrics": str(args.metrics_out),
                "report": str(args.report_out),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
