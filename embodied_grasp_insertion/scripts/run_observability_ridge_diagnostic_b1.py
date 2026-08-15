#!/usr/bin/env python3
"""Run P0-Obs-B1 future-drift falsification on D1 pack (no policy training)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from embodied_grasp_insertion.io_paths import path_for_manifest
from embodied_grasp_insertion.observability.ridge_diagnostic_b1 import (
    CONDITION_ORDER,
    PROTOCOL,
    run_b1_diagnostic,
)

DEFAULT_PACK = Path(
    "/mnt/hdd/dexjoco/datasets/embodied_grasp_insertion/observability_eval_v1"
)
PROJECT = Path(__file__).resolve().parents[1]


def _fmt(m: dict) -> str:
    return (
        f"tMAE={m['translation_mae_m']:.5f}m "
        f"rMAE={m['rotation_geodesic_mae_deg']:.3f}deg"
    )


def write_report(result: dict, path: Path) -> None:
    v = result["verdict"]
    lines = [
        f"# Observability Future-Drift Falsification ({PROTOCOL})",
        "",
        f"- 日期：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- overall_verdict：**{v['overall_verdict']}**",
        f"- research_decision：**{v['research_decision']}**",
        f"- pack：`{result['pack_root']}`；samples={result['n_samples']}；单几何；test ep=15",
        f"- 目标：从 t={result['target']['t_obs_index']} 预测未来 o2h 漂移 Δ∈{result['target']['deltas']}",
        f"- 正式判定：配对 episode bootstrap CI（平移+旋转均显著，且 val+test）",
        f"- oracle：仅目标帧之前的 privileged history（含 t，不含 t+Δ）",
        f"- any_deploy_real_signal={v['any_deploy_real_signal']}；any_ft_helps={v['any_ft_helps_claim']}",
        f"- claims_observability_p0_pass=false；allow_policy_training=false",
        "",
        "## B0 修正（本轮前提）",
        "",
        "- B0「稳定优于」仅点估计，偏强；旋转配对 CI 跨 0。",
        "- B0 ceiling 含目标帧，近零属必然，不能证明可学。",
        "- act44 含 wrist pose，同时刻 o2h 回归可能有运动学捷径。",
        "",
    ]
    for d, verd in result["verdicts_by_delta"].items():
        lines.append(f"## Δ={d}")
        lines.append("")
        lines.append(
            f"- verdict：`{verd['overall_verdict']}`；"
            f"real_signal={verd['deploy_real_signal'] or '无'}；"
            f"ft_helps={verd['ft_helps_claim']}"
        )
        lines.append("")
        lines.append("| condition | val | test |")
        lines.append("|---|---|---|")
        for name in CONDITION_ORDER:
            cond = result["by_delta"][d][name]
            lines.append(
                f"| {name} | {_fmt(cond['splits']['val']['metrics'])} | "
                f"{_fmt(cond['splits']['test']['metrics'])} |"
            )
        lines.append("")
        hist = verd["history_A_H8_vs_A_H1_test"]
        lines.append(
            f"- A_H8−A_H1 test 配对：trans mean={hist['translation']['mean_diff']:.5f} "
            f"CI[{hist['translation']['ci95_lo']:.5f},{hist['translation']['ci95_hi']:.5f}]；"
            f"rot mean={hist['rotation']['mean_diff']:.3f} "
            f"CI[{hist['rotation']['ci95_lo']:.3f},{hist['rotation']['ci95_hi']:.3f}]"
        )
        lines.append("")
        # Key FT pairs on test
        for key, blob in verd["ft_paired_comparisons"].items():
            if not key.endswith("_test"):
                continue
            lines.append(
                f"- {key}: trans CI[{blob['translation']['ci95_lo']:.5f},"
                f"{blob['translation']['ci95_hi']:.5f}] "
                f"sig={blob['translation']['significantly_better']}；"
                f"rot CI[{blob['rotation']['ci95_lo']:.3f},"
                f"{blob['rotation']['ci95_hi']:.3f}] "
                f"sig={blob['rotation']['significantly_better']}"
            )
        lines.append("")

    lines.extend(
        [
            "## 研究判断规则",
            "",
            "- 若 A/B 未能在未来漂移上、以配对 CI 击败全部代理基线 → **停止 A/B sensing 路线**。",
            "- 若 FT 仍不优于 shuffled-FT → 腕力时序信息未证实。",
            "- 不自动换复杂网络；不训策略；不宣称 Obs P0。",
            "",
            "## Guards",
            "",
            f"- {json.dumps(result['guards'], ensure_ascii=False)}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-root", type=Path, default=DEFAULT_PACK)
    ap.add_argument(
        "--metrics-out",
        type=Path,
        default=PROJECT / "data" / "manifests" / "observability_ridge_b1_metrics.json",
    )
    ap.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT / "docs" / "OBSERVABILITY_RIDGE_B1.md",
    )
    args = ap.parse_args()

    result = run_b1_diagnostic(args.pack_root)
    # Drop bulky per-episode before writing metrics.
    result.pop("_per_episode_by_delta", None)
    # Also drop nested paired_comparisons proxy detail if huge — keep FT + verdicts.
    # paired_comparisons kept for auditability (small).

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    result["metrics_path"] = path_for_manifest(args.metrics_out, project_root=PROJECT)
    result["report_path"] = path_for_manifest(args.report_out, project_root=PROJECT)
    args.metrics_out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    write_report(result, args.report_out)
    print(
        json.dumps(
            {
                "overall_verdict": result["verdict"]["overall_verdict"],
                "research_decision": result["verdict"]["research_decision"],
                "any_deploy_real_signal": result["verdict"]["any_deploy_real_signal"],
                "any_ft_helps_claim": result["verdict"]["any_ft_helps_claim"],
                "by_delta": {
                    d: {
                        "overall": v["overall_verdict"],
                        "signal": v["deploy_real_signal"],
                        "ft": v["ft_helps_claim"],
                    }
                    for d, v in result["verdicts_by_delta"].items()
                },
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
