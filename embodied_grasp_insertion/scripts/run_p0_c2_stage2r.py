#!/usr/bin/env python3
"""P0-C2 Stage-2R: one-shot privilege-complete action-conditioned Ridge."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT), str(PROJECT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.observability.c2_stage2r import PROTOCOL, run_stage2r  # noqa: E402
from embodied_grasp_insertion.pilot import WRITE_IMPLEMENTATION_ENABLED  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--branch-dir",
        type=Path,
        default=PROJECT / "outputs" / "p0_c2_s1b_v1" / "branches",
    )
    ap.add_argument(
        "--sidecar",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly"),
    )
    ap.add_argument(
        "--export-cache",
        type=Path,
        default=PROJECT / "outputs" / "p0_c2_s2r_v1" / "root_feature_export.json",
    )
    ap.add_argument(
        "--metrics-out",
        type=Path,
        default=PROJECT / "data" / "manifests" / "p0_c2_stage2r_metrics.json",
    )
    ap.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT / "docs" / "P0_C2_STAGE2R.md",
    )
    args = ap.parse_args()
    if WRITE_IMPLEMENTATION_ENABLED:
        raise SystemExit("WRITE_IMPLEMENTATION_ENABLED must stay False")

    result = run_stage2r(
        branch_dir=args.branch_dir,
        sidecar=args.sidecar,
        export_cache=args.export_cache,
    )
    result["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (args.export_cache.parent / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    (args.export_cache.parent / "RETAIN.md").write_text(
        "# Retain Stage-2R export cache and summary.\n", encoding="utf-8"
    )

    v = result["verdict"]
    lines = [
        f"# P0-C2 Stage-2R Privilege-Complete ({PROTOCOL})",
        "",
        f"- 日期：{result['created_at']}",
        f"- overall_verdict：**{v['overall_verdict']}**",
        f"- decision_tree：**{v['decision_tree']}**",
        f"- research_decision：**{v['research_decision']}**",
        f"- summary：{v.get('summary')}",
        f"- n_samples={result['n_samples']}；roots exported={result['n_roots_exported']}",
        "- 复用 S1b roots/outcomes；MjData 补导出 o2h pose/vel、finger q/qdot、wrist state/FT",
        "- Ridge only；禁止新采集/复杂网络/策略/触觉视觉/pilot",
        f"- enter_stage3={v['enter_stage3']}；allow_policy_training={v['allow_policy_training']}",
        "",
        "## Split",
        "",
        f"- {json.dumps(result['split'], ensure_ascii=False)}",
        "",
        "## Per-target",
        "",
    ]
    for yk, blob in result["by_target"].items():
        vv = blob["verdict"]
        lines.append(
            f"- **{yk}**: tree={vv['decision_tree_branch']} oracle_ok={vv['oracle_ok']} "
            f"q_ok={vv['deploy_q_ok']} qft_ok={vv['deploy_qft_ok']}"
        )
        lines.append("| condition | val MAE | test MAE |")
        lines.append("|---|---|---|")
        for cond, c in blob["conditions"].items():
            lines.append(
                f"| {cond} | {c['splits']['val']['mae']:.5f} | {c['splits']['test']['mae']:.5f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Stop",
            "",
            "- 本轮结束；等待人工决策；不自动扩展。",
            "",
        ]
    )
    args.report_out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(v, indent=2, ensure_ascii=False))
    print("report", args.report_out)
    print("metrics", args.metrics_out)


if __name__ == "__main__":
    main()
