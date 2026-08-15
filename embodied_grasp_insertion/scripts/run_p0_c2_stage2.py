#!/usr/bin/env python3
"""Run P0-C2 Stage-2 action-conditioned Ridge on existing S1b branch JSONs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from embodied_grasp_insertion.observability.c2_stage2_action_conditioned import (
    PROTOCOL,
    run_stage2,
)
from embodied_grasp_insertion.pilot import WRITE_IMPLEMENTATION_ENABLED

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BRANCHES = PROJECT / "outputs" / "p0_c2_s1b_v1" / "branches"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch-dir", type=Path, default=DEFAULT_BRANCHES)
    ap.add_argument(
        "--metrics-out",
        type=Path,
        default=PROJECT / "data" / "manifests" / "p0_c2_stage2_metrics.json",
    )
    ap.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT / "docs" / "P0_C2_STAGE2.md",
    )
    args = ap.parse_args()
    if WRITE_IMPLEMENTATION_ENABLED:
        raise SystemExit("WRITE_IMPLEMENTATION_ENABLED must stay False")

    result = run_stage2(args.branch_dir)
    result["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    v = result["verdict"]
    lines = [
        f"# P0-C2 Stage-2 Action-Conditioned H4 ({PROTOCOL})",
        "",
        f"- 日期：{result['created_at']}",
        f"- overall_verdict：**{v['overall_verdict']}**",
        f"- decision_tree：**{v['decision_tree']}**",
        f"- research_decision：**{v['research_decision']}**",
        f"- enter_stage3：{v['enter_stage3']}；allow_policy_training：{v['allow_policy_training']}",
        f"- 数据：仅 S1b matched-branch JSON（n_samples={result['n_samples']}）",
        f"- 目标：动作相对 hold 的有符号后果（不要求跨 root 同向）",
        f"- 模型：Ridge only；episode/root held-out；配对 bootstrap",
        "",
        "## 数据边界（重要）",
        "",
        f"- `{json.dumps(result['data_limits'], ensure_ascii=False)}`",
        "- wrist FT / root qdot / 预 root command history：**S1b 导出中不存在**，本轮标为 unavailable，不作假阴性。",
        "",
        "## Split",
        "",
        f"- train eps {result['split']['train_episodes']} roots {result['split']['train_root_ids']}",
        f"- val eps {result['split']['val_episodes']} roots {result['split']['val_root_ids']}",
        f"- test（S1b held-out）roots {result['split']['test_root_ids']}",
        "",
        "## Primary target `d_trans_drift_max_m`",
        "",
    ]
    pv = result["primary_verdict"]
    lines.append(f"- oracle_ok={pv['oracle_ok']}；deploy_qpos_ok={pv['deploy_qpos_ok']}")
    lines.append(f"- checks：{json.dumps(pv['checks'], ensure_ascii=False)}")
    lines.append("")
    lines.append("| condition | val MAE | test MAE | alpha |")
    lines.append("|---|---|---|---|")
    for cond, blob in result["by_target"]["d_trans_drift_max_m"]["conditions"].items():
        lines.append(
            f"| {cond} | {blob['splits']['val']['mae']:.5f} | "
            f"{blob['splits']['test']['mae']:.5f} | {blob.get('alpha')} |"
        )
    lines.extend(
        [
            "",
            "## All targets (decision branches)",
            "",
        ]
    )
    for yk, blob in result["by_target"].items():
        vv = blob["verdict"]
        lines.append(
            f"- **{yk}**: tree={vv['decision_tree_branch']}；"
            f"oracle_ok={vv['oracle_ok']}；deploy_qpos_ok={vv['deploy_qpos_ok']}"
        )
    lines.extend(
        [
            "",
            "## Stop rules applied",
            "",
            "- B：Oracle+action 预测不了 → 任务/标签无效，停止",
            "- C：Oracle 能、部署不能 → sensing gap",
            "- D：部署能区分动作后果 → H4 初步证据，仍不训策略",
            "- **不进入 Stage-3**；完成后等人决策",
            "",
            "## Retain",
            "",
            "- `outputs/p0_c2_s1b_v1/` 与 `outputs/p0_c2_stage1_v1/` 继续保留",
            "",
        ]
    )
    args.report_out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(v, indent=2, ensure_ascii=False))
    print("metrics", args.metrics_out)
    print("report", args.report_out)


if __name__ == "__main__":
    main()
