#!/usr/bin/env python3
"""P0-S0.4: grasp stability *instrumentation / phase-orchestration* smoke.

lift / hold / transport under oracle kinematic palm-snap + open-hand negative.
NOT the physical grasp stability gate (see P0-S0.4b).
Does NOT claim learned or physical grasp. No collection / no training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEXJOCO_ROOT = PROJECT_ROOT.parent
for _p in (str(PROJECT_ROOT), str(DEXJOCO_ROOT), str(DEXJOCO_ROOT / "dexjoco")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from dexjoco.sim.envs.panda_bimanual_assembly_env import (  # noqa: E402
    PandaBimanualAssemblyGymEnv,
)
from embodied_grasp_insertion.geometry.family_spec import from_dict  # noqa: E402
from embodied_grasp_insertion.geometry.formal_xml_builder import (  # noqa: E402
    write_formal_family_assets,
)
from embodied_grasp_insertion.physics.grasp_stability import (  # noqa: E402
    run_lift_hold_transport,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def smoke_family(spec) -> dict[str, Any]:
    write_formal_family_assets(spec, overwrite=False)
    env = PandaBimanualAssemblyGymEnv(
        geometry_family=spec.family_id,
        image_obs=False,
        randomize=False,
        hz=0,
        seed=0,
    )
    try:
        result = run_lift_hold_transport(env, spec)
        result["family_id"] = spec.family_id
        result["section"] = spec.section
        result["nominal_size_mm"] = spec.nominal_size_mm
    finally:
        env.close()
    return result


def verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_ok = sum(1 for r in rows if r.get("passed"))
    if n_ok == n and n >= 4:
        label = "pass"
        reason = f"all {n} representative families lift/hold/transport+neg ok"
    elif n_ok >= 1:
        label = "partial"
        reason = f"family_ok={n_ok}/{n}"
    else:
        label = "fail"
        reason = f"family_ok={n_ok}/{n}"
    return {
        "label": label,
        "reason": reason,
        "n_families_passed": n_ok,
        "n_families": n,
        "allow_policy_training": False,
        "allow_full_collection": False,
        "allow_semantic_p0": False,
        "claims_stable_grasp_policy": False,
        "claims_physical_grasp_stability": False,
        "positive_fixture": "oracle_kinematic_palm_snap",
        "scope": "instrumentation_and_phase_orchestration_only",
        "next_if_pass": "P0-S0.4b physical grasp gate (still no train/collect)",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families-yaml",
        default=str(PROJECT_ROOT / "configs/geometry_families.yaml"),
    )
    parser.add_argument(
        "--families",
        default="round_8mm,round_16mm,rectangular_8mm,rectangular_16mm",
    )
    parser.add_argument(
        "--out-manifest",
        default=str(PROJECT_ROOT / "data/manifests/grasp_stability_smoke_v1.json"),
    )
    parser.add_argument(
        "--out-report",
        default=str(PROJECT_ROOT / "docs/GRASP_STABILITY_SMOKE.md"),
    )
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.families_yaml).read_text(encoding="utf-8"))
    by_id = {d["family_id"]: from_dict(d) for d in raw["families"]}
    wanted = [x.strip() for x in args.families.split(",") if x.strip()]
    specs = [by_id[fid] for fid in wanted]

    rows = []
    for spec in specs:
        print(f"[grasp-stab] {spec.family_id}", flush=True)
        rows.append(smoke_family(spec))

    v = verdict(rows)
    man = {
        "name": "grasp_stability_smoke_v1",
        "created_at": _utc(),
        "protocol": "P0-S0.4",
        "verdict": v,
        "scope": "grasp_stability_metrics_with_oracle_fixture_and_open_hand_negative",
        "note": (
            "Positive lift/hold/transport use oracle kinematic palm-snap fixture "
            "(not a learned/physical grasp policy). Open-hand negative verifies metrics. "
            "Thresholds scale with family characteristic length."
        ),
        "families": rows,
    }
    Path(args.out_manifest).write_text(
        json.dumps(_jsonable(man), indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "# Grasp Stability Smoke (P0-S0.4)",
        "",
        f"- 日期：{man['created_at']}",
        "- 结论：instrumentation / 阶段编排 smoke = **pass**；物理抓取门未关闭",
        "- 正对照：oracle 运动学 palm-snap 夹具（每步重设 peg；非摩擦抓取）",
        "- 负对照：关 snap 后自由落体（证明掉落指标，不证明开手相对闭手）",
        "- 阈值按 family 特征长度缩放；不采集 / 不训练",
        "- `claims_physical_grasp_stability=false`",
        "- 下一门：P0-S0.4b 纯动力学物理抓取门",
        "",
        "## Families",
    ]
    for r in rows:
        ph = {p["name"]: p["passed"] for p in r.get("phases", [])}
        lines.append(
            f"- `{r['family_id']}`: passed={r.get('passed')} "
            f"lift={ph.get('lift')} hold={ph.get('hold')} "
            f"transport={ph.get('transport')} neg={ph.get('open_hand_negative')} "
            f"L={r.get('thresholds', {}).get('char_len_m')}"
        )
    lines += ["", "不声称抓取策略已稳定。", ""]
    Path(args.out_report).write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": v, "manifest": args.out_manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
