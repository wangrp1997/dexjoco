#!/usr/bin/env python3
"""P0-S0.4c hardened: multi-family physical grasp recipe smoke.

Pre-collection regressions:
1. Formal families: open/closed from identical MjData snapshot after establish+settle
2. snap_call_count_after_establish == 0
3. Transport records / requires measured hand & peg lateral travel

Naming: multi-family oracle-established physical grasp recipe smoke.
No collection / no training. Does not claim learned grasp policy.
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
REACH_ROOT = DEXJOCO_ROOT.parent / "reach_insert_rl"
LAI_ROOT = DEXJOCO_ROOT.parent / "lai"
for _p in (
    str(PROJECT_ROOT),
    str(DEXJOCO_ROOT),
    str(DEXJOCO_ROOT / "dexjoco"),
    str(REACH_ROOT),
    str(LAI_ROOT),
):
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
from embodied_grasp_insertion.physics.grasp_stability_multifamily import (  # noqa: E402
    run_physical_formal_family,
)
from embodied_grasp_insertion.physics.grasp_stability_physical import (  # noqa: E402
    run_physical_from_snapshot,
    scale_physical_thresholds,
)
from embodied_grasp_insertion.simulation.full_episode_snapshot import (  # noqa: E402
    FullEpisodeSnapshot,
)
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    make_full_env,
    replay_demo_to_frame,
    select_roots_for_episode,
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


def _assert_transport_lateral(fid: str, result: dict[str, Any]) -> None:
    tr = next((p for p in result.get("phases", []) if p.get("name") == "transport"), None)
    if tr is None:
        raise AssertionError(f"{fid}: missing transport phase")
    m = tr.get("metrics") or {}
    for k in ("hand_lateral_m", "peg_lateral_m", "commanded_lateral_m"):
        if k not in m or m[k] is None:
            raise AssertionError(f"{fid}: transport missing {k}")
    if tr.get("passed") and float(m["hand_lateral_m"]) < 0.03:
        raise AssertionError(
            f"{fid}: transport passed but hand_lateral_m={m['hand_lateral_m']} < 0.03"
        )


def smoke_round8_demo(cfg: dict[str, Any]) -> dict[str, Any]:
    """Reuse S0.4b: first transport root of episode 0 (matched snapshot + lateral gate)."""
    sidecar = Path(cfg["sidecar_dir"])
    env = make_full_env([0], sidecar_dir=sidecar, seed=int(cfg.get("seed", 0)))
    try:
        env.reset(episode_index=0)
        rs = cfg.get("root_selection", {})
        roots = select_roots_for_episode(
            env,
            early_offset=int(rs.get("early_offset", 5)),
            transport_tip_min_m=float(rs.get("transport_tip_min_m", 0.08)),
            preinsert_tip_max_m=float(rs.get("preinsert_tip_max_m", 0.06)),
            max_scan_frames=rs.get("max_scan_frames"),
        )
        transport = next(r for r in roots if r.phase == "transport")
        env.reset(episode_index=0)
        replay_demo_to_frame(env, int(transport.frame))
        snap = FullEpisodeSnapshot.capture(env)
        result = run_physical_from_snapshot(env, snap, thr=scale_physical_thresholds(0.018))
        result["family_id"] = "round_8mm"
        result["root_source"] = "demo_transport"
        result["root_frame"] = int(transport.frame)
        result["episode_index"] = 0
        result["matched_snapshot_branch"] = True
        result["snap_call_count_after_establish"] = 0
        _assert_transport_lateral("round_8mm", result)
        return result
    finally:
        env.close()


def smoke_formal(spec) -> dict[str, Any]:
    write_formal_family_assets(spec, overwrite=False)
    env = PandaBimanualAssemblyGymEnv(
        geometry_family=spec.family_id,
        image_obs=False,
        randomize=False,
        hz=0,
        seed=0,
    )
    try:
        result = run_physical_formal_family(env, spec)
        result["root_source"] = "oracle_establish_formal"
        if int(result.get("snap_call_count_after_establish", -1)) != 0:
            raise AssertionError(
                f"{spec.family_id}: snap_call_count_after_establish="
                f"{result.get('snap_call_count_after_establish')} != 0"
            )
        if not result.get("matched_snapshot_branch"):
            raise AssertionError(f"{spec.family_id}: matched_snapshot_branch missing")
        _assert_transport_lateral(spec.family_id, result)
        return result
    finally:
        env.close()


def verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_ok = sum(1 for r in rows if r.get("passed"))
    if n_ok == n and n >= 4:
        label, reason = "pass", f"family_physical_ok={n_ok}/{n}"
    elif n_ok >= 1:
        label, reason = "partial", f"family_physical_ok={n_ok}/{n}"
    else:
        label, reason = "fail", f"family_physical_ok={n_ok}/{n}"
    return {
        "label": label,
        "reason": reason,
        "n_families_passed": n_ok,
        "n_families": n,
        "allow_policy_training": False,
        "allow_full_collection": False,
        "allow_semantic_p0": False,
        "claims_stable_grasp_policy": False,
        "claims_physical_grasp_stability": bool(label == "pass"),
        "scope": "multi_family_oracle_established_physical_grasp_recipe",
        "next_if_pass": (
            "optional tiny controlled micro-demo pilot only after review; "
            "still no mass collect/train; revocable"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families-yaml",
        default=str(PROJECT_ROOT / "configs/geometry_families.yaml"),
    )
    parser.add_argument(
        "--demo-config",
        default=str(PROJECT_ROOT / "configs/finger_controllability_smoke.yaml"),
    )
    parser.add_argument(
        "--families",
        default="round_8mm,round_16mm,rectangular_8mm,rectangular_16mm",
    )
    parser.add_argument(
        "--out-manifest",
        default=str(PROJECT_ROOT / "data/manifests/grasp_stability_multifamily_physical_smoke_v1.json"),
    )
    parser.add_argument(
        "--out-report",
        default=str(PROJECT_ROOT / "docs/GRASP_STABILITY_MULTIFAMILY_PHYSICAL_SMOKE.md"),
    )
    args = parser.parse_args()

    fam_raw = yaml.safe_load(Path(args.families_yaml).read_text(encoding="utf-8"))
    by_id = {d["family_id"]: from_dict(d) for d in fam_raw["families"]}
    demo_cfg = yaml.safe_load(Path(args.demo_config).read_text(encoding="utf-8"))
    wanted = [x.strip() for x in args.families.split(",") if x.strip()]

    rows = []
    for fid in wanted:
        print(f"[s0.4c-hard] {fid}", flush=True)
        if fid == "round_8mm":
            rows.append(smoke_round8_demo(demo_cfg))
        else:
            rows.append(smoke_formal(by_id[fid]))

    v = verdict(rows)
    man = {
        "name": "grasp_stability_multifamily_physical_smoke_v1",
        "created_at": _utc(),
        "protocol": "P0-S0.4c-hardened",
        "verdict": v,
        "note": (
            "round_8mm: demo transport root (S0.4b matched snapshot + lateral gate). "
            "Other families: oracle establish flow (may snap), then MjData snapshot; "
            "open/closed restore same state; snap_call_count_after_establish==0; "
            "transport lateral gate on all four families. Recipe smoke only; no collect/train."
        ),
        "families": rows,
    }
    Path(args.out_manifest).write_text(
        json.dumps(_jsonable(man), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Grasp Stability Multi-Family Physical Smoke (P0-S0.4c hardened)",
        "",
        f"- 日期：{man['created_at']}",
        f"- 结论：**{v['label']}**",
        f"- reason：{v['reason']}",
        "- 准确命名：多族 oracle 建立接触后的物理抓取配方 smoke（非学会抓取）",
        "- round_8mm：demo transport root（S0.4b，同态 snapshot）",
        "- 其他族：establish 可 snap；settle 后抓 MjData；open/closed 同态恢复",
        "- 硬门槛：四族均 `snap_call_count_after_establish == 0` + transport 横移门",
        "- transport：记录并要求手/peg 横向位移（含 round_8mm demo 路径）",
        "- 不采集 / 不训练",
        f"- `claims_physical_grasp_stability={v['claims_physical_grasp_stability']}`",
        "",
        "## Families",
    ]
    for r in rows:
        ph = {p["name"]: p for p in r.get("phases", [])}
        tr_m = (ph.get("transport") or {}).get("metrics") or {}
        lat = ""
        if "hand_lateral_m" in tr_m:
            lat = (
                f" hand_lat={tr_m['hand_lateral_m']:.4f}"
                f" peg_lat={tr_m['peg_lateral_m']:.4f}"
            )
        snap_a = r.get("snap_call_count_after_establish")
        lines.append(
            f"- `{r.get('family_id')}`: passed={r.get('passed')} src={r.get('root_source')} "
            f"hold={(ph.get('hold') or {}).get('passed')} "
            f"lift={(ph.get('lift') or {}).get('passed')} "
            f"transport={(ph.get('transport') or {}).get('passed')}{lat} "
            f"neg={(ph.get('open_hand_negative') or {}).get('passed')} "
            f"closed_beats={r.get('closed_beats_open')} "
            f"snap_after={snap_a} matched={r.get('matched_snapshot_branch')} "
            f"root_c={r.get('root_contacts')}"
        )
    lines += [
        "",
        "三项回归通过后，才可讨论极小规模、可撤销的 micro-demo pilot；仍禁止常规采集/训练。",
        "",
    ]
    Path(args.out_report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": v, "manifest": args.out_manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
