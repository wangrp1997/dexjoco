#!/usr/bin/env python3
"""P0-S0.4b: physical grasp stability gate (demo transport roots).

Single restore oracle → pure-dynamics hold/lift/transport + open-hand negative.
No per-step snap. No collection / no training.
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


def smoke_episode(ep: int, cfg: dict[str, Any]) -> dict[str, Any]:
    sidecar = Path(cfg["sidecar_dir"])
    env = make_full_env([ep], sidecar_dir=sidecar, seed=int(cfg.get("seed", 0)))
    try:
        env.reset(episode_index=ep)
        rs = cfg.get("root_selection", {})
        roots = select_roots_for_episode(
            env,
            early_offset=int(rs.get("early_offset", 5)),
            transport_tip_min_m=float(rs.get("transport_tip_min_m", 0.08)),
            preinsert_tip_max_m=float(rs.get("preinsert_tip_max_m", 0.06)),
            max_scan_frames=rs.get("max_scan_frames"),
        )
        transport = next((r for r in roots if r.phase == "transport"), None)
        if transport is None:
            return {
                "episode_index": ep,
                "passed": False,
                "error": "no_transport_root",
            }
        env.reset(episode_index=ep)
        replay_demo_to_frame(env, int(transport.frame))
        # Ensure enough remaining steps for gate (~120).
        remaining = int(env.max_episode_steps) - int(env._t)
        if remaining < 130:
            return {
                "episode_index": ep,
                "passed": False,
                "error": f"insufficient_remaining_steps={remaining}",
                "frame": int(transport.frame),
                "max_episode_steps": int(env.max_episode_steps),
            }
        snap = FullEpisodeSnapshot.capture(env)
        # round_8mm demo peg characteristic length ~ radius
        thr = scale_physical_thresholds(0.018)
        result = run_physical_from_snapshot(env, snap, thr=thr)
        result["episode_index"] = ep
        result["root_frame"] = int(transport.frame)
        result["root_phase"] = "transport"
        result["geometry_family_id"] = "round_8mm"
        result["remaining_steps_at_root"] = remaining
        return result
    finally:
        env.close()


def verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if "phases" in r]
    n_ok = sum(1 for r in usable if r.get("passed"))
    n = len(usable)
    if n == 0:
        label, reason = "fail", "no_usable_roots"
    elif n_ok == n and n >= 1:
        label, reason = "pass", f"physical_roots_ok={n_ok}/{n}"
    elif n_ok >= 1:
        label, reason = "partial", f"physical_roots_ok={n_ok}/{n}"
    else:
        label, reason = "fail", f"physical_roots_ok={n_ok}/{n}"
    return {
        "label": label,
        "reason": reason,
        "n_roots_passed": n_ok,
        "n_roots": n,
        "allow_policy_training": False,
        "allow_full_collection": False,
        "allow_semantic_p0": False,
        "claims_stable_grasp_policy": False,
        "claims_physical_grasp_stability": bool(label == "pass"),
        "fixture": "demo_transport_root_restore_once",
        "scope": "physical_dynamics_after_single_demo_root_restore",
        "not_claimed": "formal_multi_family_scripted_physical_grasp",
        "next_if_pass": "still no train/collect; optional multi-family physical recipes",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/finger_controllability_smoke.yaml"),
    )
    parser.add_argument("--episodes", default="0,2,4")
    parser.add_argument(
        "--out-manifest",
        default=str(PROJECT_ROOT / "data/manifests/grasp_stability_physical_smoke_v1.json"),
    )
    parser.add_argument(
        "--out-report",
        default=str(PROJECT_ROOT / "docs/GRASP_STABILITY_PHYSICAL_SMOKE.md"),
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    episodes = [int(x) for x in args.episodes.split(",") if x.strip()]
    rows = []
    for ep in episodes:
        print(f"[grasp-phys] episode={ep}", flush=True)
        rows.append(smoke_episode(ep, cfg))

    v = verdict(rows)
    man = {
        "name": "grasp_stability_physical_smoke_v1",
        "created_at": _utc(),
        "protocol": "P0-S0.4b",
        "verdict": v,
        "scope": "physical_grasp_gate_demo_transport_roots",
        "note": (
            "Oracle = single FullEpisodeSnapshot restore at demo transport root. "
            "No per-step snap/weld/teleport during phases. "
            "Closed hold/lift/transport under dynamics; open-hand from same root must be worse. "
            "Does not claim learned grasp or formal multi-family scripted physical grasp."
        ),
        "roots": rows,
    }
    Path(args.out_manifest).write_text(
        json.dumps(_jsonable(man), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Grasp Stability Physical Smoke (P0-S0.4b)",
        "",
        f"- 日期：{man['created_at']}",
        f"- 结论：**{v['label']}**",
        f"- reason：{v['reason']}",
        "- 范围：demo transport root 单次 restore 后的**纯动力学** hold/lift/transport",
        "- 负对照：同 root open-hand，且闭手必须显著优于开手",
        "- 无逐步 snap/weld；不采集 / 不训练",
        f"- `claims_physical_grasp_stability={v['claims_physical_grasp_stability']}`",
        "- 不声称：学会的抓取策略、正式多族脚本物理抓取",
        "",
        "## Roots",
    ]
    for r in rows:
        if "phases" not in r:
            lines.append(f"- ep{r.get('episode_index')}: error={r.get('error')}")
            continue
        ph = {p["name"]: p["passed"] for p in r.get("phases", [])}
        lines.append(
            f"- ep{r['episode_index']} f{r.get('root_frame')}: passed={r.get('passed')} "
            f"hold={ph.get('hold')} lift={ph.get('lift')} transport={ph.get('transport')} "
            f"neg={ph.get('open_hand_negative')} closed_beats_open={r.get('closed_beats_open')} "
            f"root_c={r.get('root_contacts')}"
        )
    lines += ["", "相对 S0.4 instrumentation：本门才是物理抓取稳定性门。", ""]
    Path(args.out_report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": v, "manifest": args.out_manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
