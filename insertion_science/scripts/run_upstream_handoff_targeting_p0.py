#!/usr/bin/env python3
"""Upstream Handoff Targeting P0: skill_replay handoff vs recoverable basin.

Final death gate for the handoff research line. No training. No force-demo.
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
for _p in (
    str(PROJECT_ROOT),
    str(DEXJOCO_ROOT),
    str(DEXJOCO_ROOT / "dexjoco"),
    str(DEXJOCO_ROOT.parent / "reach_insert_rl"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROTOCOL = "UpstreamHandoffTargetingP0"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _patch_execute_peg_lift_compat() -> None:
    """run_hybrid_insert passes for_insert=; current lift API may not accept it."""
    import interaction_retarget.grasp.lift as lift_mod

    orig = lift_mod.execute_peg_lift

    def _wrapped(*args, for_insert=None, **kwargs):  # noqa: ARG001
        return orig(*args, **kwargs)

    lift_mod.execute_peg_lift = _wrapped  # type: ignore[assignment]


def _install_handoff_capture(store: list[dict[str, Any]]) -> None:
    """Capture tip/lat/along/axis right after hybrid handoff force/confirm."""
    import interaction_retarget.skill_replay.insert as insert_mod
    from reach_insert_rl.env.full_obs import privileged_full_features
    from reach_insert_rl.env.obs import privileged_geom_features

    orig = insert_mod._try_force_handoff

    def _wrapped(env, raw_env, insert_runner, labeler, *, cfg, peg_rest_z):
        ok = orig(env, raw_env, insert_runner, labeler, cfg=cfg, peg_rest_z=peg_rest_z)
        try:
            feat = privileged_full_features(raw_env)
        except Exception:
            feat = privileged_geom_features(raw_env)
        active = bool(insert_runner.controller and insert_runner.active)
        store.append(
            {
                "force_ok": bool(ok),
                "controller_active": active,
                "tip_m": float(feat["tip_dist"]),
                "lat_m": float(feat["lat_err"]),
                "along_m": float(feat["along"]),
                "axis_err_rad": float(feat["axis_err"]),
            }
        )
        return ok

    insert_mod._try_force_handoff = _wrapped  # type: ignore[assignment]

    # Also capture when approach activates controller without force.
    orig_approach = insert_mod._privileged_approach_to_handoff

    def _approach_wrapped(env, raw_env, insert_runner, labeler, *, cfg, max_steps=900):
        orig_approach(env, raw_env, insert_runner, labeler, cfg=cfg, max_steps=max_steps)
        if insert_runner.controller is not None and insert_runner.active:
            if store and store[-1].get("controller_active"):
                return
            try:
                feat = privileged_full_features(raw_env)
            except Exception:
                feat = privileged_geom_features(raw_env)
            store.append(
                {
                    "force_ok": False,
                    "controller_active": True,
                    "tip_m": float(feat["tip_dist"]),
                    "lat_m": float(feat["lat_err"]),
                    "along_m": float(feat["along"]),
                    "axis_err_rad": float(feat["axis_err"]),
                    "source": "approach_active",
                }
            )

    insert_mod._privileged_approach_to_handoff = _approach_wrapped  # type: ignore[assignment]


def load_roots_and_limits(boundary_path: Path) -> tuple[list[dict], dict[str, float], dict[str, float]]:
    payload = json.loads(boundary_path.read_text(encoding="utf-8"))
    roots = list(payload.get("success_roots") or [])
    limits = {k: float(v) for k, v in (payload.get("limits") or {}).items()}
    base = {k: float(v) for k, v in (payload.get("config", {}).get("base") or {}).items()}
    return roots, limits, base


def nearest_root(feat: dict, roots: list[dict]) -> tuple[dict, float]:
    fv = np.asarray([feat["tip_m"], feat["lat_m"], feat["along_m"]], dtype=np.float64)
    best = None
    best_d = float("inf")
    for r in roots:
        rv = np.asarray([r["tip_m"], r["lat_m"], r["along_m"]], dtype=np.float64)
        d = float(np.linalg.norm(fv - rv))
        if d < best_d:
            best_d = d
            best = r
    assert best is not None
    return best, best_d


def in_basin(feat: dict, root: dict, *, base: dict, limits: dict) -> tuple[bool, dict[str, float], list[str]]:
    scales = {
        "s_lat": abs(float(feat["lat_m"]) - float(root["lat_m"])) / float(base["tip_lat_m"]),
        "s_along": abs(float(feat["along_m"]) - float(root["along_m"])) / float(base["tip_along_m"]),
        "s_axis": abs(float(feat["axis_err_rad"]) - float(root["axis_err_rad"]))
        / float(base["axis_rad"]),
    }
    outside = []
    if scales["s_lat"] > float(limits.get("tip_lat", 0.0)) + 1e-9:
        outside.append("tip_lat")
    if scales["s_along"] > float(limits.get("tip_along", 0.0)) + 1e-9:
        outside.append("tip_along")
    if scales["s_axis"] > float(limits.get("axis", 0.0)) + 1e-9:
        outside.append("axis")
    return len(outside) == 0, scales, outside


def judge(rows: list[dict], cfg: dict) -> dict[str, Any]:
    n = len(rows)
    n_handoff = sum(1 for r in rows if r.get("reached_handoff"))
    n_basin = sum(1 for r in rows if r.get("basin_hit"))
    n_basin_ins = sum(1 for r in rows if r.get("basin_hit") and r.get("insert_success"))
    handoff_rate = n_handoff / n if n else 0.0
    basin_hit_rate = n_basin / n_handoff if n_handoff else 0.0
    basin_insert_rate = n_basin_ins / n_basin if n_basin else 0.0
    checks = {
        "handoff_rate_ok": handoff_rate >= float(cfg["min_handoff_rate"]),
        "basin_hit_rate_ok": basin_hit_rate >= float(cfg["min_basin_hit_rate"]),
        "basin_insert_rate_ok": basin_insert_rate >= float(cfg["min_basin_insert_rate"]),
    }
    if all(checks.values()):
        verdict = "pass_upstream_targeting"
        decision = "handoff_upstream_viable_continue_research"
        reason = (
            f"handoff={handoff_rate:.3f} basin_hit={basin_hit_rate:.3f} "
            f"basin_insert={basin_insert_rate:.3f}"
        )
    else:
        verdict = "fail_stop_handoff_direction"
        decision = "stop_handoff_direction"
        reason = (
            f"handoff={handoff_rate:.3f} basin_hit={basin_hit_rate:.3f} "
            f"basin_insert={basin_insert_rate:.3f} checks={checks}"
        )
    return {
        "verdict": verdict,
        "decision": decision,
        "reason": reason,
        "n": n,
        "n_handoff": n_handoff,
        "n_basin_hit": n_basin,
        "n_basin_insert_ok": n_basin_ins,
        "handoff_rate": handoff_rate,
        "basin_hit_rate": basin_hit_rate,
        "basin_insert_rate": basin_insert_rate,
        "checks": checks,
    }


def write_report(path: Path, judgment: dict, rows: list[dict], cfg: dict) -> None:
    lines = [
        "# Upstream Handoff Targeting P0 — Result",
        "",
        f"- UTC: `{_utc()}`",
        f"- Protocol: `{PROTOCOL}`",
        f"- Verdict: `{judgment['verdict']}`",
        f"- Decision: `{judgment['decision']}`",
        f"- Reason: {judgment['reason']}",
        "",
        "## Rates",
        "",
        f"- seeds: `{judgment['n']}`",
        f"- handoff_rate: `{judgment['handoff_rate']:.3f}` ({judgment['n_handoff']}/{judgment['n']})",
        f"- basin_hit_rate (among handoff): `{judgment['basin_hit_rate']:.3f}` "
        f"({judgment['n_basin_hit']}/{judgment['n_handoff']})",
        f"- basin_insert_rate (among basin hits): `{judgment['basin_insert_rate']:.3f}` "
        f"({judgment['n_basin_insert_ok']}/{judgment['n_basin_hit']})",
        "",
        "## Gates",
        "",
        f"- min_handoff_rate: `{cfg['min_handoff_rate']}`",
        f"- min_basin_hit_rate: `{cfg['min_basin_hit_rate']}`",
        f"- min_basin_insert_rate: `{cfg['min_basin_insert_rate']}`",
        f"- checks: `{judgment['checks']}`",
        "",
        "## Per-seed",
        "",
    ]
    for r in rows:
        lines.append(
            f"- seed=`{r['seed']}` demo=`{r.get('demo_episode')}` "
            f"handoff=`{r.get('reached_handoff')}` basin=`{r.get('basin_hit')}` "
            f"insert=`{r.get('insert_success')}` reason=`{r.get('fail_reason')}` "
            f"outside=`{r.get('outside_axes')}`"
        )
    lines += [
        "",
        "## Note",
        "",
        "无 force-demo / restore；不训练。失败则停止整个 handoff 方向。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "upstream_handoff_targeting_p0.yaml",
    )
    ap.add_argument("--smoke", action="store_true", help="2 seeds only")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = [int(s) for s in cfg["seeds"]]
    if args.smoke:
        seeds = seeds[:2]

    roots, limits, base_from_boundary = load_roots_and_limits(
        PROJECT_ROOT / cfg["boundary_results"]
    )
    base = {k: float(v) for k, v in cfg["base"].items()}
    if not base_from_boundary:
        pass
    if not roots:
        raise RuntimeError("no success_roots in boundary results")

    _patch_execute_peg_lift_compat()
    captures: list[dict[str, Any]] = []
    _install_handoff_capture(captures)

    from interaction_retarget.constants import TASK_ID, default_sidecar_dir
    from interaction_retarget.skill_replay.deploy import run_skill_replay

    sidecar = Path(cfg["sidecar_dir"]) if cfg.get("sidecar_dir") else default_sidecar_dir(TASK_ID)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        captures.clear()
        print(f"=== seed {seed} ===", flush=True)
        try:
            report = run_skill_replay(
                sidecar_dir=sidecar,
                seed=int(seed),
                hold_steps=int(cfg.get("hold_steps", 20)),
                tray_hold_max_steps=int(cfg.get("tray_hold_max_steps", 72)),
                skip_insert=False,
                skip_peg_lift=False,
                force_demo_episode=None,
                restore_demo_layout=False,
                fast=bool(cfg.get("fast", True)),
            )
            err = None
        except Exception as e:
            report = None
            err = str(e)
            print(f"  exception: {e}", flush=True)

        feat = None
        for c in reversed(captures):
            if c.get("controller_active") or c.get("force_ok"):
                feat = c
                break
        if feat is None and captures:
            feat = captures[-1]

        ins = getattr(report, "insert", None) if report is not None else None
        reached = bool(ins is not None and (ins.handoff or (feat and feat.get("controller_active"))))
        insert_success = bool(report.success) if report is not None else False
        basin_hit = False
        scales = None
        outside = None
        nearest_ep = None
        if feat is not None and reached:
            root, dist = nearest_root(feat, roots)
            nearest_ep = int(root["episode_index"])
            basin_hit, scales, outside = in_basin(feat, root, base=base, limits=limits)
            # If axis limit is 0, almost everything fails axis — that's intentional and harsh.
            # Keep as pre-registered conservative boundary.
        row = {
            "seed": int(seed),
            "demo_episode": int(report.demo_episode_index) if report is not None else None,
            "retrieval_m": float(report.retrieval_distance_m) if report is not None else None,
            "fail_reason": (report.fail_reason if report is not None else err),
            "reached_handoff": reached,
            "insert_success": insert_success,
            "insert_report": None
            if ins is None
            else {
                "success": bool(ins.success),
                "insert_ok": bool(ins.insert_ok),
                "handoff": bool(ins.handoff),
                "phase": str(ins.phase),
                "fail_reason": str(ins.fail_reason),
                "peg_lift_m": float(ins.peg_lift_m),
            },
            "handoff_feat": feat,
            "basin_hit": bool(basin_hit),
            "equivalent_scales": scales,
            "outside_axes": outside,
            "nearest_success_ep": nearest_ep,
            "n_captures": len(captures),
        }
        rows.append(row)
        print(
            f"  handoff={reached} basin={basin_hit} insert={insert_success} "
            f"reason={row['fail_reason']} outside={outside}",
            flush=True,
        )

    judgment = judge(rows, cfg)
    out_dir = PROJECT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / cfg["report_path"]
    manifest_path = PROJECT_ROOT / cfg["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "protocol": PROTOCOL,
        "utc": _utc(),
        "config": cfg,
        "limits": limits,
        "rows": rows,
        "judgment": judgment,
        "smoke": bool(args.smoke),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(report_path, judgment, rows, cfg)
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "utc": _utc(),
                "verdict": judgment["verdict"],
                "decision": judgment["decision"],
                "handoff_rate": judgment["handoff_rate"],
                "basin_hit_rate": judgment["basin_hit_rate"],
                "basin_insert_rate": judgment["basin_insert_rate"],
                "results": str(out_dir / "results.json"),
                "report": str(report_path),
                "smoke": bool(args.smoke),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    state_path = PROJECT_ROOT / "outputs" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    stop = judgment["decision"] == "stop_handoff_direction"
    state.update(
        {
            "date": "2026-08-16",
            "phase": "upstream_handoff_targeting_p0",
            "busy": False,
            "training_allowed": False,
            "collection_allowed": False,
            "handoff_direction": {
                "status": "stopped" if stop else "viable_research",
                "verdict": judgment["verdict"],
                "decision": judgment["decision"],
            },
            "upstream_handoff_targeting_p0": {
                "status": "done",
                "verdict": judgment["verdict"],
                "decision": judgment["decision"],
                "smoke": bool(args.smoke),
            },
            "next_action": judgment["decision"],
        }
    )
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    prog = PROJECT_ROOT / "PROGRESS.md"
    block = (
        f"\n## 2026-08-16：Upstream Handoff Targeting P0\n\n"
        f"- 判定：`{judgment['verdict']}` → `{judgment['decision']}`\n"
        f"- handoff/basin_hit/basin_insert："
        f"`{judgment['handoff_rate']:.3f}` / `{judgment['basin_hit_rate']:.3f}` / "
        f"`{judgment['basin_insert_rate']:.3f}`\n"
        f"- 报告：`docs/UPSTREAM_HANDOFF_TARGETING_P0_RESULT.md`\n"
        f"- smoke=`{bool(args.smoke)}`\n"
    )
    if prog.exists():
        text = prog.read_text(encoding="utf-8")
        if "Upstream Handoff Targeting P0" not in text:
            parts = text.split("\n## 下一步", 1)
            next_blk = (
                "\n## 下一步\n\n"
                + (
                    "1. **停止整个 handoff 方向**；不训练；不跑 along_far P0.1。\n"
                    if stop
                    else "1. handoff 上游门通过；仍不自动开训，另开协议。\n"
                )
                + "2. 不抢救 PrivHI / compliance / affordance。\n"
            )
            if len(parts) == 2:
                prog.write_text(parts[0] + block + next_blk, encoding="utf-8")
            else:
                prog.write_text(text + block + next_blk, encoding="utf-8")

    print(json.dumps(judgment, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
