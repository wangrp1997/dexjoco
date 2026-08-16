#!/usr/bin/env python3
"""Handoff Datagen Redesign P0: sample handoffs inside recoverable basin; death-gate.

No policy training. Does not rewrite the production sidecar library.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEXJOCO_ROOT = PROJECT_ROOT.parent
EMBODIED = DEXJOCO_ROOT / "embodied_grasp_insertion"
for _p in (
    str(PROJECT_ROOT),
    str(DEXJOCO_ROOT),
    str(DEXJOCO_ROOT / "dexjoco"),
    str(EMBODIED),
    str(DEXJOCO_ROOT.parent / "reach_insert_rl"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.physics.grasp_metrics import (  # noqa: E402
    object_in_hand_pose,
    peg_hand_contact_counts,
)
from embodied_grasp_insertion.simulation.full_episode_snapshot import (  # noqa: E402
    FullEpisodeSnapshot,
)
from embodied_grasp_insertion.simulation.full_episode_utils import make_full_env  # noqa: E402
from insertion_science.physics.handoff_perturb import apply_perturbation  # noqa: E402
from interaction_retarget.sim.replay import raw_flat_to_dict  # noqa: E402
from pose_insert.pre_insert import resolve_peg_lift_end_frame  # noqa: E402
from reach_insert_rl.env.full_obs import current_action44, privileged_full_features  # noqa: E402
from reach_insert_rl.env.handoff_env import load_manifest_entries  # noqa: E402

PROTOCOL = "HandoffDatagenRedesignP0"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _demo_abs_step(env) -> dict[str, Any]:
    assert env._actions is not None
    if env._done or int(env._t) >= len(env._actions):
        raise RuntimeError("cannot step")
    env._raw.step(raw_flat_to_dict(env._actions[int(env._t)]))
    env._hold44 = current_action44(env._raw).copy()
    env._t += 1
    outcome = env._labeler.compute(env._raw)
    feat = privileged_full_features(env._raw)
    if not outcome.peg_ok:
        env._peg_lost += 1
    else:
        env._peg_lost = 0
    _ = env._shaped_reward(outcome, feat, success=bool(outcome.insert_ok))
    peg_lost = env._peg_ok_seen and env._peg_lost >= env.peg_lost_abort
    timeout = env._t >= env.max_episode_steps
    terminated = bool(outcome.insert_ok or peg_lost)
    truncated = bool(timeout and not terminated)
    env._done = terminated or truncated
    return {
        "insert_ok": bool(outcome.insert_ok),
        "peg_ok": bool(outcome.peg_ok),
        "tip_m": float(feat["tip_dist"]),
        "lat_m": float(feat["lat_err"]),
        "along_m": float(feat["along"]),
        "axis_err_rad": float(feat["axis_err"]),
        "peg_lost_abort": bool(peg_lost),
    }


def _feat(env) -> dict[str, Any]:
    feat = privileged_full_features(env._raw)
    o2h = object_in_hand_pose(env._raw)
    contact = peg_hand_contact_counts(env._raw)
    outcome = env._labeler.compute(env._raw)
    return {
        "tip_m": float(feat["tip_dist"]),
        "lat_m": float(feat["lat_err"]),
        "along_m": float(feat["along"]),
        "axis_err_rad": float(feat["axis_err"]),
        "contact_total": int(contact.total),
        "o2h_t": o2h.translation.tolist(),
        "peg_ok": bool(outcome.peg_ok),
        "insert_ok": bool(outcome.insert_ok),
    }


def continue_demo(env, *, max_steps: int) -> dict[str, Any]:
    tip_min = float("inf")
    insert_ok = False
    steps = 0
    reason = "horizon_end"
    while steps < int(max_steps):
        if env._done:
            reason = "already_done"
            break
        if env._actions is None or int(env._t) >= len(env._actions):
            reason = "demo_exhausted"
            break
        info = _demo_abs_step(env)
        steps += 1
        tip_min = min(tip_min, float(info["tip_m"]))
        if info["insert_ok"]:
            insert_ok = True
            reason = "insert_ok"
            break
        if info.get("peg_lost_abort"):
            reason = "peg_lost_abort"
            break
    return {
        "insert_ok": bool(insert_ok),
        "steps": steps,
        "tip_min_m": float(tip_min if np.isfinite(tip_min) else float("nan")),
        "term_reason": reason,
    }


def capture_root(env, entry, sidecar: Path) -> tuple[FullEpisodeSnapshot, dict[str, Any]]:
    env.reset(entry=entry)
    ple = int(resolve_peg_lift_end_frame(entry, sidecar))
    while int(env._t) < ple and not env._done:
        _demo_abs_step(env)
    if env._done and int(env._t) < ple:
        raise RuntimeError(f"done before handoff ep={entry['episode_index']}")
    snap = FullEpisodeSnapshot.capture(env)
    root = _feat(env)
    root["frame"] = int(env._t)
    root["peg_lift_end"] = ple
    root["episode_index"] = int(env._spec.episode_index)
    return snap, root


def kind_limits(boundary_payload: dict, safety: float) -> dict[str, float]:
    lim = dict(boundary_payload.get("limits") or {})
    out = {}
    for k, v in lim.items():
        if k == "axis":
            out[k] = 0.0  # never generate axis tilt in-basin
        else:
            out[k] = float(v) * float(safety)
    return out


def resolve_in_scale(spec: dict, limits: dict[str, float], rng: np.random.Generator) -> float:
    kind = str(spec["kind"])
    key = {
        "tip_lat": "tip_lat",
        "tip_along": "tip_along",
        "o2h": "o2h",
        "finger": "finger",
        "axis": "axis",
    }[kind]
    max_s = float(limits.get(key, 0.0))
    if max_s <= 0:
        return 0.0
    lo, hi = [float(x) for x in spec.get("scale_frac", [0.25, 1.0])]
    frac = float(rng.uniform(lo, hi))
    return max_s * frac


def run_cell(
    env,
    snap: FullEpisodeSnapshot,
    *,
    pert_cfg: dict,
    scale: float,
    base: dict[str, float],
    max_steps: int,
    basin: str,
) -> dict[str, Any]:
    snap.restore(env)
    if bool(env._done):
        raise RuntimeError("restore done")
    pmeta = apply_perturbation(
        env._raw,
        kind=str(pert_cfg["kind"]),
        scale=float(scale),
        base=base,
        pert_cfg=pert_cfg,
    )
    after = _feat(env)
    cont = continue_demo(env, max_steps=max_steps)
    return {
        "basin": basin,
        "pert_name": str(pert_cfg.get("name")),
        "kind": str(pert_cfg["kind"]),
        "scale": float(scale),
        "perturb_meta": pmeta,
        "after_perturb": after,
        **cont,
    }


def load_archived_fails(globs: list[str]) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for g in globs:
        paths.extend(Path(p) for p in sorted(glob(g)))
    out = []
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        traj = d.get("traj") or []
        if not traj:
            continue
        t0 = traj[0]
        out.append(
            {
                "tip_m": float(t0["tip_mm"]) * 1e-3,
                "lat_m": float(t0["lat_mm"]) * 1e-3,
                "along_m": float(t0["along_mm"]) * 1e-3,
                "episode_index": int(d.get("summary", {}).get("ep", -1)),
                "path": str(p),
            }
        )
    return out


def coverage_outside_frac(
    fails: list[dict],
    accepted_roots: list[dict],
    *,
    base: dict[str, float],
    limits: dict[str, float],
) -> dict[str, Any]:
    if not fails or not accepted_roots:
        return {"outside_frac": float("nan"), "n": 0, "n_outside": 0}
    n_out = 0
    for f in fails:
        best = None
        best_d = float("inf")
        fv = np.asarray([f["tip_m"], f["lat_m"], f["along_m"]], dtype=np.float64)
        for r in accepted_roots:
            rv = np.asarray([r["tip_m"], r["lat_m"], r["along_m"]], dtype=np.float64)
            d = float(np.linalg.norm(fv - rv))
            if d < best_d:
                best_d = d
                best = r
        assert best is not None
        s_lat = abs(f["lat_m"] - best["lat_m"]) / float(base["tip_lat_m"])
        s_along = abs(f["along_m"] - best["along_m"]) / float(base["tip_along_m"])
        if s_lat > float(limits.get("tip_lat", 0.0)) or s_along > float(limits.get("tip_along", 0.0)):
            n_out += 1
    return {
        "outside_frac": n_out / len(fails),
        "n": len(fails),
        "n_outside": n_out,
        "n_roots": len(accepted_roots),
    }


def judge(in_rows: list[dict], out_rows: list[dict], cov: dict, cfg: dict) -> dict[str, Any]:
    in_ok = [bool(r["insert_ok"]) for r in in_rows]
    out_ok = [bool(r["insert_ok"]) for r in out_rows]
    in_rate = float(np.mean(in_ok)) if in_ok else 0.0
    out_rate = float(np.mean(out_ok)) if out_ok else 0.0
    accepted = [r for r in in_rows if r["insert_ok"]]
    gap = in_rate - out_rate
    checks = {
        "in_basin_rate_ok": in_rate >= float(cfg["in_basin_min_rate"]),
        "out_basin_rate_ok": out_rate <= float(cfg["out_basin_max_rate"]),
        "gap_ok": gap >= float(cfg["min_rate_gap"]),
        "min_accepted_ok": len(accepted) >= int(cfg["min_accepted"]),
    }
    if all(checks.values()):
        verdict = "pass_datagen_redesign"
        decision = "use_constrained_handoff_set_for_next_research"
        reason = (
            f"in={in_rate:.3f} out={out_rate:.3f} gap={gap:.3f} accepted={len(accepted)}; "
            f"archived_fail_outside_frac={cov.get('outside_frac')}"
        )
    elif not checks["in_basin_rate_ok"] or not checks["min_accepted_ok"]:
        verdict = "fail_basin_not_generative"
        decision = "pause_basin_or_replay_recheck"
        reason = f"in={in_rate:.3f} accepted={len(accepted)} checks={checks}"
    else:
        verdict = "fail_no_contrast"
        decision = "pause_boundary_not_operational"
        reason = f"in={in_rate:.3f} out={out_rate:.3f} gap={gap:.3f} checks={checks}"
    return {
        "verdict": verdict,
        "decision": decision,
        "reason": reason,
        "in_basin_rate": in_rate,
        "out_basin_rate": out_rate,
        "gap": gap,
        "n_in": len(in_rows),
        "n_out": len(out_rows),
        "n_accepted": len(accepted),
        "checks": checks,
        "coverage_recheck": cov,
    }


def write_report(path: Path, judgment: dict, limits: dict, cfg: dict) -> None:
    lines = [
        "# Handoff Datagen Redesign P0 — Result",
        "",
        f"- UTC: `{_utc()}`",
        f"- Protocol: `{PROTOCOL}`",
        f"- Verdict: `{judgment['verdict']}`",
        f"- Decision: `{judgment['decision']}`",
        f"- Reason: {judgment['reason']}",
        "",
        "## Rates",
        "",
        f"- In-basin insert_ok: `{judgment['in_basin_rate']:.3f}` (n={judgment['n_in']})",
        f"- Out-basin insert_ok: `{judgment['out_basin_rate']:.3f}` (n={judgment['n_out']})",
        f"- Gap: `{judgment['gap']:.3f}`",
        f"- Accepted: `{judgment['n_accepted']}`",
        "",
        "## Generation limits (conservative × safety)",
        "",
    ]
    for k, v in limits.items():
        lines.append(f"- `{k}`: `{v}`")
    cov = judgment.get("coverage_recheck") or {}
    lines += [
        "",
        "## Coverage recheck (archived fails vs accepted)",
        "",
        f"- outside_frac: `{cov.get('outside_frac')}` (n={cov.get('n')}, outside={cov.get('n_outside')})",
        "",
        "## Gates",
        "",
        f"- in_basin_min_rate: `{cfg['in_basin_min_rate']}`",
        f"- out_basin_max_rate: `{cfg['out_basin_max_rate']}`",
        f"- min_rate_gap: `{cfg['min_rate_gap']}`",
        f"- min_accepted: `{cfg['min_accepted']}`",
        f"- checks: `{judgment['checks']}`",
        "",
        "## Note",
        "",
        "不训练；不写回生产 sidecar；旧失败仍应在盆地外（预期）。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "handoff_datagen_redesign_p0.yaml",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    boundary_payload = json.loads(
        (PROJECT_ROOT / cfg["boundary_results"]).read_text(encoding="utf-8")
    )
    limits = kind_limits(boundary_payload, float(cfg["safety_factor"]))
    base = {k: float(v) for k, v in cfg["base"].items()}
    roots = [int(x) for x in cfg["root_episodes"]]
    n_in = int(cfg["n_in_per_root"])
    n_out = int(cfg["n_out_per_root"])
    if args.smoke:
        roots = roots[:2]
        n_in = 1
        n_out = 1

    sidecar = Path(cfg["sidecar_dir"])
    entries = load_manifest_entries(sidecar, episode_indices=roots)
    by_ep = {int(e["episode_index"]): e for e in entries}
    env = make_full_env(roots, sidecar_dir=sidecar, seed=int(cfg.get("seed", 0)))
    max_steps = int(cfg["max_continuation_steps"])

    in_rows: list[dict] = []
    out_rows: list[dict] = []
    accepted_feats: list[dict] = []

    for ep in roots:
        print(f"[ep {ep}] capture root", flush=True)
        snap, root = capture_root(env, by_ep[ep], sidecar)
        print(
            f"  tip={root['tip_m']:.4f} lat={root['lat_m']:.4f} axis={root['axis_err_rad']:.4f}",
            flush=True,
        )
        # identity sanity
        snap.restore(env)
        id_cont = continue_demo(env, max_steps=max_steps)
        if not id_cont["insert_ok"]:
            print(f"  SKIP root ep{ep}: identity failed", flush=True)
            continue

        for i in range(n_in):
            spec = dict(cfg["in_basin_specs"][int(rng.integers(0, len(cfg["in_basin_specs"])))])
            scale = resolve_in_scale(spec, limits, rng)
            if scale <= 0 and str(spec["kind"]) != "none":
                # skip zero-limit kinds
                continue
            print(f"  in {spec['name']} scale={scale:.3f}", flush=True)
            row = run_cell(
                env, snap, pert_cfg=spec, scale=scale, base=base, max_steps=max_steps, basin="in"
            )
            row["episode_index"] = ep
            row["root"] = root
            in_rows.append(row)
            print(f"    insert_ok={row['insert_ok']} tip_min={row['tip_min_m']}", flush=True)
            if row["insert_ok"]:
                feat = dict(row["after_perturb"])
                feat["episode_index"] = ep
                feat["pert_name"] = row["pert_name"]
                feat["scale"] = row["scale"]
                accepted_feats.append(feat)

        for _ in range(n_out):
            spec = dict(cfg["out_basin_specs"][int(rng.integers(0, len(cfg["out_basin_specs"])))])
            scale = float(spec["scale"])
            print(f"  out {spec['name']} scale={scale:.3f}", flush=True)
            row = run_cell(
                env, snap, pert_cfg=spec, scale=scale, base=base, max_steps=max_steps, basin="out"
            )
            row["episode_index"] = ep
            row["root"] = root
            out_rows.append(row)
            print(f"    insert_ok={row['insert_ok']} tip_min={row['tip_min_m']}", flush=True)

    fails = load_archived_fails(list(cfg.get("archived_fail_globs") or []))
    # Use accepted after-perturb features as new roots; fallback to original roots if empty
    cov_roots = accepted_feats if accepted_feats else []
    if not cov_roots:
        # placeholder empty
        cov = {"outside_frac": float("nan"), "n": len(fails), "n_outside": 0, "n_roots": 0}
    else:
        # coverage uses generation limits (safety-scaled), same as basin
        cov = coverage_outside_frac(fails, cov_roots, base=base, limits=limits)

    judgment = judge(in_rows, out_rows, cov, cfg)

    out_dir = PROJECT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / cfg["report_path"]
    manifest_path = PROJECT_ROOT / cfg["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "protocol": PROTOCOL,
        "utc": _utc(),
        "config": cfg,
        "limits_used": limits,
        "in_rows": in_rows,
        "out_rows": out_rows,
        "accepted_handoffs": accepted_feats,
        "judgment": judgment,
        "smoke": bool(args.smoke),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "accepted_handoffs.json").write_text(
        json.dumps(accepted_feats, indent=2), encoding="utf-8"
    )
    write_report(report_path, judgment, limits, cfg)
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "utc": _utc(),
                "verdict": judgment["verdict"],
                "decision": judgment["decision"],
                "in_basin_rate": judgment["in_basin_rate"],
                "out_basin_rate": judgment["out_basin_rate"],
                "n_accepted": judgment["n_accepted"],
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
    state.update(
        {
            "date": "2026-08-16",
            "phase": "handoff_datagen_redesign_p0",
            "busy": False,
            "training_allowed": False,
            "collection_allowed": False,
            "handoff_datagen_redesign_p0": {
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
        f"\n## 2026-08-16：Handoff Datagen Redesign P0\n\n"
        f"- 判定：`{judgment['verdict']}` → `{judgment['decision']}`\n"
        f"- in/out rate：`{judgment['in_basin_rate']:.3f}` / `{judgment['out_basin_rate']:.3f}` "
        f"(accepted={judgment['n_accepted']})\n"
        f"- 报告：`docs/HANDOFF_DATAGEN_REDESIGN_P0_RESULT.md`\n"
        f"- smoke=`{bool(args.smoke)}`\n"
    )
    if prog.exists():
        text = prog.read_text(encoding="utf-8")
        if "Datagen Redesign P0" not in text:
            parts = text.split("\n## 下一步", 1)
            if len(parts) == 2:
                next_block = (
                    "\n## 下一步\n\n"
                    "1. 按 Datagen Redesign 判定行动；`training_allowed=false`。\n"
                    "2. 不抢救 PrivHI / compliance / affordance。\n"
                )
                prog.write_text(parts[0] + block + next_block, encoding="utf-8")
            else:
                prog.write_text(text + block, encoding="utf-8")

    print(json.dumps(judgment, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
