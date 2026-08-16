#!/usr/bin/env python3
"""Handoff Recoverability Boundary / Coverage P0.

Build anisotropic recoverable boundary from prior perturb-recoverability results,
then place archived + demo-identity-fail handoffs inside/outside.
No policy training; archived fail trajs are readonly state samples only.
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
from embodied_grasp_insertion.simulation.full_episode_utils import make_full_env  # noqa: E402
from insertion_science.physics.recoverability_boundary import (  # noqa: E402
    build_cell_rates,
    build_direction_boundaries,
    kind_boundary_summary,
)
from interaction_retarget.sim.replay import raw_flat_to_dict  # noqa: E402
from pose_insert.pre_insert import resolve_peg_lift_end_frame  # noqa: E402
from reach_insert_rl.env.full_obs import current_action44, privileged_full_features  # noqa: E402
from reach_insert_rl.env.handoff_env import load_manifest_entries  # noqa: E402

PROTOCOL = "HandoffRecoverabilityBoundaryCoverageP0"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _feat_vec(h: dict[str, Any], *, use_axis: bool) -> np.ndarray:
    tip = float(h["tip_m"])
    lat = float(h["lat_m"])
    along = float(h["along_m"])
    if use_axis and h.get("axis_err_rad") is not None:
        return np.asarray([tip, lat, along, float(h["axis_err_rad"])], dtype=np.float64)
    return np.asarray([tip, lat, along], dtype=np.float64)


def success_roots_from_recoverability(rows: list[dict]) -> list[dict[str, Any]]:
    roots = {}
    for r in rows:
        if str(r.get("kind")) not in ("none", "identity"):
            continue
        if not bool(r.get("insert_ok")):
            continue
        ep = int(r["episode_index"])
        h = dict(r["handoff"])
        h["episode_index"] = ep
        h["source"] = "recoverability_identity_success"
        roots[ep] = h
    return [roots[k] for k in sorted(roots)]


def load_archived_fail_handoffs(globs: list[str]) -> list[dict[str, Any]]:
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
        # InsertHandoffEnv traj starts at peg_lift_end handoff.
        out.append(
            {
                "source": "archived_eval_fail_traj",
                "path": str(p),
                "episode_index": int(d.get("summary", {}).get("ep", -1)),
                "fail_reason": "fail" if "_fail_" in p.name else "unknown",
                "tip_m": float(t0["tip_mm"]) * 1e-3,
                "lat_m": float(t0["lat_mm"]) * 1e-3,
                "along_m": float(t0["along_mm"]) * 1e-3,
                "axis_err_rad": None,  # not in traj schema
                "peg_ok": bool(t0.get("peg_ok")),
                "insert_ok": bool(t0.get("insert_ok")),
            }
        )
    return out


def _demo_abs_step(env) -> dict[str, Any]:
    assert env._actions is not None
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
    }


def capture_demo_identity_fail_handoffs(
    episodes: list[int],
    *,
    sidecar: Path,
    seed: int,
) -> list[dict[str, Any]]:
    if not episodes:
        return []
    entries = load_manifest_entries(sidecar, episode_indices=list(episodes))
    by_ep = {int(e["episode_index"]): e for e in entries}
    env = make_full_env(list(episodes), sidecar_dir=sidecar, seed=seed)
    out = []
    for ep in episodes:
        entry = by_ep[ep]
        env.reset(entry=entry)
        ple = int(resolve_peg_lift_end_frame(entry, sidecar))
        while int(env._t) < ple and not env._done:
            _demo_abs_step(env)
        if env._done and int(env._t) < ple:
            out.append({"episode_index": ep, "source": "demo_identity_fail", "error": "done_before_handoff"})
            continue
        feat = privileged_full_features(env._raw)
        o2h = object_in_hand_pose(env._raw)
        contact = peg_hand_contact_counts(env._raw)
        outcome = env._labeler.compute(env._raw)
        handoff = {
            "source": "demo_identity_fail_handoff",
            "episode_index": int(ep),
            "frame": int(env._t),
            "peg_lift_end": ple,
            "tip_m": float(feat["tip_dist"]),
            "lat_m": float(feat["lat_err"]),
            "along_m": float(feat["along"]),
            "axis_err_rad": float(feat["axis_err"]),
            "contact_total": int(contact.total),
            "o2h_t": o2h.translation.tolist(),
            "peg_ok": bool(outcome.peg_ok),
            "insert_ok": bool(outcome.insert_ok),
        }
        # Confirm identity continuation fails (sanity).
        tip_min = float("inf")
        insert_ok = False
        while not env._done and env._actions is not None and env._t < len(env._actions):
            info = _demo_abs_step(env)
            tip_min = min(tip_min, float(info["tip_m"]))
            if info["insert_ok"]:
                insert_ok = True
                break
        handoff["identity_insert_ok"] = bool(insert_ok)
        handoff["identity_tip_min_m"] = float(tip_min if np.isfinite(tip_min) else float("nan"))
        out.append(handoff)
    return out


def nearest_root(fail: dict, roots: list[dict], *, use_axis: bool) -> tuple[dict, float]:
    fv = _feat_vec(fail, use_axis=use_axis and fail.get("axis_err_rad") is not None)
    best = None
    best_d = float("inf")
    for r in roots:
        rv = _feat_vec(r, use_axis=use_axis and r.get("axis_err_rad") is not None and fail.get("axis_err_rad") is not None)
        # align dims
        n = min(len(fv), len(rv))
        d = float(np.linalg.norm(fv[:n] - rv[:n]))
        if d < best_d:
            best_d = d
            best = r
    assert best is not None
    return best, best_d


def equivalent_scales(fail: dict, root: dict, base: dict[str, float]) -> dict[str, float]:
    s = {
        "s_lat": abs(float(fail["lat_m"]) - float(root["lat_m"])) / float(base["tip_lat_m"]),
        "s_along": abs(float(fail["along_m"]) - float(root["along_m"])) / float(base["tip_along_m"]),
        "s_tip": abs(float(fail["tip_m"]) - float(root["tip_m"])) / float(base["tip_lat_m"]),
    }
    if fail.get("axis_err_rad") is not None and root.get("axis_err_rad") is not None:
        s["s_axis"] = abs(float(fail["axis_err_rad"]) - float(root["axis_err_rad"])) / float(base["axis_rad"])
    return s


def direction_limits(boundaries: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Map logical axis -> conservative (min) recoverable scale among related pert names."""
    groups = {
        "tip_lat": [n for n, b in boundaries.items() if b["kind"] == "tip_lat"],
        "tip_along": [n for n, b in boundaries.items() if b["kind"] == "tip_along"],
        "axis": [n for n, b in boundaries.items() if b["kind"] == "axis"],
        "o2h": [n for n, b in boundaries.items() if b["kind"] == "o2h"],
        "finger": [n for n, b in boundaries.items() if b["kind"] == "finger"],
    }
    out = {}
    for g, names in groups.items():
        if not names:
            out[g] = 0.0
            continue
        out[g] = float(min(boundaries[n]["max_recoverable_scale"] for n in names))
    return out


def classify_fail(
    fail: dict,
    roots: list[dict],
    *,
    base: dict[str, float],
    limits: dict[str, float],
    pair_l2_max: float,
) -> dict[str, Any]:
    use_axis = fail.get("axis_err_rad") is not None
    root, dist = nearest_root(fail, roots, use_axis=use_axis)
    scales = equivalent_scales(fail, root, base)
    # Compare measured axes only.
    checks = {
        "tip_lat": (scales["s_lat"], limits.get("tip_lat", 0.0)),
        "tip_along": (scales["s_along"], limits.get("tip_along", 0.0)),
    }
    if "s_axis" in scales:
        checks["axis"] = (scales["s_axis"], limits.get("axis", 0.0))
    outside_axes = [name for name, (s, lim) in checks.items() if float(s) > float(lim) + 1e-9]
    inside = len(outside_axes) == 0
    # Pair evidence: another success root within feature L2, or nearest root itself is success.
    pair = False
    pair_ep = None
    for r in roots:
        n = 3
        fv = _feat_vec(fail, use_axis=False)
        rv = _feat_vec(r, use_axis=False)
        if float(np.linalg.norm(fv[:n] - rv[:n])) <= float(pair_l2_max):
            pair = True
            pair_ep = int(r["episode_index"])
            break
    return {
        "fail": {k: fail[k] for k in fail if k != "o2h_t"},
        "nearest_success_ep": int(root["episode_index"]),
        "nearest_l2": dist,
        "equivalent_scales": scales,
        "limits_used": {k: checks[k][1] for k in checks},
        "outside_axes": outside_axes,
        "inside_boundary": bool(inside),
        "has_nearby_success_pair": bool(pair),
        "pair_success_ep": pair_ep,
    }


def judge(classifications: list[dict], cfg: dict) -> dict[str, Any]:
    n = len(classifications)
    if n == 0:
        return {
            "verdict": "insufficient_fail_samples",
            "decision": "collect_or_locate_fail_handoffs",
            "reason": "no fail handoffs",
            "outside_frac": float("nan"),
            "n": 0,
        }
    n_out = sum(1 for c in classifications if not c["inside_boundary"])
    n_in = n - n_out
    n_in_pair = sum(1 for c in classifications if c["inside_boundary"] and c["has_nearby_success_pair"])
    outside_frac = n_out / n
    maj = float(cfg["outside_majority"])
    min_pair = int(cfg["min_inside_with_pair"])
    if outside_frac >= maj:
        verdict = "branch1_fails_mostly_outside_boundary"
        decision = "redesign_handoff_data_generation"
        reason = f"outside_frac={outside_frac:.3f} >= {maj}"
    elif n_in_pair < min_pair:
        verdict = "branch2_inside_but_lacking_correction_pairs"
        decision = "generate_same_root_success_fail_pairs"
        reason = f"inside={n_in}, inside_with_pair={n_in_pair} < {min_pair}"
    else:
        verdict = "branch3_inside_with_distinguishable_pairs"
        decision = "allow_min_supervised_policy_p0"
        reason = f"inside_with_pair={n_in_pair} >= {min_pair}, outside_frac={outside_frac:.3f}"
    return {
        "verdict": verdict,
        "decision": decision,
        "reason": reason,
        "outside_frac": outside_frac,
        "n": n,
        "n_outside": n_out,
        "n_inside": n_in,
        "n_inside_with_pair": n_in_pair,
    }


def write_report(
    path: Path,
    *,
    judgment: dict,
    boundaries: dict,
    kind_sum: dict,
    limits: dict,
    classifications: list[dict],
    cfg: dict,
) -> None:
    lines = [
        "# Handoff Recoverability Boundary / Coverage P0 — Result",
        "",
        f"- UTC: `{_utc()}`",
        f"- Protocol: `{PROTOCOL}`",
        f"- Verdict: `{judgment['verdict']}`",
        f"- Decision: `{judgment['decision']}`",
        f"- Reason: {judgment['reason']}",
        "",
        "## Anisotropic boundary (max recoverable scale, rate≥"
        f"{cfg['inside_rate_min']})",
        "",
    ]
    for name, b in sorted(boundaries.items()):
        lines.append(f"- `{name}` ({b['kind']}): max_scale=`{b['max_recoverable_scale']}`")
    lines += ["", "## Kind envelope (min across directions = fragile)", ""]
    for kind, s in kind_sum.items():
        lines.append(
            f"- `{kind}`: min_dir=`{s['min_among_dirs']}`, max_dir=`{s['max_among_dirs']}`, n_dirs=`{s['n_dirs']}`"
        )
    lines += ["", "## Conservative limits used for coverage", ""]
    for k, v in limits.items():
        lines.append(f"- `{k}`: `{v}`")
    lines += [
        "",
        "## Coverage",
        "",
        f"- fails: `{judgment['n']}` outside=`{judgment['n_outside']}` "
        f"inside=`{judgment['n_inside']}` inside_with_pair=`{judgment['n_inside_with_pair']}`",
        f"- outside_frac: `{judgment['outside_frac']:.3f}`",
        "",
        "## Per-fail (compact)",
        "",
    ]
    for c in classifications:
        f = c["fail"]
        lines.append(
            f"- src=`{f.get('source')}` ep=`{f.get('episode_index')}` "
            f"inside=`{c['inside_boundary']}` outside_axes=`{c['outside_axes']}` "
            f"scales=`{ {k: round(v, 2) for k, v in c['equivalent_scales'].items()} }` "
            f"pair=`{c['has_nearby_success_pair']}`"
        )
    lines += [
        "",
        "## Note",
        "",
        "归档 fail traj 仅作 handoff 状态样本；不训练、不复活 PrivHI 主线。",
        "axis 在归档 traj 中缺失时，覆盖判定不加 axis 轴。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "handoff_recoverability_boundary_coverage_p0.yaml",
    )
    ap.add_argument("--skip-demo-fails", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    rec_path = PROJECT_ROOT / cfg["recoverability_results"]
    payload = json.loads(rec_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    cells = build_cell_rates(rows, min_n_held_out=int(cfg["min_n_held_out"]))
    boundaries = build_direction_boundaries(cells, inside_rate_min=float(cfg["inside_rate_min"]))
    kind_sum = kind_boundary_summary(boundaries)
    limits = direction_limits(boundaries)
    roots = success_roots_from_recoverability(rows)

    fails = load_archived_fail_handoffs(list(cfg["archived_fail_globs"]))
    demo_fails: list[dict] = []
    if not args.skip_demo_fails:
        print("capturing demo identity-fail handoffs...", flush=True)
        demo_fails = capture_demo_identity_fail_handoffs(
            [int(x) for x in cfg["demo_identity_fail_episodes"]],
            sidecar=Path(cfg["sidecar_dir"]),
            seed=int(cfg.get("seed", 0)),
        )
        # Only keep those that truly fail identity (or error).
        kept = []
        for h in demo_fails:
            if h.get("error"):
                kept.append(h)
            elif h.get("identity_insert_ok") is False:
                kept.append(h)
            else:
                print(f"  skip ep{h.get('episode_index')}: identity unexpectedly ok", flush=True)
        demo_fails = kept

    all_fails = fails + [h for h in demo_fails if "tip_m" in h]
    print(f"archived_fails={len(fails)} demo_fails={len(demo_fails)} classify={len(all_fails)}", flush=True)

    classifications = [
        classify_fail(
            f,
            roots,
            base={k: float(v) for k, v in cfg["base"].items()},
            limits=limits,
            pair_l2_max=float(cfg["pair_feature_l2_max"]),
        )
        for f in all_fails
    ]
    judgment = judge(classifications, cfg)

    out_dir = PROJECT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / cfg["report_path"]
    manifest_path = PROJECT_ROOT / cfg["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "protocol": PROTOCOL,
        "utc": _utc(),
        "config": cfg,
        "boundaries": boundaries,
        "kind_summary": kind_sum,
        "limits": limits,
        "success_roots": roots,
        "classifications": classifications,
        "judgment": judgment,
        "n_archived_fails": len(fails),
        "n_demo_fails": len(demo_fails),
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(
        report_path,
        judgment=judgment,
        boundaries=boundaries,
        kind_sum=kind_sum,
        limits=limits,
        classifications=classifications,
        cfg=cfg,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "utc": _utc(),
                "verdict": judgment["verdict"],
                "decision": judgment["decision"],
                "outside_frac": judgment.get("outside_frac"),
                "results": str(out_dir / "results.json"),
                "report": str(report_path),
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
            "phase": "handoff_recoverability_boundary_coverage_p0",
            "busy": False,
            "training_allowed": False,
            "collection_allowed": False,
            "handoff_recoverability_boundary_coverage_p0": {
                "status": "done",
                "verdict": judgment["verdict"],
                "decision": judgment["decision"],
            },
            "next_action": judgment["decision"],
        }
    )
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    prog = PROJECT_ROOT / "PROGRESS.md"
    block = (
        f"\n## 2026-08-16：Handoff Recoverability Boundary / Coverage P0\n\n"
        f"- 判定：`{judgment['verdict']}` → `{judgment['decision']}`\n"
        f"- 原因：{judgment['reason']}\n"
        f"- outside_frac：`{judgment.get('outside_frac')}`\n"
        f"- 报告：`docs/HANDOFF_RECOVERABILITY_BOUNDARY_COVERAGE_P0_RESULT.md`\n"
    )
    if prog.exists():
        text = prog.read_text(encoding="utf-8")
        if "Boundary / Coverage P0" not in text:
            parts = text.split("\n## 下一步", 1)
            if len(parts) == 2:
                prog.write_text(parts[0] + block + "\n## 下一步" + parts[1], encoding="utf-8")
            else:
                prog.write_text(text + block, encoding="utf-8")

    print(json.dumps(judgment, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
