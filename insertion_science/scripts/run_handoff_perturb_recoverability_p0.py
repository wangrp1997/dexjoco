#!/usr/bin/env python3
"""Demo Handoff Perturbation Recoverability P0.

Matched micro-perturbations at successful demo handoff + original demo continuation.
No PrivHI, no policy training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
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
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    make_full_env,
)
from insertion_science.physics.handoff_perturb import apply_perturbation  # noqa: E402
from interaction_retarget.sim.replay import raw_flat_to_dict  # noqa: E402
from pose_insert.pre_insert import resolve_peg_lift_end_frame  # noqa: E402
from reach_insert_rl.env.full_obs import privileged_full_features  # noqa: E402
from reach_insert_rl.env.handoff_env import load_manifest_entries  # noqa: E402

PROTOCOL = "DemoHandoffPerturbRecoverabilityP0"


def _demo_abs_step(env) -> dict[str, Any]:
    """Force raw_flat demo step; terminate only on insert_ok / peg_lost / timeout.

    FullEpisodeEnv geom-seat success (~4.5cm tip) aborts before socket-bottom contact;
    InsertHandoffEnv uses insert_ok — we match that here.
    """
    assert env._actions is not None
    t = int(env._t)
    if t >= len(env._actions):
        raise RuntimeError("demo actions exhausted")
    if env._done:
        raise RuntimeError("episode already done")
    env._raw.step(raw_flat_to_dict(env._actions[t]))
    from reach_insert_rl.env.full_obs import current_action44, privileged_full_features

    now44 = current_action44(env._raw)
    env._hold44 = now44.copy()
    env._t += 1
    outcome = env._labeler.compute(env._raw)
    feat = privileged_full_features(env._raw)
    if not outcome.peg_ok:
        env._peg_lost += 1
    else:
        env._peg_lost = 0
    # Side effects for reward bookkeeping only (do not use geom seat as done).
    _ = env._shaped_reward(outcome, feat, success=bool(outcome.insert_ok))
    peg_lost = env._peg_ok_seen and env._peg_lost >= env.peg_lost_abort
    timeout = env._t >= env.max_episode_steps
    terminated = bool(outcome.insert_ok or peg_lost)
    truncated = bool(timeout and not terminated)
    env._done = terminated or truncated
    return {
        "tray_ok": bool(outcome.tray_ok),
        "peg_ok": bool(outcome.peg_ok),
        "insert_ok": bool(outcome.insert_ok),
        "tip_dist_m": float(feat["tip_dist"]),
        "lat_err_m": float(feat["lat_err"]),
        "along_m": float(feat["along"]),
        "terminated": terminated,
        "truncated": truncated,
        "success": bool(outcome.insert_ok),
        "peg_lost_abort": bool(peg_lost),
    }


def _replay_to_frame_raw(env, frame: int) -> None:
    target = int(frame)
    if target < int(env._t):
        raise RuntimeError(f"cannot rewind without restore: at {env._t}, want {target}")
    while int(env._t) < target:
        _demo_abs_step(env)
        if env._done and int(env._t) < target:
            raise RuntimeError(f"terminated at t={env._t} before frame {target}")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _feat(env) -> dict[str, Any]:
    raw = env._raw
    feat = privileged_full_features(raw)
    o2h = object_in_hand_pose(raw)
    contact = peg_hand_contact_counts(raw)
    outcome = env._labeler.compute(raw)
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


def continue_demo(
    env,
    *,
    max_steps: int,
) -> dict[str, Any]:
    tip_min = float("inf")
    tip_end = float("nan")
    lat_end = float("nan")
    insert_ok = False
    peg_lost_abort = False
    steps = 0
    term_reason = "horizon_end"
    while steps < int(max_steps):
        if env._done:
            term_reason = "already_done"
            break
        if env._actions is None or int(env._t) >= len(env._actions):
            term_reason = "demo_exhausted"
            break
        info = _demo_abs_step(env)
        steps += 1
        tip = float(info.get("tip_dist_m", float("nan")))
        lat = float(info.get("lat_err_m", float("nan")))
        if np.isfinite(tip):
            tip_min = min(tip_min, tip)
            tip_end = tip
        if np.isfinite(lat):
            lat_end = lat
        if bool(info.get("insert_ok")):
            insert_ok = True
            term_reason = "insert_ok"
            break
        if bool(info.get("peg_lost_abort")):
            peg_lost_abort = True
            term_reason = "peg_lost_abort"
            break
        if bool(info.get("terminated")) or bool(info.get("truncated")):
            term_reason = "terminated" if info.get("terminated") else "truncated"
            break
    if not np.isfinite(tip_min):
        tip_min = float("nan")
    return {
        "insert_ok": bool(insert_ok),
        "peg_lost_abort": bool(peg_lost_abort),
        "steps": int(steps),
        "tip_min_m": float(tip_min),
        "tip_end_m": float(tip_end),
        "lat_end_m": float(lat_end),
        "term_reason": term_reason,
    }


def capture_handoff(env, entry: dict, sidecar: Path) -> tuple[FullEpisodeSnapshot, dict[str, Any]]:
    env.reset(entry=entry)
    peg_lift_end = int(resolve_peg_lift_end_frame(entry, sidecar))
    _replay_to_frame_raw(env, peg_lift_end)
    if bool(env._done):
        raise RuntimeError(f"done before handoff frame={peg_lift_end}")
    snap = FullEpisodeSnapshot.capture(env)
    root = _feat(env)
    root["frame"] = int(env._t)
    root["peg_lift_end"] = peg_lift_end
    root["episode_index"] = int(env._spec.episode_index)
    return snap, root


def run_cell(
    env,
    snap: FullEpisodeSnapshot,
    *,
    pert_cfg: dict[str, Any],
    scale: float,
    base: dict[str, float],
    max_steps: int,
) -> dict[str, Any]:
    snap.restore(env)
    if bool(env._done):
        raise RuntimeError("restore left done=True")
    pmeta = apply_perturbation(
        env._raw,
        kind=str(pert_cfg.get("kind", "none")),
        scale=float(scale),
        base=base,
        pert_cfg=pert_cfg,
    )
    after = _feat(env)
    cont = continue_demo(env, max_steps=max_steps)
    return {
        "pert_name": str(pert_cfg.get("name", pert_cfg.get("kind"))),
        "kind": str(pert_cfg.get("kind")),
        "scale": float(scale),
        "perturb_meta": pmeta,
        "after_perturb": after,
        **cont,
    }


def rate_by_scale(rows: list[dict], *, non_identity_only: bool = False) -> dict[str, float]:
    buckets: dict[float, list[bool]] = defaultdict(list)
    for r in rows:
        if non_identity_only and str(r.get("kind")) in ("none", "identity"):
            continue
        buckets[float(r["scale"])].append(bool(r["insert_ok"]))
    return {str(k): float(np.mean(v)) if v else float("nan") for k, v in sorted(buckets.items())}


def judge(all_rows: list[dict], cfg: dict) -> dict[str, Any]:
    disc_ids = set(int(x) for x in cfg["episodes"]["discovery"])
    hold_ids = set(int(x) for x in cfg["episodes"]["held_out"])
    baseline = [r for r in all_rows if str(r.get("kind")) in ("none", "identity")]
    base_rate = float(np.mean([bool(r["insert_ok"]) for r in baseline])) if baseline else 0.0
    hold_nonid = [
        r
        for r in all_rows
        if int(r["episode_index"]) in hold_ids and str(r.get("kind")) not in ("none", "identity")
    ]
    disc_nonid = [
        r
        for r in all_rows
        if int(r["episode_index"]) in disc_ids and str(r.get("kind")) not in ("none", "identity")
    ]
    hold_by_s = rate_by_scale(hold_nonid)
    disc_by_s = rate_by_scale(disc_nonid)
    r05 = float(hold_by_s.get("0.5", float("nan")))
    r10 = float(hold_by_s.get("1.0", float("nan")))
    r20 = float(hold_by_s.get("2.0", float("nan")))
    slack = float(cfg["monotone_slack"])
    monotone = True
    if np.isfinite(r05) and np.isfinite(r10) and r05 + slack < r10:
        monotone = False
    if np.isfinite(r10) and np.isfinite(r20) and r10 + slack < r20:
        monotone = False

    min_base = float(cfg["min_baseline_insert_ok_rate"])
    if base_rate < min_base:
        verdict = "branch3_replay_env_broken"
        decision = "pause_algo_fix_replay_env"
        reason = f"identity continuation insert_ok_rate={base_rate:.3f} < {min_base}"
    elif np.isfinite(r05) and r05 <= float(cfg["island_max_rate_at_0_5"]):
        verdict = "branch2_narrow_island"
        decision = "redesign_data_generation"
        reason = f"held_out non-id rate@0.5={r05:.3f} <= island_max"
    elif (
        np.isfinite(r05)
        and r05 >= float(cfg["neighborhood_min_rate_at_0_5"])
        and monotone
    ):
        verdict = "branch1_continuous_recoverable_neighborhood"
        decision = "neighborhood_exists_study_coverage"
        reason = f"held_out rate@0.5={r05:.3f}, monotone_ok={monotone}"
    elif np.isfinite(r05) and r05 >= float(cfg["neighborhood_min_rate_at_0_5"]):
        verdict = "branch1_partial_neighborhood_nonmonotone"
        decision = "neighborhood_exists_study_coverage_with_caveat"
        reason = f"held_out rate@0.5={r05:.3f} but not monotone (slack={slack})"
    else:
        verdict = "branch2_narrow_or_sparse"
        decision = "redesign_data_generation"
        reason = f"held_out rate@0.5={r05:.3f} below neighborhood_min"

    return {
        "verdict": verdict,
        "decision": decision,
        "reason": reason,
        "baseline_insert_ok_rate": base_rate,
        "held_out_nonid_rate_by_scale": hold_by_s,
        "discovery_nonid_rate_by_scale": disc_by_s,
        "monotone_ok": bool(monotone),
        "n_baseline": len(baseline),
        "n_held_out_nonid": len(hold_nonid),
        "n_discovery_nonid": len(disc_nonid),
    }


def write_report(path: Path, judgment: dict, cfg: dict, n_rows: int) -> None:
    lines = [
        "# Demo Handoff Perturbation Recoverability P0 — Result",
        "",
        f"- UTC: `{_utc()}`",
        f"- Protocol: `{PROTOCOL}`",
        f"- Rows: `{n_rows}`",
        f"- Verdict: `{judgment['verdict']}`",
        f"- Decision: `{judgment['decision']}`",
        f"- Reason: {judgment['reason']}",
        "",
        "## Rates",
        "",
        f"- Baseline (identity) insert_ok: `{judgment['baseline_insert_ok_rate']:.3f}`",
        f"- Held-out non-id by scale: `{json.dumps(judgment['held_out_nonid_rate_by_scale'])}`",
        f"- Discovery non-id by scale: `{json.dumps(judgment['discovery_nonid_rate_by_scale'])}`",
        f"- Monotone ok: `{judgment['monotone_ok']}`",
        "",
        "## Gates (pre-registered)",
        "",
        f"- min_baseline_insert_ok_rate: `{cfg['min_baseline_insert_ok_rate']}`",
        f"- neighborhood_min_rate_at_0_5: `{cfg['neighborhood_min_rate_at_0_5']}`",
        f"- island_max_rate_at_0_5: `{cfg['island_max_rate_at_0_5']}`",
        "",
        "## Note",
        "",
        "不依赖 PrivHI；不做策略训练；纯 demo continuation + 预注册微扰。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "handoff_perturb_recoverability_p0.yaml",
    )
    ap.add_argument("--smoke", action="store_true", help="1 discovery ep, 2 scales, few perts")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sidecar = Path(cfg["sidecar_dir"])
    out_dir = PROJECT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = PROJECT_ROOT / cfg["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / cfg["report_path"]

    disc = [int(x) for x in cfg["episodes"]["discovery"]]
    hold = [int(x) for x in cfg["episodes"]["held_out"]]
    scales = [float(x) for x in cfg["amplitude_scales"]]
    perts = list(cfg["perturbations"])
    base = {k: float(v) for k, v in cfg["base"].items()}
    max_steps = int(cfg["max_continuation_steps"])

    if args.smoke:
        disc = disc[:1]
        hold = hold[:1]
        scales = [0.0, 1.0]
        perts = [p for p in perts if p.get("kind") in ("none", "tip_lat", "finger")][:4]

    episode_ids = sorted(set(disc + hold))
    entries = load_manifest_entries(sidecar, episode_indices=episode_ids)
    by_ep = {int(e["episode_index"]): e for e in entries}

    all_rows: list[dict[str, Any]] = []
    env = make_full_env(episode_ids, sidecar_dir=sidecar, seed=int(cfg.get("seed", 0)))

    for ep in episode_ids:
        entry = by_ep[ep]
        split = "discovery" if ep in disc else "held_out"
        print(f"[ep {ep}] capture handoff ({split})", flush=True)
        snap, root = capture_handoff(env, entry, sidecar)
        print(
            f"[ep {ep}] handoff tip={root['tip_m']:.4f} lat={root['lat_m']:.4f} "
            f"peg_ok={root['peg_ok']} frame={root['frame']}",
            flush=True,
        )
        for pcfg in perts:
            kind = str(pcfg.get("kind", "none"))
            for scale in scales:
                if kind in ("none", "identity") and scale != 0.0:
                    continue
                if kind not in ("none", "identity") and scale == 0.0:
                    continue
                print(f"  run {pcfg.get('name')} scale={scale}", flush=True)
                try:
                    row = run_cell(
                        env,
                        snap,
                        pert_cfg=pcfg,
                        scale=scale,
                        base=base,
                        max_steps=max_steps,
                    )
                except Exception as e:
                    row = {
                        "pert_name": str(pcfg.get("name")),
                        "kind": kind,
                        "scale": float(scale),
                        "insert_ok": False,
                        "error": str(e),
                        "term_reason": "exception",
                    }
                row["episode_index"] = int(ep)
                row["split"] = split
                row["handoff"] = root
                all_rows.append(row)
                print(
                    f"    insert_ok={row.get('insert_ok')} tip_min={row.get('tip_min_m')} "
                    f"reason={row.get('term_reason')}",
                    flush=True,
                )

    judgment = judge(all_rows, cfg)
    payload = {
        "protocol": PROTOCOL,
        "utc": _utc(),
        "config": cfg,
        "judgment": judgment,
        "rows": all_rows,
        "smoke": bool(args.smoke),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "utc": _utc(),
                "verdict": judgment["verdict"],
                "decision": judgment["decision"],
                "baseline_insert_ok_rate": judgment["baseline_insert_ok_rate"],
                "held_out_nonid_rate_by_scale": judgment["held_out_nonid_rate_by_scale"],
                "n_rows": len(all_rows),
                "results": str(out_dir / "results.json"),
                "report": str(report_path),
                "smoke": bool(args.smoke),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(report_path, judgment, cfg, len(all_rows))

    state_path = PROJECT_ROOT / "outputs" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state.update(
        {
            "date": "2026-08-15",
            "phase": "handoff_perturb_recoverability_p0",
            "busy": False,
            "training_allowed": False,
            "collection_allowed": False,
            "handoff_support_audit": {
                "status": "superseded",
                "reason": "replaced_by_perturb_recoverability_p0",
            },
            "handoff_perturb_recoverability_p0": {
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
        f"\n## 2026-08-15：Demo Handoff Perturbation Recoverability P0\n\n"
        f"- 判定：`{judgment['verdict']}` → `{judgment['decision']}`\n"
        f"- 原因：{judgment['reason']}\n"
        f"- Baseline insert_ok：`{judgment['baseline_insert_ok_rate']:.3f}`\n"
        f"- Held-out non-id by scale：`{judgment['held_out_nonid_rate_by_scale']}`\n"
        f"- 报告：`docs/HANDOFF_PERTURB_RECOVERABILITY_P0_RESULT.md`\n"
        f"- smoke=`{bool(args.smoke)}`\n"
    )
    if prog.exists():
        text = prog.read_text(encoding="utf-8")
        if "Perturbation Recoverability P0" not in text:
            # insert after title block
            parts = text.split("\n## 下一步", 1)
            if len(parts) == 2:
                prog.write_text(parts[0] + block + "\n## 下一步" + parts[1], encoding="utf-8")
            else:
                prog.write_text(text + block, encoding="utf-8")
    print(json.dumps(judgment, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
