#!/usr/bin/env python3
"""P0-C1.1: calibrate matched transport-load dose (no finger intervention)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.physics.grasp_metrics import (  # noqa: E402
    compute_step_metrics,
    control_dt_seconds,
    object_in_hand_pose,
    peg_hand_contact_counts,
    summarize_rollout_metrics_v2,
)
from embodied_grasp_insertion.physics.unstable_root_criteria import (  # noqa: E402
    label_root,
)
from embodied_grasp_insertion.simulation.full_episode_snapshot import (  # noqa: E402
    FullEpisodeSnapshot,
)
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    build_wrist_sequence,
    load_yaml,
    make_full_env,
    replay_demo_to_frame,
    select_roots_for_episode,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pad44(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).ravel()
    out = np.zeros(44, dtype=np.float64)
    out[: min(44, a.size)] = a[:44]
    return out


def _screen(env, snap, wrist, horizon, root_o2h, root_contact, root_z, dt):
    snap.restore(env)
    steps = []
    prev = root_o2h
    for a in wrist[:horizon]:
        if env._done:
            break
        env.step(a)
        m = compute_step_metrics(env, root_o2h=root_o2h, prev_o2h=prev, dt=dt)
        steps.append(m)
        prev = m.object_in_hand
        if env._done:
            break
    return summarize_rollout_metrics_v2(
        steps,
        root_o2h=root_o2h,
        root_contact=root_contact,
        control_dt_s=dt,
        root_peg_world_z=root_z,
    )


def _score_dose(rows: list[dict[str, Any]], criteria: dict[str, Any]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"ok": False, "reason": "no_rows"}
    n_drop = sum(1 for r in rows if r["load"].get("object_dropped_proxy"))
    n_unstable = 0
    n_stable = 0
    n_load_reason = 0
    drift_deltas = []
    for r in rows:
        lab = label_root(
            r["hold"],
            r["load"],
            root_contact_total=int(r["root_contact_total"]),
            criteria=criteria,
        )
        if lab["unstable_flag"]:
            n_unstable += 1
            if any(x.startswith("load_") for x in lab["unstable_reasons"]):
                n_load_reason += 1
        elif lab["stable_control_flag"]:
            n_stable += 1
        drift_deltas.append(
            float(r["load"]["trans_drift_max_m"] - r["hold"]["trans_drift_max_m"])
        )
    drop_frac = n_drop / n
    # Prefer: some unstable with load reasons, enough stable, not mostly dropping.
    ok = (
        n_stable >= 3
        and n_unstable >= 4
        and n_load_reason >= 2
        and drop_frac <= 0.35
        and float(np.median(drift_deltas)) > 0.0005
    )
    return {
        "ok": ok,
        "n": n,
        "n_stable": n_stable,
        "n_unstable": n_unstable,
        "n_load_reason": n_load_reason,
        "drop_frac": drop_frac,
        "median_drift_delta_m": float(np.median(drift_deltas)),
        "mean_drift_delta_m": float(np.mean(drift_deltas)),
    }


def _set_busy(busy: bool, *, phase: str, extra: dict[str, Any] | None = None) -> None:
    state_path = PROJECT_ROOT / "outputs" / "state.json"
    if not state_path.exists():
        return
    st = json.loads(state_path.read_text(encoding="utf-8"))
    st["busy"] = bool(busy)
    st["phase"] = phase
    st["updated_at"] = _utc()
    if extra:
        st.update(extra)
    state_path.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs/finger_controllability_c1_1.yaml"),
    )
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))
    load_cfg = cfg.get("transport_load_calibration", {})
    criteria = cfg.get("unstable_criteria", {})
    episodes = [int(x) for x in load_cfg.get("episodes", cfg.get("episodes", [0, 2, 4, 6, 8, 10]))]
    horizon = int(load_cfg.get("horizon", 16))
    scales = [float(x) for x in load_cfg.get("scales", [1.0, 2.0, 3.0, 4.0])]
    profiles = [
        str(x)
        for x in load_cfg.get("profiles", ["constant", "shake", "impulse_hold", "go_return"])
    ]
    base = _pad44(cfg.get("mild_transport_delta44_base", cfg.get("mild_transport_delta44", [])))
    bases = {"default": base}
    if load_cfg.get("shake_delta44_base"):
        bases["shake"] = _pad44(load_cfg["shake_delta44_base"])
    sidecar = Path(cfg["sidecar_dir"])
    seed = int(cfg.get("seed", 0))
    max_scan = load_cfg.get("max_scan_frames", 500)

    _set_busy(True, phase="transport_load_calibration_c1_1")
    dose_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    diagnostic_best: dict[str, Any] | None = None
    n_candidates = 0
    exit_code = 0
    try:
        # Stream per episode to avoid holding all MjData deepcopies at once.
        from interaction_retarget.constants import PEG_BODY

        env = make_full_env(episodes, sidecar_dir=sidecar, seed=seed)
        try:
            dt = control_dt_seconds(env)
            # First pass: hold metrics only (light); store root identity, re-capture for loads.
            hold_rows: list[dict[str, Any]] = []
            for ep in episodes:
                print(f"[load-cal] collect hold ep {ep}", flush=True)
                env.reset(episode_index=ep)
                rs = cfg.get("root_selection", {})
                roots = select_roots_for_episode(
                    env,
                    early_offset=int(rs.get("early_offset", 5)),
                    transport_tip_min_m=float(rs.get("transport_tip_min_m", 0.08)),
                    preinsert_tip_max_m=float(rs.get("preinsert_tip_max_m", 0.06)),
                    max_scan_frames=max_scan,
                )
                for root in roots:
                    env.reset(episode_index=ep)
                    replay_demo_to_frame(env, int(root.frame))
                    outcome = env._labeler.compute(env._raw)
                    if not outcome.peg_ok or outcome.insert_ok:
                        continue
                    snap = FullEpisodeSnapshot.capture(env)
                    root_o2h = object_in_hand_pose(env._raw)
                    root_contact = peg_hand_contact_counts(env._raw)
                    peg_id = int(env._raw._model.body(PEG_BODY).id)
                    root_z = float(env._raw._data.xpos[peg_id, 2])
                    wrist_hold = build_wrist_sequence(
                        source="hold", horizon=horizon, mild_transport_delta=base
                    )
                    hold_m = _screen(
                        env, snap, wrist_hold, horizon, root_o2h, root_contact, root_z, dt
                    )
                    hold_rows.append(
                        {
                            "episode_index": ep,
                            "frame": int(root.frame),
                            "phase": root.phase,
                            "root_contact_total": int(root_contact.total),
                            "hold": hold_m,
                        }
                    )
                    del snap  # release MjData deepcopy ASAP
                n_candidates = len(hold_rows)
                print(f"[load-cal] hold roots so far={n_candidates}", flush=True)

            for profile in profiles:
                base_use = bases.get(profile, bases["default"])
                for scale in scales:
                    mild = base_use * float(scale)
                    wrist_load = build_wrist_sequence(
                        source="mild_transport",
                        horizon=horizon,
                        mild_transport_delta=mild,
                        profile=profile,
                    )
                    rows: list[dict[str, Any]] = []
                    # Re-capture root snapshots episode-by-episode (deterministic replay).
                    by_ep: dict[int, list[dict[str, Any]]] = {}
                    for c in hold_rows:
                        by_ep.setdefault(int(c["episode_index"]), []).append(c)
                    for ep, clist in by_ep.items():
                        print(
                            f"[load-cal] profile={profile} scale={scale} ep={ep}",
                            flush=True,
                        )
                        env.reset(episode_index=ep)
                        for c in clist:
                            env.reset(episode_index=ep)
                            replay_demo_to_frame(env, int(c["frame"]))
                            snap = FullEpisodeSnapshot.capture(env)
                            root_o2h = object_in_hand_pose(env._raw)
                            root_contact = peg_hand_contact_counts(env._raw)
                            peg_id = int(env._raw._model.body(PEG_BODY).id)
                            root_z = float(env._raw._data.xpos[peg_id, 2])
                            load_m = _screen(
                                env,
                                snap,
                                wrist_load,
                                horizon,
                                root_o2h,
                                root_contact,
                                root_z,
                                dt,
                            )
                            del snap
                            rows.append(
                                {
                                    "episode_index": c["episode_index"],
                                    "frame": c["frame"],
                                    "phase": c["phase"],
                                    "root_contact_total": c["root_contact_total"],
                                    "hold": c["hold"],
                                    "load": load_m,
                                    "drift_delta_m": float(
                                        load_m["trans_drift_max_m"] - c["hold"]["trans_drift_max_m"]
                                    ),
                                }
                            )
                    score = _score_dose(rows, criteria)
                    entry = {
                        "profile": profile,
                        "scale": scale,
                        "delta44": mild.tolist(),
                        "score": score,
                        "per_root": [
                            {
                                "episode_index": r["episode_index"],
                                "frame": r["frame"],
                                "phase": r["phase"],
                                "hold_drift": r["hold"]["trans_drift_max_m"],
                                "load_drift": r["load"]["trans_drift_max_m"],
                                "drift_delta_m": r["drift_delta_m"],
                                "hold_ret": r["hold"]["contact_retention_vs_root_mean"],
                                "load_ret": r["load"]["contact_retention_vs_root_mean"],
                                "hold_slip": r["hold"]["slip_proxy_tangential_rel_vel_mean_mps"],
                                "load_slip": r["load"]["slip_proxy_tangential_rel_vel_mean_mps"],
                                "load_drop": r["load"].get("object_dropped_proxy"),
                                "load_peg_ok": r["load"].get("terminal_peg_ok"),
                                "hold_peg_ok": r["hold"].get("terminal_peg_ok"),
                            }
                            for r in rows
                        ],
                    }
                    dose_rows.append(entry)
                    print(
                        f"[load-cal] profile={profile} scale={scale} "
                        f"score={json.dumps(score, ensure_ascii=False)}",
                        flush=True,
                    )
                    if selected is None and score.get("ok"):
                        selected = entry
            if selected is None and dose_rows:
                ranked = sorted(
                    dose_rows,
                    key=lambda e: (
                        e["score"].get("drop_frac", 1.0) <= 0.5,
                        e["score"].get("n_stable", 0),
                        e["score"].get("n_load_reason", 0),
                        e["score"].get("n_unstable", 0),
                        e["score"].get("median_drift_delta_m", 0.0),
                    ),
                    reverse=True,
                )
                best = ranked[0]
                diagnostic_best = {
                    "profile": best.get("profile"),
                    "scale": best.get("scale"),
                    "score": best.get("score"),
                    "delta44": best.get("delta44"),
                    "note": "diagnostic only; NOT a valid selected load",
                }
        finally:
            env.close()

        selection_ok = bool(selected and selected["score"].get("ok"))
        verdict = "ok" if selection_ok else "load_calibration_fail"
        manifest = {
            "name": "transport_load_calibration_v1",
            "created_at": _utc(),
            "protocol": "P0-C1.1",
            "verdict": verdict,
            "episodes": episodes,
            "base_delta44": base.tolist(),
            "scales": scales,
            "profiles": profiles,
            "criteria": criteria,
            "n_candidates": n_candidates,
            "dose_response": [
                {
                    "profile": e["profile"],
                    "scale": e["scale"],
                    "score": e["score"],
                    "n_per_root": len(e["per_root"]),
                }
                for e in dose_rows
            ],
            "dose_response_detail": dose_rows,
            # Valid selection only when score.ok; never write fallback into selected_*.
            "selected_profile": None if not selection_ok else selected.get("profile"),
            "selected_scale": None if not selection_ok else selected.get("scale"),
            "selected_delta44": None if not selection_ok else selected.get("delta44"),
            "selected_score": None if not selection_ok else selected.get("score"),
            "selected_fallback": False,
            "selection_ok": selection_ok,
            "diagnostic_best_failed_dose": None if selection_ok else diagnostic_best,
        }
        out = PROJECT_ROOT / cfg.get(
            "transport_load_manifest", "data/manifests/transport_load_calibration_v1.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report = [
            "# Transport Load Calibration (P0-C1.1)",
            "",
            f"- 日期：{manifest['created_at']}",
            f"- verdict：{verdict}",
            f"- candidates：{manifest['n_candidates']}",
            f"- selected_profile：{manifest['selected_profile']}",
            f"- selected_scale：{manifest['selected_scale']}",
            f"- selection_ok：{manifest['selection_ok']}",
            f"- selected_score：{json.dumps(manifest['selected_score'], ensure_ascii=False)}",
            f"- diagnostic_best_failed_dose：{json.dumps(manifest['diagnostic_best_failed_dose'], ensure_ascii=False)}",
            "",
            "## Dose response",
        ]
        for e in dose_rows:
            report.append(
                f"- profile={e['profile']} scale={e['scale']}: unstable={e['score'].get('n_unstable')} "
                f"stable={e['score'].get('n_stable')} load_reason={e['score'].get('n_load_reason')} "
                f"drop_frac={e['score'].get('drop_frac'):.3f} "
                f"median_d_drift={e['score'].get('median_drift_delta_m'):.5f}"
            )
        rep = PROJECT_ROOT / cfg.get(
            "transport_load_report", "docs/TRANSPORT_LOAD_CALIBRATION.md"
        )
        rep.write_text("\n".join(report) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "manifest": str(out),
                    "verdict": verdict,
                    "selection_ok": selection_ok,
                    "selected_scale": manifest["selected_scale"],
                    "selected_profile": manifest["selected_profile"],
                },
                ensure_ascii=False,
            )
        )
        if not selection_ok:
            exit_code = 2
    except Exception:
        exit_code = 1
        raise
    finally:
        _set_busy(
            False,
            phase="transport_load_calibration_c1_1",
            extra={
                "verdict": "screening_fail" if exit_code else "partial",
                "p0c1_1": {
                    "verdict": "ok" if exit_code == 0 else "load_calibration_fail",
                    "selection_ok": exit_code == 0,
                    "selected_fallback": False,
                    "allow_extended_controllability_p0": False,
                    "allow_observability_p0": False,
                    "allow_semantic_p0": False,
                    "allow_policy_training": False,
                },
            },
        )
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
