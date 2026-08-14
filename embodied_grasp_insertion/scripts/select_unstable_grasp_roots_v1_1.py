#!/usr/bin/env python3
"""P0-C1.1: screen unstable/stable grasp roots with absolute reasons (no finger intervention)."""

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
    screening_gate,
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


def _screen_branch(env, snap, wrist_seq, horizon, root_o2h, root_contact, root_z, dt):
    snap.restore(env)
    steps = []
    prev = root_o2h
    for a in wrist_seq[:horizon]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs/finger_controllability_c1_1.yaml"),
    )
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))
    screen = cfg.get("root_screening", {})
    criteria = cfg.get("unstable_criteria", {})
    episodes = [int(x) for x in screen.get("episodes", cfg.get("episodes", [0, 2, 4, 6, 8, 10]))]
    horizon = int(screen.get("horizon", 16))
    sidecar = Path(cfg["sidecar_dir"])
    seed = int(cfg.get("seed", 0))
    max_scan = screen.get("max_scan_frames", cfg.get("root_selection", {}).get("max_scan_frames"))

    mild = _pad44(cfg.get("mild_transport_delta44", np.zeros(44)))
    load_profile = str(cfg.get("wrist_load_profile", "constant"))
    load_man_path = cfg.get("transport_load_manifest")
    if not load_man_path:
        print(json.dumps({"label": "screening_fail", "reason": "missing transport_load_manifest"}))
        raise SystemExit(2)
    lp = PROJECT_ROOT / load_man_path
    if not lp.exists():
        print(json.dumps({"label": "screening_fail", "reason": f"missing {lp}"}))
        raise SystemExit(2)
    lm = json.loads(lp.read_text(encoding="utf-8"))
    if not lm.get("selection_ok"):
        print(
            json.dumps(
                {
                    "label": "screening_fail",
                    "reason": "transport load selection_ok=false",
                    "load_verdict": lm.get("verdict"),
                    "selected_fallback_forbidden": True,
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)
    if lm.get("selected_fallback"):
        print(
            json.dumps(
                {
                    "label": "screening_fail",
                    "reason": "selected_fallback is forbidden for root screening",
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)
    if not lm.get("selected_delta44"):
        print(
            json.dumps(
                {
                    "label": "screening_fail",
                    "reason": "selected_delta44 is null",
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)
    mild = _pad44(lm["selected_delta44"])
    if lm.get("selected_profile"):
        load_profile = str(lm["selected_profile"])

    wrist_hold = build_wrist_sequence(source="hold", horizon=horizon, mild_transport_delta=mild)
    wrist_load = build_wrist_sequence(
        source="mild_transport",
        horizon=horizon,
        mild_transport_delta=mild,
        profile=load_profile,
    )

    candidates: list[dict[str, Any]] = []
    env = make_full_env(episodes, sidecar_dir=sidecar, seed=seed)
    try:
        dt = control_dt_seconds(env)
        for ep in episodes:
            print(f"[screen] episode {ep}", flush=True)
            env.reset(episode_index=ep)
            rs = cfg.get("root_selection", {})
            roots = select_roots_for_episode(
                env,
                early_offset=int(rs.get("early_offset", 5)),
                transport_tip_min_m=float(rs.get("transport_tip_min_m", 0.08)),
                preinsert_tip_max_m=float(rs.get("preinsert_tip_max_m", 0.06)),
                max_scan_frames=max_scan,
            )
            from interaction_retarget.constants import PEG_BODY

            for root in roots:
                env.reset(episode_index=ep)
                replay_demo_to_frame(env, int(root.frame))
                outcome = env._labeler.compute(env._raw)
                if not outcome.peg_ok or outcome.insert_ok:
                    candidates.append(
                        {
                            "episode_index": ep,
                            "frame": root.frame,
                            "phase": root.phase,
                            "excluded": True,
                            "reason": f"peg_ok={outcome.peg_ok}, insert_ok={outcome.insert_ok}",
                            "unstable_reasons": [],
                        }
                    )
                    continue
                snap = FullEpisodeSnapshot.capture(env)
                root_o2h = object_in_hand_pose(env._raw)
                root_contact = peg_hand_contact_counts(env._raw)
                peg_id = int(env._raw._model.body(PEG_BODY).id)
                root_z = float(env._raw._data.xpos[peg_id, 2])
                hold_m = _screen_branch(
                    env, snap, wrist_hold, horizon, root_o2h, root_contact, root_z, dt
                )
                load_m = _screen_branch(
                    env, snap, wrist_load, horizon, root_o2h, root_contact, root_z, dt
                )
                candidates.append(
                    {
                        "episode_index": ep,
                        "frame": int(root.frame),
                        "phase": root.phase,
                        "reason": root.reason,
                        "excluded": False,
                        "root_contact_total": int(root_contact.total),
                        "hold_screen": hold_m,
                        "load_screen": load_m,
                        "load_delta_used": mild.tolist(),
                    }
                )
    finally:
        env.close()

    valid = [c for c in candidates if not c.get("excluded")]
    hold_drift = np.array([c["hold_screen"]["trans_drift_max_m"] for c in valid], dtype=np.float64)
    # Informative only — never sole unstable reason.
    drift_q = (
        float(np.quantile(hold_drift, float(screen.get("info_drift_quantile", 0.55))))
        if len(hold_drift)
        else 0.0
    )

    unstable: list[dict[str, Any]] = []
    stable: list[dict[str, Any]] = []
    excluded = [c for c in candidates if c.get("excluded")]
    for c in valid:
        hm, lm = c["hold_screen"], c["load_screen"]
        if hm.get("object_dropped_proxy"):
            excluded.append(
                {
                    **c,
                    "excluded": True,
                    "reason": "hold_object_dropped_proxy",
                    "unstable_reasons": [],
                }
            )
            continue
        if not hm.get("terminal_peg_ok") and hm.get("peg_contact_absent_steps", 0) >= horizon // 2:
            excluded.append(
                {
                    **c,
                    "excluded": True,
                    "reason": "hold_lost_contact_early",
                    "unstable_reasons": [],
                }
            )
            continue
        lab = label_root(
            hm,
            lm,
            root_contact_total=int(c["root_contact_total"]),
            criteria=criteria,
            hold_drift_threshold=drift_q,
        )
        row = {
            **c,
            "thresholds": {
                "info_hold_drift_quantile": drift_q,
                "criteria": criteria,
            },
            **lab,
        }
        if lab["unstable_flag"]:
            unstable.append(row)
        elif lab["stable_control_flag"]:
            stable.append(row)
        else:
            excluded.append({**row, "excluded": True, "reason": "not_intervenable"})

    max_roots = int(screen.get("max_unstable_roots", 8))
    max_stable = int(screen.get("max_stable_controls", 8))
    # Prefer phase/episode diversity for unstable.
    unstable_sel: list[dict[str, Any]] = []
    for phase in ("pre_insert", "transport", "early_grasp"):
        for u in unstable:
            if u["phase"] != phase:
                continue
            if len(unstable_sel) >= max_roots:
                break
            if u in unstable_sel:
                continue
            unstable_sel.append(u)
        if len(unstable_sel) >= max_roots:
            break
    for u in unstable:
        if len(unstable_sel) >= max_roots:
            break
        if u in unstable_sel:
            continue
        unstable_sel.append(u)

    # Prefer low-drift stables as controls.
    stable_sorted = sorted(stable, key=lambda r: float(r["hold_screen"]["trans_drift_max_m"]))
    stable_sel = stable_sorted[:max_stable]

    gate = screening_gate(
        unstable_sel,
        stable_sel,
        min_unstable=int(screen.get("min_unstable", 4)),
        min_stable=int(screen.get("min_stable", 3)),
        min_unstable_episodes=int(screen.get("min_unstable_episodes", 3)),
    )
    # Load must change at least one physical metric on some unstable root.
    load_effect = False
    for u in unstable_sel:
        reasons = u.get("unstable_reasons") or []
        if any(
            r.startswith("load_") or r.endswith("_vs_hold")
            for r in reasons
        ):
            load_effect = True
            break
        hm, lm = u["hold_screen"], u["load_screen"]
        if abs(lm["trans_drift_max_m"] - hm["trans_drift_max_m"]) > 1e-4:
            load_effect = True
            break
        if abs(lm["contact_retention_vs_root_mean"] - hm["contact_retention_vs_root_mean"]) > 1e-4:
            load_effect = True
            break
    gate["load_changes_metric"] = load_effect
    if not load_effect:
        gate["passed"] = False
        gate["label"] = "screening_fail"
        gate["checks"] = {**(gate.get("checks") or {}), "load_changes_metric": False}

    manifest = {
        "name": "unstable_grasp_roots_v1_1",
        "created_at": _utc(),
        "protocol": "P0-C1.1",
        "episodes_scanned": episodes,
        "n_candidates": len(candidates),
        "n_valid": len(valid),
        "load_delta44": mild.tolist(),
        "load_profile": load_profile,
        "thresholds_frozen_before_hand_interventions": {
            "info_hold_drift_quantile_only": drift_q,
            "note": "quantile drift is informational; unstable requires absolute reasons",
            "criteria": criteria,
        },
        "unstable_roots": unstable_sel,
        "stable_control_roots": stable_sel,
        "excluded_roots": excluded,
        "screening_gate": gate,
        "summary": {
            "n_unstable": len(unstable_sel),
            "n_stable_controls": len(stable_sel),
            "n_episodes_unstable": len({u["episode_index"] for u in unstable_sel}),
            "n_elevated_hold_drift_only": sum(
                1 for c in valid if c.get("elevated_hold_drift_only")
            ),
        },
    }
    out = PROJECT_ROOT / cfg.get(
        "unstable_roots_manifest", "data/manifests/unstable_grasp_roots_v1_1.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(out), **gate}, ensure_ascii=False))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
