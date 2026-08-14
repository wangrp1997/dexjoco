#!/usr/bin/env python3
"""Screen unstable-but-recoverable grasp roots (P0-C1), no finger interventions."""

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
        default=str(PROJECT_ROOT / "configs/finger_controllability_calibrated_smoke.yaml"),
    )
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))
    screen = cfg.get("root_screening", {})
    episodes = [int(x) for x in screen.get("episodes", cfg.get("episodes", [0, 2, 4, 6, 8, 10]))]
    horizon = int(screen.get("horizon", 24))
    sidecar = Path(cfg["sidecar_dir"])
    seed = int(cfg.get("seed", 0))
    max_scan = screen.get("max_scan_frames", cfg.get("root_selection", {}).get("max_scan_frames"))

    mild = np.asarray(cfg.get("mild_transport_delta44", np.zeros(44)), dtype=np.float64)
    if mild.size < 44:
        tmp = np.zeros(44, dtype=np.float64)
        tmp[: mild.size] = mild
        mild = tmp
    else:
        mild = mild[:44]
    wrist_hold = build_wrist_sequence(source="hold", horizon=horizon, mild_transport_delta=mild)
    wrist_load = build_wrist_sequence(
        source="mild_transport", horizon=horizon, mild_transport_delta=mild
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
                    }
                )
    finally:
        env.close()

    valid = [c for c in candidates if not c.get("excluded")]
    # Freeze thresholds from hold-screen distribution BEFORE labeling unstable.
    hold_drift = np.array([c["hold_screen"]["trans_drift_max_m"] for c in valid], dtype=np.float64)
    hold_ret = np.array(
        [c["hold_screen"]["contact_retention_vs_root_mean"] for c in valid], dtype=np.float64
    )
    drift_q = float(np.quantile(hold_drift, float(screen.get("unstable_drift_quantile", 0.6))))
    ret_q = float(np.quantile(hold_ret, float(screen.get("unstable_retention_quantile", 0.4))))

    unstable = []
    stable = []
    excluded = [c for c in candidates if c.get("excluded")]
    for c in valid:
        hm, lm = c["hold_screen"], c["load_screen"]
        # Must not immediately drop under hold.
        if hm.get("object_dropped_proxy"):
            excluded.append({**c, "excluded": True, "reason": "hold_object_dropped_proxy"})
            continue
        if not hm.get("terminal_peg_ok") and hm.get("peg_contact_absent_steps", 0) >= horizon // 2:
            excluded.append({**c, "excluded": True, "reason": "hold_lost_contact_early"})
            continue
        # Unstable if hold or load shows elevated drift / reduced retention / slip, but still has window.
        unstable_flag = (
            (hm["trans_drift_max_m"] >= drift_q)
            or (hm["contact_retention_vs_root_mean"] <= ret_q)
            or (lm["trans_drift_max_m"] >= drift_q)
            or (lm["contact_retention_vs_root_mean"] <= ret_q)
            or (lm["slip_proxy_tangential_rel_vel_mean_mps"] > hm["slip_proxy_tangential_rel_vel_mean_mps"] * 1.2)
        )
        # Still intervenable: contact present at root and not dropped on hold.
        intervenable = c["root_contact_total"] > 0 and hm.get("terminal_peg_ok", False)
        row = {
            **c,
            "thresholds": {"hold_drift_q": drift_q, "hold_retention_q": ret_q},
            "unstable_flag": bool(unstable_flag),
            "intervenable": bool(intervenable),
        }
        if unstable_flag and intervenable:
            unstable.append(row)
        elif intervenable:
            stable.append(row)
        else:
            excluded.append({**row, "excluded": True, "reason": "not_intervenable"})

    # Cap and ensure diversity.
    max_roots = int(screen.get("max_unstable_roots", 8))
    max_stable = int(screen.get("max_stable_controls", 3))
    # Prefer covering phases and episodes.
    unstable_sel = []
    seen_ep = set()
    for phase in ("pre_insert", "transport", "early_grasp"):
        for u in unstable:
            if u["phase"] != phase:
                continue
            if len(unstable_sel) >= max_roots:
                break
            unstable_sel.append(u)
            seen_ep.add(u["episode_index"])
        if len(unstable_sel) >= max_roots:
            break
    for u in unstable:
        if len(unstable_sel) >= max_roots:
            break
        if u in unstable_sel:
            continue
        unstable_sel.append(u)
        seen_ep.add(u["episode_index"])

    stable_sel = stable[:max_stable]

    manifest = {
        "name": "unstable_grasp_roots_v1",
        "created_at": _utc(),
        "episodes_scanned": episodes,
        "n_candidates": len(candidates),
        "n_valid": len(valid),
        "thresholds_frozen_before_hand_interventions": {
            "hold_drift_quantile": drift_q,
            "hold_retention_quantile": ret_q,
        },
        "unstable_roots": unstable_sel,
        "stable_control_roots": stable_sel,
        "excluded_roots": excluded,
        "summary": {
            "n_unstable": len(unstable_sel),
            "n_stable_controls": len(stable_sel),
            "n_episodes_unstable": len({u["episode_index"] for u in unstable_sel}),
        },
    }
    out = PROJECT_ROOT / cfg.get(
        "unstable_roots_manifest", "data/manifests/unstable_grasp_roots_v1.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "n_unstable": len(unstable_sel),
                "n_stable": len(stable_sel),
                "n_episodes": manifest["summary"]["n_episodes_unstable"],
                "manifest": str(out),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
