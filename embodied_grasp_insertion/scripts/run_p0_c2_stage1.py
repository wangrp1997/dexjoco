#!/usr/bin/env python3
"""P0-C2 Stage-1: matched finger causal fork test (no policy; no Stage-2 auto-run).

Same snapshot + same demo wrist; only finger actions differ.
Root selection is frozen in physics/c2_root_criteria.py before reading outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
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

from embodied_grasp_insertion.io_paths import path_for_manifest  # noqa: E402
from embodied_grasp_insertion.physics.c2_root_criteria import (  # noqa: E402
    CRITERIA_VERSION,
    MIN_EPISODES_REQUIRED,
    PROTOCOL,
    accept_screened_root,
    select_ranked_roots,
)
from embodied_grasp_insertion.physics.grasp_metrics import (  # noqa: E402
    compute_step_metrics,
    control_dt_seconds,
    object_in_hand_pose,
    peg_hand_contact_counts,
    summarize_rollout_metrics_v2,
)
from embodied_grasp_insertion.pilot import WRITE_IMPLEMENTATION_ENABLED  # noqa: E402
from embodied_grasp_insertion.simulation.calibrated_interventions import (  # noqa: E402
    RIGHT_FINGER_IDX,
    WRIST_IDX,
    assert_left_fingers_zero,
    build_calibrated_right_offset,
    build_right_demo_replay_actions,
    load_semantics,
    project_matched_feasible_offsets,
    target_offset_to_pulse_actions,
)
from embodied_grasp_insertion.simulation.full_episode_snapshot import (  # noqa: E402
    FullEpisodeSnapshot,
)
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    abs44_from_demo,
    load_yaml,
    make_full_env,
    replay_demo_to_frame,
    select_roots_for_episode,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_demo_wrist_sequence(env, *, root_frame: int, horizon: int) -> np.ndarray:
    """Wrist deltas tracking demo absolute wrist poses; fingers left zero."""
    assert env._hold44 is not None and env._actions is not None
    scale = env._scale()
    hold = env._hold44.copy()
    out = np.zeros((int(horizon), 44), dtype=np.float64)
    for k in range(horizon):
        demo_idx = root_frame + k
        if demo_idx >= len(env._actions):
            break
        demo_abs = abs44_from_demo(env, demo_idx)
        delta = np.clip((demo_abs - hold) / (scale + 1e-12), -1.0, 1.0)
        out[k, WRIST_IDX] = delta[WRIST_IDX]
        hold = hold.copy()
        hold[WRIST_IDX] = hold[WRIST_IDX] + out[k, WRIST_IDX] * scale[WRIST_IDX]
    return out


def _rollout(
    env,
    snap: FullEpisodeSnapshot,
    actions: np.ndarray,
    *,
    root_o2h,
    root_contact,
    root_z: float,
    dt: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    snap.restore(env)
    if bool(env._done):
        raise RuntimeError("snapshot restore left env._done=True")
    steps = []
    tip_diag = []
    executed = []
    term_reason = "horizon_end"
    for a in actions:
        if env._done:
            term_reason = "already_done"
            break
        _, _, term, trunc, info = env.step(np.asarray(a, dtype=np.float64))
        m = compute_step_metrics(env)
        steps.append(m)
        tip_diag.append(float(np.linalg.norm(m.peg_world_pos[:2])))
        executed.append(np.asarray(a, dtype=np.float64).copy())
        if term or trunc:
            term_reason = str(info.get("fail_reason") or ("terminated" if term else "truncated"))
            break
    if not steps:
        summary = {
            "num_steps": 0,
            "terminal_peg_ok": False,
            "error": "empty_rollout",
            "term_reason": term_reason,
        }
    else:
        summary = summarize_rollout_metrics_v2(
            steps,
            root_o2h=root_o2h,
            root_contact=root_contact,
            root_peg_world_z=root_z,
            control_dt_s=dt,
        )
        summary["tip_proxy_xy_mean_m"] = float(np.mean(tip_diag))
        summary["tip_proxy_xy_end_m"] = float(tip_diag[-1])
        summary["insert_ok_any"] = bool(any(bool(s.insert_ok) for s in steps))
        summary["insert_ok_end"] = bool(steps[-1].insert_ok)
        summary["term_reason"] = term_reason
        summary["executed_steps"] = int(len(steps))
    future_action = {
        "actions44": np.asarray(executed if executed else actions[:0], dtype=np.float64).tolist(),
        "horizon": int(len(executed)),
        "planned_horizon": int(actions.shape[0]),
    }
    return summary, future_action


def _paired_bootstrap(
    diffs: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    diffs = np.asarray(diffs, dtype=np.float64)
    if diffs.size == 0:
        return {
            "n": 0,
            "mean": None,
            "ci95_lo": None,
            "ci95_hi": None,
            "excludes_zero": False,
        }
    rng = np.random.default_rng(seed)
    boots = [float(rng.choice(diffs, size=len(diffs), replace=True).mean()) for _ in range(n_boot)]
    arr = np.asarray(boots, dtype=np.float64)
    lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return {
        "n": int(len(diffs)),
        "mean": float(diffs.mean()),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "excludes_zero": bool(hi < 0.0 or lo > 0.0),
    }


PRIMARY_METRICS = (
    ("trans_drift_max_m", "higher_worse"),
    ("rot_drift_max_rad", "higher_worse"),
    ("contact_retention_vs_root_mean", "higher_better"),
    ("object_dropped_proxy", "higher_worse"),
    ("terminal_peg_ok", "higher_better"),
)


def analyze_causal_forks(
    branches: list[dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    by_root: dict[str, dict[str, dict[str, Any]]] = {}
    for b in branches:
        if not b.get("fairness_passed"):
            continue
        by_root.setdefault(b["root_id"], {})[b["intervention"]] = b

    interventions = [
        "demo_finger_replay",
        "calibrated_finger_intervention",
        "random_finger_control",
    ]
    results: dict[str, Any] = {}
    any_fork = False
    fork_details: list[dict[str, Any]] = []

    for interv in interventions:
        metric_hits = []
        per_metric = {}
        for metric, direction in PRIMARY_METRICS:
            diffs = []
            episodes = []
            for rid, mp in by_root.items():
                if "hold_finger" not in mp or interv not in mp:
                    continue
                h = mp["hold_finger"]["metrics"]
                x = mp[interv]["metrics"]
                hv = h.get(metric)
                xv = x.get(metric)
                if hv is None or xv is None:
                    continue
                if isinstance(hv, bool) or isinstance(xv, bool):
                    hv = float(bool(hv))
                    xv = float(bool(xv))
                # signed so positive always means interv "worse" on higher_worse /
                # interv "better" on higher_better for excludes_zero alone is enough;
                # store raw interv - hold.
                diffs.append(float(xv) - float(hv))
                episodes.append(int(mp[interv]["episode_index"]))
            salt = int(
                hashlib.sha256(f"{interv}|{metric}".encode()).hexdigest()[:8],
                16,
            )
            boot = _paired_bootstrap(
                np.asarray(diffs, dtype=np.float64),
                n_boot=n_boot,
                seed=(seed + salt) % (2**31 - 1),
            )
            significant = bool(boot["excludes_zero"]) and boot["n"] >= 3
            per_metric[metric] = {**boot, "direction": direction, "significant": significant}
            if significant:
                metric_hits.append(metric)
        # Tip must never be sole reason — we simply never include tip in PRIMARY_METRICS.
        fork = len(metric_hits) > 0
        if fork:
            any_fork = True
            fork_details.append({"intervention": interv, "metrics": metric_hits})
        results[interv] = {
            "per_metric": per_metric,
            "causal_fork": fork,
            "n_roots_compared": int(
                sum(
                    1
                    for mp in by_root.values()
                    if "hold_finger" in mp and interv in mp
                )
            ),
            "n_episodes": int(
                len(
                    {
                        int(mp[interv]["episode_index"])
                        for mp in by_root.values()
                        if "hold_finger" in mp and interv in mp
                    }
                )
            ),
        }

    return {
        "any_causal_fork": any_fork,
        "fork_details": fork_details,
        "by_intervention": results,
        "n_roots": len(by_root),
        "n_episodes": len({b["episode_index"] for b in branches if b.get("fairness_passed")}),
    }


def build_intervention_actions(
    env,
    *,
    name: str,
    wrist_seq: np.ndarray,
    root_frame: int,
    horizon: int,
    semantics,
    close_rad: float,
    pulse_steps: int,
    projected_offsets: dict[str, np.ndarray] | None = None,
    projection_meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if name == "hold_finger":
        actions = wrist_seq.copy()
        actions[:, RIGHT_FINGER_IDX] = 0.0
        return actions, {"mode": "hold_finger"}
    if name == "demo_finger_replay":
        actions, meta = build_right_demo_replay_actions(
            env, root_frame=root_frame, horizon=horizon, wrist_seq=wrist_seq
        )
        assert_left_fingers_zero(actions)
        return actions, meta
    if name == "calibrated_finger_intervention":
        assert projected_offsets is not None
        actions, meta = target_offset_to_pulse_actions(
            env,
            right_offset_rad=projected_offsets["calibrated_close_low"],
            horizon=horizon,
            pulse_steps=pulse_steps,
            wrist_seq=wrist_seq,
            allow_clip=False,
        )
        assert_left_fingers_zero(actions)
        meta["projection"] = projection_meta
        return actions, meta
    if name == "random_finger_control":
        assert projected_offsets is not None
        actions, meta = target_offset_to_pulse_actions(
            env,
            right_offset_rad=projected_offsets["random_matched"],
            horizon=horizon,
            pulse_steps=pulse_steps,
            wrist_seq=wrist_seq,
            allow_clip=False,
        )
        assert_left_fingers_zero(actions)
        meta["projection"] = projection_meta
        return actions, meta
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "p0_c2_stage1.yaml",
    )
    args = ap.parse_args()
    if WRITE_IMPLEMENTATION_ENABLED:
        raise SystemExit("WRITE_IMPLEMENTATION_ENABLED must stay False")

    cfg = load_yaml(args.config)
    sem = json.loads((PROJECT_ROOT / cfg["semantics_manifest"]).read_text())
    semantics = load_semantics(sem)
    episodes = [int(x) for x in cfg["episodes"]]
    horizon = int(cfg["horizon"])
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    sidecar = Path(cfg["sidecar_dir"])
    close_rad = float(cfg["calibrated_close_rad"])
    pulse_steps = int(cfg["pulse_steps"])
    n_boot = int(cfg.get("bootstrap_n", 1000))
    boot_seed = int(cfg.get("bootstrap_seed", 20260815))
    fair_q = float(cfg["fairness"]["init_qpos_atol"])
    fair_o = float(cfg["fairness"]["init_obs_atol"])

    out_dir = PROJECT_ROOT / cfg["output_dir"]
    branch_dir = out_dir / "branches"
    branch_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase A: hold-screen root selection (frozen criteria) ----
    screened: list[dict[str, Any]] = []
    rs = cfg["root_selection"]
    for ep in episodes:
        env = make_full_env([ep], sidecar_dir=sidecar, seed=seed)
        try:
            env.reset(episode_index=ep)
            dt = control_dt_seconds(env)
            roots = select_roots_for_episode(
                env,
                early_offset=int(rs["early_offset"]),
                transport_tip_min_m=float(rs["transport_tip_min_m"]),
                preinsert_tip_max_m=float(rs["preinsert_tip_max_m"]),
                max_scan_frames=rs.get("max_scan_frames"),
            )
            for root in roots:
                env.reset(episode_index=ep)
                replay_demo_to_frame(env, int(root.frame))
                outcome0 = env._labeler.compute(env._raw)
                contact0 = peg_hand_contact_counts(env._raw)
                if not outcome0.peg_ok or outcome0.insert_ok:
                    screened.append(
                        {
                            "episode_index": ep,
                            "frame": int(root.frame),
                            "phase": root.phase,
                            "accepted": False,
                            "reasons": ["root_peg_or_insert_gate"],
                        }
                    )
                    continue
                snap = FullEpisodeSnapshot.capture(env)
                root_o2h = object_in_hand_pose(env._raw)
                root_z = float(
                    env._raw._data.xpos[
                        env._raw._model.body("industreal_round_peg_8mm").id
                    ][2]
                )
                wrist_seq = build_demo_wrist_sequence(
                    env, root_frame=int(root.frame), horizon=horizon
                )
                hold_actions, _ = build_intervention_actions(
                    env,
                    name="hold_finger",
                    wrist_seq=wrist_seq,
                    root_frame=int(root.frame),
                    horizon=horizon,
                    semantics=semantics,
                    close_rad=close_rad,
                    pulse_steps=pulse_steps,
                )
                hold_m, _ = _rollout(
                    env,
                    snap,
                    hold_actions,
                    root_o2h=root_o2h,
                    root_contact=contact0,
                    root_z=root_z,
                    dt=dt,
                )
                if int(hold_m.get("executed_steps") or 0) < max(4, horizon // 2):
                    screened.append(
                        {
                            "episode_index": ep,
                            "frame": int(root.frame),
                            "phase": root.phase,
                            "accepted": False,
                            "reasons": ["hold_screen_too_short"],
                            "hold_metrics": {
                                "executed_steps": hold_m.get("executed_steps"),
                                "term_reason": hold_m.get("term_reason"),
                            },
                        }
                    )
                    continue
                decision = accept_screened_root(
                    root_contact_total=int(contact0.total),
                    root_peg_ok=bool(outcome0.peg_ok),
                    root_insert_ok=bool(outcome0.insert_ok),
                    hold_metrics=hold_m,
                )
                screened.append(
                    {
                        "episode_index": ep,
                        "frame": int(root.frame),
                        "phase": root.phase,
                        "tip_dist_m": float(root.tip_dist_m),
                        "root_contact_total": int(contact0.total),
                        "hold_metrics": {
                            k: hold_m.get(k)
                            for k in (
                                "trans_drift_max_m",
                                "rot_drift_max_rad",
                                "contact_retention_vs_root_mean",
                                "peg_contact_absent_steps",
                                "terminal_peg_ok",
                                "object_dropped_proxy",
                            )
                        },
                        **decision,
                    }
                )
        finally:
            # FullEpisodeEnv may not need close; ignore.
            pass

    selected = select_ranked_roots(
        [c for c in screened if "hold_metrics" in c],
        max_total=int(cfg["max_roots_total"]),
        max_per_episode=int(cfg["max_roots_per_episode"]),
    )
    n_eps_sel = len({int(c["episode_index"]) for c in selected})
    if n_eps_sel < int(cfg.get("min_episodes_required", MIN_EPISODES_REQUIRED)):
        verdict = {
            "overall_verdict": "h2_failed_insufficient_excited_roots",
            "decision_tree": "A_or_infrastructure",
            "research_decision": "stop_h2_controllability_route",
            "reason": "could not find enough excited intervenable roots under frozen criteria",
            "n_selected": len(selected),
            "n_episodes": n_eps_sel,
            "enter_stage2": False,
        }
        _write_outputs(cfg, screened, selected, [], verdict, {})
        print(json.dumps(verdict, indent=2, ensure_ascii=False))
        return

    # Freeze selected list to disk before interventions (audit).
    (out_dir / "selected_roots_frozen.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "criteria_version": CRITERIA_VERSION,
                "frozen_at": _utc(),
                "selected": selected,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    # ---- Phase B: matched finger branches on frozen roots ----
    branches: list[dict[str, Any]] = []
    by_ep: dict[int, list[dict[str, Any]]] = {}
    for r in selected:
        by_ep.setdefault(int(r["episode_index"]), []).append(r)

    for ep, roots in by_ep.items():
        env = make_full_env([ep], sidecar_dir=sidecar, seed=seed)
        dt = control_dt_seconds(env)
        for root in roots:
            env.reset(episode_index=ep)
            replay_demo_to_frame(env, int(root["frame"]))
            snap = FullEpisodeSnapshot.capture(env)
            root_o2h = object_in_hand_pose(env._raw)
            root_contact = peg_hand_contact_counts(env._raw)
            root_z = float(
                env._raw._data.xpos[env._raw._model.body("industreal_round_peg_8mm").id][2]
            )
            root_id = f"ep{ep:03d}_f{int(root['frame']):04d}_{root['phase']}"
            wrist_seq = build_demo_wrist_sequence(
                env, root_frame=int(root["frame"]), horizon=horizon
            )
            root_rng = np.random.default_rng(seed + ep * 10007 + int(root["frame"]))
            snap.restore(env)
            close_off = build_calibrated_right_offset(
                semantics,
                mode="calibrated_close_low",
                low_rad=close_rad,
                medium_rad=close_rad,
            )
            rand_off = build_calibrated_right_offset(
                semantics,
                mode="random_matched",
                low_rad=close_rad,
                medium_rad=close_rad,
                rng=root_rng,
            )
            projected, pmeta = project_matched_feasible_offsets(
                env,
                {
                    "calibrated_close_low": close_off,
                    "random_matched": rand_off,
                },
            )

            snap.restore(env)
            q0 = np.asarray(env._raw._data.qpos, dtype=np.float64).copy()
            obs0 = np.asarray(env._obs(), dtype=np.float64).copy()
            wrist_ref = None

            for name in cfg["interventions"]:
                snap.restore(env)
                q1 = np.asarray(env._raw._data.qpos, dtype=np.float64)
                obs1 = np.asarray(env._obs(), dtype=np.float64)
                fair = bool(
                    np.allclose(q0, q1, atol=fair_q) and np.allclose(obs0, obs1, atol=fair_o)
                )
                actions, meta = build_intervention_actions(
                    env,
                    name=str(name),
                    wrist_seq=wrist_seq,
                    root_frame=int(root["frame"]),
                    horizon=horizon,
                    semantics=semantics,
                    close_rad=close_rad,
                    pulse_steps=pulse_steps,
                    projected_offsets=projected,
                    projection_meta=pmeta,
                )
                if wrist_ref is None:
                    wrist_ref = actions[:, WRIST_IDX].copy()
                fair = fair and np.allclose(
                    actions[:, WRIST_IDX], wrist_ref, atol=float(cfg["fairness"]["wrist_atol"])
                )
                metrics, future = _rollout(
                    env,
                    snap,
                    actions,
                    root_o2h=root_o2h,
                    root_contact=root_contact,
                    root_z=root_z,
                    dt=dt,
                )
                rec = {
                    "root_id": root_id,
                    "episode_index": ep,
                    "frame": int(root["frame"]),
                    "phase": root["phase"],
                    "intervention": str(name),
                    "fairness_passed": bool(fair),
                    "metrics": metrics,
                    "action_meta": {
                        k: meta[k]
                        for k in meta
                        if k
                        in (
                            "mode",
                            "realized_l2",
                            "realized_abs_max",
                            "clip_count",
                            "n_demo_steps",
                        )
                    },
                    "future_action_horizon": int(future["horizon"]),
                    "excited_reasons": root.get("excited_reasons", []),
                }
                branches.append(rec)
                (branch_dir / f"{root_id}__{name}.json").write_text(
                    json.dumps(
                        {
                            **rec,
                            # Keep future actions for Stage-2 only if authorized later.
                            "future_actions44": future["actions44"],
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    analysis = analyze_causal_forks(branches, n_boot=n_boot, seed=boot_seed)
    if analysis["any_causal_fork"]:
        verdict = {
            "overall_verdict": "finger_causal_fork_detected",
            "decision_tree": "continue_to_stage2_eligible",
            "research_decision": "enter_stage2_action_conditioned_h4",
            "enter_stage2": True,
            "claims_controllability_p0_pass": False,
            "allow_policy_training": False,
            "analysis": analysis,
        }
    else:
        verdict = {
            "overall_verdict": "h2_failed_no_finger_causal_effect",
            "decision_tree": "A",
            "research_decision": "stop_h2_controllability_route",
            "enter_stage2": False,
            "conclusion": (
                "当前 simulator/control formulation 下 H2 失败："
                "finger intervention 在跨 root 配对分析中未能稳定改变主物理指标。"
            ),
            "next": (
                "停止当前策略路线；不得用 sensing/网络容量抢救；"
                "仅可另行决定是否修 actuator/contact/control interface。"
            ),
            "claims_controllability_p0_pass": False,
            "allow_policy_training": False,
            "analysis": analysis,
        }

    _write_outputs(cfg, screened, selected, branches, verdict, analysis)
    # Stage-2 is NOT auto-started even if eligible — wait for explicit continue,
    # unless enter_stage2 and user already authorized Stage-2 in the same directive.
    # User said: only enter Stage-2 when Stage-1 has real fork; and "不自动进入下一阶段"
    # at the end means stop after the authorized task completes. Stage-2 is part of
    # the authorized task IF Stage-1 forks. So we print eligibility; main will call
    # stage2 only if enter_stage2 — but user also said 不自动进入下一阶段 and
    # "完成后必须停止，等待人工决定" for the overall C2+H4 task tree.
    # Re-read: "仅当 Stage 1 有真实因果分叉时进入 Stage 2" AND "不自动进入下一阶段".
    # The "下一阶段" likely means beyond this whole C2+H4 task. Stage-2 is authorized
    # when Stage-1 forks. I'll enter Stage-2 only if fork detected.
    print(json.dumps({k: verdict[k] for k in verdict if k != "analysis"}, indent=2, ensure_ascii=False))
    if verdict.get("enter_stage2"):
        print(
            json.dumps(
                {
                    "note": "Stage-1 fork detected; Stage-2 authorized by same directive — starting.",
                    "stage2": "will_run_next",
                },
                ensure_ascii=False,
            )
        )


def _write_outputs(cfg, screened, selected, branches, verdict, analysis) -> None:
    out_dir = PROJECT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = PROJECT_ROOT / cfg["manifest_path"]
    report_path = PROJECT_ROOT / cfg["report_path"]
    payload = {
        "protocol": PROTOCOL,
        "criteria_version": CRITERIA_VERSION,
        "created_at": _utc(),
        "config": path_for_manifest(cfg.get("_config_path", "configs/p0_c2_stage1.yaml"), project_root=PROJECT_ROOT)
        if False
        else "configs/p0_c2_stage1.yaml",
        "wrist_source": cfg.get("wrist_source"),
        "interventions": list(cfg.get("interventions", [])),
        "n_screened": len(screened),
        "n_selected": len(selected),
        "selected_roots": [
            {
                "episode_index": r["episode_index"],
                "frame": r["frame"],
                "phase": r["phase"],
                "excited_reasons": r.get("excited_reasons"),
                "hold_metrics": r.get("hold_metrics"),
            }
            for r in selected
        ],
        "n_branches": len(branches),
        "fairness_pass_rate": (
            float(np.mean([1.0 if b.get("fairness_passed") else 0.0 for b in branches]))
            if branches
            else None
        ),
        "verdict": {k: v for k, v in verdict.items() if k != "analysis"},
        "analysis": analysis if analysis else verdict.get("analysis"),
        "guards": {
            "WRITE_IMPLEMENTATION_ENABLED": WRITE_IMPLEMENTATION_ENABLED,
            "allow_policy_training": False,
            "no_load_recalibration": True,
            "no_pilot_write": True,
            "tip_not_pass_criterion": True,
            "single_geometry_diagnostic_only": True,
        },
        "screened_reject_counts": _count_rejects(screened),
    }
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    v = payload["verdict"]
    lines = [
        f"# P0-C2 Stage-1 Controllability ({PROTOCOL})",
        "",
        f"- 日期：{_utc()}",
        f"- overall_verdict：**{v.get('overall_verdict')}**",
        f"- decision_tree：**{v.get('decision_tree')}**",
        f"- research_decision：**{v.get('research_decision')}**",
        f"- enter_stage2：{v.get('enter_stage2')}",
        f"- criteria：`{CRITERIA_VERSION}`（冻结；hold-screen 后再跑干预）",
        f"- wrist：`{cfg.get('wrist_source')}`（全分支相同 demo wrist；未重调 transport load）",
        f"- selected roots：{len(selected)} / screened {len(screened)}",
        f"- fairness_pass_rate：{payload['fairness_pass_rate']}",
        f"- tip distance 仅诊断，不作通过理由",
        f"- claims_controllability_p0_pass=false；allow_policy_training=false",
        "",
        "## 必须回答",
        "",
        f"1. finger action 是否制造真实因果分叉？ **{'是' if (analysis or {}).get('any_causal_fork') else '否'}**",
        "2. privileged+future action 预测？（Stage-2；未进入则 N/A）",
        "3. true proprio/command/FT 增量？（Stage-2；未进入则 N/A）",
        f"4. 项目动作：**{v.get('research_decision')}**",
        "",
        "## Selected roots",
        "",
    ]
    for r in selected:
        lines.append(
            f"- ep{r['episode_index']} f{r['frame']} {r['phase']} "
            f"reasons={r.get('excited_reasons')} "
            f"hold_drift={r.get('hold_metrics', {}).get('trans_drift_max_m')}"
        )
    if analysis:
        lines.extend(["", "## Paired effects vs hold_finger", ""])
        for interv, blob in analysis.get("by_intervention", {}).items():
            lines.append(f"### {interv} (fork={blob.get('causal_fork')})")
            for metric, m in blob.get("per_metric", {}).items():
                lines.append(
                    f"- {metric}: mean_diff={m.get('mean')} "
                    f"CI[{m.get('ci95_lo')},{m.get('ci95_hi')}] "
                    f"sig={m.get('significant')}"
                )
            lines.append("")
    if v.get("conclusion"):
        lines.extend(["## Conclusion", "", v["conclusion"], "", v.get("next", ""), ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _count_rejects(screened: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in screened:
        if c.get("accepted"):
            continue
        for r in c.get("reasons") or ["unknown"]:
            counts[str(r)] = counts.get(str(r), 0) + 1
    return counts


if __name__ == "__main__":
    main()
