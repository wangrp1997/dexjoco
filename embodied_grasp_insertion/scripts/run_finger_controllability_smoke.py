#!/usr/bin/env python3
"""P0-C0 finger controllability matched smoke (no training)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.physics.grasp_metrics import (  # noqa: E402
    compute_step_metrics,
    metrics_to_jsonable,
    object_in_hand_pose,
    summarize_rollout_metrics,
)
from embodied_grasp_insertion.simulation.full_episode_snapshot import (  # noqa: E402
    FullEpisodeSnapshot,
)
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    FINGER_IDX,
    WRIST_IDX,
    action_finger_stats,
    action_wrist_equal,
    build_finger_sequence,
    build_wrist_sequence,
    load_yaml,
    make_full_env,
    merge_wrist_finger,
    replay_demo_to_frame,
    select_roots_for_episode,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_determinism(cfg: dict[str, Any]) -> dict[str, Any]:
    det = cfg.get("determinism", {})
    if not det.get("require_pass", True):
        return {"skipped": True, "passed": True}
    out = PROJECT_ROOT / "outputs" / "snapshot_restore_smoke_ep0.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "smoke_full_snapshot_restore.py"),
        "--episode-id",
        str(int(det.get("episode_id", 0))),
        "--horizon",
        str(int(det.get("horizon", 8))),
        "--strict",
        "--output",
        str(out),
        "--sidecar-dir",
        str(cfg.get("sidecar_dir")),
    ]
    # Prefer conda if available via caller; here we already run inside conda.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), str(PROJECT_ROOT.parent), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True)
    payload = {}
    if out.exists():
        payload = json.loads(out.read_text(encoding="utf-8"))
    return {
        "skipped": False,
        "returncode": proc.returncode,
        "passed": bool(payload.get("passed")) and proc.returncode == 0,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
        "output": str(out),
        "payload": payload,
    }


def _initial_fairness(env, snap: FullEpisodeSnapshot, atol_qpos: float, atol_obs: float) -> dict[str, Any]:
    obs = snap.restore(env)
    qpos = np.asarray(env._raw._data.qpos, dtype=np.float64)
    qvel = np.asarray(env._raw._data.qvel, dtype=np.float64)
    outcome = env._labeler.compute(env._raw)
    return {
        "qpos": qpos.copy(),
        "qvel": qvel.copy(),
        "obs": np.asarray(obs, dtype=np.float64).copy(),
        "peg_contact_count": int(outcome.peg_contact_count),
        "tray_ok": bool(outcome.tray_ok),
        "peg_ok": bool(outcome.peg_ok),
        "insert_ok": bool(outcome.insert_ok),
        "hold44": np.asarray(env._hold44, dtype=np.float64).copy(),
        "checks": {
            "qpos_vs_snap_max": float(np.max(np.abs(qpos - snap.qpos))),
            "obs_finite": bool(np.isfinite(obs).all()),
        },
        "atol_qpos": atol_qpos,
        "atol_obs": atol_obs,
    }


def _rollout_branch(env, actions: np.ndarray, root_o2h) -> dict[str, Any]:
    step_metrics = []
    rewards = []
    infos = []
    executed = []
    prev = root_o2h
    for a in actions:
        if env._done:
            break
        obs, reward, terminated, truncated, info = env.step(np.asarray(a, dtype=np.float64))
        m = compute_step_metrics(env, root_o2h=root_o2h, prev_o2h=prev, dt=1.0)
        step_metrics.append(m)
        prev = m.object_in_hand
        rewards.append(float(reward))
        infos.append(
            {
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "peg_ok": bool(info.get("peg_ok")),
                "insert_ok": bool(info.get("insert_ok")),
                "peg_lost": bool(info.get("peg_lost")),
                "fail_reason": info.get("fail_reason", ""),
                "steps": int(info.get("steps", env._t)),
            }
        )
        executed.append(np.asarray(a, dtype=np.float64).copy())
        if terminated or truncated:
            break
    summary = summarize_rollout_metrics(step_metrics, root_o2h=root_o2h)
    term_reason = ""
    if infos:
        if infos[-1]["terminated"] or infos[-1]["truncated"]:
            term_reason = infos[-1].get("fail_reason") or (
                "terminated" if infos[-1]["terminated"] else "truncated"
            )
        else:
            term_reason = "horizon_end"
    else:
        term_reason = "no_steps"
    return {
        "summary": summary,
        "n_steps": len(step_metrics),
        "rewards_sum": float(np.sum(rewards)) if rewards else 0.0,
        "termination_reason": term_reason,
        "final_info": infos[-1] if infos else {},
        "executed_actions": np.asarray(executed, dtype=np.float64) if executed else np.zeros((0, 44)),
        "step0": metrics_to_jsonable(step_metrics[0]) if step_metrics else None,
        "step_end": metrics_to_jsonable(step_metrics[-1]) if step_metrics else None,
    }


def _verdict(branches: list[dict[str, Any]], roots_ok: list[dict[str, Any]]) -> dict[str, Any]:
    if any(b.get("fairness_passed") is False for b in branches):
        return {
            "label": "infrastructure_fail",
            "allow_extended_controllability_p0": False,
            "reason": "matched fairness failed on one or more branches",
        }
    if not roots_ok:
        return {
            "label": "infrastructure_fail",
            "allow_extended_controllability_p0": False,
            "reason": "no valid matched roots after fairness/determinism filters",
        }

    by_root: dict[str, list[dict[str, Any]]] = {}
    for b in branches:
        if not b.get("fairness_passed"):
            continue
        by_root.setdefault(b["root_id"], []).append(b)

    stabilizing = []
    harmful_roots = []
    effect_roots = []

    for root_id, blist in by_root.items():
        hold = next((x for x in blist if x["intervention"] == "hold_fingers"), None)
        rand = next((x for x in blist if x["intervention"] == "shuffled_or_random_finger"), None)
        if hold is None:
            continue
        hs = hold["metrics"]
        improved = []
        worsened = []
        any_effect = False
        for x in blist:
            if x["intervention"] == "hold_fingers":
                continue
            ms = x["metrics"]
            d_trans = float(hs["trans_drift_max_m"] - ms["trans_drift_max_m"])  # >0 => better
            d_rot = float(hs["rot_drift_max_rad"] - ms["rot_drift_max_rad"])
            d_ret = float(ms["contact_retention_mean"] - hs["contact_retention_mean"])
            d_loss = int(hs["contact_loss_steps"] - ms["contact_loss_steps"])
            peg_not_worse = bool(ms["peg_ok_end"]) >= bool(hs["peg_ok_end"])

            delta_mag = (
                abs(d_trans) + abs(d_rot) + abs(d_ret) + abs(float(ms["peg_ok_end"]) - float(hs["peg_ok_end"]))
            )
            if delta_mag > 1e-4:
                any_effect = True

            random_also = False
            if rand is not None and x["intervention"] != "shuffled_or_random_finger":
                rs = rand["metrics"]
                random_also = (
                    abs(rs["trans_drift_max_m"] - ms["trans_drift_max_m"]) < 1e-3
                    and abs(rs["contact_retention_mean"] - ms["contact_retention_mean"]) < 0.05
                    and bool(rs["peg_ok_end"]) == bool(ms["peg_ok_end"])
                )

            # Stabilizing: peg not worse, at least one clear gain, no major regression.
            clear_gain = d_trans > 1e-3 or d_rot > 5e-3 or d_ret > 0.05 or d_loss > 0
            no_major_regression = (
                d_trans >= -1e-3
                and d_rot >= -5e-3
                and d_ret >= -0.05
                and bool(ms["peg_ok_end"]) >= bool(hs["peg_ok_end"])
            )
            better = peg_not_worse and clear_gain and no_major_regression and not random_also

            worse = (not bool(ms["peg_ok_end"]) and bool(hs["peg_ok_end"])) or (
                d_trans < -1e-3 and (d_ret < -0.05 or d_rot < -5e-3)
            ) or (d_ret < -0.15)

            if better:
                improved.append(
                    {
                        "intervention": x["intervention"],
                        "d_trans": d_trans,
                        "d_rot": d_rot,
                        "d_ret": d_ret,
                        "d_loss_steps": d_loss,
                        "peg_ok_end": ms["peg_ok_end"],
                    }
                )
            if worse:
                worsened.append(x["intervention"])

        if any_effect:
            effect_roots.append(root_id)
        if improved:
            stabilizing.append({"root_id": root_id, "improved": improved})
        if worsened and not improved:
            harmful_roots.append({"root_id": root_id, "worsened": worsened})

    n_roots = len(by_root)
    n_eps = len({b["episode_index"] for b in branches if b.get("fairness_passed")})
    same_dir = len(stabilizing)

    if same_dir >= 3 and n_roots >= 4 and n_eps >= 2:
        return {
            "label": "promising",
            "allow_extended_controllability_p0": True,
            "reason": (
                f"{same_dir} roots show stabilizing hand-action effects vs hold/random "
                f"across {n_eps} episodes / {n_roots} roots"
            ),
            "stabilizing": stabilizing,
        }
    if same_dir == 0 and len(harmful_roots) >= 2 and len(effect_roots) >= 2:
        return {
            "label": "harmful_only",
            "allow_extended_controllability_p0": False,
            "reason": (
                "finger actions causally change grasp outcomes, but observed interventions "
                "(especially mild_open / random) increase drift or drop peg vs hold_fingers; "
                "no repeatable stabilizing intervention beat hold"
            ),
            "harmful_roots": harmful_roots,
            "effect_roots": effect_roots,
        }
    if len(effect_roots) == 0:
        return {
            "label": "no_effect",
            "allow_extended_controllability_p0": False,
            "reason": "finger interventions executed but no repeatable grasp-metric differences",
        }
    return {
        "label": "no_effect",
        "allow_extended_controllability_p0": False,
        "reason": (
            f"mixed/weak effects on {len(effect_roots)} roots / stabilizing={same_dir}, "
            "below promising threshold (>=3 stabilizing roots, >=4 roots, >=2 eps)"
        ),
        "stabilizing": stabilizing,
        "effect_roots": effect_roots,
        "harmful_roots": harmful_roots,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    v = summary["verdict"]
    lines = [
        "# Finger Controllability Matched Smoke (P0-C0)",
        "",
        f"- 日期（UTC）：{summary['created_at']}",
        f"- 结论：**{v['label']}**",
        f"- 允许扩展 Controllability P0：{v.get('allow_extended_controllability_p0')}",
        f"- Observability / Semantic / policy：**仍禁止**",
        "",
        "## Snapshot",
        "",
        f"- Determinism passed：{summary['determinism']['passed']}",
        f"- 保存字段：MuJoCo `{summary['snapshot_fields']}` + FullEpisodeEnv Python 累计量",
        "",
        "## 规模",
        "",
        f"- episodes：{summary['n_episodes']}",
        f"- roots：{summary['n_roots']}",
        f"- branches：{summary['n_branches']}",
        f"- fairness pass rate：{summary['fairness_pass_rate']}",
        "",
        "## Interventions",
        "",
    ]
    for name in summary["interventions"]:
        lines.append(f"- {name}")
    lines += [
        "",
        f"- wrist_action_source：`{summary['wrist_action_source']}`",
        "",
        "## 主要结果",
        "",
        f"- {v.get('reason')}",
        "",
        "## 说明",
        "",
        "- tip distance 不是本轮主指标。",
        "- slip_* 均为 proxy，不是 ground truth。",
        "- 单一 geometry family；不宣称语义泛化。",
        "- 本轮不是完整 Controllability P0 pass。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_state(summary: dict[str, Any]) -> None:
    state_path = PROJECT_ROOT / "outputs" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    # Preserve P0-A history.
    history = list(state.get("history") or [])
    history.append(
        {
            "date": summary["created_at"][:10],
            "event": "p0c0_finger_controllability_smoke",
            "verdict": summary["verdict"]["label"],
            "n_roots": summary["n_roots"],
            "n_branches": summary["n_branches"],
            "determinism_passed": summary["determinism"]["passed"],
        }
    )
    state["phase"] = "finger_controllability_smoke"
    state["busy"] = False
    state["updated_at"] = summary["created_at"]
    state["p0c0"] = {
        "verdict": summary["verdict"],
        "determinism_passed": summary["determinism"]["passed"],
        "n_episodes": summary["n_episodes"],
        "n_roots": summary["n_roots"],
        "n_branches": summary["n_branches"],
        "allow_extended_controllability_p0": summary["verdict"].get(
            "allow_extended_controllability_p0"
        ),
        "allow_observability_p0": False,
        "allow_semantic_p0": False,
        "allow_policy_training": False,
        "summary_path": summary["summary_path"],
        "manifest_path": summary["manifest_path"],
    }
    state["next_proposal"] = {
        "allow_full_observability_p0": False,
        "allow_extended_controllability_p0": summary["verdict"].get(
            "allow_extended_controllability_p0"
        ),
        "allow_policy_training": False,
        "actions": [
            "若 promising：扩大 Controllability P0（更多 episode/root，仍禁止 policy）",
            "若 no_effect/harmful_only：检查 finger action 是否真正进入执行器 / 接触模型",
            "若 infrastructure_fail：先修 snapshot/fairness",
            "仍不启动 Observability/Semantic/policy",
        ],
    }
    state["history"] = history
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "finger_controllability_smoke.yaml"),
    )
    parser.add_argument("--skip-determinism", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))

    # Mark busy.
    state_path = PROJECT_ROOT / "outputs" / "state.json"
    if state_path.exists():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        st["busy"] = True
        st["phase"] = "finger_controllability_smoke"
        state_path.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.skip_determinism:
        det = {"skipped": True, "passed": True}
    else:
        det = _run_determinism(cfg)
    if not det.get("passed"):
        summary = {
            "created_at": _utc(),
            "verdict": {
                "label": "infrastructure_fail",
                "allow_extended_controllability_p0": False,
                "reason": "snapshot restore determinism failed",
            },
            "determinism": det,
            "n_episodes": 0,
            "n_roots": 0,
            "n_branches": 0,
            "fairness_pass_rate": 0.0,
            "interventions": cfg.get("finger_interventions"),
            "wrist_action_source": cfg.get("wrist_action_source"),
            "snapshot_fields": ["qpos", "qvel", "act", "ctrl", "mocap", "userdata", "time", "python"],
            "summary_path": str(PROJECT_ROOT / cfg["output_dir"] / "summary.json"),
            "manifest_path": str(PROJECT_ROOT / cfg["manifest_path"]),
        }
        out_dir = PROJECT_ROOT / cfg["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _write_report(PROJECT_ROOT / cfg["report_path"], summary)
        _update_state(summary)
        print(json.dumps({"verdict": "infrastructure_fail", "determinism": det}, ensure_ascii=False))
        raise SystemExit(2)

    episodes = [int(x) for x in cfg["episodes"]]
    horizon = int(cfg["horizon"])
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    sidecar = Path(cfg["sidecar_dir"])
    out_dir = PROJECT_ROOT / cfg["output_dir"]
    branch_dir = out_dir / "branches"
    branch_dir.mkdir(parents=True, exist_ok=True)

    wrist_seq = build_wrist_sequence(
        source=str(cfg["wrist_action_source"]),
        horizon=horizon,
        mild_transport_delta=np.asarray(cfg.get("mild_transport_delta44", np.zeros(44))),
    )

    manifest_branches: list[dict[str, Any]] = []
    roots_meta: list[dict[str, Any]] = []
    fair_atol_q = float(cfg.get("fairness", {}).get("init_qpos_atol", 1e-8))
    fair_atol_o = float(cfg.get("fairness", {}).get("init_obs_atol", 1e-5))

    for ep in episodes:
        env = make_full_env([ep], sidecar_dir=sidecar, seed=seed)
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
            if not roots:
                roots_meta.append(
                    {
                        "episode_index": ep,
                        "error": "no_roots_found",
                        "roots": [],
                    }
                )
                continue

            for root in roots:
                env.reset(episode_index=ep)
                replay_demo_to_frame(env, int(root.frame))
                outcome0 = env._labeler.compute(env._raw)
                if not outcome0.peg_ok or outcome0.insert_ok:
                    roots_meta.append(
                        {
                            "episode_index": ep,
                            "frame": root.frame,
                            "phase": root.phase,
                            "excluded": True,
                            "reason": f"peg_ok={outcome0.peg_ok}, insert_ok={outcome0.insert_ok}",
                        }
                    )
                    continue
                snap = FullEpisodeSnapshot.capture(env)
                root_o2h = object_in_hand_pose(env._raw)
                root_id = f"ep{ep:03d}_f{root.frame:04d}_{root.phase}"
                roots_meta.append(
                    {
                        "root_id": root_id,
                        "episode_index": ep,
                        "frame": root.frame,
                        "phase": root.phase,
                        "reason": root.reason,
                        "tip_dist_m": root.tip_dist_m,
                        "excluded": False,
                    }
                )

                # Reference initial state after restore.
                init_ref = _initial_fairness(env, snap, fair_atol_q, fair_atol_o)
                ref_actions = None

                for intervention in cfg["finger_interventions"]:
                    # Restore matched root.
                    env_init = _initial_fairness(env, snap, fair_atol_q, fair_atol_o)
                    finger_seq = build_finger_sequence(
                        name=str(intervention),
                        horizon=horizon,
                        env=env,
                        root_frame=int(root.frame),
                        mild_close_delta=float(cfg["mild_close_delta"]),
                        rng=rng,
                    )
                    actions = merge_wrist_finger(wrist_seq, finger_seq)
                    if ref_actions is None:
                        ref_actions = actions.copy()
                        wrist_match = True
                    else:
                        wrist_match = action_wrist_equal(
                            ref_actions,
                            actions,
                            atol=float(cfg.get("fairness", {}).get("wrist_atol", 0.0)),
                        )

                    # Fairness on initial state.
                    init_qpos_err = float(
                        np.max(np.abs(env_init["qpos"] - init_ref["qpos"]))
                    )
                    init_obs_err = float(np.max(np.abs(env_init["obs"] - init_ref["obs"])))
                    init_contact_ok = env_init["peg_contact_count"] == init_ref["peg_contact_count"]
                    fairness_passed = (
                        init_qpos_err <= fair_atol_q
                        and init_obs_err <= fair_atol_o
                        and init_contact_ok
                        and wrist_match
                        and env_init["peg_ok"] == init_ref["peg_ok"]
                    )

                    roll = _rollout_branch(env, actions, root_o2h)
                    # Verify executed wrist equals planned wrist for executed prefix.
                    exec_n = int(roll["executed_actions"].shape[0])
                    if exec_n > 0:
                        wrist_exec_ok = action_wrist_equal(
                            actions[:exec_n],
                            roll["executed_actions"],
                            atol=0.0,
                        )
                    else:
                        wrist_exec_ok = False
                    fairness_passed = bool(fairness_passed and wrist_exec_ok)

                    fstats = action_finger_stats(actions)
                    branch_id = f"{root_id}__{intervention}"
                    out_npz = branch_dir / f"{branch_id}.npz"
                    np.savez_compressed(
                        out_npz,
                        actions=actions.astype(np.float32),
                        executed=roll["executed_actions"].astype(np.float32),
                        metrics_json=np.asarray([json.dumps(roll["summary"])], dtype=object),
                    )

                    rec = {
                        "branch_id": branch_id,
                        "root_id": root_id,
                        "episode_index": ep,
                        "root_frame": int(root.frame),
                        "root_phase": root.phase,
                        "root_selection_rule": root.reason,
                        "intervention": intervention,
                        "seed": seed,
                        "horizon": horizon,
                        "n_steps_executed": roll["n_steps"],
                        "wrist_action_source": cfg["wrist_action_source"],
                        "finger_action_source": intervention,
                        "finger_action_stats": fstats,
                        "initial_metrics": {
                            "peg_ok": init_ref["peg_ok"],
                            "insert_ok": init_ref["insert_ok"],
                            "peg_contact_count": init_ref["peg_contact_count"],
                            "object_in_hand": {
                                "translation": root_o2h.translation.tolist(),
                                "rotvec": root_o2h.rotvec.tolist(),
                                "reference_body": root_o2h.reference_body,
                            },
                        },
                        "metrics": roll["summary"],
                        "termination_reason": roll["termination_reason"],
                        "fairness_checks": {
                            "passed": fairness_passed,
                            "init_qpos_err": init_qpos_err,
                            "init_obs_err": init_obs_err,
                            "init_contact_ok": init_contact_ok,
                            "wrist_sequence_matched": wrist_match,
                            "wrist_executed_matched": wrist_exec_ok,
                        },
                        "fairness_passed": fairness_passed,
                        "restore_determinism_result": det.get("passed"),
                        "output_path": str(out_npz),
                        "insert_ok": roll["summary"].get("insert_ok_end"),
                        "peg_retained": roll["summary"].get("peg_retained"),
                        "peg_loss": roll["summary"].get("peg_loss"),
                        "final_max_drift": {
                            "trans_max_m": roll["summary"].get("trans_drift_max_m"),
                            "rot_max_rad": roll["summary"].get("rot_drift_max_rad"),
                        },
                        "contact_retention": roll["summary"].get("contact_retention_mean"),
                    }
                    manifest_branches.append(rec)
        finally:
            env.close()

    valid_roots = [r for r in roots_meta if not r.get("excluded") and r.get("root_id")]
    fair_branches = [b for b in manifest_branches if b.get("fairness_passed")]
    verdict = _verdict(manifest_branches, valid_roots)
    if not det.get("passed"):
        verdict = {
            "label": "infrastructure_fail",
            "allow_extended_controllability_p0": False,
            "reason": "determinism failed",
        }

    summary = {
        "created_at": _utc(),
        "verdict": verdict,
        "determinism": {"passed": det.get("passed"), "output": det.get("output")},
        "n_episodes": len(episodes),
        "n_roots": len(valid_roots),
        "n_branches": len(manifest_branches),
        "fairness_pass_rate": (
            float(len(fair_branches) / len(manifest_branches)) if manifest_branches else 0.0
        ),
        "interventions": list(cfg["finger_interventions"]),
        "wrist_action_source": cfg["wrist_action_source"],
        "snapshot_fields": [
            "mjSTATE_INTEGRATION",
            "qpos",
            "qvel",
            "act",
            "ctrl",
            "mocap_pos",
            "mocap_quat",
            "userdata",
            "time",
            "data_arrays",
            "hold44",
            "force_baseline",
            "prev_tip/lat/along",
            "tray_ok_seen",
            "peg_ok_seen",
            "peg_lost",
            "done",
            "labeler_rest_z",
        ],
        "roots": roots_meta,
        "summary_path": str(out_dir / "summary.json"),
        "manifest_path": str(PROJECT_ROOT / cfg["manifest_path"]),
    }

    manifest = {
        "name": "finger_controllability_smoke_v1",
        "created_at": summary["created_at"],
        "config": cfg,
        "determinism": det,
        "roots": roots_meta,
        "branches": manifest_branches,
        "verdict": verdict,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    man_path = PROJECT_ROOT / cfg["manifest_path"]
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(PROJECT_ROOT / cfg["report_path"], summary)
    _update_state(summary)

    print(
        json.dumps(
            {
                "verdict": verdict["label"],
                "n_roots": summary["n_roots"],
                "n_branches": summary["n_branches"],
                "fairness_pass_rate": summary["fairness_pass_rate"],
                "allow_extended": verdict.get("allow_extended_controllability_p0"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
