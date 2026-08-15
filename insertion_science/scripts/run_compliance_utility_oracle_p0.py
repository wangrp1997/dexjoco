#!/usr/bin/env python3
"""Compliance Utility/Oracle P0: task benefit of gain configs vs best fixed."""

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
EMBODIED = DEXJOCO_ROOT / "embodied_grasp_insertion"
for _p in (
    str(PROJECT_ROOT),
    str(DEXJOCO_ROOT),
    str(EMBODIED),
    str(DEXJOCO_ROOT.parent / "reach_insert_rl"),
):
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
from embodied_grasp_insertion.scripts.run_p0_c2_stage1 import (  # noqa: E402
    build_demo_wrist_sequence,
)
from embodied_grasp_insertion.simulation.calibrated_interventions import (  # noqa: E402
    build_right_demo_replay_actions,
)
from embodied_grasp_insertion.simulation.full_episode_snapshot import (  # noqa: E402
    FullEpisodeSnapshot,
)
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    make_full_env,
    replay_demo_to_frame,
)
from insertion_science.physics.osc_gain_patch import osc_gains, set_task_axis  # noqa: E402
from reach_insert_rl.env.full_obs import privileged_full_features  # noqa: E402

PROTOCOL = "ComplianceUtilityOracleP0"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_actions(env, *, name: str, root_frame: int, horizon: int):
    wrist = build_demo_wrist_sequence(env, root_frame=root_frame, horizon=horizon)
    if name == "hold":
        return np.zeros((horizon, 44), dtype=np.float64), {"mode": "hold"}
    if name == "demo_matched":
        return build_right_demo_replay_actions(
            env, root_frame=root_frame, horizon=horizon, wrist_seq=wrist
        )
    raise ValueError(name)


def _wrist_ft_norm(env) -> float:
    fl = getattr(env, "_force_labeler", None)
    if fl is None:
        return float("nan")
    frame = fl.compute(env._raw)
    return float(np.linalg.norm(np.asarray(frame.wrist_ft_right[:3], dtype=np.float64)))


def _contact_force_proxy(raw) -> float:
    import mujoco

    model, data = raw._model, raw._data
    total = 0.0
    force = np.zeros(6, dtype=np.float64)
    for i in range(int(data.ncon)):
        mujoco.mj_contactForce(model, data, i, force)
        total += float(np.linalg.norm(force[:3]))
    return total


def resolve_gain_kwargs(cfg: dict, gcfg: dict, hole_axis: np.ndarray) -> dict[str, Any]:
    base_pos = tuple(cfg["baseline_pos_gains"])
    base_ori = tuple(cfg["baseline_ori_gains"])
    damp = float(cfg["baseline_damping_ratio"])
    mode = gcfg["mode"]
    if mode == "iso":
        s = float(gcfg["stiffness_scale"])
        return {
            "mode": "iso",
            "pos_gains": tuple(v * s for v in base_pos),
            "ori_gains": tuple(v * s for v in base_ori),
            "damping_ratio": damp,
        }
    if mode == "task_aniso":
        ka = float(base_pos[0]) * float(gcfg["k_axial_scale"])
        kl = float(base_pos[0]) * float(gcfg["k_lateral_scale"])
        os_ = float(gcfg.get("ori_scale", 1.0))
        return {
            "mode": "task_aniso",
            "pos_gains": base_pos,  # unused in aniso path except logging
            "ori_gains": tuple(v * os_ for v in base_ori),
            "damping_ratio": damp,
            "k_axial": ka,
            "k_lateral": kl,
            "task_axis": np.asarray(hole_axis, dtype=np.float64).copy(),
        }
    raise ValueError(mode)


def rollout_once(
    env,
    snap,
    actions,
    *,
    gain_kwargs: dict,
    hole_axis: np.ndarray,
    root_o2h,
    root_contact,
    root_z: float,
    tip0: float,
    lat0: float,
    dt: float,
    jam_force_thresh: float,
) -> dict[str, Any]:
    snap.restore(env)
    if bool(env._done):
        raise RuntimeError("restore left done=True")
    steps_m = []
    tip_series, lat_series, wrist_ft, contact_f = [], [], [], []
    nan_hit = False
    term_reason = "horizon_end"
    with osc_gains(**gain_kwargs) as g:
        set_task_axis(hole_axis)
        for a in actions:
            if env._done:
                term_reason = "already_done"
                break
            # keep axis frozen at root (matched); still refresh for safety
            set_task_axis(hole_axis)
            obs, _, term, trunc, info = env.step(np.asarray(a, dtype=np.float64))
            if not np.isfinite(obs).all():
                nan_hit = True
                term_reason = "nonfinite_obs"
                break
            steps_m.append(compute_step_metrics(env))
            feat = privileged_full_features(env._raw)
            tip_series.append(float(feat["tip_dist"]))
            lat_series.append(float(feat["lat_err"]))
            wrist_ft.append(_wrist_ft_norm(env))
            contact_f.append(_contact_force_proxy(env._raw))
            if term or trunc:
                term_reason = str(info.get("fail_reason") or ("terminated" if term else "truncated"))
                break

    if not steps_m:
        return {"metrics": {"num_steps": 0, "error": "empty", "term_reason": term_reason}, "gains": g.as_dict()}

    metrics = summarize_rollout_metrics_v2(
        steps_m,
        root_o2h=root_o2h,
        root_contact=root_contact,
        root_peg_world_z=root_z,
        control_dt_s=dt,
    )
    tip = np.asarray(tip_series, dtype=np.float64)
    lat = np.asarray(lat_series, dtype=np.float64)
    wft = np.asarray(wrist_ft, dtype=np.float64)
    cf = np.asarray(contact_f, dtype=np.float64)
    tip_progress = float(tip0 - tip[-1])
    lat_progress = float(lat0 - lat[-1])
    jam_proxy = bool(float(np.mean(cf)) > float(jam_force_thresh) and tip_progress < 0.001)
    metrics.update(
        {
            "term_reason": term_reason,
            "executed_steps": len(steps_m),
            "tip_progress_m": tip_progress,
            "lat_progress_m": lat_progress,
            "tip_end_m": float(tip[-1]),
            "lat_end_m": float(lat[-1]),
            "wrist_ft_mean_n": float(np.nanmean(wft)),
            "contact_force_mean_n": float(np.mean(cf)),
            "jam_proxy": jam_proxy,
            "nonfinite_obs": bool(nan_hit),
            "insert_ok_end": bool(steps_m[-1].insert_ok),
            "object_dropped_proxy": bool(metrics.get("object_dropped_proxy", False)),
            "contact_retention_vs_root_mean": float(
                metrics["contact_retention_vs_root_mean"]
            ),
        }
    )
    return {"metrics": metrics, "gains": g.as_dict()}


def utility_tuple(m: dict[str, Any]) -> tuple:
    """Lexicographic utility: insert_ok, tip_progress, not jam, lat_progress."""
    return (
        1 if m.get("insert_ok_end") else 0,
        float(m.get("tip_progress_m", 0.0)),
        0 if m.get("jam_proxy") else 1,
        float(m.get("lat_progress_m", 0.0)),
    )


def retention_ok(m: dict, baseline_ret: float, slack: float) -> bool:
    if m.get("object_dropped_proxy"):
        return False
    return float(m.get("contact_retention_vs_root_mean", 0.0)) >= float(baseline_ret) - float(slack)


def mean_metrics(reps: list[dict]) -> dict[str, Any]:
    keys = [
        "insert_ok_end",
        "tip_progress_m",
        "lat_progress_m",
        "jam_proxy",
        "contact_retention_vs_root_mean",
        "object_dropped_proxy",
        "wrist_ft_mean_n",
        "contact_force_mean_n",
        "nonfinite_obs",
    ]
    out = {}
    for k in keys:
        vals = [float(r["metrics"].get(k, 0.0)) for r in reps]
        mean = float(np.mean(vals))
        if k in ("insert_ok_end", "jam_proxy", "object_dropped_proxy", "nonfinite_obs"):
            out[k] = bool(round(mean))
        else:
            out[k] = mean
    return out


def judge(heldout_cells: list[dict], cfg: dict) -> dict[str, Any]:
    """heldout_cells: one row per root×action×gain with metrics_mean."""
    tip_eps = float(cfg["tip_improve_eps"])
    fixed_eps = float(cfg["oracle_vs_fixed_eps"])
    slack = float(cfg["retention_slack"])

    # group by (ep, frame, action)
    from collections import defaultdict

    groups: dict[tuple, list] = defaultdict(list)
    for c in heldout_cells:
        key = (int(c["episode_index"]), int(c["frame"]), c["action"])
        groups[key].append(c)

    oracle_picks = []
    for key, cells in groups.items():
        base = next(c for c in cells if c["gain_name"] == "baseline")
        base_ret = float(base["metrics_mean"]["contact_retention_vs_root_mean"])
        feasible = [
            c
            for c in cells
            if retention_ok(c["metrics_mean"], base_ret, slack)
            and not c["metrics_mean"].get("nonfinite_obs")
        ]
        if not feasible:
            pick = base
            reason = "no_feasible_fallback_baseline"
        else:
            pick = max(feasible, key=lambda c: utility_tuple(c["metrics_mean"]))
            reason = "max_utility_among_retention_ok"
        oracle_picks.append(
            {
                "episode_index": key[0],
                "frame": key[1],
                "action": key[2],
                "gain_name": pick["gain_name"],
                "metrics": pick["metrics_mean"],
                "baseline_metrics": base["metrics_mean"],
                "reason": reason,
                "beats_baseline_tip": float(pick["metrics_mean"]["tip_progress_m"])
                > float(base["metrics_mean"]["tip_progress_m"]) + tip_eps,
                "beats_baseline_insert": bool(pick["metrics_mean"]["insert_ok_end"])
                and not bool(base["metrics_mean"]["insert_ok_end"]),
                "jam_improved": (not pick["metrics_mean"]["jam_proxy"])
                and bool(base["metrics_mean"]["jam_proxy"]),
            }
        )

    # Best fixed: among gain names, maximize mean utility on held-out (all cells)
    gain_names = sorted({c["gain_name"] for c in heldout_cells})
    fixed_scores = []
    for name in gain_names:
        subset = [c for c in heldout_cells if c["gain_name"] == name]
        # also require mean retention not worse than baseline mean by slack
        base_subset = [c for c in heldout_cells if c["gain_name"] == "baseline"]
        mean_ret = float(np.mean([c["metrics_mean"]["contact_retention_vs_root_mean"] for c in subset]))
        base_ret = float(
            np.mean([c["metrics_mean"]["contact_retention_vs_root_mean"] for c in base_subset])
        )
        mean_tip = float(np.mean([c["metrics_mean"]["tip_progress_m"] for c in subset]))
        mean_ins = float(np.mean([float(c["metrics_mean"]["insert_ok_end"]) for c in subset]))
        mean_jam = float(np.mean([float(c["metrics_mean"]["jam_proxy"]) for c in subset]))
        ret_ok = mean_ret >= base_ret - slack
        score = (mean_ins, mean_tip if ret_ok else -1e9, -mean_jam, mean_ret)
        fixed_scores.append(
            {
                "gain_name": name,
                "mean_insert_ok": mean_ins,
                "mean_tip_progress_m": mean_tip,
                "mean_jam_proxy": mean_jam,
                "mean_retention": mean_ret,
                "retention_ok_vs_baseline": ret_ok,
                "score": score,
            }
        )
    best_fixed = max(fixed_scores, key=lambda s: s["score"])

    # Oracle aggregate vs baseline / best fixed
    def agg(picks, field_metrics_key):
        tips = [p["metrics"]["tip_progress_m"] for p in picks]
        bases = [p["baseline_metrics"]["tip_progress_m"] for p in picks]
        return {
            "mean_tip": float(np.mean(tips)),
            "mean_baseline_tip": float(np.mean(bases)),
            "mean_insert": float(np.mean([float(p["metrics"]["insert_ok_end"]) for p in picks])),
            "mean_baseline_insert": float(
                np.mean([float(p["baseline_metrics"]["insert_ok_end"]) for p in picks])
            ),
            "n_beats_tip": int(sum(1 for p in picks if p["beats_baseline_tip"])),
            "n_beats_insert": int(sum(1 for p in picks if p["beats_baseline_insert"])),
            "n_jam_improved": int(sum(1 for p in picks if p["jam_improved"])),
        }

    oracle_vs_base = agg(oracle_picks, "metrics")
    # oracle vs best fixed: compare oracle tip to that config's tip on same cells
    bf_name = best_fixed["gain_name"]
    oracle_tips = []
    fixed_tips = []
    for p in oracle_picks:
        key = (p["episode_index"], p["frame"], p["action"])
        cell = next(
            c
            for c in heldout_cells
            if (c["episode_index"], c["frame"], c["action"]) == key and c["gain_name"] == bf_name
        )
        oracle_tips.append(float(p["metrics"]["tip_progress_m"]))
        fixed_tips.append(float(cell["metrics_mean"]["tip_progress_m"]))
    oracle_mean_tip = float(np.mean(oracle_tips))
    best_fixed_mean_tip = float(np.mean(fixed_tips))
    distinct_oracle_gains = sorted({p["gain_name"] for p in oracle_picks})
    state_dependent = len(distinct_oracle_gains) >= 2

    beats_baseline = bool(
        oracle_vs_base["mean_insert"] > oracle_vs_base["mean_baseline_insert"] + 1e-9
        or oracle_vs_base["mean_tip"] > oracle_vs_base["mean_baseline_tip"] + tip_eps
        or oracle_vs_base["n_jam_improved"] > 0
        and oracle_vs_base["mean_tip"] >= oracle_vs_base["mean_baseline_tip"] - tip_eps
    )
    # require not just jam with tip collapse
    if oracle_vs_base["mean_tip"] + tip_eps < oracle_vs_base["mean_baseline_tip"] and oracle_vs_base[
        "mean_insert"
    ] <= oracle_vs_base["mean_baseline_insert"]:
        beats_baseline = False

    beats_best_fixed = bool(oracle_mean_tip > best_fixed_mean_tip + fixed_eps) or (
        float(np.mean([float(p["metrics"]["insert_ok_end"]) for p in oracle_picks]))
        > float(best_fixed["mean_insert_ok"]) + 1e-9
    )

    if not beats_baseline:
        branch = 1
        verdict = "abandon_compliance"
        summary = "Oracle 在 held-out 上不优于 baseline；放弃 compliance 方案。"
        allow_wrapper = False
        fix_default_only = False
    elif best_fixed["gain_name"] != "baseline" and best_fixed["retention_ok_vs_baseline"] and not (
        beats_best_fixed and state_dependent
    ):
        # fixed gain better than baseline; oracle not clearly needing state-dependence over best fixed
        if beats_best_fixed and state_dependent:
            branch = 3
            verdict = "state_dependent_compliance_warranted"
            summary = "Oracle 状态依赖且明显优于最佳固定刚度；可进入 wrapper/learnability。"
            allow_wrapper = True
            fix_default_only = False
        else:
            branch = 2
            verdict = "fix_default_gains_only"
            summary = (
                f"固定配置 `{best_fixed['gain_name']}` 优于 baseline，"
                "且 Oracle 未明显超过该固定值；只改默认增益，不训练。"
            )
            allow_wrapper = False
            fix_default_only = True
    elif beats_best_fixed and state_dependent:
        branch = 3
        verdict = "state_dependent_compliance_warranted"
        summary = "不同状态需要不同刚度，且 Oracle 明显优于 Best fixed；可进入 wrapper。"
        allow_wrapper = True
        fix_default_only = False
    elif beats_baseline and not beats_best_fixed:
        branch = 2
        verdict = "fix_default_gains_only"
        summary = (
            f"存在优于 baseline 的固定刚度 `{best_fixed['gain_name']}`；"
            "Oracle 未明显超过 Best fixed → 只改默认，不做动态 compliance。"
        )
        allow_wrapper = False
        fix_default_only = True
    else:
        branch = 1
        verdict = "abandon_compliance"
        summary = "无清晰任务收益路径；放弃 compliance 方案。"
        allow_wrapper = False
        fix_default_only = False

    return {
        "branch": branch,
        "verdict": verdict,
        "summary": summary,
        "allow_wrapper": allow_wrapper,
        "fix_default_only": fix_default_only,
        "oracle_picks": oracle_picks,
        "oracle_vs_baseline": oracle_vs_base,
        "best_fixed": best_fixed,
        "fixed_scores": fixed_scores,
        "oracle_mean_tip": oracle_mean_tip,
        "best_fixed_mean_tip_on_oracle_cells": best_fixed_mean_tip,
        "beats_baseline": beats_baseline,
        "beats_best_fixed": beats_best_fixed,
        "state_dependent_oracle": state_dependent,
        "distinct_oracle_gains": distinct_oracle_gains,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "compliance_utility_oracle_p0.yaml",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    out_dir = PROJECT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = PROJECT_ROOT / cfg["manifest_path"]
    man_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / cfg["report_path"]

    roots = [{**r, "role": "discovery"} for r in cfg["discovery_roots"]] + [
        {**r, "role": "held_out"} for r in cfg["held_out_roots"]
    ]
    frozen = {
        "protocol": PROTOCOL,
        "frozen_at": _utc(),
        "roots": roots,
        "gain_configs": cfg["gain_configs"],
        "actions": cfg["actions"],
    }
    (out_dir / "roots_and_gains_frozen.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )

    episodes = sorted({int(r["episode_index"]) for r in roots})
    env = make_full_env(episodes, sidecar_dir=Path(cfg["sidecar_dir"]), seed=int(cfg["seed"]))
    dt = control_dt_seconds(env)

    cells: list[dict[str, Any]] = []
    for root in roots:
        ep, frame = int(root["episode_index"]), int(root["frame"])
        entry = next(e for e in env.entries if int(e["episode_index"]) == ep)
        env.reset(entry=entry)
        replay_demo_to_frame(env, frame)
        snap = FullEpisodeSnapshot.capture(env)
        root_o2h = object_in_hand_pose(env._raw)
        root_contact = peg_hand_contact_counts(env._raw)
        root_z = float(compute_step_metrics(env).peg_world_pos[2])
        feat0 = privileged_full_features(env._raw)
        tip0, lat0 = float(feat0["tip_dist"]), float(feat0["lat_err"])
        hole_axis = np.asarray(feat0["hole"], dtype=np.float64).reshape(3)

        for action_name in cfg["actions"]:
            actions, ameta = build_actions(
                env, name=action_name, root_frame=frame, horizon=int(cfg["horizon"])
            )
            for gcfg in cfg["gain_configs"]:
                gkw = resolve_gain_kwargs(cfg, gcfg, hole_axis)
                reps = []
                for rep in range(int(cfg["repeats"])):
                    out = rollout_once(
                        env,
                        snap,
                        actions,
                        gain_kwargs=gkw,
                        hole_axis=hole_axis,
                        root_o2h=root_o2h,
                        root_contact=root_contact,
                        root_z=root_z,
                        tip0=tip0,
                        lat0=lat0,
                        dt=dt,
                        jam_force_thresh=float(cfg["jam_force_thresh_n"]),
                    )
                    out["repeat"] = rep
                    reps.append(out)
                mm = mean_metrics(reps)
                cell = {
                    "role": root["role"],
                    "episode_index": ep,
                    "frame": frame,
                    "phase": root["phase"],
                    "action": action_name,
                    "gain_name": gcfg["name"],
                    "gain_config": gcfg,
                    "gains": reps[0]["gains"],
                    "metrics_mean": mm,
                    "repeats": [{"repeat": r["repeat"], "metrics": r["metrics"]} for r in reps],
                }
                cells.append(cell)
                fname = f"ep{ep:03d}_f{frame:04d}__{root['role']}__{action_name}__{gcfg['name']}.json"
                (out_dir / fname).write_text(
                    json.dumps(cell, indent=2, default=float), encoding="utf-8"
                )

    heldout = [c for c in cells if c["role"] == "held_out"]
    verdict = judge(heldout, cfg)

    manifest = {
        "protocol": PROTOCOL,
        "finished_at": _utc(),
        "config": cfg,
        "n_cells": len(cells),
        "verdict": verdict,
        "index": [
            {
                "role": c["role"],
                "episode_index": c["episode_index"],
                "frame": c["frame"],
                "action": c["action"],
                "gain_name": c["gain_name"],
                "metrics_mean": c["metrics_mean"],
            }
            for c in cells
        ],
    }
    man_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, default=float), encoding="utf-8"
    )

    lines = [
        "# Compliance Utility / Oracle P0 Result",
        "",
        f"- 完成：{_utc()}",
        f"- 判定：`{verdict['verdict']}`",
        f"- 分支：{verdict['branch']}",
        f"- 允许 wrapper：{verdict['allow_wrapper']}",
        f"- 只改默认增益：{verdict['fix_default_only']}",
        f"- 摘要：{verdict['summary']}",
        "",
        "## Oracle vs baseline",
        "",
        "```json",
        json.dumps(verdict["oracle_vs_baseline"], indent=2),
        "```",
        "",
        "## Best fixed",
        "",
        "```json",
        json.dumps(verdict["best_fixed"], indent=2),
        "```",
        "",
        "## Oracle picks (held-out)",
        "",
        "```json",
        json.dumps(verdict["oracle_picks"], indent=2, default=float),
        "```",
        "",
        "## 决策",
        "",
    ]
    if verdict["branch"] == 1:
        lines += ["- **放弃** compliance 方案。", "- 不实现 wrapper，不开训。", ""]
    elif verdict["branch"] == 2:
        lines += [
            f"- 仅考虑把默认增益改为 `{verdict['best_fixed']['gain_name']}`。",
            "- 不做动态 compliance wrapper / 策略训练。",
            "",
        ]
    else:
        lines += ["- Utility 支持状态依赖 compliance；仍需单独授权才实现 wrapper。", ""]

    report_path.write_text("\n".join(lines), encoding="utf-8")

    state = {
        "date": "2026-08-15",
        "phase": "compliance_utility_oracle_p0_complete",
        "busy": False,
        "training_allowed": False,
        "collection_allowed": False,
        "wrapper_allowed": bool(verdict["allow_wrapper"]),
        "controller_compliance_p0": {
            "causal_pass": True,
            "utility_pass": bool(verdict["branch"] == 3),
            "utility_verdict": verdict["verdict"],
            "branch": verdict["branch"],
        },
        "compliance_utility_oracle_p0": {
            "manifest": str(man_path.relative_to(PROJECT_ROOT)),
            "report": str(report_path.relative_to(PROJECT_ROOT)),
            "verdict": verdict["verdict"],
            "branch": verdict["branch"],
        },
        "next_action": {
            1: "abandon_compliance_line",
            2: "optional_fix_default_gains_no_training",
            3: "await_approval_compliance_wrapper",
        }[verdict["branch"]],
    }
    (PROJECT_ROOT / "outputs" / "state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )

    print(json.dumps({"verdict": verdict["verdict"], "branch": verdict["branch"], "summary": verdict["summary"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
