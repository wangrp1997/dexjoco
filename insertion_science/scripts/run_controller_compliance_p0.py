#!/usr/bin/env python3
"""Controller Compliance Causal P0: same snapshot+actions, vary OSC gains only."""

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
for _p in (str(PROJECT_ROOT), str(DEXJOCO_ROOT), str(EMBODIED), str(DEXJOCO_ROOT.parent / "reach_insert_rl")):
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
from insertion_science.physics.osc_gain_patch import (  # noqa: E402
    osc_gains,
    scale_gains,
)
from reach_insert_rl.env.full_obs import privileged_full_features  # noqa: E402

PROTOCOL = "ControllerComplianceCausalP0"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_cfg(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_actions(env, *, name: str, root_frame: int, horizon: int) -> tuple[np.ndarray, dict]:
    wrist = build_demo_wrist_sequence(env, root_frame=root_frame, horizon=horizon)
    if name == "hold":
        a = np.zeros((horizon, 44), dtype=np.float64)
        return a, {"mode": "hold"}
    if name == "demo_matched":
        a, meta = build_right_demo_replay_actions(
            env, root_frame=root_frame, horizon=horizon, wrist_seq=wrist
        )
        return a, meta
    raise ValueError(name)


def _wrist_ft_norm(env) -> float:
    fl = getattr(env, "_force_labeler", None)
    if fl is None:
        return float("nan")
    frame = fl.compute(env._raw)
    ft = np.asarray(frame.wrist_ft_right[:3], dtype=np.float64)
    return float(np.linalg.norm(ft))


def _contact_force_proxy(raw) -> float:
    """Sum of contact force magnitudes (MuJoCo contact force buffer)."""
    import mujoco

    model, data = raw._model, raw._data
    total = 0.0
    force = np.zeros(6, dtype=np.float64)
    for i in range(int(data.ncon)):
        mujoco.mj_contactForce(model, data, i, force)
        total += float(np.linalg.norm(force[:3]))
    return total


def rollout_once(
    env,
    snap: FullEpisodeSnapshot,
    actions: np.ndarray,
    *,
    gains_ctx,
    root_o2h,
    root_contact,
    root_z: float,
    tip0: float,
    lat0: float,
    dt: float,
) -> dict[str, Any]:
    snap.restore(env)
    if bool(env._done):
        raise RuntimeError("restore left done=True")
    steps_m = []
    tip_series = []
    lat_series = []
    wrist_ft = []
    contact_f = []
    nan_hit = False
    term_reason = "horizon_end"
    with gains_ctx:
        for a in actions:
            if env._done:
                term_reason = "already_done"
                break
            obs, _, term, trunc, info = env.step(np.asarray(a, dtype=np.float64))
            if not np.isfinite(obs).all():
                nan_hit = True
                term_reason = "nonfinite_obs"
                break
            sm = compute_step_metrics(env)
            steps_m.append(sm)
            feat = privileged_full_features(env._raw)
            tip_series.append(float(feat["tip_dist"]))
            lat_series.append(float(feat["lat_err"]))
            wrist_ft.append(_wrist_ft_norm(env))
            contact_f.append(_contact_force_proxy(env._raw))
            if term or trunc:
                term_reason = str(info.get("fail_reason") or ("terminated" if term else "truncated"))
                break

    if not steps_m:
        return {
            "metrics": {"num_steps": 0, "error": "empty_rollout", "term_reason": term_reason},
            "extra": {},
        }

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
    tip_progress = float(tip0 - tip[-1])  # positive = closer to hole
    lat_progress = float(lat0 - lat[-1])
    # jam proxy: high mean contact force and tip not improving
    jam_proxy = bool(float(np.mean(cf)) > 50.0 and tip_progress < 0.001)
    metrics.update(
        {
            "term_reason": term_reason,
            "executed_steps": len(steps_m),
            "tip_start_m": float(tip0),
            "tip_end_m": float(tip[-1]),
            "tip_progress_m": tip_progress,
            "lat_start_m": float(lat0),
            "lat_end_m": float(lat[-1]),
            "lat_progress_m": lat_progress,
            "wrist_ft_mean_n": float(np.nanmean(wft)),
            "wrist_ft_max_n": float(np.nanmax(wft)),
            "contact_force_mean_n": float(np.mean(cf)),
            "contact_force_max_n": float(np.max(cf)),
            "jam_proxy": jam_proxy,
            "nonfinite_obs": bool(nan_hit),
            "insert_ok_end": bool(steps_m[-1].insert_ok),
        }
    )
    return {"metrics": metrics, "extra": {"tip_series": tip.tolist(), "wrist_ft": wft.tolist()}}


PRIMARY_KEYS = (
    "trans_drift_max_m",
    "rot_drift_max_rad",
    "contact_retention_vs_root_mean",
    "wrist_ft_mean_n",
    "contact_force_mean_n",
    "tip_progress_m",
)


def _mv(m: dict[str, Any], key: str) -> float:
    v = m.get(key)
    if isinstance(v, bool):
        return float(v)
    if v is None:
        return 0.0
    return float(v)


def replay_spread(repeats: list[dict[str, Any]], key: str) -> float:
    vals = np.asarray([_mv(r["metrics"], key) for r in repeats], dtype=np.float64)
    if vals.size < 2:
        return 0.0
    return float(vals.max() - vals.min())


def existence_vs_baseline(
    base_m: dict[str, Any],
    alt_m: dict[str, Any],
    *,
    spreads: dict[str, float],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    hits = []
    details = {}
    gate_map = {
        "trans_drift_max_m": cfg["existence_trans_drift_m"],
        "rot_drift_max_rad": cfg["existence_rot_drift_rad"],
        "contact_retention_vs_root_mean": cfg["existence_retention_abs"],
        "wrist_ft_mean_n": cfg["existence_wrist_ft_n"],
        "contact_force_mean_n": cfg["existence_contact_force_n"],
        "tip_progress_m": cfg["existence_tip_progress_m"],
    }
    k = float(cfg["replay_tol_k"])
    for key in PRIMARY_KEYS:
        bv, av = _mv(base_m, key), _mv(alt_m, key)
        diff = av - bv
        thr = max(float(gate_map[key]), k * float(spreads.get(key, 0.0)))
        sig = bool(abs(diff) > thr)
        details[key] = {
            "baseline": bv,
            "alt": av,
            "diff": float(diff),
            "threshold": thr,
            "significant": sig,
        }
        if sig:
            hits.append(key)
    return {"exists": bool(hits), "hit_metrics": hits, "details": details}


def harmful_only(ex: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """True if force drops while retention/progress significantly worsen — and little else."""
    d = ex["details"]
    force_down = (
        d["wrist_ft_mean_n"]["diff"] < -float(cfg["existence_wrist_ft_n"])
        or d["contact_force_mean_n"]["diff"] < -float(cfg["existence_contact_force_n"])
    )
    ret_worse = d["contact_retention_vs_root_mean"]["diff"] < -float(cfg["harm_retention_abs"])
    tip_worse = d["tip_progress_m"]["diff"] < -float(cfg["harm_tip_progress_m"])
    other_hits = [
        h
        for h in ex["hit_metrics"]
        if h
        not in (
            "wrist_ft_mean_n",
            "contact_force_mean_n",
            "contact_retention_vs_root_mean",
            "tip_progress_m",
        )
    ]
    if force_down and (ret_worse or tip_worse) and not other_hits:
        # also fail if force-down + harm is the dominant story even with drift hits that
        # indicate loss of grasp rather than useful compliance
        if "trans_drift_max_m" in ex["hit_metrics"] or "rot_drift_max_rad" in ex["hit_metrics"]:
            # drift up with retention down = drop/slip, still harmful-only for compliance claim
            if ret_worse or tip_worse:
                return True
        return True
    return False


def directionality(diffs: list[float]) -> dict[str, Any]:
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "same_sign": False, "mean": None}
    mean = float(arr.mean())
    if abs(mean) < 1e-15:
        return {"n": int(arr.size), "same_sign": False, "mean": mean}
    same = bool(np.all(np.sign(arr) == np.sign(mean)))
    return {"n": int(arr.size), "same_sign": same, "mean": mean, "diffs": arr.tolist()}


def check_restore(env, snap: FullEpisodeSnapshot, atol: float) -> dict[str, Any]:
    snap.restore(env)
    q0 = np.asarray(env._raw._data.qpos, dtype=np.float64).copy()
    snap.restore(env)
    q1 = np.asarray(env._raw._data.qpos, dtype=np.float64).copy()
    err = float(np.max(np.abs(q0 - q1)))
    return {"ok": bool(err <= float(atol)), "max_abs_qpos_delta": err, "atol": float(atol)}


def judge(
    *,
    restore_ok: bool,
    heldout_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    if not restore_ok:
        return {
            "verdict": "fail_restore_not_reproducible",
            "pass": False,
            "stop": True,
            "summary": "matched restore 不可复现；停止。",
        }

    # Evaluate each pre-registered scale independently (no post-hoc selection).
    scales = [s for s in cfg["stiffness_scales"] if float(s) != 1.0]
    scale_reports = []
    any_pass = False
    for scale in scales:
        # per action mode; require demo_matched or hold — prefer any action that meets criteria
        for action in cfg["actions"]:
            rows = [
                r
                for r in heldout_rows
                if float(r["stiffness_scale"]) == float(scale) and r["action"] == action
            ]
            if len(rows) < 2:
                continue
            n_exist = sum(1 for r in rows if r["existence"]["exists"])
            # direction: for each primary key, check same-sign diffs across held-out
            dir_ok = False
            dir_keys = []
            for key in PRIMARY_KEYS:
                diffs = [float(r["existence"]["details"][key]["diff"]) for r in rows]
                # only consider keys that were significant on ≥1 root
                if not any(r["existence"]["details"][key]["significant"] for r in rows):
                    continue
                dstat = directionality(diffs)
                if dstat.get("same_sign"):
                    dir_ok = True
                    dir_keys.append(key)
            harm_flags = [harmful_only(r["existence"], cfg) for r in rows]
            # harmful-only if ALL held-out effects that exist are harmful-only
            all_harm = bool(n_exist >= 2 and all(harm_flags[i] for i, r in enumerate(rows) if r["existence"]["exists"]))
            passed = bool(n_exist >= 2 and dir_ok and not all_harm)
            rep = {
                "stiffness_scale": float(scale),
                "action": action,
                "n_heldout": len(rows),
                "n_existence": int(n_exist),
                "direction_consistent_keys": dir_keys,
                "direction_ok": dir_ok,
                "all_effects_harmful_only": all_harm,
                "pass": passed,
            }
            scale_reports.append(rep)
            if passed:
                any_pass = True

    if any_pass:
        return {
            "verdict": "pass_compliance_causal_effect",
            "pass": True,
            "stop": False,
            "scale_reports": scale_reports,
            "summary": "held-out 上存在方向一致的 compliance 因果效应；可进入动作接口设计。",
        }
    return {
        "verdict": "fail_no_stable_compliance_effect",
        "pass": False,
        "stop": True,
        "scale_reports": scale_reports,
        "summary": "held-out 无稳定同向效应，或仅有 force↓+retention/progress 恶化；停止本线。",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "controller_compliance_p0.yaml",
    )
    args = ap.parse_args()
    cfg = _load_cfg(args.config)

    out_dir = PROJECT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = PROJECT_ROOT / cfg["manifest_path"]
    man_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / cfg["report_path"]

    roots = []
    for r in cfg["discovery_roots"]:
        roots.append({**r, "role": "discovery"})
    for r in cfg["held_out_roots"]:
        roots.append({**r, "role": "held_out"})

    # freeze root list to disk before outcomes
    frozen = {
        "protocol": PROTOCOL,
        "frozen_at": _utc(),
        "roots": roots,
        "stiffness_scales": cfg["stiffness_scales"],
        "damping_ratios": cfg["damping_ratios"],
        "actions": cfg["actions"],
    }
    (out_dir / "roots_frozen.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    episodes = sorted({int(r["episode_index"]) for r in roots})
    env = make_full_env(episodes, sidecar_dir=Path(cfg["sidecar_dir"]), seed=int(cfg["seed"]))
    dt = control_dt_seconds(env)
    baseline_pos = tuple(cfg["baseline_pos_gains"])
    baseline_ori = tuple(cfg["baseline_ori_gains"])

    all_rows: list[dict[str, Any]] = []
    restore_checks: list[dict[str, Any]] = []

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

        rc = check_restore(env, snap, cfg["restore_qpos_atol"])
        rc["root"] = f"ep{ep}_f{frame:04d}"
        restore_checks.append(rc)

        for action_name in cfg["actions"]:
            actions, ameta = build_actions(
                env, name=action_name, root_frame=frame, horizon=int(cfg["horizon"])
            )
            # run all scales; keep repeats for baseline+alts
            by_scale: dict[float, list[dict[str, Any]]] = {}
            for scale in cfg["stiffness_scales"]:
                for damp in cfg["damping_ratios"]:
                    g = scale_gains(
                        baseline_pos=baseline_pos,
                        baseline_ori=baseline_ori,
                        stiffness_scale=float(scale),
                        damping_ratio=float(damp),
                    )
                    repeats = []
                    for rep in range(int(cfg["repeats"])):
                        ctx = osc_gains(
                            pos_gains=g.pos_gains,
                            ori_gains=g.ori_gains,
                            damping_ratio=g.damping_ratio,
                        )
                        out = rollout_once(
                            env,
                            snap,
                            actions,
                            gains_ctx=ctx,
                            root_o2h=root_o2h,
                            root_contact=root_contact,
                            root_z=root_z,
                            tip0=tip0,
                            lat0=lat0,
                            dt=dt,
                        )
                        out["repeat"] = rep
                        out["gains"] = g.as_dict()
                        repeats.append(out)
                    by_scale[float(scale)] = repeats

            spreads: dict[str, float] = {}
            for key in PRIMARY_KEYS:
                # max spread across scales' repeats (conservative noise floor)
                spreads[key] = max(
                    replay_spread(reps, key) for reps in by_scale.values()
                )

            base_reps = by_scale[1.0]
            base_mean = {
                k: float(np.mean([_mv(r["metrics"], k) for r in base_reps]))
                for k in list(PRIMARY_KEYS)
                + [
                    "jam_proxy",
                    "terminal_peg_ok",
                    "insert_ok_end",
                    "nonfinite_obs",
                ]
            }
            # merge mean metrics from first repeat structure
            base_m = dict(base_reps[0]["metrics"])
            for k, v in base_mean.items():
                if k in ("jam_proxy", "terminal_peg_ok", "insert_ok_end", "nonfinite_obs"):
                    base_m[k] = bool(round(v))
                else:
                    base_m[k] = v

            for scale, reps in by_scale.items():
                alt_m = dict(reps[0]["metrics"])
                alt_mean = {
                    k: float(np.mean([_mv(r["metrics"], k) for r in reps]))
                    for k in PRIMARY_KEYS
                }
                alt_m.update(alt_mean)
                for bk in ("jam_proxy", "terminal_peg_ok", "insert_ok_end", "nonfinite_obs"):
                    alt_m[bk] = bool(
                        round(float(np.mean([_mv(r["metrics"], bk) for r in reps])))
                    )
                if float(scale) == 1.0:
                    ex = {
                        "exists": False,
                        "hit_metrics": [],
                        "details": {
                            k: {
                                "baseline": _mv(base_m, k),
                                "alt": _mv(alt_m, k),
                                "diff": 0.0,
                                "threshold": 0.0,
                                "significant": False,
                            }
                            for k in PRIMARY_KEYS
                        },
                    }
                else:
                    ex = existence_vs_baseline(base_m, alt_m, spreads=spreads, cfg=cfg)

                row = {
                    "role": root["role"],
                    "episode_index": ep,
                    "frame": frame,
                    "phase": root["phase"],
                    "action": action_name,
                    "action_meta": ameta,
                    "stiffness_scale": float(scale),
                    "damping_ratio": float(cfg["damping_ratios"][0]),
                    "gains": reps[0]["gains"],
                    "metrics_mean": {k: _mv(alt_m, k) for k in list(PRIMARY_KEYS) + [
                        "jam_proxy",
                        "terminal_peg_ok",
                        "insert_ok_end",
                        "tip_end_m",
                        "lat_progress_m",
                        "wrist_ft_max_n",
                        "contact_force_max_n",
                        "nonfinite_obs",
                    ]},
                    "replay_spreads": spreads,
                    "existence": ex,
                    "harmful_only": bool(harmful_only(ex, cfg)) if float(scale) != 1.0 else False,
                    "repeats": [
                        {"repeat": r["repeat"], "metrics": r["metrics"]} for r in reps
                    ],
                }
                all_rows.append(row)
                # slim per-branch dump
                branch_name = (
                    f"ep{ep:03d}_f{frame:04d}__{root['role']}__{action_name}__k{scale}"
                )
                (out_dir / f"{branch_name}.json").write_text(
                    json.dumps(row, indent=2, default=float), encoding="utf-8"
                )

    restore_ok = all(r["ok"] for r in restore_checks)
    heldout_rows = [r for r in all_rows if r["role"] == "held_out" and r["stiffness_scale"] != 1.0]
    verdict = judge(restore_ok=restore_ok, heldout_rows=heldout_rows, cfg=cfg)

    manifest = {
        "protocol": PROTOCOL,
        "finished_at": _utc(),
        "config": cfg,
        "restore_checks": restore_checks,
        "restore_ok": restore_ok,
        "n_rows": len(all_rows),
        "verdict": verdict,
        "rows_index": [
            {
                "role": r["role"],
                "episode_index": r["episode_index"],
                "frame": r["frame"],
                "action": r["action"],
                "stiffness_scale": r["stiffness_scale"],
                "existence": r["existence"]["exists"],
                "hit_metrics": r["existence"]["hit_metrics"],
                "harmful_only": r["harmful_only"],
                "metrics_mean": r["metrics_mean"],
            }
            for r in all_rows
        ],
    }
    man_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, default=float), encoding="utf-8"
    )

    # report
    lines = [
        f"# Controller Compliance Causal P0 Result",
        "",
        f"- 完成时间：{_utc()}",
        f"- 判定：`{verdict['verdict']}`",
        f"- 通过：{verdict['pass']}",
        f"- 停止：{verdict['stop']}",
        f"- 摘要：{verdict['summary']}",
        "",
        "## Restore",
        "",
        f"- ok: {restore_ok}",
        "",
        "## Held-out scale reports",
        "",
        "```json",
        json.dumps(verdict.get("scale_reports", []), indent=2),
        "```",
        "",
        "## Row index (compact)",
        "",
        "```json",
        json.dumps(manifest["rows_index"], indent=2, default=float),
        "```",
        "",
    ]
    if verdict["stop"]:
        lines += [
            "## 决策",
            "",
            "- **正式停止** Controller Compliance 线。",
            "- 不设计动作接口，不设训练硬门，不进入 Stage 后续。",
            "",
        ]
    else:
        lines += [
            "## 决策",
            "",
            "- P0 通过：下一步可设计将 compliance/gains 纳入动作接口的方案与训练硬门。",
            "- 仍禁止直接开训。",
            "",
        ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # state
    state_path = PROJECT_ROOT / "outputs" / "state.json"
    state = {
        "date": "2026-08-15",
        "phase": "controller_compliance_p0_complete",
        "busy": False,
        "training_allowed": False,
        "collection_allowed": False,
        "simulation_experiment_allowed": False,
        "controller_compliance_p0": {
            "pass": verdict["pass"],
            "verdict": verdict["verdict"],
            "stop": verdict["stop"],
            "manifest": str(man_path.relative_to(PROJECT_ROOT)),
            "report": str(report_path.relative_to(PROJECT_ROOT)),
        },
        "next_action": (
            "design_compliance_action_interface_hard_gates"
            if verdict["pass"]
            else "stop_compliance_line_no_policy"
        ),
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print(json.dumps({"verdict": verdict, "manifest": str(man_path)}, indent=2, default=str))
    return 0 if verdict["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
