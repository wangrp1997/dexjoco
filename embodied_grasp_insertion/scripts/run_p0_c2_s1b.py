#!/usr/bin/env python3
"""P0-C2-S1b: actuation + heterogeneous fork audit (no Stage-2 auto-run).

Withdraws Stage-1 decision-tree A. Checks whether finger commands move joints,
whether matched branches create per-root physical forks (existence), and whether
effects are same-signed across roots (directionality; optional generality).
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

from embodied_grasp_insertion.physics.c2_root_criteria import (  # noqa: E402
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
from embodied_grasp_insertion.scripts.run_p0_c2_stage1 import (  # noqa: E402
    build_demo_wrist_sequence,
)
from embodied_grasp_insertion.simulation.calibrated_interventions import (  # noqa: E402
    RIGHT_FINGER_ACTUATOR_NAMES,
    RIGHT_FINGER_IDX,
    RIGHT_FINGER_JOINT_NAMES,
    WRIST_IDX,
    assert_left_fingers_zero,
    build_calibrated_right_offset,
    build_right_demo_replay_actions,
    load_semantics,
    project_independent_feasible_offset,
    target_offset_to_pulse_actions,
)
from embodied_grasp_insertion.simulation.full_episode_snapshot import (  # noqa: E402
    FullEpisodeSnapshot,
)
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    load_yaml,
    make_full_env,
    replay_demo_to_frame,
    select_roots_for_episode,
)

PROTOCOL = "P0-C2-S1b"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def right_finger_state(env) -> dict[str, np.ndarray]:
    model, data = env._raw._model, env._raw._data
    qpos = np.zeros(16, dtype=np.float64)
    qvel = np.zeros(16, dtype=np.float64)
    ctrl = np.zeros(16, dtype=np.float64)
    for i, jn in enumerate(RIGHT_FINGER_JOINT_NAMES):
        jid = int(model.joint(jn).id)
        qpos[i] = float(data.qpos[int(model.jnt_qposadr[jid])])
        qvel[i] = float(data.qvel[int(model.jnt_dofadr[jid])])
    for i, an in enumerate(RIGHT_FINGER_ACTUATOR_NAMES):
        aid = int(model.actuator(an).id)
        ctrl[i] = float(data.ctrl[aid])
    hold = np.asarray(env._hold44, dtype=np.float64)[RIGHT_FINGER_IDX]
    return {"qpos": qpos, "qvel": qvel, "ctrl": ctrl, "hold_target": hold.copy()}


def build_actions(
    env,
    *,
    name: str,
    wrist_seq: np.ndarray,
    root_frame: int,
    horizon: int,
    semantics,
    close_rad: float,
    pulse_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    if name == "hold_finger":
        a = wrist_seq.copy()
        a[:, RIGHT_FINGER_IDX] = 0.0
        return a, {"mode": "hold_finger", "requested_l2": 0.0, "realized_l2": 0.0}
    if name == "demo_finger_replay":
        a, meta = build_right_demo_replay_actions(
            env, root_frame=root_frame, horizon=horizon, wrist_seq=wrist_seq
        )
        assert_left_fingers_zero(a)
        return a, meta
    if name == "calibrated_finger_intervention":
        raw = build_calibrated_right_offset(
            semantics, mode="calibrated_close_low", low_rad=close_rad, medium_rad=close_rad
        )
        proj, pmeta = project_independent_feasible_offset(env, raw)
        a, meta = target_offset_to_pulse_actions(
            env,
            right_offset_rad=proj,
            horizon=horizon,
            pulse_steps=pulse_steps,
            wrist_seq=wrist_seq,
            allow_clip=False,
        )
        assert_left_fingers_zero(a)
        meta["projection"] = pmeta
        meta["dose_coupling"] = "independent"
        return a, meta
    if name == "random_finger_control":
        raw = build_calibrated_right_offset(
            semantics,
            mode="random_matched",
            low_rad=close_rad,
            medium_rad=close_rad,
            rng=rng,
        )
        proj, pmeta = project_independent_feasible_offset(env, raw)
        a, meta = target_offset_to_pulse_actions(
            env,
            right_offset_rad=proj,
            horizon=horizon,
            pulse_steps=pulse_steps,
            wrist_seq=wrist_seq,
            allow_clip=False,
        )
        assert_left_fingers_zero(a)
        meta["projection"] = pmeta
        meta["dose_coupling"] = "independent"
        return a, meta
    raise ValueError(name)


def rollout_tracked(
    env,
    snap: FullEpisodeSnapshot,
    actions: np.ndarray,
    *,
    root_o2h,
    root_contact,
    root_z: float,
    dt: float,
) -> dict[str, Any]:
    snap.restore(env)
    if bool(env._done):
        raise RuntimeError("restore left done=True")
    s0 = right_finger_state(env)
    steps_m = []
    track = []
    executed = []
    term_reason = "horizon_end"
    for a in actions:
        if env._done:
            term_reason = "already_done"
            break
        cmd = np.asarray(a, dtype=np.float64)
        _, _, term, trunc, info = env.step(cmd)
        st = right_finger_state(env)
        track.append(
            {
                "cmd_finger_delta": cmd[RIGHT_FINGER_IDX].tolist(),
                "ctrl": st["ctrl"].tolist(),
                "qpos": st["qpos"].tolist(),
                "qvel": st["qvel"].tolist(),
                "tracking_err_ctrl_minus_qpos": (st["ctrl"] - st["qpos"]).tolist(),
                "contact_total": int(peg_hand_contact_counts(env._raw).total),
            }
        )
        steps_m.append(compute_step_metrics(env))
        executed.append(cmd.copy())
        if term or trunc:
            term_reason = str(info.get("fail_reason") or ("terminated" if term else "truncated"))
            break
    if not steps_m:
        metrics = {"num_steps": 0, "error": "empty_rollout", "term_reason": term_reason}
        q_delta = np.zeros(16)
    else:
        metrics = summarize_rollout_metrics_v2(
            steps_m,
            root_o2h=root_o2h,
            root_contact=root_contact,
            root_peg_world_z=root_z,
            control_dt_s=dt,
        )
        metrics["term_reason"] = term_reason
        metrics["executed_steps"] = len(steps_m)
        q_end = np.asarray(track[-1]["qpos"], dtype=np.float64)
        q_delta = q_end - s0["qpos"]
    force_delta = None
    if steps_m:
        f0 = np.asarray(steps_m[0].right_finger_force_norm, dtype=np.float64)
        f1 = np.asarray(steps_m[-1].right_finger_force_norm, dtype=np.float64)
        force_delta = (f1 - f0).tolist()
    return {
        "metrics": metrics,
        "actuation": {
            "qpos0": s0["qpos"].tolist(),
            "qpos_delta_max_abs": float(np.max(np.abs(q_delta))) if q_delta.size else 0.0,
            "qpos_delta_l2": float(np.linalg.norm(q_delta)),
            "qpos_delta": q_delta.tolist(),
            "mean_abs_qvel": float(
                np.mean([np.mean(np.abs(np.asarray(t["qvel"]))) for t in track])
            )
            if track
            else 0.0,
            "mean_abs_tracking_err": float(
                np.mean(
                    [
                        np.mean(np.abs(np.asarray(t["tracking_err_ctrl_minus_qpos"])))
                        for t in track
                    ]
                )
            )
            if track
            else 0.0,
            "contact_total_start": int(track[0]["contact_total"]) if track else None,
            "contact_total_end": int(track[-1]["contact_total"]) if track else None,
            "finger_force_norm_delta": force_delta,
            "n_track_steps": len(track),
        },
        "track_tail": track[-3:] if track else [],
        "future_actions44": np.asarray(executed, dtype=np.float64).tolist(),
    }


PRIMARY = (
    "trans_drift_max_m",
    "rot_drift_max_rad",
    "contact_retention_vs_root_mean",
    "terminal_peg_ok",
    "object_dropped_proxy",
)


def _metric_val(m: dict[str, Any], key: str) -> float:
    v = m.get(key)
    if isinstance(v, bool):
        return float(v)
    return float(v if v is not None else 0.0)


def replay_spread(repeats_metrics: list[dict[str, Any]], key: str) -> float:
    vals = np.asarray([_metric_val(m, key) for m in repeats_metrics], dtype=np.float64)
    if vals.size < 2:
        return 0.0
    return float(vals.max() - vals.min())


def existence_fork(
    hold_m: dict[str, Any],
    interv_m: dict[str, Any],
    *,
    hold_spreads: dict[str, float],
    interv_spreads: dict[str, float],
    gates: dict[str, float],
    k: float,
) -> dict[str, Any]:
    hits = []
    details = {}
    for key in PRIMARY:
        hv = _metric_val(hold_m, key)
        iv = _metric_val(interv_m, key)
        diff = abs(iv - hv)
        # boolean metrics: any disagreement is existence
        if key in ("terminal_peg_ok", "object_dropped_proxy"):
            sig = bool(hv != iv)
            thr = 0.0
        else:
            base = {
                "trans_drift_max_m": gates["existence_trans_drift_m"],
                "rot_drift_max_rad": gates["existence_rot_drift_rad"],
                "contact_retention_vs_root_mean": gates["existence_retention_abs"],
            }[key]
            thr = max(base, k * max(hold_spreads.get(key, 0.0), interv_spreads.get(key, 0.0)))
            sig = bool(diff > thr)
        details[key] = {"abs_diff": diff, "threshold": thr, "significant": sig}
        if sig:
            hits.append(key)
    return {"exists": bool(hits), "hit_metrics": hits, "details": details}


def directionality(diffs: list[float]) -> dict[str, Any]:
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "same_sign_fraction": None, "ci_excludes_zero": False}
    rng = np.random.default_rng(20260815)
    boots = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(1000)]
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    signs = np.sign(arr)
    same = float(np.mean(signs == np.sign(arr.mean()))) if abs(arr.mean()) > 1e-15 else 0.0
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "ci_excludes_zero": bool(hi < 0 or lo > 0),
        "same_sign_fraction": same,
    }


def judge_s1b(
    *,
    actuation_moved: bool,
    existence_on_frozen: bool,
    existence_on_heldout: bool | None,
    directional: bool,
) -> dict[str, Any]:
    if not actuation_moved:
        return {
            "overall_verdict": "h2_untested_actuation_dead",
            "research_decision": "fix_actuator_control_interface",
            "enter_stage2": False,
            "summary": "实际关节基本没动：H2 尚未被测试，修 actuator/control interface。",
        }
    if existence_on_heldout is False or (
        existence_on_heldout is None and not existence_on_frozen
    ):
        # held-out fully no forks, or no existence anywhere
        if existence_on_heldout is False and not existence_on_frozen:
            pass
        if existence_on_heldout is False:
            return {
                "overall_verdict": "h2_failed_no_physical_fork_heldout",
                "research_decision": "stop_h2_controllability_route",
                "enter_stage2": False,
                "summary": "关节动了，held-out roots 仍无物理分叉：当前 H2 正式失败。",
            }
        if not existence_on_frozen:
            return {
                "overall_verdict": "h2_failed_no_physical_fork",
                "research_decision": "stop_h2_controllability_route",
                "enter_stage2": False,
                "summary": "关节动了，但 matched branches 无超过重放容差的物理分叉。",
            }
    if existence_on_frozen and (existence_on_heldout is True or existence_on_heldout is None):
        # existence with optional held-out confirmation
        if existence_on_heldout is True or (
            existence_on_heldout is None and existence_on_frozen
        ):
            # Prefer requiring held-out when available
            if existence_on_heldout is False:
                pass  # handled above
            ok_held = existence_on_heldout is True
            if ok_held or (existence_on_heldout is None and existence_on_frozen):
                if ok_held or existence_on_frozen:
                    return {
                        "overall_verdict": "h2_controllability_exists_heterogeneous",
                        "research_decision": "enter_stage2_action_conditioned_eligible",
                        "enter_stage2": True,
                        "directional_universal_intervention": bool(directional),
                        "summary": (
                            "关节动了，并出现可重复但方向可能异质的物理分叉："
                            "可控性存在；尚无通用动作则需 action-conditioned Stage-2。"
                        ),
                    }
    # frozen exists but held-out missing/false already returned
    if existence_on_frozen and existence_on_heldout is True:
        return {
            "overall_verdict": "h2_controllability_exists_heterogeneous",
            "research_decision": "enter_stage2_action_conditioned_eligible",
            "enter_stage2": True,
            "directional_universal_intervention": bool(directional),
            "summary": "关节动了且 frozen+held-out 均有存在性分叉。",
        }
    return {
        "overall_verdict": "c2_s1b_inconclusive",
        "research_decision": "await_human",
        "enter_stage2": False,
        "summary": "S1b 结果未落入三路预注册判定，需人工审。",
    }


def run_root_block(
    env,
    root: dict[str, Any],
    *,
    cfg: dict[str, Any],
    semantics,
    repeats: int,
    seed: int,
    out_branch_dir: Path,
    pool: str,
) -> list[dict[str, Any]]:
    ep = int(root["episode_index"])
    frame = int(root["frame"])
    phase = str(root["phase"])
    horizon = int(cfg["horizon"])
    close_rad = float(cfg["calibrated_close_rad"])
    pulse_steps = int(cfg["pulse_steps"])
    fair_q = float(cfg["fairness"]["init_qpos_atol"])
    fair_o = float(cfg["fairness"]["init_obs_atol"])
    dt = control_dt_seconds(env)

    env.reset(episode_index=ep)
    replay_demo_to_frame(env, frame)
    snap = FullEpisodeSnapshot.capture(env)
    root_o2h = object_in_hand_pose(env._raw)
    root_contact = peg_hand_contact_counts(env._raw)
    root_z = float(
        env._raw._data.xpos[env._raw._model.body("industreal_round_peg_8mm").id][2]
    )
    root_id = f"ep{ep:03d}_f{frame:04d}_{phase}"
    wrist_seq = build_demo_wrist_sequence(env, root_frame=frame, horizon=horizon)
    root_rng = np.random.default_rng(seed + ep * 10007 + frame)

    snap.restore(env)
    q0 = np.asarray(env._raw._data.qpos, dtype=np.float64).copy()
    obs0 = np.asarray(env._obs(), dtype=np.float64).copy()

    records: list[dict[str, Any]] = []
    for name in cfg["interventions"]:
        for rep in range(repeats):
            snap.restore(env)
            fair = bool(
                np.allclose(q0, env._raw._data.qpos, atol=fair_q)
                and np.allclose(obs0, env._obs(), atol=fair_o)
            )
            actions, ameta = build_actions(
                env,
                name=str(name),
                wrist_seq=wrist_seq,
                root_frame=frame,
                horizon=horizon,
                semantics=semantics,
                close_rad=close_rad,
                pulse_steps=pulse_steps,
                rng=root_rng,
            )
            fair = fair and np.allclose(
                actions[:, WRIST_IDX],
                wrist_seq[:, WRIST_IDX],
                atol=float(cfg["fairness"]["wrist_atol"]),
            )
            out = rollout_tracked(
                env,
                snap,
                actions,
                root_o2h=root_o2h,
                root_contact=root_contact,
                root_z=root_z,
                dt=dt,
            )
            rec = {
                "pool": pool,
                "root_id": root_id,
                "episode_index": ep,
                "frame": frame,
                "phase": phase,
                "intervention": str(name),
                "repeat": int(rep),
                "fairness_passed": bool(fair),
                "metrics": out["metrics"],
                "actuation": out["actuation"],
                "action_meta": {
                    k: ameta.get(k)
                    for k in (
                        "mode",
                        "realized_l2",
                        "realized_abs_max",
                        "clip_count",
                        "dose_coupling",
                        "n_demo_steps",
                    )
                    if k in ameta or True
                },
                "projection": ameta.get("projection"),
            }
            # clean action_meta Nones
            rec["action_meta"] = {k: v for k, v in rec["action_meta"].items() if v is not None}
            records.append(rec)
            (out_branch_dir / f"{root_id}__{name}__r{rep}.json").write_text(
                json.dumps(
                    {**rec, "track_tail": out["track_tail"], "future_actions44": out["future_actions44"]},
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
    return records


def analyze_records(
    records: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
    pool: str,
) -> dict[str, Any]:
    gates = {
        "existence_trans_drift_m": float(cfg["existence_trans_drift_m"]),
        "existence_rot_drift_rad": float(cfg["existence_rot_drift_rad"]),
        "existence_retention_abs": float(cfg["existence_retention_abs"]),
    }
    k = float(cfg["replay_tol_k"])
    min_j = float(cfg["min_joint_abs_delta_rad"])

    by_root: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in records:
        if r.get("pool") != pool or not r.get("fairness_passed"):
            continue
        by_root.setdefault(r["root_id"], {}).setdefault(r["intervention"], []).append(r)

    # Actuation: non-hold interventions
    moved_flags = []
    dose_report = []
    for rid, mp in by_root.items():
        for name, reps in mp.items():
            if name == "hold_finger":
                continue
            deltas = [float(x["actuation"]["qpos_delta_max_abs"]) for x in reps]
            moved_flags.append(bool(np.mean(deltas) >= min_j))
            dose_report.append(
                {
                    "root_id": rid,
                    "intervention": name,
                    "mean_qpos_delta_max_abs": float(np.mean(deltas)),
                    "realized_l2_mean": float(
                        np.mean(
                            [
                                float((x.get("projection") or {}).get("realized_l2") or 0.0)
                                for x in reps
                            ]
                        )
                    ),
                }
            )
    actuation_moved = bool(moved_flags) and (sum(moved_flags) / len(moved_flags) >= 0.5)

    existence_roots = []
    dir_diffs: dict[str, dict[str, list[float]]] = {}
    for rid, mp in by_root.items():
        if "hold_finger" not in mp:
            continue
        hold_reps = mp["hold_finger"]
        hold_metrics = [x["metrics"] for x in hold_reps]
        hold_med = {key: float(np.median([_metric_val(m, key) for m in hold_metrics])) for key in PRIMARY}
        hold_spreads = {key: replay_spread(hold_metrics, key) for key in PRIMARY}
        root_exists = False
        for name, reps in mp.items():
            if name == "hold_finger":
                continue
            interv_metrics = [x["metrics"] for x in reps]
            interv_med = {
                key: float(np.median([_metric_val(m, key) for m in interv_metrics]))
                for key in PRIMARY
            }
            interv_spreads = {key: replay_spread(interv_metrics, key) for key in PRIMARY}
            ex = existence_fork(
                hold_med,
                interv_med,
                hold_spreads=hold_spreads,
                interv_spreads=interv_spreads,
                gates=gates,
                k=k,
            )
            if ex["exists"]:
                root_exists = True
            for key in PRIMARY:
                dir_diffs.setdefault(name, {}).setdefault(key, []).append(
                    _metric_val(interv_med, key) - _metric_val(hold_med, key)
                )
            existence_roots.append({"root_id": rid, "intervention": name, **ex})
        # mark per root
        # (also aggregate below)

    roots_with_existence = sorted(
        {
            e["root_id"]
            for e in existence_roots
            if e["exists"]
        }
    )
    existence_any = bool(roots_with_existence)

    directional = {}
    any_dir = False
    for name, by_m in dir_diffs.items():
        directional[name] = {}
        for key, diffs in by_m.items():
            d = directionality(diffs)
            directional[name][key] = d
            if d.get("ci_excludes_zero"):
                any_dir = True

    # repeatability
    rep_ok = True
    for rid, mp in by_root.items():
        for name, reps in mp.items():
            for key in ("trans_drift_max_m", "rot_drift_max_rad"):
                if replay_spread([x["metrics"] for x in reps], key) > 1e-2:
                    # soft: large spread flags but doesn't alone fail
                    pass
            # bit-ish: fairness all true already
            if len(reps) < int(cfg["repeats"]):
                rep_ok = False

    return {
        "pool": pool,
        "n_roots": len(by_root),
        "actuation_moved_fraction": float(sum(moved_flags) / len(moved_flags))
        if moved_flags
        else 0.0,
        "actuation_moved": actuation_moved,
        "min_joint_abs_delta_rad": min_j,
        "dose_report": dose_report,
        "existence_any": existence_any,
        "roots_with_existence": roots_with_existence,
        "existence_details": existence_roots,
        "directionality": directional,
        "directional_any_metric": any_dir,
        "repeatability_structure_ok": rep_ok,
    }


def select_held_out(cfg: dict[str, Any], frozen_keys: set[tuple[int, int]]) -> list[dict[str, Any]]:
    """Hold-screen only; never reads intervention outcomes."""
    s1 = load_yaml(PROJECT_ROOT / cfg["stage1_config"])
    sidecar = Path(cfg["sidecar_dir"])
    seed = int(cfg["seed"])
    horizon = int(cfg["horizon"])
    semantics = load_semantics(
        json.loads((PROJECT_ROOT / cfg["semantics_manifest"]).read_text())
    )
    screened = []
    rs = s1["root_selection"]
    for ep in [int(x) for x in cfg["held_out"]["episodes"]]:
        env = make_full_env([ep], sidecar_dir=sidecar, seed=seed)
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
            key = (ep, int(root.frame))
            if key in frozen_keys:
                continue
            env.reset(episode_index=ep)
            replay_demo_to_frame(env, int(root.frame))
            outcome0 = env._labeler.compute(env._raw)
            contact0 = peg_hand_contact_counts(env._raw)
            if not outcome0.peg_ok or outcome0.insert_ok:
                continue
            snap = FullEpisodeSnapshot.capture(env)
            root_o2h = object_in_hand_pose(env._raw)
            root_z = float(
                env._raw._data.xpos[env._raw._model.body("industreal_round_peg_8mm").id][2]
            )
            wrist = build_demo_wrist_sequence(
                env, root_frame=int(root.frame), horizon=horizon
            )
            hold_a = wrist.copy()
            hold_a[:, RIGHT_FINGER_IDX] = 0.0
            out = rollout_tracked(
                env,
                snap,
                hold_a,
                root_o2h=root_o2h,
                root_contact=contact0,
                root_z=root_z,
                dt=dt,
            )
            hold_m = out["metrics"]
            if int(hold_m.get("executed_steps") or 0) < max(4, horizon // 2):
                continue
            decision = accept_screened_root(
                root_contact_total=int(contact0.total),
                root_peg_ok=True,
                root_insert_ok=False,
                hold_metrics=hold_m,
            )
            screened.append(
                {
                    "episode_index": ep,
                    "frame": int(root.frame),
                    "phase": root.phase,
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
    return select_ranked_roots(
        screened,
        max_total=int(cfg["held_out"]["max_roots"]),
        max_per_episode=int(cfg["held_out"]["max_per_episode"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "p0_c2_s1b.yaml")
    args = ap.parse_args()
    if WRITE_IMPLEMENTATION_ENABLED:
        raise SystemExit("WRITE_IMPLEMENTATION_ENABLED must stay False")

    cfg = load_yaml(args.config)
    sem = json.loads((PROJECT_ROOT / cfg["semantics_manifest"]).read_text())
    semantics = load_semantics(sem)
    frozen_doc = json.loads((PROJECT_ROOT / cfg["stage1_frozen_roots"]).read_text())
    frozen = list(frozen_doc["selected"])
    frozen_keys = {(int(r["episode_index"]), int(r["frame"])) for r in frozen}

    out_dir = PROJECT_ROOT / cfg["output_dir"]
    branch_dir = out_dir / "branches"
    branch_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"phase": "frozen_block", "n": len(frozen)}, ensure_ascii=False), flush=True)
    all_recs: list[dict[str, Any]] = []
    by_ep: dict[int, list[dict[str, Any]]] = {}
    for r in frozen:
        by_ep.setdefault(int(r["episode_index"]), []).append(r)
    for ep, roots in by_ep.items():
        env = make_full_env([ep], sidecar_dir=Path(cfg["sidecar_dir"]), seed=int(cfg["seed"]))
        for root in roots:
            print(
                json.dumps(
                    {"running": "frozen", "ep": ep, "frame": root["frame"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            all_recs.extend(
                run_root_block(
                    env,
                    root,
                    cfg=cfg,
                    semantics=semantics,
                    repeats=int(cfg["repeats"]),
                    seed=int(cfg["seed"]),
                    out_branch_dir=branch_dir,
                    pool="frozen",
                )
            )

    print(json.dumps({"phase": "held_out_select"}, ensure_ascii=False), flush=True)
    held = select_held_out(cfg, frozen_keys)
    (out_dir / "held_out_roots_frozen.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "frozen_at": _utc(),
                "selection": "hold_screen_only",
                "selected": held,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(json.dumps({"phase": "held_out_block", "n": len(held)}, ensure_ascii=False), flush=True)
    by_ep = {}
    for r in held:
        by_ep.setdefault(int(r["episode_index"]), []).append(r)
    for ep, roots in by_ep.items():
        env = make_full_env([ep], sidecar_dir=Path(cfg["sidecar_dir"]), seed=int(cfg["seed"]))
        for root in roots:
            print(
                json.dumps(
                    {"running": "held_out", "ep": ep, "frame": root["frame"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            all_recs.extend(
                run_root_block(
                    env,
                    root,
                    cfg=cfg,
                    semantics=semantics,
                    repeats=int(cfg["repeats"]),
                    seed=int(cfg["seed"]),
                    out_branch_dir=branch_dir,
                    pool="held_out",
                )
            )

    frozen_an = analyze_records(all_recs, cfg=cfg, pool="frozen")
    held_an = analyze_records(all_recs, cfg=cfg, pool="held_out")
    verdict = judge_s1b(
        actuation_moved=bool(frozen_an["actuation_moved"]),
        existence_on_frozen=bool(frozen_an["existence_any"]),
        existence_on_heldout=(
            bool(held_an["existence_any"]) if held_an["n_roots"] > 0 else None
        ),
        directional=bool(frozen_an["directional_any_metric"]),
    )
    # Refine: user rule — joints moved AND held-out fully no fork => H2 fail
    if frozen_an["actuation_moved"] and held_an["n_roots"] > 0 and not held_an["existence_any"]:
        verdict = {
            "overall_verdict": "h2_failed_no_physical_fork_heldout",
            "research_decision": "stop_h2_controllability_route",
            "enter_stage2": False,
            "summary": "关节动了，held-out roots 仍完全无物理分叉：当前 H2 正式失败。",
        }
    elif (
        frozen_an["actuation_moved"]
        and frozen_an["existence_any"]
        and held_an["n_roots"] > 0
        and held_an["existence_any"]
    ):
        verdict = {
            "overall_verdict": "h2_controllability_exists_heterogeneous",
            "research_decision": "enter_stage2_action_conditioned_eligible",
            "enter_stage2": True,
            "directional_universal_intervention": bool(frozen_an["directional_any_metric"]),
            "summary": (
                "关节动了，并出现可重复物理分叉（含 held-out）："
                "H2 可控性存在；方向同号另计，可进入 action-conditioned Stage-2。"
            ),
        }
    elif not frozen_an["actuation_moved"]:
        verdict = {
            "overall_verdict": "h2_untested_actuation_dead",
            "research_decision": "fix_actuator_control_interface",
            "enter_stage2": False,
            "summary": "实际关节基本没动：H2 尚未被测试，修 actuator/control interface。",
        }

    payload = {
        "protocol": PROTOCOL,
        "created_at": _utc(),
        "stage1_verdict_withdrawn": "h2_failed_no_finger_causal_effect",
        "prior_status": "c2_inconclusive_heterogeneous_forks_actuation_unverified",
        "n_records": len(all_recs),
        "fairness_pass_rate": float(
            np.mean([1.0 if r["fairness_passed"] else 0.0 for r in all_recs])
        )
        if all_recs
        else None,
        "frozen_analysis": frozen_an,
        "held_out_analysis": held_an,
        "held_out_roots": [
            {"episode_index": r["episode_index"], "frame": r["frame"], "phase": r["phase"]}
            for r in held
        ],
        "verdict": verdict,
        "guards": {
            "WRITE_IMPLEMENTATION_ENABLED": WRITE_IMPLEMENTATION_ENABLED,
            "allow_policy_training": False,
            "enter_stage2_auto": False,
            "independent_dose_projection": True,
            "stage1_outputs_retained": True,
        },
    }
    man = PROJECT_ROOT / cfg["manifest_path"]
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    lines = [
        f"# P0-C2-S1b Actuation/Fork Audit ({PROTOCOL})",
        "",
        f"- 日期：{_utc()}",
        f"- overall_verdict：**{verdict['overall_verdict']}**",
        f"- research_decision：**{verdict['research_decision']}**",
        f"- enter_stage2（仅资格，不自动开跑）：{verdict.get('enter_stage2')}",
        f"- Stage-1 A 判定：**已撤回**；本轮验证执行与存在性分叉",
        f"- independent dose projection：是（calibrated/random 不再 common-scale 互拖）",
        f"- frozen actuation_moved={frozen_an['actuation_moved']} "
        f"(frac={frozen_an['actuation_moved_fraction']:.2f})",
        f"- frozen existence_any={frozen_an['existence_any']} "
        f"roots={frozen_an['roots_with_existence']}",
        f"- held-out existence_any={held_an['existence_any']} "
        f"roots={held_an['roots_with_existence']}",
        f"- directional_any_metric(frozen)={frozen_an['directional_any_metric']} "
        f"（方向性≠存在性）",
        "",
        "## Summary",
        "",
        verdict.get("summary", ""),
        "",
        "## Notes",
        "",
        "- 存在性：同 root matched branches 超过重放容差的物理差异。",
        "- 方向性：跨 root 同号；只说明是否已有通用干预。",
        "- `outputs/p0_c2_stage1_v1/` 仍保留，勿删。",
        "",
    ]
    (PROJECT_ROOT / cfg["report_path"]).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(verdict, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
