"""P0-C2 Stage-2R: privilege-complete action-conditioned Ridge (one-shot).

Reuses S1b roots/outcomes; re-exports root MjData features via demo replay.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from embodied_grasp_insertion.labels.privileged_schema import (
    null_velocity,
    o2h_velocity_from_poses,
)
from embodied_grasp_insertion.observability.c2_stage2_action_conditioned import (
    ALPHA_GRID,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    PRIMARY_Y,
    beat,
    episode_equal_mae,
    load_s1b_samples,
    make_mismatch_map,
    paired_bootstrap_mae_diff,
    strip_internal,
)
from embodied_grasp_insertion.physics.grasp_metrics import (
    control_dt_seconds,
    object_in_hand_pose,
    peg_hand_contact_counts,
)
from embodied_grasp_insertion.physics.grasp_metrics import ObjectInHandPose
from embodied_grasp_insertion.simulation.calibrated_interventions import (
    RIGHT_FINGER_ACTUATOR_NAMES,
    RIGHT_FINGER_JOINT_NAMES,
)
from embodied_grasp_insertion.simulation.full_episode_utils import (
    make_full_env,
    replay_demo_one_step,
    replay_demo_to_frame,
)

PROTOCOL = "P0-C2-S2R"


def _finger_qpos_qvel(env) -> tuple[np.ndarray, np.ndarray]:
    model, data = env._raw._model, env._raw._data
    qpos = np.zeros(16, dtype=np.float64)
    qvel = np.zeros(16, dtype=np.float64)
    for i, jn in enumerate(RIGHT_FINGER_JOINT_NAMES):
        jid = int(model.joint(jn).id)
        qpos[i] = float(data.qpos[int(model.jnt_qposadr[jid])])
        qvel[i] = float(data.qvel[int(model.jnt_dofadr[jid])])
    return qpos, qvel


def _wrist_state44(env) -> np.ndarray:
    """Commanded/held wrist pose slice from hold44 (12D: right+left wrist)."""
    hold = np.asarray(env._hold44, dtype=np.float64)
    # indices 0:6 and 22:28
    return np.concatenate([hold[0:6], hold[22:28]])


def _wrist_ft(env) -> np.ndarray:
    if env._force_labeler is None:
        return np.zeros(6, dtype=np.float64)
    fr = env._force_labeler.compute(env._raw)
    return np.asarray(fr.wrist_ft_right, dtype=np.float64).reshape(6)


def export_root_features(
    *,
    episode_index: int,
    frame: int,
    sidecar: Path,
    seed: int = 0,
) -> dict[str, Any]:
    env = make_full_env([episode_index], sidecar_dir=sidecar, seed=seed)
    env.reset(episode_index=episode_index)
    dt = float(control_dt_seconds(env))
    prev_o2h = None
    if int(frame) > 0:
        replay_demo_to_frame(env, int(frame) - 1)
        prev_o2h = object_in_hand_pose(env._raw)
        replay_demo_one_step(env)
        # ensure at frame
        if int(env._t) != int(frame):
            # fallback exact
            env.reset(episode_index=episode_index)
            replay_demo_to_frame(env, int(frame))
            # velocity unavailable in fallback
            prev_o2h = None
    else:
        replay_demo_to_frame(env, 0)

    o2h = object_in_hand_pose(env._raw)
    contact = peg_hand_contact_counts(env._raw)
    if prev_o2h is not None and dt > 0:
        vel = o2h_velocity_from_poses(prev_o2h, o2h, dt)
    else:
        vel = null_velocity()
    qpos, qvel = _finger_qpos_qvel(env)
    wrist = _wrist_state44(env)
    ft = _wrist_ft(env)
    return {
        "episode_index": int(episode_index),
        "frame": int(frame),
        "control_dt_s": dt,
        "o2h_translation_m": o2h.translation.astype(np.float64).tolist(),
        "o2h_rotvec_rad": o2h.rotvec.astype(np.float64).tolist(),
        "o2h_vel_available": bool(vel.available),
        "o2h_vel_linear_mps": list(vel.linear_mps) if vel.available else [0.0, 0.0, 0.0],
        "o2h_vel_angular_radps": list(vel.angular_radps) if vel.available else [0.0, 0.0, 0.0],
        "root_contact_total": int(contact.total),
        "root_contact_by_class": {k: int(v) for k, v in contact.by_class.items()},
        "finger_qpos": qpos.tolist(),
        "finger_qvel": qvel.tolist(),
        "wrist_state12": wrist.tolist(),
        "wrist_ft6": ft.tolist(),
    }


def compact_action_features(actions44: np.ndarray, realized_l2: float) -> np.ndarray:
    """Low-dim future-action summary (no intervention one-hot)."""
    finger = np.asarray(actions44, dtype=np.float64)[:, 6:22]
    wrist = np.asarray(actions44, dtype=np.float64)[:, [0, 1, 2, 3, 4, 5, 22, 23, 24, 25, 26, 27]]
    first = finger[:2].reshape(-1) if finger.shape[0] >= 2 else np.zeros(32)
    # keep first-2 pulse (32) is a bit high; compress to norms per step + global stats
    step_l2 = np.linalg.norm(finger, axis=1)
    return np.concatenate(
        [
            [float(realized_l2), float(np.mean(np.abs(finger))), float(np.max(np.abs(finger))), float(np.std(finger))],
            [float(np.mean(step_l2)), float(np.max(step_l2)), float(np.linalg.norm(wrist))],
            np.linalg.norm(finger[:2], axis=1) if finger.shape[0] >= 2 else np.zeros(2),
            finger[0] if finger.shape[0] >= 1 else np.zeros(16),  # first-step signed direction (16)
        ]
    ).astype(np.float64)


def privilege_vec(feat: dict[str, Any], phase: str) -> np.ndarray:
    by = feat["root_contact_by_class"]
    keys = sorted(by.keys())
    phase_oh = np.asarray(
        [1.0 if phase == p else 0.0 for p in ("early_grasp", "transport", "pre_insert")],
        dtype=np.float64,
    )
    vel_flag = np.asarray([1.0 if feat["o2h_vel_available"] else 0.0], dtype=np.float64)
    return np.concatenate(
        [
            np.asarray(feat["o2h_translation_m"], dtype=np.float64),
            np.asarray(feat["o2h_rotvec_rad"], dtype=np.float64),
            np.asarray(feat["o2h_vel_linear_mps"], dtype=np.float64),
            np.asarray(feat["o2h_vel_angular_radps"], dtype=np.float64),
            vel_flag,
            [float(feat["root_contact_total"])],
            np.asarray([float(by[k]) for k in keys], dtype=np.float64),
            phase_oh,
        ]
    )


def proprio_vec(feat: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(feat["finger_qpos"], dtype=np.float64),
            np.asarray(feat["finger_qvel"], dtype=np.float64),
            np.asarray(feat["wrist_state12"], dtype=np.float64),
        ]
    )


def ft_vec(feat: dict[str, Any]) -> np.ndarray:
    return np.asarray(feat["wrist_ft6"], dtype=np.float64)


def build_x(
    sample: dict[str, Any],
    root_feat: dict[str, Any],
    condition: str,
    *,
    action_override: np.ndarray | None = None,
) -> np.ndarray | None:
    if condition in ("train_mean", "phase_mean"):
        return None
    act = action_override if action_override is not None else sample["interv"]["future_actions44"]
    a = compact_action_features(act, float(sample["interv"].get("realized_l2", 0.0)))
    priv = privilege_vec(root_feat, sample["phase"])
    prop = proprio_vec(root_feat)
    ft = ft_vec(root_feat)
    if condition == "action_only":
        return a
    if condition == "privilege_plus_action":
        return np.concatenate([priv, a])
    if condition == "qpos_qdot_plus_action":
        return np.concatenate([prop, a])
    if condition == "qpos_qdot_ft_plus_action":
        return np.concatenate([prop, ft, a])
    if condition == "mismatched_action":
        return np.concatenate([priv, a])
    raise ValueError(condition)


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(0)
    std = X.std(0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std):
    return (X - mean) / std


def _predict_ridge(Xtr, ytr, Xev, alpha: float) -> np.ndarray:
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(Xtr, ytr)
    return m.predict(Xev)


def run_condition(
    samples: list[dict[str, Any]],
    root_feats: dict[str, dict[str, Any]],
    condition: str,
    *,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
    y_key: str,
    mismatch_map: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    def subset(ids: set[str]) -> list[dict[str, Any]]:
        return [s for s in samples if s["root_id"] in ids]

    tr, va, te = subset(train_ids), subset(val_ids), subset(test_ids)
    y = {
        n: np.asarray([s["y"][y_key] for s in lst], dtype=np.float64)
        for n, lst in (("train", tr), ("val", va), ("test", te))
    }
    eps = {
        n: np.asarray([s["episode_index"] for s in lst], dtype=np.int64)
        for n, lst in (("train", tr), ("val", va), ("test", te))
    }
    out: dict[str, Any] = {"condition": condition, "y_key": y_key}

    if condition == "train_mean":
        mu = float(y["train"].mean()) if len(y["train"]) else 0.0
        preds = {sp: np.full_like(y[sp], mu) for sp in y}
        out["alpha"] = None
    elif condition == "phase_mean":
        buckets: dict[str, list[float]] = {}
        for s, yy in zip(tr, y["train"]):
            buckets.setdefault(s["phase"], []).append(float(yy))
        phase_mu = {k: float(np.mean(v)) for k, v in buckets.items()}
        g = float(y["train"].mean()) if len(y["train"]) else 0.0

        def pred_list(lst):
            return np.asarray([phase_mu.get(s["phase"], g) for s in lst], dtype=np.float64)

        preds = {"train": pred_list(tr), "val": pred_list(va), "test": pred_list(te)}
        out["alpha"] = None
    else:

        def feat(s):
            ov = None
            if condition == "mismatched_action":
                assert mismatch_map is not None
                ov = mismatch_map[s["root_id"] + "|" + s["intervention"]]
            return build_x(s, root_feats[s["root_id"]], condition, action_override=ov)

        Xtr = np.stack([feat(s) for s in tr])
        Xva = np.stack([feat(s) for s in va])
        Xte = np.stack([feat(s) for s in te])
        mean, std = fit_standardizer(Xtr)
        Xtrn, Xvan, Xten = (
            apply_standardizer(Xtr, mean, std),
            apply_standardizer(Xva, mean, std),
            apply_standardizer(Xte, mean, std),
        )
        best_a, best_sc = ALPHA_GRID[0], float("inf")
        for a in ALPHA_GRID:
            pv = _predict_ridge(Xtrn, y["train"], Xvan, a)
            sc = episode_equal_mae(eps["val"], pv, y["val"])
            if sc < best_sc:
                best_sc, best_a = sc, a
        out["alpha"] = float(best_a)
        out["feature_dim"] = int(Xtr.shape[1])
        preds = {
            "train": _predict_ridge(Xtrn, y["train"], Xtrn, best_a),
            "val": _predict_ridge(Xtrn, y["train"], Xvan, best_a),
            "test": _predict_ridge(Xtrn, y["train"], Xten, best_a),
        }

    splits = {}
    for sp in ("train", "val", "test"):
        err = np.abs(preds[sp] - y[sp])
        splits[sp] = {
            "mae": episode_equal_mae(eps[sp], preds[sp], y[sp]),
            "n_samples": int(len(y[sp])),
            "n_episodes": int(len(np.unique(eps[sp]))) if len(eps[sp]) else 0,
            "_pred": preds[sp],
            "_y": y[sp],
            "_eps": eps[sp],
            "_err": err,
        }
    out["splits"] = splits
    return out


def _sig_better(a, b, split, salt: int) -> bool:
    d = paired_bootstrap_mae_diff(
        a["splits"][split]["_eps"],
        a["splits"][split]["_err"],
        b["splits"][split]["_err"],
        seed=(BOOTSTRAP_SEED + salt) % (2**31 - 1),
    )
    return bool(d["a_significantly_better"])


def judge_target(results: dict[str, dict[str, Any]], y_key: str) -> dict[str, Any]:
    oracle = results["privilege_plus_action"]
    action = results["action_only"]
    mean = results["train_mean"]
    mism = results["mismatched_action"]
    q = results["qpos_qdot_plus_action"]
    qft = results["qpos_qdot_ft_plus_action"]

    oracle_ok = all(
        [
            _sig_better(oracle, mean, "val", 1),
            _sig_better(oracle, mean, "test", 2),
            _sig_better(oracle, action, "val", 3),
            _sig_better(oracle, action, "test", 4),
        ]
    )
    # also prefer beating mismatch on test (action relevance)
    oracle_vs_mism = _sig_better(oracle, mism, "test", 5)

    def deploy_ok(model):
        return all(
            [
                _sig_better(model, mean, "val", 10),
                _sig_better(model, mean, "test", 11),
                _sig_better(model, action, "val", 12),
                _sig_better(model, action, "test", 13),
            ]
        )

    q_ok = deploy_ok(q)
    qft_ok = deploy_ok(qft)

    if not oracle_ok:
        branch, decision = "B", "stop_project_privilege_cannot_predict"
    elif oracle_ok and not (q_ok or qft_ok):
        branch, decision = "C", "sensing_gap_consider_tactile"
    else:
        branch, decision = "D", "h4_deploy_signal_no_policy"

    return {
        "y_key": y_key,
        "decision_tree_branch": branch,
        "research_decision": decision,
        "oracle_ok": oracle_ok,
        "oracle_beats_mismatch_test": oracle_vs_mism,
        "deploy_q_ok": q_ok,
        "deploy_qft_ok": qft_ok,
        "paired_test": {
            "oracle_vs_mean": beat(oracle, mean, "test", seed=BOOTSTRAP_SEED + 50),
            "oracle_vs_action": beat(oracle, action, "test", seed=BOOTSTRAP_SEED + 51),
            "q_vs_mean": beat(q, mean, "test", seed=BOOTSTRAP_SEED + 52),
            "qft_vs_mean": beat(qft, mean, "test", seed=BOOTSTRAP_SEED + 53),
        },
    }


def run_stage2r(
    *,
    branch_dir: Path,
    sidecar: Path,
    export_cache: Path,
) -> dict[str, Any]:
    samples = load_s1b_samples(branch_dir)
    # unique roots
    roots = {}
    for s in samples:
        roots[s["root_id"]] = {
            "episode_index": s["episode_index"],
            "frame": s["frame"],
            "phase": s["phase"],
            "pool": s["pool"],
            "root_id": s["root_id"],
        }

    export_cache.parent.mkdir(parents=True, exist_ok=True)
    root_feats: dict[str, dict[str, Any]] = {}
    if export_cache.is_file():
        cached = json.loads(export_cache.read_text())
        root_feats = cached.get("roots", {})
    missing = [rid for rid in roots if rid not in root_feats]
    for rid in missing:
        meta = roots[rid]
        print(json.dumps({"export_root": rid, "ep": meta["episode_index"], "frame": meta["frame"]}), flush=True)
        root_feats[rid] = export_root_features(
            episode_index=int(meta["episode_index"]),
            frame=int(meta["frame"]),
            sidecar=sidecar,
        )
        root_feats[rid]["root_id"] = rid
        root_feats[rid]["phase"] = meta["phase"]
        root_feats[rid]["pool"] = meta["pool"]
    export_cache.write_text(
        json.dumps({"protocol": PROTOCOL, "roots": root_feats}, indent=2, ensure_ascii=False) + "\n"
    )

    frozen = [s for s in samples if s["pool"] == "frozen"]
    held = [s for s in samples if s["pool"] == "held_out"]
    ep_counts = defaultdict(int)
    for s in frozen:
        ep_counts[s["episode_index"]] += 1
    frozen_eps = sorted(ep_counts.keys())
    val_eps = set(frozen_eps[-2:]) if len(frozen_eps) >= 2 else set(frozen_eps)
    train_eps = set(frozen_eps) - val_eps
    train_ids = {s["root_id"] for s in frozen if s["episode_index"] in train_eps}
    val_ids = {s["root_id"] for s in frozen if s["episode_index"] in val_eps}
    test_ids = {s["root_id"] for s in held}

    conditions = [
        "train_mean",
        "phase_mean",
        "action_only",
        "privilege_plus_action",
        "qpos_qdot_plus_action",
        "qpos_qdot_ft_plus_action",
        "mismatched_action",
    ]
    mism = make_mismatch_map(samples, train_ids | val_ids | test_ids, seed=BOOTSTRAP_SEED)

    by_y = {}
    for y_key in PRIMARY_Y:
        results = {}
        for cond in conditions:
            results[cond] = run_condition(
                samples,
                root_feats,
                cond,
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=test_ids,
                y_key=y_key,
                mismatch_map=mism if cond == "mismatched_action" else None,
            )
        by_y[y_key] = {
            "verdict": judge_target(results, y_key),
            "conditions": {c: strip_internal(results[c]) for c in conditions},
        }

    # Aggregate: privileged must work on at least primary physical deltas (trans or rot)
    phys = ["d_trans_drift_max_m", "d_rot_drift_max_rad", "d_contact_retention"]
    oracle_any = any(by_y[k]["verdict"]["oracle_ok"] for k in phys)
    deploy_any = any(
        by_y[k]["verdict"]["deploy_q_ok"] or by_y[k]["verdict"]["deploy_qft_ok"] for k in phys
    )

    if not oracle_any:
        overall = {
            "overall_verdict": "stage2r_privilege_cannot_predict",
            "decision_tree": "B",
            "research_decision": "stop_project_formal",
            "allow_policy_training": False,
            "enter_stage3": False,
            "summary": "完整 privileged+action 仍不能预测物理分叉 → 正式停止当前项目。",
        }
    elif oracle_any and not deploy_any:
        overall = {
            "overall_verdict": "stage2r_sensing_gap",
            "decision_tree": "C",
            "research_decision": "sensing_gap_await_human_for_tactile",
            "allow_policy_training": False,
            "enter_stage3": False,
            "summary": "Oracle 可预测，部署 q/qdot(+FT) 不能 → sensing gap；停等人工是否开触觉。",
        }
    else:
        overall = {
            "overall_verdict": "stage2r_deploy_can_predict",
            "decision_tree": "D",
            "research_decision": "h4_preliminary_stop_no_policy",
            "allow_policy_training": False,
            "enter_stage3": False,
            "summary": "部署输入可区分动作后果 → H4 初步证据；仍不训策略，停等人工。",
        }

    return {
        "protocol": PROTOCOL,
        "n_samples": len(samples),
        "n_roots_exported": len(root_feats),
        "split": {
            "train_episodes": sorted(train_eps),
            "val_episodes": sorted(val_eps),
            "train_root_ids": sorted(train_ids),
            "val_root_ids": sorted(val_ids),
            "test_root_ids": sorted(test_ids),
        },
        "export_cache": str(export_cache),
        "by_target": by_y,
        "verdict": overall,
        "guards": {
            "no_new_collection": True,
            "ridge_only": True,
            "reexport_mjdata_only": True,
            "allow_policy_training": False,
            "no_tactile_vision_pilot": True,
            "one_shot_stage2r": True,
        },
    }
