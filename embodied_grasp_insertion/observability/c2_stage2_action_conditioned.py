"""P0-C2 Stage-2: action-conditioned consequence prediction from S1b JSONs only.

Predicts signed (intervention - hold) physical deltas. Ridge only. No new collection.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

PROTOCOL = "P0-C2-S2"
ALPHA_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 20260815
HORIZON_PAD = 16
FINGER_DIM = 16
PRIMARY_Y = (
    "d_trans_drift_max_m",
    "d_rot_drift_max_rad",
    "d_contact_retention",
    "d_terminal_peg_ok",
)


def _median_metrics(recs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "trans_drift_max_m",
        "rot_drift_max_rad",
        "contact_retention_vs_root_mean",
        "terminal_peg_ok",
        "object_dropped_proxy",
        "root_contact_total",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        vals = []
        for r in recs:
            v = r["metrics"].get(k)
            if isinstance(v, bool):
                vals.append(float(v))
            elif v is None:
                continue
            else:
                vals.append(float(v))
        out[k] = float(np.median(vals)) if vals else 0.0
    # class contact from first
    by = recs[0]["metrics"].get("root_contact_by_class") or {}
    out["root_contact_by_class"] = {str(k): float(by[k]) for k in sorted(by)}
    out["phase"] = recs[0]["phase"]
    out["pool"] = recs[0]["pool"]
    out["episode_index"] = int(recs[0]["episode_index"])
    out["frame"] = int(recs[0]["frame"])
    out["root_id"] = recs[0]["root_id"]
    # root proprio / action from first repeat (deterministic enough)
    out["qpos0"] = np.asarray(recs[0]["actuation"]["qpos0"], dtype=np.float64)
    # pad future actions to HORIZON_PAD x 44
    acts = []
    for r in recs:
        a = np.asarray(r["future_actions44"], dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 44:
            raise ValueError(f"bad actions {a.shape}")
        pad = np.zeros((HORIZON_PAD, 44), dtype=np.float64)
        n = min(HORIZON_PAD, a.shape[0])
        pad[:n] = a[:n]
        acts.append(pad)
    out["future_actions44"] = np.median(np.stack(acts, axis=0), axis=0)
    meta_l2 = [float((r.get("action_meta") or {}).get("realized_l2") or 0.0) for r in recs]
    out["realized_l2"] = float(np.median(meta_l2))
    return out


def load_s1b_samples(branch_dir: Path) -> list[dict[str, Any]]:
    by: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(branch_dir).glob("*.json")):
        d = json.loads(path.read_text())
        if not d.get("fairness_passed", True):
            continue
        by[d["root_id"]][d["intervention"]].append(d)

    samples: list[dict[str, Any]] = []
    for root_id, mp in by.items():
        if "hold_finger" not in mp:
            continue
        hold = _median_metrics(mp["hold_finger"])
        for interv in ("demo_finger_replay", "calibrated_finger_intervention", "random_finger_control"):
            if interv not in mp:
                continue
            x = _median_metrics(mp[interv])
            y = {
                "d_trans_drift_max_m": x["trans_drift_max_m"] - hold["trans_drift_max_m"],
                "d_rot_drift_max_rad": x["rot_drift_max_rad"] - hold["rot_drift_max_rad"],
                "d_contact_retention": x["contact_retention_vs_root_mean"]
                - hold["contact_retention_vs_root_mean"],
                "d_terminal_peg_ok": x["terminal_peg_ok"] - hold["terminal_peg_ok"],
                "d_object_dropped": x["object_dropped_proxy"] - hold["object_dropped_proxy"],
            }
            samples.append(
                {
                    "root_id": root_id,
                    "episode_index": hold["episode_index"],
                    "frame": hold["frame"],
                    "phase": hold["phase"],
                    "pool": hold["pool"],
                    "intervention": interv,
                    "hold": hold,
                    "interv": x,
                    "y": y,
                }
            )
    return samples


def _phase_oh(phase: str) -> np.ndarray:
    names = ("early_grasp", "transport", "pre_insert")
    return np.asarray([1.0 if phase == n else 0.0 for n in names], dtype=np.float64)


def _contact_vec(hold: dict[str, Any]) -> np.ndarray:
    by = hold["root_contact_by_class"]
    keys = sorted(by.keys())
    return np.asarray([hold["root_contact_total"]] + [by[k] for k in keys], dtype=np.float64)


def _action_finger_flat(actions44: np.ndarray) -> np.ndarray:
    # right fingers 6:22 in act44 layout
    finger = actions44[:, 6:22]
    return finger.reshape(-1)


def build_x(sample: dict[str, Any], condition: str, *, action_override: np.ndarray | None = None) -> np.ndarray | None:
    hold = sample["hold"]
    interv = sample["interv"]
    act = action_override if action_override is not None else interv["future_actions44"]
    a_flat = _action_finger_flat(act)
    q0 = np.asarray(hold["qpos0"], dtype=np.float64)
    priv = np.concatenate([_contact_vec(hold), _phase_oh(hold["phase"]), [hold["frame"] / 500.0]])

    if condition == "train_mean":
        return None
    if condition == "phase_mean":
        return None
    if condition == "action_only":
        return a_flat
    if condition == "command_future_action":
        # candidate future command sequence only (S1b has no separate command history)
        return a_flat
    if condition == "qpos_plus_action":
        return np.concatenate([q0, a_flat])
    if condition == "privilege_contact_plus_action":
        return np.concatenate([priv, a_flat])
    if condition == "privilege_contact_only":
        return priv
    if condition == "mismatched_action":
        # placeholder — caller supplies action_override
        return np.concatenate([priv, a_flat]) if action_override is not None else np.concatenate([priv, a_flat])
    if condition == "ft_plus_action":
        raise KeyError("wrist_ft_unavailable_in_s1b_export")
    raise ValueError(condition)


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(0)
    std = X.std(0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std):
    return (X - mean) / std


def episode_equal_mae(episodes: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> float:
    maes = []
    for e in np.unique(episodes):
        m = episodes == e
        maes.append(float(np.mean(np.abs(pred[m] - gt[m]))))
    return float(np.mean(maes))


def paired_bootstrap_mae_diff(
    episodes: np.ndarray,
    err_a: np.ndarray,
    err_b: np.ndarray,
    *,
    seed: int,
    n_boot: int = BOOTSTRAP_N,
) -> dict[str, Any]:
    """err = |pred-gt|; diff = mae_a - mae_b per episode; negative => a better."""
    ep_ids = np.unique(episodes)
    diffs = []
    for e in ep_ids:
        m = episodes == e
        diffs.append(float(np.mean(err_a[m]) - np.mean(err_b[m])))
    diffs = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = [float(rng.choice(diffs, size=len(diffs), replace=True).mean()) for _ in range(n_boot)]
    arr = np.asarray(boots)
    lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return {
        "mean_diff": float(diffs.mean()),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "a_significantly_better": bool(hi < 0.0),
        "n_episodes": int(len(ep_ids)),
    }


def _predict_ridge(Xtr, ytr, Xev, alpha: float) -> np.ndarray:
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(Xtr, ytr)
    return m.predict(Xev)


def run_condition(
    samples: list[dict[str, Any]],
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
    y = {name: np.asarray([s["y"][y_key] for s in lst], dtype=np.float64) for name, lst in (("train", tr), ("val", va), ("test", te))}
    eps = {
        name: np.asarray([s["episode_index"] for s in lst], dtype=np.int64)
        for name, lst in (("train", tr), ("val", va), ("test", te))
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
        def feat_for(s: dict[str, Any]) -> np.ndarray:
            if condition == "mismatched_action":
                assert mismatch_map is not None
                key = s["root_id"] + "|" + s["intervention"]
                return build_x(
                    s,
                    "privilege_contact_plus_action",
                    action_override=mismatch_map[key],
                )
            return build_x(s, condition)

        Xtr = np.stack([feat_for(s) for s in tr])
        Xva = np.stack([feat_for(s) for s in va])
        Xte = np.stack([feat_for(s) for s in te])
        mean, std = fit_standardizer(Xtr)
        Xtrn = apply_standardizer(Xtr, mean, std)
        Xvan = apply_standardizer(Xva, mean, std)
        Xten = apply_standardizer(Xte, mean, std)
        best_a, best_score = ALPHA_GRID[0], float("inf")
        for a in ALPHA_GRID:
            pv = _predict_ridge(Xtrn, y["train"], Xvan, a)
            sc = episode_equal_mae(eps["val"], pv, y["val"])
            if sc < best_score:
                best_score, best_a = sc, a
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
            "rmse": float(
                np.mean(
                    [
                        np.sqrt(np.mean((preds[sp][eps[sp] == e] - y[sp][eps[sp] == e]) ** 2))
                        for e in np.unique(eps[sp])
                    ]
                )
            )
            if len(eps[sp])
            else None,
            "n_samples": int(len(y[sp])),
            "n_episodes": int(len(np.unique(eps[sp]))) if len(eps[sp]) else 0,
            "_pred": preds[sp],
            "_y": y[sp],
            "_eps": eps[sp],
            "_err": err,
        }
    out["splits"] = splits
    return out


def make_mismatch_map(samples: list[dict[str, Any]], ids: set[str], seed: int) -> dict[str, np.ndarray]:
    """Assign each sample another root's future action within the same id pool."""
    pool = [s for s in samples if s["root_id"] in ids]
    rng = np.random.default_rng(seed)
    actions = [s["interv"]["future_actions44"].copy() for s in pool]
    keys = [s["root_id"] + "|" + s["intervention"] for s in pool]
    perm = rng.permutation(len(pool))
    # ensure not identical pairing when possible
    for i in range(len(perm)):
        if perm[i] == i and len(perm) > 1:
            perm[i] = (i + 1) % len(perm)
    return {keys[i]: actions[perm[i]] for i in range(len(keys))}


def beat(a: dict[str, Any], b: dict[str, Any], split: str, *, seed: int) -> dict[str, Any]:
    return paired_bootstrap_mae_diff(
        a["splits"][split]["_eps"],
        a["splits"][split]["_err"],
        b["splits"][split]["_err"],
        seed=seed,
    )


def judge(results: dict[str, dict[str, Any]], y_key: str) -> dict[str, Any]:
    """Stop-tree for one target; aggregate later."""
    oracle = results["privilege_contact_plus_action"]
    action = results["action_only"]
    mean = results["train_mean"]
    mism = results["mismatched_action"]
    qpos = results["qpos_plus_action"]

    def ok_pair(better, worse, split, salt: int):
        return beat(better, worse, split, seed=BOOTSTRAP_SEED + salt).get("a_significantly_better", False)

    oracle_beats_mean_val = ok_pair(oracle, mean, "val", 11)
    oracle_beats_mean_test = ok_pair(oracle, mean, "test", 12)
    oracle_beats_action_val = ok_pair(oracle, action, "val", 13)
    oracle_beats_action_test = ok_pair(oracle, action, "test", 14)
    oracle_beats_mism_val = ok_pair(oracle, mism, "val", 15)
    oracle_beats_mism_test = ok_pair(oracle, mism, "test", 16)

    oracle_ok = all(
        [
            oracle_beats_mean_val,
            oracle_beats_mean_test,
            oracle_beats_action_val,
            oracle_beats_action_test,
        ]
    )

    deploy_ok = all(
        [
            ok_pair(qpos, mean, "val", 21),
            ok_pair(qpos, mean, "test", 22),
            ok_pair(qpos, action, "val", 23),
            ok_pair(qpos, action, "test", 24),
            ok_pair(qpos, mism, "val", 25),
            ok_pair(qpos, mism, "test", 26),
        ]
    )

    if not oracle_ok:
        branch = "B_oracle_cannot_predict"
        decision = "stop_task_or_label_invalid"
    elif oracle_ok and not deploy_ok:
        branch = "C_sensing_gap"
        decision = "sensing_gap_then_consider_tactile"
    else:
        branch = "D_deploy_can_distinguish"
        decision = "h4_preliminary_evidence_no_policy"

    return {
        "y_key": y_key,
        "decision_tree_branch": branch,
        "research_decision": decision,
        "oracle_ok": oracle_ok,
        "deploy_qpos_ok": deploy_ok,
        "checks": {
            "oracle_beats_mean_val": oracle_beats_mean_val,
            "oracle_beats_mean_test": oracle_beats_mean_test,
            "oracle_beats_action_val": oracle_beats_action_val,
            "oracle_beats_action_test": oracle_beats_action_test,
            "oracle_beats_mismatch_val": oracle_beats_mism_val,
            "oracle_beats_mismatch_test": oracle_beats_mism_test,
            "qpos_beats_mean_test": ok_pair(qpos, mean, "test", 31),
            "qpos_beats_action_test": ok_pair(qpos, action, "test", 32),
            "qpos_beats_mismatch_test": ok_pair(qpos, mism, "test", 33),
        },
        "paired": {
            "oracle_vs_mean_test": beat(oracle, mean, "test", seed=BOOTSTRAP_SEED + 40),
            "oracle_vs_action_test": beat(oracle, action, "test", seed=BOOTSTRAP_SEED + 41),
            "qpos_vs_mean_test": beat(qpos, mean, "test", seed=BOOTSTRAP_SEED + 42),
            "qpos_vs_mismatch_test": beat(qpos, mism, "test", seed=BOOTSTRAP_SEED + 43),
        },
    }


def strip_internal(res: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in res.items() if k != "splits"}
    out["splits"] = {
        sp: {k: v for k, v in blob.items() if not k.startswith("_")}
        for sp, blob in res["splits"].items()
    }
    return out


def run_stage2(branch_dir: Path) -> dict[str, Any]:
    samples = load_s1b_samples(branch_dir)
    frozen = [s for s in samples if s["pool"] == "frozen"]
    held = [s for s in samples if s["pool"] == "held_out"]
    # val: two frozen episodes with most samples; train: other frozen; test: held_out
    ep_counts = defaultdict(int)
    for s in frozen:
        ep_counts[s["episode_index"]] += 1
    # deterministic pick: highest episode ids for val among frozen
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
        "command_future_action",
        "qpos_plus_action",
        "privilege_contact_only",
        "privilege_contact_plus_action",
        "mismatched_action",
    ]

    # mismatch maps per split pool of ids
    all_ids = train_ids | val_ids | test_ids
    mism = make_mismatch_map(samples, all_ids, seed=BOOTSTRAP_SEED)

    by_y: dict[str, Any] = {}
    for y_key in PRIMARY_Y:
        results = {}
        for cond in conditions:
            results[cond] = run_condition(
                samples,
                cond,
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=test_ids,
                y_key=y_key,
                mismatch_map=mism if cond == "mismatched_action" else None,
            )
        verdict = judge(results, y_key)
        by_y[y_key] = {
            "verdict": verdict,
            "conditions": {c: strip_internal(results[c]) for c in conditions},
        }

    # Aggregate: primary focus on signed translation drift consequence
    primary = by_y["d_trans_drift_max_m"]["verdict"]
    # Also require oracle gate on at least one physical metric to avoid single-target fluke
    oracle_any = any(by_y[k]["verdict"]["oracle_ok"] for k in PRIMARY_Y)
    deploy_any = any(by_y[k]["verdict"]["deploy_qpos_ok"] for k in PRIMARY_Y)
    if not oracle_any:
        overall = {
            "overall_verdict": "stage2_oracle_cannot_predict",
            "decision_tree": "B",
            "research_decision": "stop_task_or_label_invalid",
            "enter_stage3": False,
            "allow_policy_training": False,
        }
    elif oracle_any and not deploy_any:
        overall = {
            "overall_verdict": "stage2_sensing_gap",
            "decision_tree": "C",
            "research_decision": "sensing_gap_consider_tactile_separately",
            "enter_stage3": False,
            "allow_policy_training": False,
        }
    else:
        overall = {
            "overall_verdict": "stage2_deploy_can_distinguish_actions",
            "decision_tree": "D",
            "research_decision": "h4_preliminary_evidence_stop_no_policy",
            "enter_stage3": False,
            "allow_policy_training": False,
        }

    return {
        "protocol": PROTOCOL,
        "n_samples": len(samples),
        "split": {
            "train_root_ids": sorted(train_ids),
            "val_root_ids": sorted(val_ids),
            "test_root_ids": sorted(test_ids),
            "train_episodes": sorted(train_eps),
            "val_episodes": sorted(val_eps),
            "method": "frozen_episode_holdout_val__s1b_heldout_as_test",
        },
        "data_limits": {
            "wrist_ft": "unavailable_in_s1b_export",
            "command_history_pre_root": "unavailable_in_s1b_export",
            "qdot_at_root": "unavailable_in_s1b_export",
            "privilege": "root_contact_total_by_class_plus_phase_only__no_o2h_in_export",
            "future_action": "finger_slice_of_future_actions44_padded",
            "target": "signed_intervention_minus_hold_metrics",
        },
        "by_target": by_y,
        "primary_target": "d_trans_drift_max_m",
        "primary_verdict": primary,
        "verdict": overall,
        "guards": {
            "no_new_collection": True,
            "ridge_only": True,
            "allow_policy_training": False,
            "enter_stage3": False,
            "s1b_outputs_retained": True,
        },
    }
