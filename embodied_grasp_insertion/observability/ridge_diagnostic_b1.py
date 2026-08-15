"""P0-Obs-B1: future o2h-drift falsification (Ridge only; not policy training).

Predicts o2h drift from observation time t to t+Δ (Δ∈{1,8}), using history that
ends at t. Formal tests use paired episode bootstrap CIs, not point estimates alone.
Oracle ceiling may use privileged history at frames < t+Δ only (never the target frame).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import Ridge

from embodied_grasp_insertion.observability.eval_pack import PRIMARY_H, STORE_H
from embodied_grasp_insertion.observability.ridge_diagnostic_b0 import (
    ALPHA_GRID,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    SHUFFLE_FT_SEED,
    apply_standardizer,
    fit_standardizer,
    rotation_geodesic_deg,
)
from embodied_grasp_insertion.pilot import WRITE_IMPLEMENTATION_ENABLED

PROTOCOL = "P0-Obs-B1"
T_OBS = PRIMARY_H - 1  # observation time index inside stored window
DELTA_HORIZONS = (1, 8)
# wrist pose in act44 (identifiability audit): xyz + assumed quat-ish first 7
WRIST_POSE_DIM = 7

DEPLOY_CONDITIONS = (
    "A_H1",
    "A_H8",
    "B_H1",
    "B_H8",
    "B_H8_shuffled_FT",
)
PROXY_CONDITIONS = (
    "train_mean",
    "phase_mean",
    "current_command",
    "time_index",
)
ORACLE_CONDITION = "privileged_o2h_causal_ceiling"
CONDITION_ORDER = PROXY_CONDITIONS + DEPLOY_CONDITIONS + (ORACLE_CONDITION,)


@dataclass(frozen=True)
class SampleRow:
    path: str
    episode_index: int
    root_id: str
    root_phase: str
    root_frame: int
    split: str
    act44: np.ndarray
    ft12: np.ndarray
    o2h_t: np.ndarray
    o2h_r: np.ndarray


def load_pack_samples(pack_root: Path) -> list[SampleRow]:
    rows: list[SampleRow] = []
    for path in sorted((Path(pack_root) / "samples").glob("*.npz")):
        data = np.load(path)
        meta = json.loads(str(data["meta_json"]))
        rows.append(
            SampleRow(
                path=str(path),
                episode_index=int(meta["episode_index"]),
                root_id=str(meta["root_id"]),
                root_phase=str(meta["root_phase"]),
                root_frame=int(meta["root_frame"]),
                split=str(meta["split"]),
                act44=np.asarray(data["act44"], dtype=np.float64),
                ft12=np.asarray(data["ft12"], dtype=np.float64),
                o2h_t=np.asarray(data["o2h_translation_m"], dtype=np.float64),
                o2h_r=np.asarray(data["o2h_rotvec_rad"], dtype=np.float64),
            )
        )
    if not rows:
        raise FileNotFoundError(f"no samples under {pack_root}/samples")
    for r in rows:
        need = T_OBS + max(DELTA_HORIZONS)
        if r.act44.shape[0] < need + 1:
            raise ValueError(f"{r.path} store too short for B1")
    return rows


def drift_targets(row: SampleRow, delta: int) -> tuple[np.ndarray, np.ndarray]:
    """Translation delta (m) and relative rotvec (rad) from t to t+delta."""
    t0 = T_OBS
    t1 = T_OBS + int(delta)
    d_t = row.o2h_t[t1] - row.o2h_t[t0]
    r0 = R.from_rotvec(row.o2h_r[t0])
    r1 = R.from_rotvec(row.o2h_r[t1])
    d_r = (r0.inv() * r1).as_rotvec()
    return d_t.astype(np.float64), np.asarray(d_r, dtype=np.float64)


def _time_shuffle_ft(ft: np.ndarray, root_id: str, h: int) -> np.ndarray:
    window = ft[:h].copy()
    seed = int(hashlib.sha256(f"{SHUFFLE_FT_SEED}:b1:{root_id}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return window[rng.permutation(h)]


def build_features(row: SampleRow, condition: str, delta: int) -> np.ndarray | None:
    """Feature vector for condition; None for mean-table predictors."""
    if condition in ("train_mean", "phase_mean"):
        return None
    if condition == "current_command":
        # Proxy: wrist pose slice only (kinematics-adjacent), not full finger proprio.
        return row.act44[T_OBS, :WRIST_POSE_DIM].copy()
    if condition == "time_index":
        # Proxy: script/time cues only.
        return np.asarray(
            [
                float(row.root_frame),
                float(T_OBS),
                float(delta),
                1.0 if row.root_phase == "early_grasp" else 0.0,
                1.0 if row.root_phase == "transport" else 0.0,
            ],
            dtype=np.float64,
        )
    if condition == "A_H1":
        return row.act44[T_OBS : T_OBS + 1].reshape(-1)
    if condition == "A_H8":
        return row.act44[:PRIMARY_H].reshape(-1)
    if condition == "B_H1":
        return np.concatenate(
            [
                row.act44[T_OBS : T_OBS + 1].reshape(-1),
                row.ft12[T_OBS : T_OBS + 1].reshape(-1),
            ]
        )
    if condition == "B_H8":
        return np.concatenate(
            [row.act44[:PRIMARY_H].reshape(-1), row.ft12[:PRIMARY_H].reshape(-1)]
        )
    if condition == "B_H8_shuffled_FT":
        ft = _time_shuffle_ft(row.ft12, row.root_id, PRIMARY_H)
        return np.concatenate([row.act44[:PRIMARY_H].reshape(-1), ft.reshape(-1)])
    if condition == ORACLE_CONDITION:
        # Causal privileged history: frames strictly before target frame t+Δ.
        # Includes observation-time o2h at t, excludes o2h at t+Δ.
        end = T_OBS + 1  # [0..t] inclusive
        return np.concatenate(
            [row.o2h_t[:end].reshape(-1), row.o2h_r[:end].reshape(-1)]
        )
    raise ValueError(f"unknown condition {condition}")


def episode_equal_metrics(
    episodes: np.ndarray,
    pred_t: np.ndarray,
    gt_t: np.ndarray,
    pred_r: np.ndarray,
    gt_r: np.ndarray,
) -> dict[str, float]:
    ep_ids = np.unique(episodes)
    t_mae_ep, t_rmse_ep, r_mae_ep = [], [], []
    for e in ep_ids:
        m = episodes == e
        err = np.linalg.norm(pred_t[m] - gt_t[m], axis=1)
        t_mae_ep.append(float(err.mean()))
        t_rmse_ep.append(float(np.sqrt((err**2).mean())))
        # Relative rotvec already; geodesic vs predicted relative rotvec
        r_mae_ep.append(float(rotation_geodesic_deg(pred_r[m], gt_r[m]).mean()))
    return {
        "translation_mae_m": float(np.mean(t_mae_ep)),
        "translation_rmse_m": float(np.mean(t_rmse_ep)),
        "rotation_geodesic_mae_deg": float(np.mean(r_mae_ep)),
        "n_episodes": int(len(ep_ids)),
        "n_samples": int(len(episodes)),
    }


def per_episode_metrics(
    episodes: np.ndarray,
    pred_t: np.ndarray,
    gt_t: np.ndarray,
    pred_r: np.ndarray,
    gt_r: np.ndarray,
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for e in np.unique(episodes):
        m = episodes == e
        err = np.linalg.norm(pred_t[m] - gt_t[m], axis=1)
        out[int(e)] = {
            "translation_mae_m": float(err.mean()),
            "translation_rmse_m": float(np.sqrt((err**2).mean())),
            "rotation_geodesic_mae_deg": float(
                rotation_geodesic_deg(pred_r[m], gt_r[m]).mean()
            ),
        }
    return out


def paired_bootstrap_diff(
    ep_a: dict[int, dict[str, float]],
    ep_b: dict[int, dict[str, float]],
    *,
    metric: str,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap CI of mean(ep_a - ep_b). Negative mean ⇒ a better (lower error)."""
    common = sorted(set(ep_a) & set(ep_b))
    if not common:
        raise ValueError("no common episodes for paired bootstrap")
    diffs = np.asarray([ep_a[e][metric] - ep_b[e][metric] for e in common], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(diffs, size=len(diffs), replace=True)
        boots.append(float(draw.mean()))
    arr = np.asarray(boots, dtype=np.float64)
    mean = float(diffs.mean())
    lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return {
        "metric": metric,
        "n_episodes": len(common),
        "mean_diff": mean,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "significantly_better": bool(hi < 0.0),  # a better than b
        "significantly_worse": bool(lo > 0.0),
    }


def _fit_predict_ridge(Xtr, ytr, Xev, alpha: float) -> np.ndarray:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Xtr, ytr)
    return model.predict(Xev)


def run_condition_for_delta(
    rows: list[SampleRow],
    condition: str,
    delta: int,
) -> dict[str, Any]:
    by = {s: [r for r in rows if r.split == s] for s in ("train", "val", "test")}
    y_t = {s: np.stack([drift_targets(r, delta)[0] for r in by[s]]) for s in by}
    y_r = {s: np.stack([drift_targets(r, delta)[1] for r in by[s]]) for s in by}
    eps = {s: np.asarray([r.episode_index for r in by[s]], dtype=np.int64) for s in by}
    phases = {s: [r.root_phase for r in by[s]] for s in by}

    out: dict[str, Any] = {"condition": condition, "delta": int(delta)}

    if condition == "train_mean":
        mean_t, mean_r = y_t["train"].mean(0), y_r["train"].mean(0)
        preds = {
            s: (
                np.broadcast_to(mean_t, y_t[s].shape).copy(),
                np.broadcast_to(mean_r, y_r[s].shape).copy(),
            )
            for s in by
        }
        out["alpha_t"] = out["alpha_r"] = None
    elif condition == "phase_mean":
        phase_t: dict[str, np.ndarray] = {}
        phase_r: dict[str, np.ndarray] = {}
        for ph in sorted(set(phases["train"])):
            m = np.asarray([p == ph for p in phases["train"]])
            phase_t[ph] = y_t["train"][m].mean(0)
            phase_r[ph] = y_r["train"][m].mean(0)
        global_t, global_r = y_t["train"].mean(0), y_r["train"].mean(0)
        preds = {}
        for s in by:
            pt = np.stack([phase_t.get(ph, global_t) for ph in phases[s]])
            pr = np.stack([phase_r.get(ph, global_r) for ph in phases[s]])
            preds[s] = (pt, pr)
        out["alpha_t"] = out["alpha_r"] = None
        out["phases"] = sorted(phase_t)
    else:
        X = {
            s: np.stack([build_features(r, condition, delta) for r in by[s]])
            for s in by
        }
        mean, std = fit_standardizer(X["train"])
        Xn = {s: apply_standardizer(X[s], mean, std) for s in by}
        best_at, best_ar = ALPHA_GRID[0], ALPHA_GRID[0]
        best_st, best_sr = float("inf"), float("inf")
        for alpha in ALPHA_GRID:
            pt = _fit_predict_ridge(Xn["train"], y_t["train"], Xn["val"], alpha)
            pr = _fit_predict_ridge(Xn["train"], y_r["train"], Xn["val"], alpha)
            mt = episode_equal_metrics(eps["val"], pt, y_t["val"], pr, y_r["val"])
            if mt["translation_mae_m"] < best_st:
                best_st, best_at = mt["translation_mae_m"], alpha
            if mt["rotation_geodesic_mae_deg"] < best_sr:
                best_sr, best_ar = mt["rotation_geodesic_mae_deg"], alpha
        out["alpha_t"], out["alpha_r"] = float(best_at), float(best_ar)
        out["feature_dim"] = int(Xn["train"].shape[1])
        preds = {
            s: (
                _fit_predict_ridge(Xn["train"], y_t["train"], Xn[s], best_at),
                _fit_predict_ridge(Xn["train"], y_r["train"], Xn[s], best_ar),
            )
            for s in by
        }

    splits: dict[str, Any] = {}
    ep_metrics: dict[str, dict[int, dict[str, float]]] = {}
    for s in ("train", "val", "test"):
        pt, pr = preds[s]
        metrics = episode_equal_metrics(eps[s], pt, y_t[s], pr, y_r[s])
        epm = per_episode_metrics(eps[s], pt, y_t[s], pr, y_r[s])
        splits[s] = {"metrics": metrics}
        ep_metrics[s] = epm
    out["splits"] = splits
    out["per_episode"] = {s: {str(k): v for k, v in ep_metrics[s].items()} for s in ep_metrics}
    return out


def _sig_better_both_metrics(pair_t: dict, pair_r: dict) -> bool:
    return bool(pair_t["significantly_better"] and pair_r["significantly_better"])


def judge_delta(
    by_cond: dict[str, dict[str, Any]],
    delta: int,
) -> dict[str, Any]:
    """Formal paired-CI judgments for one horizon."""
    comparisons: dict[str, Any] = {}
    deploy_real_signal: list[str] = []

    def pair(a: str, b: str, split: str, metric: str) -> dict[str, Any]:
        salt = int(
            hashlib.sha256(f"{a}|{b}|{split}|{metric}|{delta}".encode()).hexdigest()[:8],
            16,
        )
        return paired_bootstrap_diff(
            {int(k): v for k, v in by_cond[a]["per_episode"][split].items()},
            {int(k): v for k, v in by_cond[b]["per_episode"][split].items()},
            metric=metric,
            seed=(BOOTSTRAP_SEED + delta * 1009 + salt) % (2**31 - 1),
        )

    proxies = list(PROXY_CONDITIONS)
    for name in DEPLOY_CONDITIONS:
        if name == "B_H8_shuffled_FT":
            continue
        ok_splits = {}
        for split in ("val", "test"):
            beats = {}
            for prox in proxies:
                pt = pair(name, prox, split, "translation_mae_m")
                pr = pair(name, prox, split, "rotation_geodesic_mae_deg")
                beats[prox] = {
                    "translation": pt,
                    "rotation": pr,
                    "both_sig_better": _sig_better_both_metrics(pt, pr),
                }
            # Real sensing signal: beat train_mean AND all other proxies on BOTH metrics
            # with paired CI, on this split. Rotation+translation both required.
            ok_splits[split] = all(beats[p]["both_sig_better"] for p in proxies)
            comparisons[f"{name}_vs_proxies_{split}"] = beats
        if ok_splits["val"] and ok_splits["test"]:
            deploy_real_signal.append(name)

    # FT claim: B_H8 vs A_H8 and vs shuffled, paired CI, val+test, both metrics
    ft_pairs = {}
    ft_helps = True
    for split in ("val", "test"):
        for rival in ("A_H8", "B_H8_shuffled_FT"):
            pt = pair("B_H8", rival, split, "translation_mae_m")
            pr = pair("B_H8", rival, split, "rotation_geodesic_mae_deg")
            ft_pairs[f"B_H8_vs_{rival}_{split}"] = {
                "translation": pt,
                "rotation": pr,
                "both_sig_better": _sig_better_both_metrics(pt, pr),
            }
            if not _sig_better_both_metrics(pt, pr):
                ft_helps = False

    # A_H8 vs A_H1 history harm check on test translation/rotation
    hist = {
        "translation": pair("A_H8", "A_H1", "test", "translation_mae_m"),
        "rotation": pair("A_H8", "A_H1", "test", "rotation_geodesic_mae_deg"),
    }

    if deploy_real_signal:
        decision = "continue_candidate_after_falsification"
        overall = "future_drift_signal"
    else:
        decision = "stop_ab_sensing_route"
        overall = "ab_sensing_falsified"

    return {
        "delta": int(delta),
        "overall_verdict": overall,
        "research_decision": decision,
        "deploy_real_signal": deploy_real_signal,
        "ft_helps_claim": bool(ft_helps),
        "history_A_H8_vs_A_H1_test": hist,
        "paired_comparisons": comparisons,
        "ft_paired_comparisons": ft_pairs,
        "claims_observability_p0_pass": False,
        "allow_policy_training": False,
    }


def run_b1_diagnostic(pack_root: Path) -> dict[str, Any]:
    if WRITE_IMPLEMENTATION_ENABLED:
        raise RuntimeError("WRITE_IMPLEMENTATION_ENABLED must stay False for B1")
    rows = load_pack_samples(pack_root)
    man_path = Path(pack_root) / "manifest.json"
    pack_man = json.loads(man_path.read_text()) if man_path.is_file() else {}

    by_delta: dict[str, Any] = {}
    verdicts: dict[str, Any] = {}
    for delta in DELTA_HORIZONS:
        conds = {}
        for name in CONDITION_ORDER:
            conds[name] = run_condition_for_delta(rows, name, delta)
        # strip bulky per-episode from exported condition blobs later
        verdicts[str(delta)] = judge_delta(conds, delta)
        by_delta[str(delta)] = conds

    # Aggregate: stop if ALL horizons falsify deploy signal; FT claim only if any horizon helps
    any_signal = any(v["deploy_real_signal"] for v in verdicts.values())
    any_ft = any(v["ft_helps_claim"] for v in verdicts.values())
    if any_signal:
        research_decision = "continue_candidate_after_falsification"
        overall = "future_drift_signal"
    else:
        research_decision = "stop_ab_sensing_route"
        overall = "ab_sensing_falsified"

    # Compact export: drop per-episode arrays from conditions (keep in verdicts' paired stats)
    compact_deltas = {}
    for d, conds in by_delta.items():
        compact_deltas[d] = {
            name: {
                k: v
                for k, v in cond.items()
                if k != "per_episode"
            }
            for name, cond in conds.items()
        }

    return {
        "protocol": PROTOCOL,
        "pack_root": str(pack_root),
        "pack_split_digest": pack_man.get("split", {}).get("digest"),
        "n_samples": len(rows),
        "n_test_episodes": 15,
        "single_geometry_only": True,
        "target": {
            "type": "future_o2h_drift",
            "t_obs_index": T_OBS,
            "deltas": list(DELTA_HORIZONS),
            "translation": "o2h_t[t+d]-o2h_t[t]",
            "rotation": "rotvec(R_t^{-1} R_{t+d})",
        },
        "formal_test": "paired_episode_bootstrap_CI_both_metrics_val_and_test",
        "oracle_rule": "privileged history frames < t+delta only; includes t, excludes target",
        "alpha_grid": list(ALPHA_GRID),
        "by_delta": compact_deltas,
        "verdicts_by_delta": verdicts,
        # Keep per-episode only inside recompute path — re-attach for judge already done
        "_per_episode_by_delta": {
            d: {name: conds[name]["per_episode"] for name in conds}
            for d, conds in by_delta.items()
        },
        "verdict": {
            "overall_verdict": overall,
            "research_decision": research_decision,
            "any_deploy_real_signal": any_signal,
            "any_ft_helps_claim": any_ft,
            "claims_observability_p0_pass": False,
            "allow_policy_training": False,
            "notes": (
                "Real signal requires beating train_mean/phase_mean/current_command/time_index "
                "on BOTH translation and rotation via paired CI on val AND test. "
                "If falsified: stop A/B sensing route (no more Ridge/NN on act44+FT)."
            ),
        },
        "b0_addendum": {
            "point_estimate_stable_was_too_strong": True,
            "contemporaneous_ceiling_was_vacuous": True,
            "act44_wrist_kinematics_caveat": True,
        },
        "guards": {
            "WRITE_IMPLEMENTATION_ENABLED": WRITE_IMPLEMENTATION_ENABLED,
            "evaluation_only": True,
            "allow_policy_training": False,
            "claims_observability_p0_pass": False,
            "no_policy_training": True,
            "no_new_collection": True,
            "no_pilot_write": True,
            "no_complex_networks": True,
        },
    }
