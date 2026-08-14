"""P0-Obs-B0: minimal Ridge observability diagnostic (not policy training).

Predicts primary-window-end object_in_hand_pose_6d from deploy inputs A/B.
Linear/Ridge only; fixed split; train-only normalization; alpha on val once.
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
from embodied_grasp_insertion.pilot import WRITE_IMPLEMENTATION_ENABLED

PROTOCOL = "P0-Obs-B0"
TARGET_FRAME_IDX = PRIMARY_H - 1  # fixed label = primary H=8 window end
ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 20260815
SHUFFLE_FT_SEED = 20260815
CONDITION_ORDER = (
    "train_mean",
    "A_H1",
    "A_H8",
    "B_H1",
    "B_H8",
    "B_H8_shuffled_FT",
    "privileged_o2h_ceiling",
)


@dataclass(frozen=True)
class SampleRow:
    path: str
    episode_index: int
    root_id: str
    split: str
    act44: np.ndarray  # (STORE_H, 44)
    ft12: np.ndarray  # (STORE_H, 12)
    o2h_t: np.ndarray  # (STORE_H, 3)
    o2h_r: np.ndarray  # (STORE_H, 3)

    @property
    def y_t(self) -> np.ndarray:
        return self.o2h_t[TARGET_FRAME_IDX].astype(np.float64)

    @property
    def y_r(self) -> np.ndarray:
        return self.o2h_r[TARGET_FRAME_IDX].astype(np.float64)


def load_pack_samples(pack_root: Path) -> list[SampleRow]:
    samples_dir = Path(pack_root) / "samples"
    rows: list[SampleRow] = []
    for path in sorted(samples_dir.glob("*.npz")):
        data = np.load(path)
        meta = json.loads(str(data["meta_json"]))
        rows.append(
            SampleRow(
                path=str(path),
                episode_index=int(meta["episode_index"]),
                root_id=str(meta["root_id"]),
                split=str(meta["split"]),
                act44=np.asarray(data["act44"], dtype=np.float64),
                ft12=np.asarray(data["ft12"], dtype=np.float64),
                o2h_t=np.asarray(data["o2h_translation_m"], dtype=np.float64),
                o2h_r=np.asarray(data["o2h_rotvec_rad"], dtype=np.float64),
            )
        )
    if not rows:
        raise FileNotFoundError(f"no samples under {samples_dir}")
    return rows


def _time_shuffle_ft(ft: np.ndarray, root_id: str, h: int) -> np.ndarray:
    """Break FT temporal alignment inside the H window (fixed per root_id)."""
    window = ft[:h].copy()
    seed = int(hashlib.sha256(f"{SHUFFLE_FT_SEED}:{root_id}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    order = rng.permutation(h)
    return window[order]


def build_features(row: SampleRow, condition: str) -> np.ndarray | None:
    """Return feature vector, or None for train_mean (no X)."""
    if condition == "train_mean":
        return None
    if condition == "A_H1":
        return row.act44[TARGET_FRAME_IDX : TARGET_FRAME_IDX + 1].reshape(-1)
    if condition == "A_H8":
        return row.act44[:PRIMARY_H].reshape(-1)
    if condition == "B_H1":
        return np.concatenate(
            [
                row.act44[TARGET_FRAME_IDX : TARGET_FRAME_IDX + 1].reshape(-1),
                row.ft12[TARGET_FRAME_IDX : TARGET_FRAME_IDX + 1].reshape(-1),
            ]
        )
    if condition == "B_H8":
        return np.concatenate(
            [row.act44[:PRIMARY_H].reshape(-1), row.ft12[:PRIMARY_H].reshape(-1)]
        )
    if condition == "B_H8_shuffled_FT":
        ft = _time_shuffle_ft(row.ft12, row.root_id, PRIMARY_H)
        return np.concatenate([row.act44[:PRIMARY_H].reshape(-1), ft.reshape(-1)])
    if condition == "privileged_o2h_ceiling":
        # Cheating ceiling: feed privileged o2h over H8 including target frame.
        return np.concatenate(
            [row.o2h_t[:PRIMARY_H].reshape(-1), row.o2h_r[:PRIMARY_H].reshape(-1)]
        )
    raise ValueError(f"unknown condition {condition}")


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def rotation_geodesic_deg(pred_rotvec: np.ndarray, gt_rotvec: np.ndarray) -> np.ndarray:
    """Geodesic angle (deg) between predicted and gt rotvecs; shape (N,)."""
    rp = R.from_rotvec(np.asarray(pred_rotvec, dtype=np.float64))
    rg = R.from_rotvec(np.asarray(gt_rotvec, dtype=np.float64))
    dR = rg.inv() * rp
    ang = np.asarray(dR.magnitude(), dtype=np.float64)  # radians
    return ang * (180.0 / np.pi)


def episode_equal_metrics(
    episodes: np.ndarray,
    pred_t: np.ndarray,
    gt_t: np.ndarray,
    pred_r: np.ndarray,
    gt_r: np.ndarray,
) -> dict[str, float]:
    """Mean over episodes of per-episode mean errors."""
    ep_ids = np.unique(episodes)
    t_mae_ep: list[float] = []
    t_rmse_ep: list[float] = []
    r_mae_ep: list[float] = []
    for e in ep_ids:
        m = episodes == e
        err = np.linalg.norm(pred_t[m] - gt_t[m], axis=1)
        t_mae_ep.append(float(err.mean()))
        t_rmse_ep.append(float(np.sqrt((err**2).mean())))
        r_mae_ep.append(float(rotation_geodesic_deg(pred_r[m], gt_r[m]).mean()))
    return {
        "translation_mae_m": float(np.mean(t_mae_ep)),
        "translation_rmse_m": float(np.mean(t_rmse_ep)),
        "rotation_geodesic_mae_deg": float(np.mean(r_mae_ep)),
        "n_episodes": int(len(ep_ids)),
        "n_samples": int(len(episodes)),
    }


def bootstrap_ci(
    episodes: np.ndarray,
    pred_t: np.ndarray,
    gt_t: np.ndarray,
    pred_r: np.ndarray,
    gt_r: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    """Episode-level bootstrap 95% CI for episode-equal means."""
    ep_ids = np.unique(episodes)
    rng = np.random.default_rng(seed)
    keys = ("translation_mae_m", "translation_rmse_m", "rotation_geodesic_mae_deg")
    bags = {k: [] for k in keys}
    for _ in range(n_boot):
        draw = rng.choice(ep_ids, size=len(ep_ids), replace=True)
        # gather samples belonging to drawn episodes (with replacement of eps)
        pred_t_b, gt_t_b, pred_r_b, gt_r_b, ep_b = [], [], [], [], []
        for e in draw:
            m = episodes == e
            pred_t_b.append(pred_t[m])
            gt_t_b.append(gt_t[m])
            pred_r_b.append(pred_r[m])
            gt_r_b.append(gt_r[m])
            ep_b.append(np.full(int(m.sum()), e, dtype=np.int64))
        metrics = episode_equal_metrics(
            np.concatenate(ep_b),
            np.concatenate(pred_t_b),
            np.concatenate(gt_t_b),
            np.concatenate(pred_r_b),
            np.concatenate(gt_r_b),
        )
        for k in keys:
            bags[k].append(metrics[k])
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        arr = np.asarray(bags[k], dtype=np.float64)
        out[k] = {
            "mean": float(arr.mean()),
            "ci95_lo": float(np.percentile(arr, 2.5)),
            "ci95_hi": float(np.percentile(arr, 97.5)),
        }
    return out


def _val_select_score(metrics: dict[str, float]) -> float:
    """Pre-registered scalar for alpha selection (lower better)."""
    return float(metrics["translation_mae_m"] + metrics["rotation_geodesic_mae_deg"] / 100.0)


def _fit_predict_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def run_condition(
    rows: list[SampleRow],
    condition: str,
) -> dict[str, Any]:
    by_split = {s: [r for r in rows if r.split == s] for s in ("train", "val", "test")}
    y_t = {s: np.stack([r.y_t for r in by_split[s]]) for s in by_split}
    y_r = {s: np.stack([r.y_r for r in by_split[s]]) for s in by_split}
    eps = {s: np.asarray([r.episode_index for r in by_split[s]], dtype=np.int64) for s in by_split}

    result: dict[str, Any] = {
        "condition": condition,
        "model": "train_mean" if condition == "train_mean" else "ridge",
    }

    if condition == "train_mean":
        mean_t = y_t["train"].mean(axis=0)
        mean_r = y_r["train"].mean(axis=0)
        result["alpha_t"] = None
        result["alpha_r"] = None
        preds = {
            s: (
                np.broadcast_to(mean_t, y_t[s].shape).copy(),
                np.broadcast_to(mean_r, y_r[s].shape).copy(),
            )
            for s in by_split
        }
    else:
        X = {s: np.stack([build_features(r, condition) for r in by_split[s]]) for s in by_split}
        mean, std = fit_standardizer(X["train"])
        Xn = {s: apply_standardizer(X[s], mean, std) for s in by_split}

        # Separate alpha for translation / rotation; each chosen on val only.
        best_alpha_t, best_alpha_r = ALPHA_GRID[0], ALPHA_GRID[0]
        best_score_t, best_score_r = float("inf"), float("inf")
        for alpha in ALPHA_GRID:
            pred_t = _fit_predict_ridge(Xn["train"], y_t["train"], Xn["val"], alpha)
            pred_r = _fit_predict_ridge(Xn["train"], y_r["train"], Xn["val"], alpha)
            mt = episode_equal_metrics(eps["val"], pred_t, y_t["val"], pred_r, y_r["val"])
            # Select independently on each head's primary metric.
            if mt["translation_mae_m"] < best_score_t:
                best_score_t = mt["translation_mae_m"]
                best_alpha_t = alpha
            if mt["rotation_geodesic_mae_deg"] < best_score_r:
                best_score_r = mt["rotation_geodesic_mae_deg"]
                best_alpha_r = alpha
        result["alpha_t"] = float(best_alpha_t)
        result["alpha_r"] = float(best_alpha_r)
        result["feature_dim"] = int(Xn["train"].shape[1])

        preds = {}
        for s in by_split:
            preds[s] = (
                _fit_predict_ridge(Xn["train"], y_t["train"], Xn[s], best_alpha_t),
                _fit_predict_ridge(Xn["train"], y_r["train"], Xn[s], best_alpha_r),
            )

    splits_out: dict[str, Any] = {}
    for s in ("train", "val", "test"):
        pt, pr = preds[s]
        metrics = episode_equal_metrics(eps[s], pt, y_t[s], pr, y_r[s])
        ci = bootstrap_ci(eps[s], pt, y_t[s], pr, y_r[s])
        splits_out[s] = {"metrics": metrics, "bootstrap_ci95": ci}
    result["splits"] = splits_out
    result["val_select_score"] = _val_select_score(splits_out["val"]["metrics"])
    return result


def _better(a: dict[str, float], b: dict[str, float]) -> bool:
    """a strictly better than b on both translation MAE and rotation geodesic MAE."""
    return (
        a["translation_mae_m"] < b["translation_mae_m"]
        and a["rotation_geodesic_mae_deg"] < b["rotation_geodesic_mae_deg"]
    )


def judge_verdict(by_cond: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mean = by_cond["train_mean"]
    deploy = ("A_H1", "A_H8", "B_H1", "B_H8")
    stable_winners: list[str] = []
    for name in deploy:
        m = by_cond[name]
        if _better(m["splits"]["val"]["metrics"], mean["splits"]["val"]["metrics"]) and _better(
            m["splits"]["test"]["metrics"], mean["splits"]["test"]["metrics"]
        ):
            stable_winners.append(name)

    ft_helps = False
    if (
        "B_H8" in stable_winners
        and _better(
            by_cond["B_H8"]["splits"]["val"]["metrics"],
            by_cond["A_H8"]["splits"]["val"]["metrics"],
        )
        and _better(
            by_cond["B_H8"]["splits"]["test"]["metrics"],
            by_cond["A_H8"]["splits"]["test"]["metrics"],
        )
        and _better(
            by_cond["B_H8"]["splits"]["val"]["metrics"],
            by_cond["B_H8_shuffled_FT"]["splits"]["val"]["metrics"],
        )
        and _better(
            by_cond["B_H8"]["splits"]["test"]["metrics"],
            by_cond["B_H8_shuffled_FT"]["splits"]["test"]["metrics"],
        )
    ):
        ft_helps = True

    if stable_winners:
        research_decision = "continue_candidate_sensing_signal"
        overall = "diagnostic_signal"
    else:
        research_decision = "stop_sensing_insufficient"
        overall = "sensing_insufficient"

    return {
        "overall_verdict": overall,
        "research_decision": research_decision,
        "stable_better_than_train_mean": stable_winners,
        "ft_helps_claim": ft_helps,
        "claims_observability_p0_pass": False,
        "allow_policy_training": False,
        "notes": (
            "Success requires A or B better than train-mean on BOTH val and test "
            "for translation MAE and rotation geodesic MAE. "
            "FT claim requires B_H8 better than A_H8 and B_H8_shuffled_FT on val+test."
        ),
    }


def run_b0_diagnostic(pack_root: Path) -> dict[str, Any]:
    if WRITE_IMPLEMENTATION_ENABLED:
        raise RuntimeError("WRITE_IMPLEMENTATION_ENABLED must stay False for B0")
    rows = load_pack_samples(pack_root)
    man_path = Path(pack_root) / "manifest.json"
    pack_manifest = json.loads(man_path.read_text()) if man_path.is_file() else {}
    conditions = {}
    for name in CONDITION_ORDER:
        conditions[name] = run_condition(rows, name)
    verdict = judge_verdict(conditions)
    return {
        "protocol": PROTOCOL,
        "pack_root": str(pack_root),
        "pack_split_digest": pack_manifest.get("split", {}).get("digest"),
        "n_samples": len(rows),
        "target": {
            "field": "object_in_hand_pose_6d",
            "frame": "primary_window_end",
            "index_in_store": TARGET_FRAME_IDX,
            "primary_h": PRIMARY_H,
            "store_h": STORE_H,
        },
        "alpha_grid": list(ALPHA_GRID),
        "normalization": "standardize_features_train_only",
        "metrics_weighting": "episode_equal",
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED},
        "conditions": conditions,
        "verdict": verdict,
        "guards": {
            "WRITE_IMPLEMENTATION_ENABLED": WRITE_IMPLEMENTATION_ENABLED,
            "evaluation_only": True,
            "allow_policy_training": False,
            "claims_observability_p0_pass": False,
            "no_policy_training": True,
            "no_new_collection": True,
            "no_pilot_write": True,
            "no_reopen_c0_c1": True,
        },
    }
