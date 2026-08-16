#!/usr/bin/env python3
"""Matched active-contact probe information audit."""

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
EMBODIED_ROOT = DEXJOCO_ROOT / "embodied_grasp_insertion"
for search_path in (
    str(PROJECT_ROOT),
    str(DEXJOCO_ROOT),
    str(EMBODIED_ROOT),
    str(DEXJOCO_ROOT.parent / "reach_insert_rl"),
):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.physics.grasp_metrics import peg_hand_contact_counts  # noqa: E402
from embodied_grasp_insertion.simulation.full_episode_snapshot import (  # noqa: E402
    FullEpisodeSnapshot,
)
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    make_full_env,
    replay_demo_to_frame,
)

PROTOCOL = "ActiveContactProbeInformationP0"
CONTACT_CLASSES = ("palm", "index", "middle", "ring", "thumb")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sensor_feature(env) -> np.ndarray:
    if env._force_labeler is None:
        raise RuntimeError("force labeler unavailable")
    force_frame = env._force_labeler.compute(env._raw)
    contact = peg_hand_contact_counts(env._raw)
    contact_values = [float(contact.total)] + [
        float(contact.by_class.get(name, 0)) for name in CONTACT_CLASSES
    ]
    return np.concatenate(
        [np.asarray(force_frame.wrist_ft_right, dtype=np.float64), contact_values]
    )


def action(index: int | None, value: float = 0.0) -> np.ndarray:
    command = np.zeros(44, dtype=np.float64)
    if index is not None:
        command[int(index)] = float(value)
    return command


def step_checked(env, command: np.ndarray) -> bool:
    if env._done:
        return True
    observation, _, _, _, _ = env.step(command)
    if not np.isfinite(observation).all():
        raise RuntimeError("nonfinite observation during probe")
    return bool(env._done)


def branch_features(
    env,
    root_snapshot: FullEpisodeSnapshot,
    *,
    perturb_index: int | None,
    perturb_value: float,
    perturb_steps: int,
    probe_specs: list[dict[str, Any]],
    probe_amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    root_snapshot.restore(env)
    for _ in range(int(perturb_steps)):
        if step_checked(env, action(perturb_index, perturb_value)):
            break
    static = np.concatenate([sensor_feature(env), [float(env._done)]])
    sequence = []
    for spec in probe_specs:
        if not env._done:
            step_checked(
                env,
                action(
                    int(spec["action_index"]),
                    float(spec["sign"]) * float(probe_amplitude),
                ),
            )
        sequence.append(np.concatenate([sensor_feature(env), [float(env._done)]]))
    return static, np.concatenate(sequence)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    n_classes: int,
    alpha: float,
) -> np.ndarray:
    train_x, test_x = standardize(train_x, test_x)
    train_design = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
    test_design = np.concatenate([test_x, np.ones((len(test_x), 1))], axis=1)
    targets = np.eye(n_classes, dtype=np.float64)[train_y]
    reg = np.eye(train_design.shape[1], dtype=np.float64) * float(alpha)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(train_design.T @ train_design + reg, train_design.T @ targets)
    return np.argmax(test_design @ weights, axis=1)


def accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(prediction == target)) if len(target) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "active_contact_probe_information_p0.yaml",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    directions = list(cfg["directions"])
    roots = [
        {**root, "role": role}
        for role in ("discovery", "held_out")
        for root in cfg[f"{role}_roots"]
    ]
    episodes = sorted({int(root["episode_index"]) for root in roots})
    env = make_full_env(episodes, sidecar_dir=Path(cfg["sidecar_dir"]), seed=int(cfg["seed"]))
    rows = []
    try:
        for root in roots:
            episode = int(root["episode_index"])
            frame = int(root["frame"])
            print(f"=== {root['role']} ep{episode} f{frame} ===", flush=True)
            env.reset(entry=next(entry for entry in env.entries if int(entry["episode_index"]) == episode))
            replay_demo_to_frame(env, frame)
            root_snapshot = FullEpisodeSnapshot.capture(env)
            baseline_static, baseline_sequence = branch_features(
                env,
                root_snapshot,
                perturb_index=None,
                perturb_value=0.0,
                perturb_steps=int(cfg["perturb_steps"]),
                probe_specs=list(cfg["probe_sequence"]),
                probe_amplitude=float(cfg["probe_action_amplitude"]),
            )
            for label, direction in enumerate(directions):
                static, sequence = branch_features(
                    env,
                    root_snapshot,
                    perturb_index=int(direction["action_index"]),
                    perturb_value=float(direction["sign"])
                    * float(cfg["perturb_action_amplitude"]),
                    perturb_steps=int(cfg["perturb_steps"]),
                    probe_specs=list(cfg["probe_sequence"]),
                    probe_amplitude=float(cfg["probe_action_amplitude"]),
                )
                rows.append(
                    {
                        "role": root["role"],
                        "episode_index": episode,
                        "frame": frame,
                        "direction": direction["name"],
                        "label": label,
                        "static_delta": (static - baseline_static).tolist(),
                        "sequence_delta": (sequence - baseline_sequence).tolist(),
                    }
                )
    finally:
        env.close()

    train_rows = [row for row in rows if row["role"] == "discovery"]
    test_rows = [row for row in rows if row["role"] == "held_out"]
    train_y = np.asarray([row["label"] for row in train_rows], dtype=np.int64)
    test_y = np.asarray([row["label"] for row in test_rows], dtype=np.int64)
    static_train = np.asarray([row["static_delta"] for row in train_rows], dtype=np.float64)
    static_test = np.asarray([row["static_delta"] for row in test_rows], dtype=np.float64)
    sequence_train = np.asarray([row["sequence_delta"] for row in train_rows], dtype=np.float64)
    sequence_test = np.asarray([row["sequence_delta"] for row in test_rows], dtype=np.float64)
    n_classes = len(directions)
    alpha = float(cfg["ridge_alpha"])
    static_pred = ridge_predict(
        static_train, train_y, static_test, n_classes=n_classes, alpha=alpha
    )
    sequence_pred = ridge_predict(
        sequence_train, train_y, sequence_test, n_classes=n_classes, alpha=alpha
    )
    static_accuracy = accuracy(static_pred, test_y)
    sequence_accuracy = accuracy(sequence_pred, test_y)
    per_root = {}
    test_episodes = np.asarray([row["episode_index"] for row in test_rows], dtype=np.int64)
    for episode in sorted(set(test_episodes.tolist())):
        mask = test_episodes == episode
        per_root[str(episode)] = accuracy(sequence_pred[mask], test_y[mask])

    rng = np.random.default_rng(int(cfg["seed"]))
    shuffle_accuracies = []
    for _ in range(int(cfg["shuffle_trials"])):
        shuffled_y = rng.permutation(train_y)
        shuffled_pred = ridge_predict(
            sequence_train,
            shuffled_y,
            sequence_test,
            n_classes=n_classes,
            alpha=alpha,
        )
        shuffle_accuracies.append(accuracy(shuffled_pred, test_y))
    shuffle_mean = float(np.mean(shuffle_accuracies))
    gain = sequence_accuracy - static_accuracy
    checks = {
        "sequence_accuracy_ok": sequence_accuracy >= float(cfg["min_sequence_accuracy"]),
        "per_root_ok": all(
            value >= float(cfg["min_per_root_accuracy"]) for value in per_root.values()
        ),
        "gain_over_static_ok": gain >= float(cfg["min_gain_over_static"]),
        "shuffle_ok": shuffle_mean <= float(cfg["max_shuffle_mean_accuracy"]),
    }
    passed = all(checks.values())
    judgment = {
        "verdict": (
            "pass_active_probe_information_exists"
            if passed
            else "fail_no_active_probe_information_gain"
        ),
        "decision": (
            "allow_probe_robustness_p0"
            if passed
            else "stop_active_probe_information_direction"
        ),
        "static_accuracy": static_accuracy,
        "sequence_accuracy": sequence_accuracy,
        "gain_over_static": gain,
        "per_root_sequence_accuracy": per_root,
        "shuffle_mean_accuracy": shuffle_mean,
        "checks": checks,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
    }

    output_dir = PROJECT_ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    report_path = PROJECT_ROOT / cfg["report_path"]
    manifest_path = PROJECT_ROOT / cfg["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "utc": _utc(),
                "config": cfg,
                "rows": rows,
                "predictions": {
                    "test_y": test_y.tolist(),
                    "static": static_pred.tolist(),
                    "sequence": sequence_pred.tolist(),
                    "shuffle_accuracies": shuffle_accuracies,
                },
                "judgment": judgment,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Active Contact Probe Information P0 — Result",
        "",
        f"- UTC: `{_utc()}`",
        f"- Protocol: `{PROTOCOL}`",
        f"- Verdict: `{judgment['verdict']}`",
        f"- Decision: `{judgment['decision']}`",
        f"- static accuracy: `{static_accuracy:.3f}`",
        f"- sequence accuracy: `{sequence_accuracy:.3f}`",
        f"- gain over static: `{gain:.3f}`",
        f"- per-root sequence: `{per_root}`",
        f"- shuffle mean: `{shuffle_mean:.3f}`",
        f"- checks: `{checks}`",
        "",
        "## Note",
        "",
        "只运行 insertion_science matched snapshot 微探针；未调用 HybridInsert/skill_replay，未训练策略。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "utc": _utc(),
                "verdict": judgment["verdict"],
                "decision": judgment["decision"],
                "results": str(results_path),
                "report": str(report_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(judgment, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
