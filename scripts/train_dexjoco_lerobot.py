#!/usr/bin/env python3
"""Launch LeRobot training with DexJoCo configs aligned to pi0.5."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASELINE_BY_ROBOT = {
    "dual_arm": _REPO_ROOT / "configs/training/dual_arm_baseline.yaml",
    "single_arm": _REPO_ROOT / "configs/training/single_arm_baseline.yaml",
}
_POLICY_DIR = _REPO_ROOT / "configs/training/policies"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _robot_type_for_task(task: str, eval_cfg: dict[str, Any]) -> str:
    return eval_cfg.get("robot_type", "single_arm")


def _build_lerobot_args(
    cfg: dict[str, Any],
    policy: str,
    task: str,
    device: str,
    output_dir: Path,
    dataset_dir: Path,
    *,
    wandb_enable: bool | None = None,
) -> list[str]:
    repo_id = cfg.get("dataset", {}).get("repo_id_template", "DexJoCo/{task}").format(task=task)
    wandb_cfg = cfg.get("wandb", {})
    if wandb_enable is None:
        wandb_enable = bool(wandb_cfg.get("enable", True))

    args = [
        "lerobot-train",
        f"--policy.type={cfg['policy_type']}",
        f"--policy.device={device}",
        f"--policy.push_to_hub={str(cfg['training']['push_to_hub']).lower()}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_dir}",
        f"--dataset.video_backend={cfg['dataset']['video_backend']}",
        f"--batch_size={cfg['training']['batch_size']}",
        f"--steps={cfg['training']['steps']}",
        f"--seed={cfg['training']['seed']}",
        f"--save_freq={cfg['training']['save_freq']}",
        f"--log_freq={cfg['training']['log_freq']}",
        f"--num_workers={cfg['training']['num_workers']}",
        f"--output_dir={output_dir}",
        f"--job_name={policy}_{task}",
        f"--wandb.enable={str(wandb_enable).lower()}",
    ]

    if wandb_enable:
        args.append(f"--wandb.project={wandb_cfg.get('project', 'dexjoco')}")
        if entity := wandb_cfg.get("entity"):
            args.append(f"--wandb.entity={entity}")

    for key, value in cfg.get("policy", {}).items():
        if isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = value
        args.append(f"--policy.{key}={rendered}")

    return args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a LeRobot policy on DexJoCo with pi0.5-aligned hyperparameters.",
    )
    parser.add_argument(
        "--policy",
        required=True,
        choices=sorted(p.stem for p in _POLICY_DIR.glob("*.yaml")),
        help="Policy config under configs/training/policies/",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task / dataset folder name, e.g. bimanual_assembly",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override dataset.root from baseline yaml",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device passed to --policy.device",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override default checkpoints/<policy>_dexjoco_ckpt/<task>/ under dexjoco repo",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (enabled by default in baseline yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the lerobot-train command without running it",
    )
    args = parser.parse_args()

    eval_cfg_path = _REPO_ROOT / "configs/rand_obj" / f"{args.task}.yaml"
    if not eval_cfg_path.exists():
        sys.exit(f"Unknown task config: {eval_cfg_path}")

    eval_cfg = _load_yaml(eval_cfg_path)
    robot_type = _robot_type_for_task(args.task, eval_cfg)
    baseline_path = _BASELINE_BY_ROBOT.get(robot_type)
    if baseline_path is None:
        sys.exit(f"Unsupported robot_type: {robot_type}")

    policy_path = _POLICY_DIR / f"{args.policy}.yaml"
    cfg = _deep_merge(_load_yaml(baseline_path), _load_yaml(policy_path))

    if args.dataset_root is not None:
        cfg["dataset"]["root"] = str(args.dataset_root.expanduser())

    dataset_dir = Path(cfg["dataset"]["root"]).expanduser() / args.task
    if not dataset_dir.exists():
        sys.exit(f"Dataset not found: {dataset_dir}")

    if args.output_dir is None:
        output_root = Path(cfg["output"]["root"]).expanduser()
        if not output_root.is_absolute():
            output_root = _REPO_ROOT / output_root
        output_dir = output_root / f"{args.policy}_dexjoco_ckpt" / args.task
    else:
        output_dir = args.output_dir.expanduser()

    if args.policy == "dexquery":
        dexquery_train = _REPO_ROOT / "dexquery" / "scripts" / "train.py"
        cmd = [
            sys.executable,
            str(dexquery_train),
            "--task",
            args.task,
            "--device",
            args.device,
            "--output-dir",
            str(output_dir),
        ]
        if args.dataset_root is not None:
            cmd.extend(["--dataset-root", str(args.dataset_root.expanduser())])
        if args.no_wandb:
            cmd.append("--no-wandb")
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"Task: {args.task} ({robot_type})")
        print(f"Policy config: {policy_path.relative_to(_REPO_ROOT)}")
        print(f"Dataset: {dataset_dir}")
        print(f"Output: {output_dir}")
        print(f"Command:\n  {' '.join(cmd)}\n")
        if args.dry_run:
            return
        subprocess.run(cmd, check=True)
        return

    lerobot_args = _build_lerobot_args(
        cfg,
        args.policy,
        args.task,
        args.device,
        output_dir,
        dataset_dir,
        wandb_enable=False if args.no_wandb else None,
    )
    cmd = " ".join(str(a) for a in lerobot_args)
    print(f"Task: {args.task} ({robot_type})")
    print(f"Baseline: {baseline_path.relative_to(_REPO_ROOT)}")
    print(f"Policy config: {policy_path.relative_to(_REPO_ROOT)}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Command:\n  {cmd}\n")

    if args.dry_run:
        return

    subprocess.run(lerobot_args, check=True)


if __name__ == "__main__":
    main()
