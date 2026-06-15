"""Build LeRobot robot configs from DexJoCo rand_obj / rand_full eval YAML files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

IMAGE_SHAPE = [640, 640, 3]
_TRAINING_POLICIES_DIR = Path(__file__).resolve().parents[2] / "configs/training/policies"
_TRAINING_BASELINE_DIR = Path(__file__).resolve().parents[2] / "configs/training"


def load_eval_yaml(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_training_policy_yaml(policy_type: str) -> dict[str, Any]:
    path = _TRAINING_POLICIES_DIR / f"{policy_type}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Training policy config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_training_baseline_yaml(robot_type: str) -> dict[str, Any]:
    name = "dual_arm_baseline.yaml" if robot_type == "dual_arm" else "single_arm_baseline.yaml"
    path = _TRAINING_BASELINE_DIR / name
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _actions_per_chunk_from_checkpoint(policy_type: str, checkpoint: Path) -> int | None:
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        return None
    with open(config_path, "r") as f:
        cfg = json.load(f)
    if policy_type == "act":
        key = "chunk_size"
    elif policy_type in ("diffusion", "multi_task_dit"):
        key = "horizon"
    else:
        return None
    value = cfg.get(key)
    return int(value) if value is not None else None


def resolve_actions_per_chunk(policy_type: str, checkpoint: Path) -> int:
    """Match policy server chunk size to the trained checkpoint when possible."""
    from_ckpt = _actions_per_chunk_from_checkpoint(policy_type, checkpoint)
    if from_ckpt is not None:
        return from_ckpt

    policy_cfg = load_training_policy_yaml(policy_type)
    if "eval" in policy_cfg and "actions_per_chunk" in policy_cfg["eval"]:
        return int(policy_cfg["eval"]["actions_per_chunk"])

    if policy_type == "act":
        return int(policy_cfg["policy"]["chunk_size"])
    if policy_type in ("diffusion", "multi_task_dit"):
        return int(policy_cfg["policy"]["horizon"])
    raise ValueError(f"Cannot resolve actions_per_chunk for policy_type={policy_type!r}")


def resolve_checkpoint_step_label(checkpoint: Path) -> str:
    """Return a stable folder suffix such as ``ckpt060000`` for eval output paths."""
    checkpoint = checkpoint.expanduser().resolve()
    step_dir = checkpoint.parent
    if step_dir.name == "pretrained_model":
        step_dir = step_dir.parent

    step_name = step_dir.resolve().name
    if step_name.isdigit():
        return f"ckpt{int(step_name):06d}"

    training_step_path = step_dir / "training_state" / "training_step.json"
    if training_step_path.exists():
        with open(training_step_path, "r") as f:
            step = json.load(f).get("step")
        if step is not None:
            return f"ckpt{int(step):06d}"

    return f"ckpt_{step_name}"


def default_replan_ratio(robot_type: str) -> float:
    baseline = load_training_baseline_yaml(robot_type)
    return float(baseline["eval"]["replan_ratio"])


def default_eval_output_dir(
    policy_type: str,
    env_name: str,
    seed: int,
    checkpoint: Path,
    *,
    rand_full: bool = False,
) -> Path:
    suffix = "_rand_full" if rand_full else ""
    ckpt_label = resolve_checkpoint_step_label(checkpoint)
    return Path("outputs") / policy_type / f"{env_name}{suffix}_seed{seed}_{ckpt_label}"


def lerobot_image_map(camera_mapping: dict[str, str], dual_arm: bool) -> dict[str, str]:
    """Map LeRobot dataset image keys to DexJoCo environment camera keys."""
    if dual_arm:
        if "base" not in camera_mapping:
            raise ValueError(
                "Dual-arm eval configs must define camera_mapping.base for the ego camera."
            )
        return {
            "ego": camera_mapping["base"],
            "wrist_left": camera_mapping["wrist_left"],
            "wrist_right": camera_mapping["wrist_right"],
        }

    if "base" not in camera_mapping or "wrist" not in camera_mapping:
        raise ValueError(
            "Single-arm eval configs must define camera_mapping.base and camera_mapping.wrist."
        )
    return {
        "front": camera_mapping["base"],
        "wrist": camera_mapping["wrist"],
    }


def build_lerobot_robot_config(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    dual_arm = eval_cfg["robot_type"] == "dual_arm"
    image_map = lerobot_image_map(eval_cfg["camera_mapping"], dual_arm)
    state_dim = 46 if dual_arm else 23
    action_dim = 44 if dual_arm else 22

    return {
        "observation_features": {
            "state": [{"state": state_dim}],
            "images": {model_key: IMAGE_SHAPE for model_key in image_map},
        },
        "action_features": [{"action": action_dim}],
        "single_arm": not dual_arm,
        "model_env_image_map": image_map,
        "task": eval_cfg["prompt"],
    }


def write_robot_config_yaml(eval_cfg: dict[str, Any], path: Path) -> None:
    robot_cfg = build_lerobot_robot_config(eval_cfg)
    with open(path, "w") as f:
        yaml.safe_dump(robot_cfg, f, sort_keys=False)


def video_camera_names(eval_cfg: dict[str, Any]) -> list[str]:
    """Environment camera names used for saved rollout videos."""
    dual_arm = eval_cfg["robot_type"] == "dual_arm"
    return list(lerobot_image_map(eval_cfg["camera_mapping"], dual_arm).values())
