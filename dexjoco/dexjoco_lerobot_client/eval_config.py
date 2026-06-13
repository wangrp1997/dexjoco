"""Build LeRobot robot configs from DexJoCo rand_obj / rand_full eval YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

IMAGE_SHAPE = [640, 640, 3]


def load_eval_yaml(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
