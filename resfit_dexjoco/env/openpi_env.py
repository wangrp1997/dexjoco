"""DexJoCo environment factory for OpenPI / ForceVLA action format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv, ForceInputMode


@dataclass
class OpenPIEnvConfig:
    env_name: str
    camera_mapping: dict[str, str]
    robot_type: str
    prompt: str
    seed: int = 0
    rand_full: bool = False
    randomize_dynamics: bool = False
    render_mode: Literal["rgb_array", "human"] = "rgb_array"
    pad_state_dim46: bool = False
    password: list[int] | None = None
    force_mode: ForceInputMode | None = None

    @property
    def dual_arm(self) -> bool:
        return self.robot_type == "dual_arm"

    @classmethod
    def from_yaml(cls, path: Path | str, **overrides) -> "OpenPIEnvConfig":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        fields = {
            "env_name": cfg["env_name"],
            "camera_mapping": cfg["camera_mapping"],
            "robot_type": cfg["robot_type"],
            "prompt": cfg["prompt"],
            "password": cfg.get("password"),
        }
        fields.update(overrides)
        return cls(**fields)


def make_openpi_env(config: OpenPIEnvConfig) -> DexJoCoOpenPIEnv:
    env = DexJoCoOpenPIEnv(
        env_name=config.env_name,
        camera_mapping=config.camera_mapping,
        seed=config.seed,
        rand_full=config.rand_full,
        randomize_dynamics=config.randomize_dynamics,
        dual_arm=config.dual_arm,
        prompt=config.prompt,
        render_mode=config.render_mode,
        pad_state_dim46=config.pad_state_dim46,
        password=config.password,
        force_mode=config.force_mode,
    )
    env.start()
    return env
