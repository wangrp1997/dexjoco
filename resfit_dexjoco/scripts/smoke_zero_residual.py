#!/usr/bin/env python3
"""Smoke test: frozen ForceVLA + zero residual should match dexjoco-openpi-eval."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEXJOco_PKG = _REPO_ROOT / "dexjoco"
if str(_DEXJOco_PKG) not in sys.path:
    sys.path.insert(0, str(_DEXJOco_PKG))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def main(
    config: Path = Path("configs/rand_obj/bimanual_assembly.yaml"),
    seed: int = 0,
    port: int = 8000,
    host: str = "0.0.0.0",
    episodes: int = 5,
    replan_ratio: float = 0.8,
    force_mode: str | None = "both",
    rand_full: bool = False,
) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    _set_seed(seed)

    from resfit_dexjoco.bc.forcevla_client import ForceVLAClient
    from resfit_dexjoco.env.openpi_env import OpenPIEnvConfig, make_openpi_env
    from resfit_dexjoco.env.residual_wrapper import ResidualEnvWrapper

    env_cfg = OpenPIEnvConfig.from_yaml(
        _REPO_ROOT / config,
        seed=seed,
        rand_full=rand_full,
        force_mode=force_mode,  # type: ignore[arg-type]
    )
    env = make_openpi_env(env_cfg)
    bc = ForceVLAClient(
        host=host,
        port=port,
        dual_arm=env_cfg.dual_arm,
        replan_ratio=replan_ratio,
    )
    bc.start()
    rollout = ResidualEnvWrapper(env, bc)

    num_success = 0
    try:
        for ep in range(episodes):
            print(f"Episode {ep + 1}/{episodes}", flush=True)
            rollout.reset()
            zero_residual = np.zeros(rollout.action_dim, dtype=np.float64)

            while True:
                result = rollout.step(zero_residual)
                if result.terminated:
                    if result.info["succeed"]:
                        num_success += 1
                        print("Success!", flush=True)
                    else:
                        print("Failed", flush=True)
                    break

            rollout.end_episode()
    finally:
        rollout.close()

    print(
        f"\nSmoke (zero residual): {num_success}/{episodes} "
        f"({100 * num_success / episodes:.1f}%)",
        flush=True,
    )


if __name__ == "__main__":
    tyro.cli(main)
