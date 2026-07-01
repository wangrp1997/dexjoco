"""Minimal env smoke test: 2 envs, 1 trajectory, reset + step only (no PPO compile)."""

from __future__ import annotations

import time

import jax
import jax.numpy as jp
import numpy as np

import track_mj as tmj
from track_mj.envs.assembly_tracking import assembly_tracking_constants as consts
from track_mj.envs.assembly_tracking.utils.wrapper import wrap_fn
from track_mj.learning.policy.ppo import train_tracking


def main() -> None:
    task = "AssemblyTrackingGeneral"
    env_class = tmj.registry.get(task, "tracking_train_env_class")
    task_cfg = tmj.registry.get(task, "tracking_config")
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config

    env_cfg.reference_traj_config.name = {consts.TASK_ID: ["ep000_full"]}
    env_cfg.episode_length = 50

    num_envs = 2
    policy_cfg.num_envs = num_envs
    policy_cfg.episode_length = env_cfg.episode_length
    policy_cfg.action_repeat = 1

    print(f"Smoke env: {num_envs} envs, 1 traj (ep000_full), episode_length={env_cfg.episode_length}")

    env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)
    trajectory_data = env.prepare_trajectory(env_cfg.reference_traj_config.name)
    env.th.traj = None

    key = jax.random.PRNGKey(0)
    key_env = jax.random.split(key, num_envs)
    env = train_tracking._maybe_wrap_env(
        env,
        wrap_env=True,
        num_envs=num_envs,
        episode_length=env_cfg.episode_length,
        action_repeat=1,
        local_device_count=1,
        key_env=key,
        wrap_env_fn=wrap_fn,
        randomization_fn=None,
    )

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    t0 = time.monotonic()
    print("Compiling reset...")
    state = reset_fn(key_env, trajectory_data)
    state.reward.block_until_ready()
    t1 = time.monotonic()
    print(f"Reset OK ({t1 - t0:.1f}s), obs state dim={state.obs['state'].shape}")

    action = jp.zeros((num_envs, env.action_size))
    print("Compiling step...")
    state = step_fn(state, action, trajectory_data)
    state.reward.block_until_ready()
    t2 = time.monotonic()
    print(f"Step OK ({t2 - t1:.1f}s), reward={np.array(state.reward)}")


if __name__ == "__main__":
    main()
