"""Diagnose NaN in assembly tracking env under random actions."""

from __future__ import annotations

import time

import jax
import jax.numpy as jp
import numpy as np

import track_mj as tmj
from track_mj.envs.assembly_tracking import assembly_tracking_constants as consts
from track_mj.envs.assembly_tracking.utils.wrapper import wrap_fn
from track_mj.learning.policy.ppo import train_tracking


def _check_state(state, label: str) -> dict:
    qpos = np.asarray(state.data.qpos)
    qvel = np.asarray(state.data.qvel)
    reward = np.asarray(state.reward)
    done = np.asarray(state.done)
    metrics = {k: np.asarray(v) for k, v in state.metrics.items()}
    info = {
        "label": label,
        "qpos_nan": int(np.isnan(qpos).sum()),
        "qvel_nan": int(np.isnan(qvel).sum()),
        "reward_nan": int(np.isnan(reward).sum()),
        "done_frac": float(done.mean()),
        "reward_mean": float(np.nanmean(reward)),
        "metric_nan": {k: int(np.isnan(v).sum()) for k, v in metrics.items()},
        "episode_len_mean": float(np.asarray(state.info["episode_metrics"]["length"]).mean()),
    }
    return info


def main() -> None:
    task = "AssemblyTrackingGeneral"
    env_class = tmj.registry.get(task, "tracking_train_env_class")
    task_cfg = tmj.registry.get(task, "tracking_config")
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config

    num_envs = 16
    n_steps = 50
    env_cfg.reference_traj_config.name = {consts.TASK_ID: ["ep000_full", "ep001_full"]}
    env_cfg.episode_length = 200

    env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)
    trajectory_data = env.prepare_trajectory(env_cfg.reference_traj_config.name)
    env.th.traj = None

    key = jax.random.PRNGKey(0)
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

    key_env = jax.random.split(key, num_envs)
    print(f"Diag: {num_envs} envs, {n_steps} steps, 2 trajectories")
    t0 = time.monotonic()
    state = reset_fn(key_env, trajectory_data)
    state.reward.block_until_ready()
    print(f"Reset ({time.monotonic() - t0:.1f}s):", _check_state(state, "reset"))

    rng = jax.random.PRNGKey(42)
    for i in range(n_steps):
        rng, act_key = jax.random.split(rng)
        # tanh-bounded actions like PPO policy output
        action = jp.tanh(jax.random.normal(act_key, (num_envs, env.action_size)))
        state = step_fn(state, action, trajectory_data)
        state.reward.block_until_ready()
        info = _check_state(state, f"step{i}")
        if info["qpos_nan"] or info["qvel_nan"] or info["reward_nan"]:
            print(f"*** NaN at step {i}:", info)
            break
        if (i + 1) % 10 == 0:
            print(f"step {i+1}:", info)
    else:
        print("No NaN in", n_steps, "steps. Final:", info)


if __name__ == "__main__":
    main()
