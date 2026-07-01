"""Quick test: load latest debug ckpt and run eval rollout."""

from __future__ import annotations

import functools
import sys

import jax
import jax.numpy as jp

import track_mj as tmj
from brax.training.agents.ppo.networks import make_ppo_networks
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from track_mj.eval.tracking.run_eval import evaluate_assembly_tracking, get_latest_ckpt
from track_mj.envs.assembly_tracking.utils.wrapper import wrap_fn
from track_mj.learning.policy.ppo import train_tracking as ppo

CKPT_ROOT = "/mnt/hdd/dexjoco/dex_track_assembly/checkpoints/07011513_AssemblyTrackingGeneral_debug_assembly_track"


def main() -> int:
    task = "AssemblyTrackingGeneral"
    env_class = tmj.registry.get(task, "tracking_train_env_class")
    task_cfg = tmj.registry.get(task, "tracking_config")
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config

    env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)
    trajectory_data = env.prepare_trajectory(env_cfg.reference_traj_config.name)
    env.th.traj = None

    num_envs = 8
    key = jax.random.PRNGKey(0)
    wrapped = ppo._maybe_wrap_env(
        env, True, num_envs, env_cfg.episode_length, 1, 1, key, wrap_fn, None
    )
    reset_fn = jax.jit(wrapped.reset)
    state = reset_fn(jax.random.split(key, num_envs), trajectory_data)
    state.reward.block_until_ready()

    ckpt_dir = get_latest_ckpt(CKPT_ROOT)
    print("Loading", ckpt_dir)
    ckpt = ppo_checkpoint.load(ckpt_dir)
    network_factory = functools.partial(
        make_ppo_networks, **policy_cfg.network_factory.to_dict()
    )
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], state.obs)
    ppo_net = network_factory(obs_shape, wrapped.action_size, preprocess_observations_fn=lambda x, y: x)
    make_policy = make_ppo_networks.make_inference_fn(ppo_net)

    metrics = evaluate_assembly_tracking(
        wrapped, trajectory_data, make_policy, ckpt, num_envs=num_envs, max_steps=50, seed=0
    )
    print("Eval OK:", metrics)
    bad = [k for k, v in metrics.items() if v != v]  # nan check
    if bad:
        print("NaN in eval metrics:", bad)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
