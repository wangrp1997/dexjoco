from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from functools import partial
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from .env import BimanualPhysicsRLEnv


def make_vec_env(
    env_count: int,
    seed: int,
    randomize: bool,
    roots: Path | None = None,
    root_max_offset: int | None = None,
    episode_steps: int | None = None,
    root_noise: float = 0.0,
    randomize_dynamics: bool = False,
    residual_cost: float = 1e-3,
    online_bias_correction: bool = False,
    causal_templates: Path | None = None,
    causal_warm_start: bool = False,
):
    factories = [
        partial(
            BimanualPhysicsRLEnv,
            seed=seed + rank,
            randomize=randomize,
            randomize_dynamics=randomize_dynamics,
            root_bank=roots,
            root_max_offset=root_max_offset,
            episode_steps=episode_steps,
            root_noise=root_noise,
            residual_cost=residual_cost,
            online_bias_correction=online_bias_correction,
            causal_templates=causal_templates,
            causal_warm_start=causal_warm_start,
        )
        for rank in range(env_count)
    ]
    if env_count == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="forkserver")


def _curriculum_env(args: argparse.Namespace, env_count: int):
    return make_vec_env(
        env_count,
        args.seed,
        args.randomize,
        args.roots,
        args.root_max_offset,
        args.episode_steps,
        args.root_noise,
        args.randomize_dynamics,
        args.residual_cost,
        args.online_bias_correction,
        args.causal_templates,
        args.causal_warm_start,
    )


def train(args: argparse.Namespace) -> None:
    if bool(args.resume_model) != bool(args.resume_stats):
        raise ValueError("--resume-model and --resume-stats must be provided together")
    torch.set_num_threads(args.torch_threads)
    run_dir = Path(
        args.output
        or (
            "/mnt/hdd/dexjoco/outputs/bimanual_physics_rl/"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {key: value for key, value in vars(args).items() if key != "handler"}
    (run_dir / "config.json").write_text(
        json.dumps(settings, default=str, indent=2) + "\n", encoding="utf-8"
    )
    base_env = VecMonitor(_curriculum_env(args, args.envs))
    if args.resume_stats:
        env = VecNormalize.load(args.resume_stats, base_env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(
            base_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
        )
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=args.rollout_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.0,
        clip_range=args.clip_range,
        n_epochs=3,
        target_kl=args.target_kl,
        policy_kwargs={
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "log_std_init": -3.0,
        },
        seed=args.seed,
        device=args.device,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
    )
    if args.resume_model:
        model = PPO.load(
            args.resume_model,
            env=env,
            device=args.device,
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
        )
    checkpoint = CheckpointCallback(
        save_freq=max(args.checkpoint_steps // args.envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=checkpoint,
            reset_num_timesteps=args.resume_model is None,
        )
        model.save(run_dir / "model")
        env.save(run_dir / "vecnormalize.pkl")
        print(run_dir)
    finally:
        env.close()


def evaluate(args: argparse.Namespace) -> None:
    base_env = VecMonitor(_curriculum_env(args, 1))
    if args.stats and not args.model:
        raise ValueError("--stats requires --model")
    if args.model:
        stats_path = (
            Path(args.stats)
            if args.stats
            else Path(args.model).with_name("vecnormalize.pkl")
        )
        if stats_path.exists():
            env = VecNormalize.load(stats_path, base_env)
            env.training = False
            env.norm_reward = False
        else:
            env = base_env
        model = PPO.load(args.model, env=env, device=args.device)
    else:
        env = base_env
        model = None
    successes = 0
    lengths: list[int] = []
    results = []
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        env.seed(args.seed)
        obs = env.reset()
        for _ in range(args.episodes):
            episode_steps = 0
            while True:
                if model is None:
                    action = np.zeros((1,) + env.action_space.shape, dtype=np.float32)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, _, done, infos = env.step(action)
                episode_steps += 1
                if done[0]:
                    success = bool(infos[0].get("is_success", False))
                    successes += int(success)
                    lengths.append(episode_steps)
                    results.append({"success": success, "steps": episode_steps})
                    break
        payload = {
            "episodes": args.episodes,
            "successes": successes,
            "success_rate": successes / args.episodes,
            "mean_episode_steps": float(np.mean(lengths)),
            "success_key": "info.succeed",
            "policy": (
                "ppo_gain"
                if model and args.causal_templates
                else "ppo_residual" if model else "zero_residual"
            ),
            "results": results,
        }
        rendered = json.dumps(payload, indent=2) + "\n"
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    finally:
        env.close()


def benchmark(args: argparse.Namespace) -> None:
    env = make_vec_env(args.envs, args.seed, randomize=False)
    try:
        env.reset()
        action = np.zeros((args.envs,) + env.action_space.shape, dtype=np.float32)
        start = time.perf_counter()
        for _ in range(args.steps):
            env.step(action)
        elapsed = time.perf_counter() - start
        transitions = args.steps * args.envs
        print(
            json.dumps(
                {
                    "envs": args.envs,
                    "transitions": transitions,
                    "seconds": elapsed,
                    "transitions_per_second": transitions / elapsed,
                },
                indent=2,
            )
        )
    finally:
        env.close()


def add_curriculum_args(command: argparse.ArgumentParser) -> None:
    command.add_argument("--roots", type=Path)
    command.add_argument("--root-max-offset", type=int)
    command.add_argument("--episode-steps", type=int)
    command.add_argument("--root-noise", type=float, default=0.0)
    command.add_argument("--residual-cost", type=float, default=1e-3)
    command.add_argument("--online-bias-correction", action="store_true")
    command.add_argument("--randomize-dynamics", action="store_true")
    command.add_argument("--causal-templates", type=Path)
    command.add_argument("--causal-warm-start", action="store_true")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="DexJoCo bimanual online physics RL")
    commands = root.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--envs", type=int, default=8)
    train_parser.add_argument("--steps", type=int, default=2_000_000)
    train_parser.add_argument("--rollout-steps", type=int, default=256)
    train_parser.add_argument("--batch-size", type=int, default=512)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--clip-range", type=float, default=0.1)
    train_parser.add_argument("--target-kl", type=float, default=0.05)
    train_parser.add_argument("--checkpoint-steps", type=int, default=100_000)
    train_parser.add_argument("--torch-threads", type=int, default=4)
    train_parser.add_argument("--resume-model", type=Path)
    train_parser.add_argument("--resume-stats", type=Path)
    train_parser.add_argument("--output")
    train_parser.add_argument("--randomize", action="store_true")
    add_curriculum_args(train_parser)
    train_parser.set_defaults(handler=train)

    eval_parser = commands.add_parser("eval")
    eval_parser.add_argument("--model")
    eval_parser.add_argument("--stats")
    eval_parser.add_argument("--episodes", type=int, default=50)
    eval_parser.add_argument("--seed", type=int, default=10_000)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--output")
    eval_parser.add_argument("--randomize", action="store_true")
    add_curriculum_args(eval_parser)
    eval_parser.set_defaults(handler=evaluate)

    bench_parser = commands.add_parser("bench")
    bench_parser.add_argument("--envs", type=int, default=8)
    bench_parser.add_argument("--steps", type=int, default=500)
    bench_parser.add_argument("--seed", type=int, default=0)
    bench_parser.set_defaults(handler=benchmark)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.handler(args)
