"""Rollout evaluation for assembly tracking policies."""

from __future__ import annotations

import time
from typing import Any, Callable

import jax
import jax.numpy as jp
import numpy as np
from absl import logging

from brax.training.types import PolicyParams

EVAL_NUM_ENVS = 4
EVAL_MAX_STEPS = 50


def get_latest_ckpt(ckpt_root) -> str:
    """Return path to the latest numeric checkpoint step under ckpt_root."""
    root = ckpt_root if hasattr(ckpt_root, "iterdir") else __import__("pathlib").Path(ckpt_root)
    steps = []
    for p in root.iterdir():
        if p.is_dir() and p.name.isdigit():
            steps.append(int(p.name))
    if not steps:
        raise FileNotFoundError(f"No checkpoints under {root}")
    return str(root / f"{max(steps):012d}")


class AssemblyTrackingEvaluator:
    """JIT-compiled eval rollout; warmup once, reuse for all checkpoints."""

    def __init__(
        self,
        env,
        trajectory_data,
        make_policy: Callable[..., Any],
        *,
        num_envs: int = EVAL_NUM_ENVS,
        max_steps: int = EVAL_MAX_STEPS,
    ):
        self.env = env
        self.trajectory_data = trajectory_data
        self.make_policy = make_policy
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.key_envs = jax.random.split(jax.random.PRNGKey(0), num_envs)
        self._warmed_up = False

        @jax.jit
        def run_eval(params: PolicyParams, key_roll: jax.Array):
            policy = make_policy(params, deterministic=True)
            state = env.reset(self.key_envs, trajectory_data)

            def body(carry, _):
                state, key = carry
                key, act_key = jax.random.split(key)
                actions, _ = policy(state.obs, act_key)
                nstate = env.step(state, actions, trajectory_data)
                return (nstate, key), None

            (final_state, _), _ = jax.lax.scan(body, (state, key_roll), (), length=max_steps)
            ep = final_state.info["episode_metrics"]
            metrics = {
                "episode_length": jp.mean(ep["length"]),
                "sum_reward": jp.mean(ep["average_sum_reward"]),
                "termination_rate": jp.mean(final_state.done),
            }
            for name, val in ep.items():
                if name.startswith("average_reward/"):
                    metrics["rew_" + name.removeprefix("average_reward/")] = jp.mean(val)
            return metrics

        self._run_eval = run_eval

    def warmup(self, params: PolicyParams, seed: int = 0) -> None:
        if self._warmed_up:
            return
        logging.info(
            "Warming up eval JIT (%d envs, %d steps, one-time compile)...",
            self.num_envs,
            self.max_steps,
        )
        t0 = time.monotonic()
        key_roll = jax.random.fold_in(jax.random.PRNGKey(seed), 0)
        out = self._run_eval(params, key_roll)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
        logging.info("Eval JIT warmup done in %.1fs", time.monotonic() - t0)
        self._warmed_up = True

    def evaluate(self, params: PolicyParams, seed: int = 0) -> dict[str, float]:
        if not self._warmed_up:
            self.warmup(params, seed)
        key_roll = jax.random.fold_in(jax.random.PRNGKey(seed), 1)
        raw = jax.device_get(self._run_eval(params, key_roll))
        out = {k: float(np.asarray(v)) for k, v in raw.items()}
        logging.info(
            "Eval: len=%.1f reward=%.3f term_rate=%.3f",
            out.get("episode_length", 0.0),
            out.get("sum_reward", 0.0),
            out.get("termination_rate", 0.0),
        )
        return out


def evaluate_assembly_tracking(
    env,
    trajectory_data,
    make_policy: Callable[..., Any],
    params: PolicyParams,
    *,
    num_envs: int = EVAL_NUM_ENVS,
    max_steps: int = EVAL_MAX_STEPS,
    seed: int = 0,
    evaluator: AssemblyTrackingEvaluator | None = None,
) -> dict[str, float]:
    """Run eval; prefer passing a pre-warmed `evaluator` from training."""
    if evaluator is None:
        evaluator = AssemblyTrackingEvaluator(
            env, trajectory_data, make_policy, num_envs=num_envs, max_steps=max_steps
        )
    return evaluator.evaluate(params, seed=seed)
