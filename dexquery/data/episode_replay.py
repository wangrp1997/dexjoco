"""Replay DexJoCo assembly episodes in sim for privileged outcome labeling."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

from .action_utils import policy_dual_arm_to_raw, rotvec_dual_arm_to_policy
from .assembly_contacts import AssemblyContactLabeler, AssemblyOutcome


def make_assembly_env(
    *,
    seed: int,
    randomize: bool = False,
    render_mode: str = "rgb_array",
):
    config = CONFIG_MAPPING["bimanual_assembly"]()
    return config.get_environment(
        policy_mode=True,
        render_mode=render_mode,
        randomize=randomize,
        randomize_dynamics=False,
        seed=seed,
    )


def replay_episode_actions(
    actions44: np.ndarray,
    *,
    seed: int,
    initial_state: np.ndarray | None = None,
    randomize: bool = False,
) -> tuple[list[AssemblyOutcome], dict[str, Any]]:
    """Replay one episode and return per-step contact outcomes.

    Labels are computed after each env step. The first list entry corresponds to
    the transition caused by ``actions44[0]`` (post-step contact state).
    """
    if actions44.ndim != 2 or actions44.shape[1] != 44:
        raise ValueError(f"Expected actions shape (T, 44), got {actions44.shape}")

    env = make_assembly_env(seed=seed, randomize=randomize)
    raw_env = env.unwrapped
    labeler = AssemblyContactLabeler(raw_env)
    config = CONFIG_MAPPING["bimanual_assembly"]()

    try:
        env.reset()
        if initial_state is not None and has_restorer("bimanual_assembly"):
            restore_initial_state(env, "bimanual_assembly", config, initial_state)
        labeler.reset_reference(raw_env)

        outcomes: list[AssemblyOutcome] = []
        for action44 in actions44:
            action46 = rotvec_dual_arm_to_policy(action44)
            raw_action = policy_dual_arm_to_raw(action46)
            raw_env.step(raw_action)
            outcomes.append(labeler.compute(raw_env))

        info = {
            "num_steps": int(actions44.shape[0]),
            "seed": int(seed),
            "used_initial_state": initial_state is not None,
        }
        return outcomes, info
    finally:
        env.close()


def replay_episode_forces(
    actions44: np.ndarray,
    *,
    seed: int,
    initial_state: np.ndarray | None = None,
    randomize: bool = False,
) -> tuple[list["ForceFrame"], dict[str, Any]]:
    """Replay one episode and return per-step finger forces + wrist wrenches."""
    from .finger_contact_forces import FingerForceLabeler, ForceFrame

    if actions44.ndim != 2 or actions44.shape[1] != 44:
        raise ValueError(f"Expected actions shape (T, 44), got {actions44.shape}")

    env = make_assembly_env(seed=seed, randomize=randomize)
    raw_env = env.unwrapped
    labeler = FingerForceLabeler(raw_env)
    config = CONFIG_MAPPING["bimanual_assembly"]()

    try:
        env.reset()
        if initial_state is not None and has_restorer("bimanual_assembly"):
            restore_initial_state(env, "bimanual_assembly", config, initial_state)
        labeler.reset_reference(raw_env)

        frames: list[ForceFrame] = []
        for action44 in actions44:
            action46 = rotvec_dual_arm_to_policy(action44)
            raw_action = policy_dual_arm_to_raw(action46)
            raw_env.step(raw_action)
            frames.append(labeler.compute(raw_env))

        info = {
            "num_steps": int(actions44.shape[0]),
            "seed": int(seed),
            "used_initial_state": initial_state is not None,
        }
        return frames, info
    finally:
        env.close()
