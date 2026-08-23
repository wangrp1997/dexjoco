"""Accurate MuJoCo replay for geometry-prior sidecar generation."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state
from dexquery.data.action_utils import policy_dual_arm_to_raw, rotvec_dual_arm_to_policy
from dexquery.data.episode_replay import make_assembly_env

from .geometry_labels import GeometryPriorFrame, PrivilegedGeometryLabeler


def replay_episode_geometry(
    actions44: np.ndarray,
    *,
    seed: int,
    initial_state: np.ndarray | None,
    randomize: bool = False,
) -> tuple[list[GeometryPriorFrame], dict[str, Any]]:
    if actions44.ndim != 2 or actions44.shape[1] != 44:
        raise ValueError(f"Expected actions shape (T, 44), got {actions44.shape}")
    if initial_state is None:
        raise ValueError("Accurate geometry replay requires the recorded initial_state")

    env = make_assembly_env(seed=seed, randomize=randomize)
    raw_env = env.unwrapped
    labeler = PrivilegedGeometryLabeler(raw_env)
    config = CONFIG_MAPPING["bimanual_assembly"]()
    try:
        env.reset()
        if not has_restorer("bimanual_assembly"):
            raise RuntimeError("bimanual_assembly initial-state restorer is unavailable")
        restore_initial_state(env, "bimanual_assembly", config, initial_state)
        labeler.reset_reference(raw_env)

        frames: list[GeometryPriorFrame] = []
        for action44 in actions44:
            action46 = rotvec_dual_arm_to_policy(action44)
            raw_action = policy_dual_arm_to_raw(action46)
            raw_env.step(raw_action)
            frames.append(labeler.compute(raw_env))
        return frames, {
            "num_steps": int(actions44.shape[0]),
            "seed": int(seed),
            "used_initial_state": True,
        }
    finally:
        env.close()
