"""Replay deployable numeric cerebellum sensors without object truth."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state
from dexquery.data.action_utils import policy_dual_arm_to_raw, rotvec_dual_arm_to_policy
from dexquery.data.episode_replay import make_assembly_env

from .geometry_labels import GeometryPriorFrame, PrivilegedGeometryLabeler
from .sensor_observation import CerebellumSensorObservation
from .sim_sensor_adapter import SimCerebellumSensorAdapter


def replay_episode_sensors(
    actions44: np.ndarray,
    states46: np.ndarray,
    *,
    seed: int,
    initial_state: np.ndarray | None,
    randomize: bool = False,
) -> tuple[list[CerebellumSensorObservation], dict[str, Any]]:
    """Replay an episode and capture only the P1 deployable sensor interface.

    Each returned observation is captured after executing the action at the same
    row. The recorded 46D proprioception is supplied to the trusted adapter so
    that the exported row stays exactly aligned with the LeRobot dataset.
    """
    actions = np.asarray(actions44, dtype=np.float32)
    states = np.asarray(states46, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 44:
        raise ValueError(f"Expected actions shape (T, 44), got {actions.shape}")
    if states.shape != (actions.shape[0], 46):
        raise ValueError(
            f"Expected states shape ({actions.shape[0]}, 46), got {states.shape}"
        )
    if initial_state is None:
        raise ValueError("Accurate sensor replay requires the recorded initial_state")

    env = make_assembly_env(seed=seed, randomize=randomize)
    raw_env = env.unwrapped
    adapter = SimCerebellumSensorAdapter(raw_env)
    config = CONFIG_MAPPING["bimanual_assembly"]()
    try:
        env.reset()
        if not has_restorer("bimanual_assembly"):
            raise RuntimeError("bimanual_assembly initial-state restorer is unavailable")
        restore_initial_state(env, "bimanual_assembly", config, initial_state)

        observations: list[CerebellumSensorObservation] = []
        for action44, state46 in zip(actions, states, strict=True):
            action46 = rotvec_dual_arm_to_policy(action44)
            raw_action = policy_dual_arm_to_raw(action46)
            raw_env.step(raw_action)
            observations.append(
                adapter.capture(
                    {"state": state46},
                    previous_action44=action44,
                )
            )
        return observations, {
            "num_steps": int(actions.shape[0]),
            "seed": int(seed),
            "used_initial_state": True,
            "randomize": bool(randomize),
        }
    finally:
        env.close()


def replay_episode_sensors_and_geometry(
    actions44: np.ndarray,
    states46: np.ndarray,
    *,
    seed: int,
    initial_state: np.ndarray | None,
    randomize: bool = False,
) -> tuple[
    list[CerebellumSensorObservation],
    list[GeometryPriorFrame],
    dict[str, Any],
]:
    """Capture deployable sensors and privileged teachers from the same steps."""
    actions = np.asarray(actions44, dtype=np.float32)
    states = np.asarray(states46, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 44:
        raise ValueError(f"Expected actions shape (T, 44), got {actions.shape}")
    if states.shape != (actions.shape[0], 46):
        raise ValueError(
            f"Expected states shape ({actions.shape[0]}, 46), got {states.shape}"
        )
    if initial_state is None:
        raise ValueError("Accurate sensor replay requires the recorded initial_state")

    env = make_assembly_env(seed=seed, randomize=randomize)
    raw_env = env.unwrapped
    adapter = SimCerebellumSensorAdapter(raw_env)
    labeler = PrivilegedGeometryLabeler(raw_env)
    config = CONFIG_MAPPING["bimanual_assembly"]()
    try:
        env.reset()
        if not has_restorer("bimanual_assembly"):
            raise RuntimeError("bimanual_assembly initial-state restorer is unavailable")
        restore_initial_state(env, "bimanual_assembly", config, initial_state)
        labeler.reset_reference(raw_env)

        observations: list[CerebellumSensorObservation] = []
        geometry_frames: list[GeometryPriorFrame] = []
        for action44, state46 in zip(actions, states, strict=True):
            action46 = rotvec_dual_arm_to_policy(action44)
            raw_action = policy_dual_arm_to_raw(action46)
            raw_env.step(raw_action)
            observations.append(
                adapter.capture(
                    {"state": state46},
                    previous_action44=action44,
                )
            )
            geometry_frames.append(labeler.compute(raw_env))
        return observations, geometry_frames, {
            "num_steps": int(actions.shape[0]),
            "seed": int(seed),
            "used_initial_state": True,
            "randomize": bool(randomize),
            "teacher_alignment": "same_replay_step",
        }
    finally:
        env.close()
