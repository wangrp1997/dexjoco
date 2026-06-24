"""Build ResFiT-style low-dim observations for residual TD3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv


@dataclass
class RLObsSpec:
    state_dim: int
    base_action_dim: int
    action_dim: int

    @property
    def prop_dim(self) -> int:
        return self.state_dim

    @property
    def actor_input_dim(self) -> int:
        return self.state_dim + self.base_action_dim


def rl_obs_spec(*, privileged_sim_state: bool, action_dim: int = 44) -> RLObsSpec:
    state_dim = 61 if privileged_sim_state else 46
    return RLObsSpec(state_dim=state_dim, base_action_dim=action_dim, action_dim=action_dim)


def read_sim_state(env: DexJoCoOpenPIEnv, *, privileged_sim_state: bool) -> np.ndarray:
    return env.get_sim_state(privileged=privileged_sim_state).astype(np.float32, copy=False)


def pack_rl_obs(
    state: np.ndarray,
    base_action_normalized: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "observation.state": state.astype(np.float32, copy=False),
        "observation.base_action": base_action_normalized.astype(np.float32, copy=False),
    }
