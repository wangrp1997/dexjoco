"""Normalized online rollout bridge for DexJoCo + ForceVLA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from resfit_dexjoco.bc.forcevla_client import ForceVLAClient
from resfit_dexjoco.env.residual_wrapper import ResidualEnvWrapper, StepResult
from resfit_dexjoco.env.rl_obs import pack_rl_obs, read_sim_state


@dataclass
class NormScalars:
    scale_action: callable
    unscale_action: callable
    standardize_state: callable


class NormalizedTrainEnv:
    """Apply ResFiT-style normalization around ``ResidualEnvWrapper``."""

    def __init__(
        self,
        rollout: ResidualEnvWrapper,
        *,
        privileged_sim_state: bool,
        norm: NormScalars,
    ):
        self.rollout = rollout
        self.privileged_sim_state = privileged_sim_state
        self.norm = norm
        self._last_rl_obs: dict[str, np.ndarray] | None = None
        self._last_base_n: np.ndarray | None = None

    @property
    def action_dim(self) -> int:
        return self.rollout.action_dim

    def reset(self) -> dict[str, np.ndarray]:
        self.rollout.reset()
        return self._refresh_obs(base_action_raw=None)

    def step_residual_normalized(self, residual_n: np.ndarray) -> tuple[dict[str, np.ndarray], StepResult]:
        prev_base_n = self.last_base_n.copy()
        combined_n = np.clip(prev_base_n + residual_n, -1.0, 1.0)
        combined_raw = self.norm.unscale_action(combined_n)
        result = self.rollout.execute_combined(combined_raw)
        next_obs = self._refresh_obs(base_action_raw=result.base_action)
        result.info["combined_normalized"] = combined_n
        result.info["prev_base_normalized"] = prev_base_n
        return next_obs, result

    def end_episode(self, *, strict_drain: bool = True) -> None:
        self.rollout.end_episode(strict_drain=strict_drain)

    def close(self) -> None:
        self.rollout.close()

    def build_transition(
        self,
        prev_obs: dict[str, np.ndarray],
        prev_base_n: np.ndarray,
        combined_n: np.ndarray,
        result: StepResult,
        next_obs: dict[str, np.ndarray],
        next_base_n: np.ndarray,
    ):
        from resfit_dexjoco.training.replay_buffer import Transition

        return Transition(
            state=prev_obs["observation.state"],
            base_action=prev_base_n,
            combined_action=combined_n,
            reward=float(result.reward),
            next_state=next_obs["observation.state"],
            next_base_action=next_base_n,
            done=bool(result.terminated),
        )

    def _refresh_obs(self, base_action_raw: np.ndarray | None) -> dict[str, np.ndarray]:
        env = self.rollout.env
        state = read_sim_state(env, privileged_sim_state=self.privileged_sim_state)
        state_n = self.norm.standardize_state(state)
        if base_action_raw is None:
            base_n = np.zeros(self.action_dim, dtype=np.float32)
        else:
            base_n = self.norm.scale_action(base_action_raw)
        self._last_rl_obs = pack_rl_obs(state_n, base_n)
        self._last_base_n = base_n
        return self._last_rl_obs

    @property
    def last_base_n(self) -> np.ndarray:
        assert self._last_base_n is not None
        return self._last_base_n

    @property
    def last_rl_obs(self) -> dict[str, np.ndarray]:
        assert self._last_rl_obs is not None
        return self._last_rl_obs
