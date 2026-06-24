"""ResFiT-style wrapper: execute a = a_bc + delta_a in OpenPI action space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv

from ..bc.forcevla_client import ForceVLAClient
from .assembly_reward import AssemblyMilestoneReward, MilestoneRewardConfig


@dataclass
class StepResult:
    obs: dict
    reward: float
    terminated: bool
    truncated: bool
    info: dict
    base_action: np.ndarray | None
    residual_action: np.ndarray
    combined_action: np.ndarray | None


class ResidualEnvWrapper:
    """Frozen BC + residual policy rollout on DexJoCo OpenPI env.

    Residual actions live in the same OpenPI rotvec layout as the base policy:
    dual-arm shape (44,), single-arm shape (22,).
    """

    def __init__(
        self,
        env: DexJoCoOpenPIEnv,
        bc_client: ForceVLAClient,
        *,
        action_dim: int | None = None,
        clip_residual: float | None = None,
        reward_mode: Literal["sparse", "milestone"] = "milestone",
        milestone_config: MilestoneRewardConfig | None = None,
    ):
        self.env = env
        self.bc = bc_client
        self.clip_residual = clip_residual
        self.reward_mode = reward_mode
        self.action_dim = action_dim or (44 if env.dual_arm else 22)
        self._timestamp = 0
        self._in_stay_state = False
        self._last_base_action: np.ndarray | None = None

        if reward_mode == "milestone":
            if env.env_name != "bimanual_assembly":
                raise ValueError(
                    "milestone reward only supports env_name='bimanual_assembly', "
                    f"got {env.env_name!r}"
                )
            assert env.env is not None
            self._milestone_reward = AssemblyMilestoneReward.for_bimanual_assembly(
                env.env, config=milestone_config
            )
        else:
            self._milestone_reward = None

    def reset(self) -> dict:
        self.env.reset()
        self._timestamp = 0
        self._in_stay_state = False
        self._last_base_action = None
        if self._milestone_reward is not None:
            assert self.env.env is not None
            self._milestone_reward.reset(self.env.env)
        obs = self.env.get_obs()
        self.bc.reset(obs)
        return self._augment_obs(obs, base_action=None)

    def step(self, residual_action: np.ndarray) -> StepResult:
        residual = np.asarray(residual_action, dtype=np.float64)
        if residual.shape != (self.action_dim,):
            raise ValueError(
                f"Expected residual shape ({self.action_dim},), got {residual.shape}"
            )
        if self.clip_residual is not None:
            residual = np.clip(residual, -self.clip_residual, self.clip_residual)

        self.bc.sync(self._timestamp)
        base_action = self.bc.pop_base_action()

        if base_action is not None:
            combined = base_action + residual
            return self._finalize_step(base_action, combined, used_stay=False)
        self.env.stay(continue_stay=self._in_stay_state)
        self._in_stay_state = True
        return self._finalize_step(None, None, used_stay=True)

    def execute_combined(self, combined_action: np.ndarray) -> StepResult:
        """Execute a pre-combined OpenPI action while advancing the BC action buffer."""
        combined = np.asarray(combined_action, dtype=np.float64)
        if combined.shape != (self.action_dim,):
            raise ValueError(
                f"Expected combined shape ({self.action_dim},), got {combined.shape}"
            )
        self.bc.sync(self._timestamp)
        base_action = self.bc.pop_base_action()
        if base_action is None:
            self.env.stay(continue_stay=self._in_stay_state)
            self._in_stay_state = True
            return self._finalize_step(None, None, used_stay=True)
        return self._finalize_step(base_action, combined, used_stay=False)

    def _finalize_step(
        self,
        base_action: np.ndarray | None,
        combined_action: np.ndarray | None,
        *,
        used_stay: bool,
    ) -> StepResult:
        if not used_stay:
            assert combined_action is not None
            self.env.step(combined_action)
            self._in_stay_state = False

        self._timestamp += 1
        obs = self.env.get_obs()
        self.bc.maybe_replan(obs, self._timestamp)

        terminated = self.env.is_done
        reward, info = self._compute_reward(terminated=terminated)
        info["residual_action"] = (
            combined_action - base_action
            if base_action is not None and combined_action is not None
            else None
        )

        return StepResult(
            obs=self._augment_obs(obs, base_action=base_action),
            reward=reward,
            terminated=terminated,
            truncated=False,
            info=info,
            base_action=base_action,
            residual_action=(
                combined_action - base_action
                if base_action is not None and combined_action is not None
                else np.zeros(self.action_dim)
            ),
            combined_action=combined_action,
        )

    def end_episode(self) -> None:
        self.bc.drain_after_episode()

    def close(self) -> None:
        self.bc.close()
        self.env.close()

    def _compute_reward(self, *, terminated: bool) -> tuple[float, dict]:
        info: dict = {
            "succeed": self.env.is_success,
            "timestamp": self._timestamp - 1,
        }
        if self._milestone_reward is not None:
            assert self.env.env is not None
            reward, milestone_info = self._milestone_reward.compute(
                self.env.env,
                terminated=terminated,
                succeed=self.env.is_success,
            )
            info.update(milestone_info.as_dict())
            return reward, info

        reward = 1.0 if terminated and self.env.is_success else 0.0
        return reward, info

    def _augment_obs(self, obs: dict, base_action: np.ndarray | None) -> dict:
        augmented = dict(obs)
        if base_action is None:
            augmented["base_action"] = np.zeros(self.action_dim, dtype=np.float32)
        else:
            augmented["base_action"] = base_action.astype(np.float32, copy=False)
        return augmented
