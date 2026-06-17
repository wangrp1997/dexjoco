"""Shared helpers to plug hybrid insert into eval loops."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from .config import HybridInsertConfig
from .controller import HybridInsertController

SUPPORTED_TASKS = frozenset({"bimanual_assembly"})


def get_raw_env(env: Any) -> Any:
    """Unwrap gym wrappers down to the MuJoCo task env."""
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


def state_to_dual_arm_action44(state46: np.ndarray) -> np.ndarray:
    """Map flattened 46d proprio state to 44d rotvec policy action."""
    state46 = np.asarray(state46, dtype=np.float64).reshape(-1)
    if state46.shape[0] < 46:
        raise ValueError(f"Expected state dim >= 46, got {state46.shape[0]}")

    r_arm = state46[:7]
    l_arm = state46[7:14]
    r_hand = state46[14:30]
    l_hand = state46[30:46]

    r_rotvec = R.from_quat(r_arm[3:7], scalar_first=True).as_rotvec()
    l_rotvec = R.from_quat(l_arm[3:7], scalar_first=True).as_rotvec()
    return np.concatenate(
        [r_arm[:3], r_rotvec, r_hand, l_arm[:3], l_rotvec, l_hand],
        dtype=np.float64,
    ).astype(np.float32)


class EvalHybridInsert:
    """Optional right-arm hybrid insert; left arm frozen at handoff by default."""

    def __init__(
        self,
        *,
        task: str,
        enabled: bool,
        config: HybridInsertConfig | None = None,
    ) -> None:
        self.task = task
        self.enabled = bool(enabled) and task in SUPPORTED_TASKS
        self.controller = HybridInsertController(config) if self.enabled else None
        if enabled and task not in SUPPORTED_TASKS:
            print(
                f"hybrid_insert requested but unsupported for task={task!r}; ignoring.",
                flush=True,
            )

    def on_reset(self, gym_env: Any) -> None:
        if not self.enabled or self.controller is None:
            return
        self.controller.reset(get_raw_env(gym_env))

    def observe(self, gym_env: Any, policy_action44: np.ndarray) -> None:
        if not self.enabled or self.controller is None or self.controller.active:
            return
        action = np.asarray(policy_action44, dtype=np.float64).reshape(-1)
        self.controller.update_handoff(get_raw_env(gym_env), action)

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.controller is not None and self.controller.active)

    @property
    def needs_policy_left(self) -> bool:
        return bool(
            self.enabled and self.controller is not None and self.controller.needs_policy_left
        )

    def merge(self, gym_env: Any, policy_action44: np.ndarray) -> np.ndarray:
        """Merge policy action with hybrid right-arm override when active."""
        action = np.asarray(policy_action44, dtype=np.float64).reshape(-1)
        if not self.enabled or self.controller is None:
            return action.astype(np.float32)
        return self.controller.merge_right_arm(get_raw_env(gym_env), action)

    def episode_summary(self) -> str:
        if not self.enabled or self.controller is None:
            return "disabled"
        return self.controller.episode_summary()
