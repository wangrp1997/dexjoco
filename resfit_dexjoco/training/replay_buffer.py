"""Replay buffer for proprio-only residual TD3 (ResFiT transition layout)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    base_action: np.ndarray
    combined_action: np.ndarray
    reward: np.ndarray | float
    next_state: np.ndarray
    next_base_action: np.ndarray
    done: np.ndarray | bool | float


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._size = 0
        self._ptr = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.base_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.combined_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_base_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        idx = self._ptr
        self.states[idx] = transition.state
        self.base_actions[idx] = transition.base_action
        self.combined_actions[idx] = transition.combined_action
        self.rewards[idx] = transition.reward
        self.next_states[idx] = transition.next_state
        self.next_base_actions[idx] = transition.next_base_action
        self.dones[idx] = float(transition.done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Transition:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        idx = np.random.randint(0, self._size, size=batch_size)
        return Transition(
            state=self.states[idx],
            base_action=self.base_actions[idx],
            combined_action=self.combined_actions[idx],
            reward=self.rewards[idx],
            next_state=self.next_states[idx],
            next_base_action=self.next_base_actions[idx],
            done=self.dones[idx] > 0.5,
        )
