"""Timestamped action-chunk buffer used by ForceVLA open-loop rollout."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class TimedAction:
    action: np.ndarray
    timestamp: int


def _interp_rotvec_geodesic(rotvec0: np.ndarray, rotvec1: np.ndarray, t: float) -> np.ndarray:
    if t <= 0.0:
        return rotvec0.copy()
    if t >= 1.0:
        return rotvec1.copy()
    r0 = R.from_rotvec(rotvec0)
    r1 = R.from_rotvec(rotvec1)
    return (r0 * R.from_rotvec((r0.inv() * r1).as_rotvec() * t)).as_rotvec()


def _interp_single_arm_action(old_action: np.ndarray, new_action: np.ndarray, t: float) -> np.ndarray:
    interp_action = (1.0 - t) * old_action + t * new_action
    rotvec_slice = slice(3, 6)
    interp_action[rotvec_slice] = _interp_rotvec_geodesic(
        old_action[rotvec_slice], new_action[rotvec_slice], t
    ).astype(interp_action.dtype, copy=False)
    return interp_action


def _interp_dual_arm_action(old_action: np.ndarray, new_action: np.ndarray, t: float) -> np.ndarray:
    interp_action = (1.0 - t) * old_action + t * new_action
    right_rotvec_slice = slice(3, 6)
    left_rotvec_slice = slice(25, 28)
    interp_action[right_rotvec_slice] = _interp_rotvec_geodesic(
        old_action[right_rotvec_slice], new_action[right_rotvec_slice], t
    ).astype(interp_action.dtype, copy=False)
    interp_action[left_rotvec_slice] = _interp_rotvec_geodesic(
        old_action[left_rotvec_slice], new_action[left_rotvec_slice], t
    ).astype(interp_action.dtype, copy=False)
    return interp_action


class DualArmActionInterpolator:
    def __init__(self, dual_arm: bool):
        self._fn = _interp_dual_arm_action if dual_arm else _interp_single_arm_action

    def __call__(self, old_action: np.ndarray, new_action: np.ndarray, t: float) -> np.ndarray:
        return self._fn(old_action, new_action, t)


@dataclass
class ActionChunk:
    action: np.ndarray
    timestamp: int


class ActionBuffer:
    """Merge overlapping ForceVLA action chunks into a per-step buffer."""

    def __init__(self, dual_arm: bool):
        self._interp = DualArmActionInterpolator(dual_arm)
        self._buffer: deque[TimedAction] = deque()

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def pop(self) -> TimedAction | None:
        if not self._buffer:
            return None
        return self._buffer.popleft()

    def ingest_queue(self, action_queue: Queue, now_timestamp: int) -> None:
        """Pull new chunks from the inference worker and merge into the buffer."""
        while self._buffer and self._buffer[0].timestamp < now_timestamp:
            self._buffer.popleft()

        while True:
            try:
                chunk: ActionChunk = action_queue.get_nowait()
            except Empty:
                break

            assert chunk.timestamp <= now_timestamp

            chunk_range = (now_timestamp, chunk.timestamp + chunk.action.shape[0])
            if chunk_range[1] <= now_timestamp:
                continue

            action = chunk.action[
                (chunk_range[0] - chunk.timestamp) : (chunk_range[1] - chunk.timestamp)
            ]

            if self._buffer:
                buffer_range = (self._buffer[0].timestamp, self._buffer[-1].timestamp + 1)
            else:
                buffer_range = (now_timestamp, now_timestamp)

            overlap_range = (
                max(chunk_range[0], buffer_range[0]),
                min(chunk_range[1], buffer_range[1]),
            )
            overlap_len = overlap_range[1] - overlap_range[0]
            for ts in range(overlap_range[0], overlap_range[1]):
                buffer_idx = ts - buffer_range[0]
                action_idx = ts - chunk_range[0]
                interp_t = (ts - overlap_range[0] + 1) / (overlap_len + 1)
                interp_action = self._interp(
                    self._buffer[buffer_idx].action,
                    action[action_idx],
                    interp_t,
                )
                self._buffer[buffer_idx] = TimedAction(action=interp_action, timestamp=ts)

            non_overlap_range = (buffer_range[1], chunk_range[1])
            for ts in range(non_overlap_range[0], non_overlap_range[1]):
                action_idx = ts - chunk_range[0]
                self._buffer.append(
                    TimedAction(action=action[action_idx], timestamp=ts)
                )
