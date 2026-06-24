"""Frozen ForceVLA / OpenPI policy client with async websocket inference."""

from __future__ import annotations

import multiprocessing as mp
import signal
import time
from dataclasses import dataclass
from multiprocessing.synchronize import Event as MpEvent
from queue import Empty, Queue

import numpy as np
from openpi_client import websocket_client_policy

from .action_buffer import ActionBuffer, ActionChunk


@dataclass
class Observation:
    obs: dict
    timestamp: int


def _get_latest(q: Queue):
    latest = None
    try:
        while True:
            latest = q.get_nowait()
    except Empty:
        pass
    return latest


def _inference_worker(
    obs_queue: mp.Queue,
    action_queue: mp.Queue,
    stop_event: MpEvent,
    host: str,
    port: int,
    inferencing_event: MpEvent,
):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    while not stop_event.is_set():
        obs: Observation | None = _get_latest(obs_queue)
        if obs is None:
            stop_event.wait(0.01)
            continue
        result = client.infer(obs.obs)
        action_queue.put(ActionChunk(action=result["actions"], timestamp=obs.timestamp))
        inferencing_event.clear()


class ForceVLAClient:
    """Async ForceVLA base policy matching dexjoco-openpi-eval chunk rollout."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        dual_arm: bool,
        action_horizon: int = 30,
        replan_ratio: float = 0.8,
    ):
        self.host = host
        self.port = port
        self.dual_arm = dual_arm
        self.action_horizon = action_horizon
        self.replan_ratio = replan_ratio

        self._obs_queue: mp.Queue | None = None
        self._action_queue: mp.Queue | None = None
        self._stop_event: MpEvent | None = None
        self._inferencing_event: MpEvent | None = None
        self._process: mp.Process | None = None
        self._buffer = ActionBuffer(dual_arm)
        self._timestamp = 0

    def start(self) -> None:
        if self._process is not None:
            return
        self._obs_queue = mp.Queue()
        self._action_queue = mp.Queue()
        self._stop_event = mp.Event()
        self._inferencing_event = mp.Event()
        self._process = mp.Process(
            target=_inference_worker,
            args=(
                self._obs_queue,
                self._action_queue,
                self._stop_event,
                self.host,
                self.port,
                self._inferencing_event,
            ),
        )
        self._process.start()

    def close(self) -> None:
        if self._process is None:
            return
        assert self._stop_event is not None
        self._stop_event.set()
        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)
        if self._obs_queue is not None:
            self._obs_queue.cancel_join_thread()
            self._obs_queue.close()
        if self._action_queue is not None:
            self._action_queue.cancel_join_thread()
            self._action_queue.close()
        self._process = None

    def reset(self, obs: dict) -> None:
        """Clear buffers and request the first action chunk."""
        self._buffer.clear()
        self._timestamp = 0
        self._drain_queues()
        self.request_replan(obs, timestamp=0)

    def request_replan(self, obs: dict, timestamp: int) -> None:
        assert self._obs_queue is not None
        assert self._inferencing_event is not None
        self._inferencing_event.set()
        self._obs_queue.put(Observation(obs=obs, timestamp=timestamp))

    def sync(self, timestamp: int) -> None:
        """Pull newly inferred chunks into the action buffer."""
        assert self._action_queue is not None
        self._buffer.ingest_queue(self._action_queue, timestamp)

    def maybe_replan(self, obs: dict, timestamp: int) -> None:
        assert self._obs_queue is not None
        assert self._action_queue is not None
        assert self._inferencing_event is not None
        should_send = (
            len(self._buffer) < self.replan_ratio * self.action_horizon
            and self._obs_queue.empty()
            and not self._inferencing_event.is_set()
            and self._action_queue.empty()
        )
        if should_send:
            self.request_replan(obs, timestamp)

    def pop_base_action(self) -> np.ndarray | None:
        timed = self._buffer.pop()
        if timed is None:
            return None
        return timed.action

    def drain_after_episode(self, *, timeout_s: float = 120.0) -> None:
        """Wait for in-flight inference before the next episode."""
        assert self._inferencing_event is not None
        assert self._obs_queue is not None
        assert self._action_queue is not None
        while True:
            try:
                self._obs_queue.get_nowait()
            except Empty:
                break
        deadline = time.monotonic() + timeout_s
        while self._inferencing_event.is_set():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"ForceVLA inference did not finish within {timeout_s:.0f}s. "
                    f"Check serve_policy on port {self.port} and kill duplicate train processes."
                )
            time.sleep(0.1)
        while not self._action_queue.empty():
            self._action_queue.get()

    def _drain_queues(self) -> None:
        if self._obs_queue is None or self._action_queue is None:
            return
        while True:
            try:
                self._obs_queue.get_nowait()
            except Empty:
                break
        while not self._action_queue.empty():
            self._action_queue.get()
