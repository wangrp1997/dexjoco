"""Thin MuJoCo bimanual_assembly wrapper for skill_graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

from skill_graph.constants import PEG_BODY, TRAY_BODY, ObjectName, Side
from skill_graph.io.actions import zarr_to_policy46, zarr_to_raw_dict


@dataclass
class AssemblySim:
    env: object
    raw: object
    seed: int
    _frame_cb: Callable[[], None] | None = field(default=None, repr=False)

    @property
    def model(self):
        return self.raw._model

    @property
    def data(self):
        return self.raw._data

    def set_frame_callback(self, cb: Callable[[], None] | None) -> None:
        self._frame_cb = cb

    def on_physics_step(self) -> None:
        if self._frame_cb is not None:
            self._frame_cb()

    def close(self) -> None:
        self.env.close()

    def object_pose(self, object_name: ObjectName) -> tuple[np.ndarray, np.ndarray]:
        body = TRAY_BODY if object_name == "tray" else PEG_BODY
        bid = int(self.model.body(body).id)
        return (
            np.asarray(self.data.xpos[bid], dtype=np.float64).copy(),
            np.asarray(self.data.xquat[bid], dtype=np.float64).copy(),
        )

    def restore_initial_state(self, state: np.ndarray | None) -> None:
        if state is None:
            return
        config = CONFIG_MAPPING["bimanual_assembly"]()
        if has_restorer("bimanual_assembly"):
            restore_initial_state(self.env, "bimanual_assembly", config, state)

    def step_policy46(self, action46: np.ndarray) -> None:
        self.env.step(np.asarray(action46, dtype=np.float32))
        self.on_physics_step()

    def replay_zarr_actions(self, actions: np.ndarray, start: int, end: int) -> None:
        for fi in range(int(start), int(end) + 1):
            self.raw.step(zarr_to_raw_dict(actions[fi]))
            self.on_physics_step()


def make_assembly_sim(*, seed: int = 0, render_mode: str = "rgb_array") -> AssemblySim:
    config = CONFIG_MAPPING["bimanual_assembly"]()
    env = config.get_environment(
        policy_mode=True,
        render_mode=render_mode,
        randomize=False,
        randomize_dynamics=False,
        seed=int(seed),
    )
    return AssemblySim(env=env, raw=env.unwrapped, seed=int(seed))
