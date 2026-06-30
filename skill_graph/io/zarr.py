"""Load replay.zarr episodes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr


def load_zarr_episode(zarr_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    key = "action" if "action" in data else "action_rotvec"
    actions = np.asarray(data[key][:], dtype=np.float64)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    initial_state = None
    if "state" in data:
        states = np.asarray(data["state"][:], dtype=np.float64)
        start = 0
        for i in range(len(actions) - 1):
            if not np.array_equal(actions[i], actions[i + 1]):
                start = i
                break
        actions = actions[start:]
        initial_state = np.asarray(states[start], dtype=np.float64).ravel()
    return actions, initial_state
