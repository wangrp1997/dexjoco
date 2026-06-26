"""Load DexJoCo replay.zarr episodes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import zarr


def _find_first_non_static_frame(action: np.ndarray) -> int:
    for i in range(len(action) - 1):
        if not np.array_equal(action[i], action[i + 1]):
            return i
    raise ValueError("All actions in episode are identical (entirely static)")


def _trim_static_prefix(
    actions: np.ndarray,
    states: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    start_idx = _find_first_non_static_frame(actions)
    if start_idx < actions.shape[0] and np.all(actions[start_idx] == 0):
        start_idx += 1
    actions = actions[start_idx:]
    initial_state = None
    if states is not None:
        initial_state = np.asarray(states[start_idx], dtype=np.float64).ravel()
    return actions, initial_state


def discover_zarr_demos(zarr_root: Path) -> list[Path]:
    zarr_root = zarr_root.expanduser()
    if not zarr_root.exists():
        raise FileNotFoundError(f"Zarr root not found: {zarr_root}")
    demos = sorted(zarr_root.rglob("replay.zarr"))
    if not demos:
        raise FileNotFoundError(f"No replay.zarr found under {zarr_root}")
    return demos


def _resolve_action_key(data) -> str:
    for key in ("action", "action_rotvec"):
        if key in data:
            return key
    raise KeyError("Expected 'action' or 'action_rotvec' in zarr data group")


def load_zarr_episode(
    zarr_path: Path,
    *,
    trim_static: bool = True,
) -> tuple[np.ndarray, str, np.ndarray | None]:
    """Return ``(actions, action_key, initial_state)``."""
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    action_key = _resolve_action_key(data)
    actions = np.asarray(data[action_key][:], dtype=np.float64)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)

    initial_state = None
    if "state" in data:
        states = np.asarray(data["state"][:], dtype=np.float64)
        if trim_static:
            actions, initial_state = _trim_static_prefix(actions, states)
        else:
            initial_state = np.asarray(states[0], dtype=np.float64).ravel()
    elif trim_static:
        actions, _ = _trim_static_prefix(actions, None)
    return actions, action_key, initial_state


def iter_zarr_episodes(
    zarr_root: Path,
) -> Iterator[tuple[int, Path, np.ndarray, str, np.ndarray | None]]:
    for episode_index, zarr_path in enumerate(discover_zarr_demos(zarr_root)):
        actions, action_key, initial_state = load_zarr_episode(zarr_path)
        yield episode_index, zarr_path, actions, action_key, initial_state
