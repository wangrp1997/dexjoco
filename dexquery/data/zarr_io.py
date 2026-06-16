"""Load DexJoCo replay.zarr episodes for privileged contact labeling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import zarr


def _find_first_non_static_frame(action: np.ndarray) -> int:
    """Match ``dexjoco_data_converter.episode_common.find_first_non_static_frame``."""
    for i in range(len(action) - 1):
        if not np.array_equal(action[i], action[i + 1]):
            return i
    raise ValueError("All actions in episode are identical (entirely static)")


def _trim_static_prefix_like_lerobot(
    actions: np.ndarray,
    states: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Drop leading static frames the same way LeRobot conversion does."""
    start_idx = _find_first_non_static_frame(actions)
    if start_idx < actions.shape[0] and np.all(actions[start_idx] == 0):
        start_idx += 1
    actions = actions[start_idx:]
    initial_state = None
    if states is not None:
        initial_state = np.asarray(states[start_idx], dtype=np.float64).ravel()
    return actions, initial_state


def discover_zarr_demos(zarr_root: Path) -> list[Path]:
    """Return sorted ``replay.zarr`` paths under ``zarr_root``."""
    zarr_root = zarr_root.expanduser()
    if not zarr_root.exists():
        raise FileNotFoundError(f"Zarr root not found: {zarr_root}")

    demos = sorted(zarr_root.rglob("replay.zarr"))
    if not demos:
        raise FileNotFoundError(f"No replay.zarr found under {zarr_root}")
    return demos


def load_zarr_episode(
    zarr_path: Path,
    *,
    action_key: str = "action_rotvec",
    state_key: str = "state",
    trim_static: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load ``(actions, initial_state)`` from one ``replay.zarr`` episode."""
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    if action_key not in data:
        raise KeyError(f"{zarr_path}: missing action key {action_key!r}")
    actions = np.asarray(data[action_key][:], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)

    initial_state = None
    if state_key in data:
        states = np.asarray(data[state_key][:], dtype=np.float64)
        if trim_static:
            actions, initial_state = _trim_static_prefix_like_lerobot(actions, states)
        else:
            initial_state = np.asarray(states[0], dtype=np.float64).ravel()
    elif trim_static:
        actions, _ = _trim_static_prefix_like_lerobot(actions, None)
    return actions, initial_state


def iter_zarr_episodes(
    zarr_root: Path,
    *,
    action_key: str = "action_rotvec",
    state_key: str = "state",
) -> Iterator[tuple[int, Path, np.ndarray, np.ndarray | None]]:
    """Yield ``(episode_index, zarr_path, actions, initial_state)`` in sorted order."""
    for episode_index, zarr_path in enumerate(discover_zarr_demos(zarr_root)):
        actions, initial_state = load_zarr_episode(
            zarr_path,
            action_key=action_key,
            state_key=state_key,
        )
        yield episode_index, zarr_path, actions, initial_state
