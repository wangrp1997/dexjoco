"""Offline demo loader (GT-as-base, sparse terminal reward) for DexJoCo LeRobot data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .replay_buffer import ReplayBuffer, Transition


def _load_episode_frames(dataset_root: Path, num_episodes: int | None) -> dict[int, list[dict]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Offline demo loading requires pyarrow. Install with: pip install pyarrow"
        ) from exc

    episodes: dict[int, list[dict]] = {}
    for parquet_path in sorted((dataset_root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(
            parquet_path,
            columns=["action", "observation.state", "episode_index", "frame_index"],
        )
        for action, state, ep_idx, frame_idx in zip(
            table.column("action").to_pylist(),
            table.column("observation.state").to_pylist(),
            table.column("episode_index").to_pylist(),
            table.column("frame_index").to_pylist(),
            strict=True,
        ):
            ep_idx = int(ep_idx)
            if num_episodes is not None and ep_idx >= num_episodes:
                continue
            episodes.setdefault(ep_idx, []).append(
                {
                    "frame_index": int(frame_idx),
                    "state": np.asarray(state, dtype=np.float32).reshape(-1),
                    "action": np.asarray(action, dtype=np.float32).reshape(-1),
                }
            )

    for frames in episodes.values():
        frames.sort(key=lambda row: row["frame_index"])
    return episodes


def populate_offline_buffer_gt_as_base(
    replay: ReplayBuffer,
    dataset_root: Path,
    *,
    state_dim: int,
    scale_action,
    standardize_state,
    num_episodes: int | None = None,
) -> int:
    """Fill replay using ResFiT GT-as-base: base_action = demo action, sparse terminal reward."""
    episodes = _load_episode_frames(dataset_root, num_episodes)
    added = 0

    for frames in episodes.values():
        if len(frames) < 2:
            continue

        for idx in range(len(frames) - 1):
            curr = frames[idx]
            nxt = frames[idx + 1]
            state_n = standardize_state(curr["state"][:state_dim])
            next_state_n = standardize_state(nxt["state"][:state_dim])
            base_n = scale_action(curr["action"])
            next_base_n = scale_action(nxt["action"])
            replay.add(
                Transition(
                    state=state_n,
                    base_action=base_n,
                    combined_action=base_n,
                    reward=0.0,
                    next_state=next_state_n,
                    next_base_action=next_base_n,
                    done=False,
                )
            )
            added += 1

        last = frames[-1]
        last_state_n = standardize_state(last["state"][:state_dim])
        last_base_n = scale_action(last["action"])
        replay.add(
            Transition(
                state=last_state_n,
                base_action=last_base_n,
                combined_action=last_base_n,
                reward=1.0,
                next_state=last_state_n,
                next_base_action=last_base_n,
                done=True,
            )
        )
        added += 1

    return added
