"""Offline demo loader (GT-as-base) for DexJoCo LeRobot data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from resfit_dexjoco.env.assembly_reward import (
    MilestoneAwardState,
    MilestoneRewardConfig,
    milestone_reward_from_flags,
)

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


def _load_dexquery_outcomes(dataset_root: Path) -> dict[tuple[int, int], tuple[bool, bool, bool]]:
    label_path = dataset_root / "dexquery_labels" / "outcomes.parquet"
    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing {label_path}. Run dexquery/scripts/label_contact.py first."
        )

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Offline milestone labels require pyarrow.") from exc

    table = pq.read_table(
        label_path,
        columns=["episode_index", "frame_index", "tray_ok", "peg_ok", "insert_ok"],
    )
    lookup: dict[tuple[int, int], tuple[bool, bool, bool]] = {}
    for ep_idx, frame_idx, tray, peg, insert in zip(
        table.column("episode_index").to_pylist(),
        table.column("frame_index").to_pylist(),
        table.column("tray_ok").to_pylist(),
        table.column("peg_ok").to_pylist(),
        table.column("insert_ok").to_pylist(),
        strict=True,
    ):
        lookup[(int(ep_idx), int(frame_idx))] = (
            bool(tray),
            bool(peg),
            bool(insert),
        )
    return lookup


def _flags_for_frame(
    lookup: dict[tuple[int, int], tuple[bool, bool, bool]],
    *,
    episode_index: int,
    frame_index: int,
) -> tuple[bool, bool, bool]:
    key = (episode_index, frame_index)
    if key not in lookup:
        raise KeyError(f"Missing dexquery label for episode={episode_index} frame={frame_index}")
    return lookup[key]


def populate_offline_buffer_gt_as_base(
    replay: ReplayBuffer,
    dataset_root: Path,
    *,
    state_dim: int,
    scale_action,
    standardize_state,
    num_episodes: int | None = None,
    offline_reward_mode: str = "milestone",
    milestone_config: MilestoneRewardConfig | None = None,
) -> int:
    """Fill replay using ResFiT GT-as-base: base_action = demo action."""
    episodes = _load_episode_frames(dataset_root, num_episodes)
    milestone_cfg = milestone_config or MilestoneRewardConfig()
    outcome_lookup = (
        _load_dexquery_outcomes(dataset_root) if offline_reward_mode == "milestone" else None
    )
    added = 0

    for ep_idx, frames in episodes.items():
        if len(frames) < 1:
            continue

        awarded = MilestoneAwardState()
        for idx, curr in enumerate(frames):
            state_n = standardize_state(curr["state"][:state_dim])
            base_n = scale_action(curr["action"])

            if idx + 1 < len(frames):
                nxt = frames[idx + 1]
                next_state_n = standardize_state(nxt["state"][:state_dim])
                next_base_n = scale_action(nxt["action"])
                done = False
                succeed = False
            else:
                next_state_n = state_n
                next_base_n = base_n
                done = True
                succeed = True

            if offline_reward_mode == "milestone":
                assert outcome_lookup is not None
                tray_ok, peg_ok, insert_ok = _flags_for_frame(
                    outcome_lookup,
                    episode_index=ep_idx,
                    frame_index=curr["frame_index"],
                )
                reward, awarded = milestone_reward_from_flags(
                    tray_ok=tray_ok,
                    peg_ok=peg_ok,
                    insert_ok=insert_ok,
                    awarded=awarded,
                    config=milestone_cfg,
                    terminated=done,
                    succeed=succeed,
                )
            elif offline_reward_mode == "sparse":
                reward = 1.0 if done else 0.0
            else:
                raise ValueError(f"Unknown offline_reward_mode: {offline_reward_mode!r}")

            replay.add(
                Transition(
                    state=state_n,
                    base_action=base_n,
                    combined_action=base_n,
                    reward=float(reward),
                    next_state=next_state_n,
                    next_base_action=next_base_n,
                    done=done,
                )
            )
            added += 1

    return added
