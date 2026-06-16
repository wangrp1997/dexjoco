"""Read DexJoCo LeRobot datasets for offline labeling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


def _require_pyarrow():
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "label_contact requires pyarrow (install lerobot or `pip install pyarrow`)."
        ) from exc
    return pq


def iter_data_parquet_files(dataset_root: Path) -> list[Path]:
    data_root = dataset_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"Missing LeRobot data directory: {data_root}")
    return sorted(data_root.rglob("file-*.parquet"))


def load_dataset_table(dataset_root: Path, columns: list[str] | None = None):
    pq = _require_pyarrow()
    files = iter_data_parquet_files(dataset_root)
    if not files:
        raise FileNotFoundError(f"No parquet shards under {dataset_root / 'data'}")
    tables = [pq.read_table(path, columns=columns) for path in files]
    if len(tables) == 1:
        return tables[0]
    import pyarrow as pa

    return pa.concat_tables(tables, promote_options="default")


def iter_episode_actions(
    dataset_root: Path,
    *,
    episode_indices: list[int] | None = None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(episode_index, global_index, frame_index, actions44)`` per episode."""
    table = load_dataset_table(
        dataset_root,
        columns=["action", "episode_index", "frame_index", "index"],
    )
    episode_col = table.column("episode_index").to_numpy(zero_copy_only=False)
    frame_col = table.column("frame_index").to_numpy(zero_copy_only=False)
    index_col = table.column("index").to_numpy(zero_copy_only=False)
    actions = table.column("action").to_pylist()

    unique_eps = sorted(set(int(x) for x in episode_col.tolist()))
    if episode_indices is not None:
        wanted = set(int(x) for x in episode_indices)
        unique_eps = [ep for ep in unique_eps if ep in wanted]

    for ep in unique_eps:
        mask = episode_col == ep
        order = np.argsort(frame_col[mask])
        ep_actions = np.asarray([actions[i] for i, keep in enumerate(mask) if keep], dtype=np.float32)[
            order
        ]
        ep_frames = frame_col[mask][order]
        ep_indices = index_col[mask][order]
        yield int(ep), ep_indices.astype(np.int64), ep_frames.astype(np.int64), ep_actions
