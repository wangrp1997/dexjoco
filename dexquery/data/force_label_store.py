"""Sidecar force/tactile label storage (does not modify source LeRobot dataset)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .finger_contact_forces import FINGER_FORCE_DIM, WRIST_FT_DIM


@dataclass
class ForceLabelManifest:
    task: str
    source_dataset: str
    created_at: str
    num_frames: int
    num_episodes: int
    columns: list[str]
    label_file: str
    seed_base: int
    randomize: bool
    notes: str = (
        "Privileged sim finger contact forces and wrist wrenches from MuJoCo replay. "
        "Original LeRobot parquet/video files are untouched."
    )


def default_force_label_dir(dataset_root: Path) -> Path:
    return dataset_root / "force_labels"


def _force_table(
    *,
    global_index: np.ndarray,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    right_finger_force: np.ndarray,
    left_finger_force: np.ndarray,
    wrist_ft_right: np.ndarray,
    wrist_ft_left: np.ndarray,
    insert_ok: np.ndarray,
):
    import pyarrow as pa

    def _list_col(matrix: np.ndarray) -> pa.Array:
        return pa.array([row.tolist() for row in matrix], type=pa.list_(pa.float32()))

    return pa.table(
        {
            "index": global_index.astype(np.int64),
            "episode_index": episode_index.astype(np.int64),
            "frame_index": frame_index.astype(np.int64),
            "right_finger_force": _list_col(right_finger_force),
            "left_finger_force": _list_col(left_finger_force),
            "wrist_ft_right": _list_col(wrist_ft_right),
            "wrist_ft_left": _list_col(wrist_ft_left),
            "insert_ok": insert_ok.astype(np.float32),
        }
    )


def episode_shard_dir(label_dir: Path) -> Path:
    return label_dir / "episodes"


def write_episode_force_parquet(
    label_dir: Path,
    episode_index: int,
    *,
    global_index: np.ndarray,
    frame_index: np.ndarray,
    right_finger_force: np.ndarray,
    left_finger_force: np.ndarray,
    wrist_ft_right: np.ndarray,
    wrist_ft_left: np.ndarray,
    insert_ok: np.ndarray,
) -> Path:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("write_episode_force_parquet requires pyarrow.") from exc

    _validate_force_shapes(
        right_finger_force,
        left_finger_force,
        wrist_ft_right,
        wrist_ft_left,
        num_frames=len(global_index),
    )

    shard_dir = episode_shard_dir(label_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    out_path = shard_dir / f"episode_{episode_index:06d}.parquet"
    ep_idx = np.full(len(global_index), int(episode_index), dtype=np.int64)
    table = _force_table(
        global_index=global_index,
        episode_index=ep_idx,
        frame_index=frame_index,
        right_finger_force=right_finger_force,
        left_finger_force=left_finger_force,
        wrist_ft_right=wrist_ft_right,
        wrist_ft_left=wrist_ft_left,
        insert_ok=insert_ok,
    )
    pq.write_table(table, out_path)
    return out_path


def merge_episode_shards(label_dir: Path) -> Path | None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("merge_episode_shards requires pyarrow.") from exc

    shard_dir = episode_shard_dir(label_dir)
    shards = sorted(shard_dir.glob("episode_*.parquet"))
    if not shards:
        return None

    tables = [pq.read_table(path) for path in shards]
    merged = pa.concat_tables(tables, promote_options="default")
    sort_cols = [
        merged.column_names.index("episode_index"),
        merged.column_names.index("frame_index"),
    ]
    merged = merged.sort_by([(sort_cols[0], "ascending"), (sort_cols[1], "ascending")])
    out_path = label_dir / "forces.parquet"
    pq.write_table(merged, out_path)
    return out_path


def load_checkpoint(label_dir: Path) -> dict | None:
    path = label_dir / "checkpoint.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_checkpoint(label_dir: Path, payload: dict) -> Path:
    label_dir.mkdir(parents=True, exist_ok=True)
    path = label_dir / "checkpoint.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def clear_label_checkpoints(label_dir: Path) -> None:
    checkpoint = label_dir / "checkpoint.json"
    if checkpoint.exists():
        checkpoint.unlink()
    shard_dir = episode_shard_dir(label_dir)
    if shard_dir.exists():
        for path in shard_dir.glob("episode_*.parquet"):
            path.unlink()
    merged = label_dir / "forces.parquet"
    if merged.exists():
        merged.unlink()


def write_manifest(label_dir: Path, manifest: ForceLabelManifest) -> Path:
    label_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = label_dir / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    return manifest_path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _validate_force_shapes(
    right_finger_force: np.ndarray,
    left_finger_force: np.ndarray,
    wrist_ft_right: np.ndarray,
    wrist_ft_left: np.ndarray,
    *,
    num_frames: int,
) -> None:
    for name, arr, dim in (
        ("right_finger_force", right_finger_force, FINGER_FORCE_DIM),
        ("left_finger_force", left_finger_force, FINGER_FORCE_DIM),
        ("wrist_ft_right", wrist_ft_right, WRIST_FT_DIM),
        ("wrist_ft_left", wrist_ft_left, WRIST_FT_DIM),
    ):
        if arr.shape != (num_frames, dim):
            raise ValueError(f"{name} expected ({num_frames}, {dim}), got {arr.shape}")
