"""Parquet storage for geometry-prior replay sidecars."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry_labels import GeometryPriorFrame


VECTOR_COLUMNS = (
    "peg_tip_world",
    "peg_axis_world",
    "hole_entry_world",
    "hole_axis_world",
    "peg_tip_in_hole_position",
    "peg_in_hole_rotvec",
    "peg_in_right_palm_position",
    "peg_in_right_palm_rotvec",
    "tray_in_left_palm_position",
    "tray_in_left_palm_rotvec",
)
SCALAR_COLUMNS = (
    "lateral_error_m",
    "axis_error_rad",
    "approach_height_m",
    "insertion_depth_m",
    "target_depth_m",
    "nominal_peg_size_m",
    "peg_ok",
    "tray_ok",
    "insert_ok",
    "peg_contact_count",
    "tray_contact_count",
)


def geometry_frames_to_columns(
    *,
    global_index: np.ndarray,
    episode_index: int,
    frame_index: np.ndarray,
    frames: list[GeometryPriorFrame],
) -> dict[str, np.ndarray]:
    """Convert same-replay geometry frames to the standard NumPy column mapping."""
    if len(frames) != len(global_index) or len(frames) != len(frame_index):
        raise ValueError("frames and index arrays must have equal lengths")
    columns: dict[str, np.ndarray] = {
        "index": np.asarray(global_index, dtype=np.int64),
        "episode_index": np.full(len(frames), int(episode_index), dtype=np.int64),
        "frame_index": np.asarray(frame_index, dtype=np.int64),
        "family_id": np.asarray([frame.family_id for frame in frames], dtype=object),
    }
    for name in VECTOR_COLUMNS:
        columns[name] = np.asarray([getattr(frame, name) for frame in frames], dtype=np.float32)
    for name in SCALAR_COLUMNS:
        columns[name] = np.asarray([getattr(frame, name) for frame in frames])
    return columns


def write_geometry_episode(
    output_dir: Path,
    episode_index: int,
    *,
    global_index: np.ndarray,
    frame_index: np.ndarray,
    frames: list[GeometryPriorFrame],
) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    if len(frames) != len(global_index) or len(frames) != len(frame_index):
        raise ValueError("frames and index arrays must have equal lengths")
    columns: dict[str, object] = {
        "index": np.asarray(global_index, dtype=np.int64),
        "episode_index": np.full(len(frames), int(episode_index), dtype=np.int64),
        "frame_index": np.asarray(frame_index, dtype=np.int64),
        "family_id": [frame.family_id for frame in frames],
    }
    for name in VECTOR_COLUMNS:
        values = np.asarray([getattr(frame, name) for frame in frames], dtype=np.float32)
        flat = pa.array(values.reshape(-1), type=pa.float32())
        columns[name] = pa.FixedSizeListArray.from_arrays(flat, 3)
    for name in SCALAR_COLUMNS[:6]:
        columns[name] = np.asarray([getattr(frame, name) for frame in frames], dtype=np.float32)
    for name in ("peg_ok", "tray_ok", "insert_ok"):
        columns[name] = np.asarray([getattr(frame, name) for frame in frames], dtype=bool)
    for name in ("peg_contact_count", "tray_contact_count"):
        columns[name] = np.asarray([getattr(frame, name) for frame in frames], dtype=np.int32)

    episode_dir = Path(output_dir) / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    path = episode_dir / f"episode_{episode_index:06d}.parquet"
    parquet.write_table(pa.table(columns), path)
    return path
