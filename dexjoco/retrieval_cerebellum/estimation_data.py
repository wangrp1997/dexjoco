"""Aligned sensor histories and privileged teacher labels for state estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .sensor_observation import CerebellumSensorObservation


METADATA_COLUMNS = (
    "index",
    "episode_index",
    "frame_index",
    "dataset_timestamp_s",
    "sensor_timestamp_s",
    "split",
    "family_id",
)
SENSOR_VECTOR_DIMS = {
    "sensor_state46": 46,
    "sensor_previous_action44": 44,
    "sensor_arm_joint_torque": 14,
    "sensor_fingertip_force_world": 24,
    "sensor_wrist_wrench_world": 12,
}
TEACHER_VECTOR_DIMS = {
    "teacher_peg_in_hole_position": 3,
    "teacher_peg_in_hole_rotvec": 3,
    "teacher_peg_in_right_palm_position": 3,
    "teacher_peg_in_right_palm_rotvec": 3,
    "teacher_tray_in_left_palm_position": 3,
    "teacher_tray_in_left_palm_rotvec": 3,
}
TEACHER_SCALAR_COLUMNS = (
    "teacher_lateral_error_m",
    "teacher_axis_error_rad",
    "teacher_approach_height_m",
    "teacher_insertion_depth_m",
    "teacher_target_depth_m",
    "teacher_nominal_peg_size_m",
    "teacher_peg_ok",
    "teacher_tray_ok",
    "teacher_insert_ok",
    "teacher_peg_contact_count",
    "teacher_tray_contact_count",
)
GEOMETRY_VECTOR_MAP = {
    "teacher_peg_in_hole_position": "peg_tip_in_hole_position",
    "teacher_peg_in_hole_rotvec": "peg_in_hole_rotvec",
    "teacher_peg_in_right_palm_position": "peg_in_right_palm_position",
    "teacher_peg_in_right_palm_rotvec": "peg_in_right_palm_rotvec",
    "teacher_tray_in_left_palm_position": "tray_in_left_palm_position",
    "teacher_tray_in_left_palm_rotvec": "tray_in_left_palm_rotvec",
}
GEOMETRY_SCALAR_MAP = {
    name: name.removeprefix("teacher_") for name in TEACHER_SCALAR_COLUMNS
}


def _fixed_list(values: np.ndarray, width: int):
    import pyarrow as pa

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != width:
        raise ValueError(f"Expected matrix shape (T, {width}), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Vector column contains non-finite values")
    flat = pa.array(matrix.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def _require_aligned(
    source: Mapping[str, np.ndarray],
    geometry: Mapping[str, np.ndarray],
) -> int:
    for name in ("index", "episode_index", "frame_index"):
        source_values = np.asarray(source[name])
        geometry_values = np.asarray(geometry[name])
        if not np.array_equal(source_values, geometry_values):
            raise ValueError(f"source and geometry {name} columns are not aligned")
    row_count = len(np.asarray(source["index"]))
    if row_count == 0:
        raise ValueError("estimation episode must contain at least one row")
    return row_count


def estimation_episode_table(
    source: Mapping[str, np.ndarray],
    observations: Sequence[CerebellumSensorObservation],
    geometry: Mapping[str, np.ndarray],
    *,
    split: str,
):
    """Create a table whose deployable and privileged columns are explicit."""
    import pyarrow as pa

    row_count = _require_aligned(source, geometry)
    if len(observations) != row_count:
        raise ValueError(
            f"Expected {row_count} sensor observations, got {len(observations)}"
        )
    family_ids = np.asarray(geometry["family_id"], dtype=object)
    if family_ids.shape != (row_count,):
        raise ValueError(f"family_id must have shape ({row_count},)")

    previous_actions = []
    for observation in observations:
        if observation.previous_action44 is None:
            raise ValueError("offline estimation rows require previous_action44")
        previous_actions.append(observation.previous_action44)

    sensor_vectors = {
        "sensor_state46": np.stack([item.state46 for item in observations]),
        "sensor_previous_action44": np.stack(previous_actions),
        "sensor_arm_joint_torque": np.stack(
            [item.arm_joint_torque.reshape(-1) for item in observations]
        ),
        "sensor_fingertip_force_world": np.stack(
            [item.fingertip_force_world.reshape(-1) for item in observations]
        ),
        "sensor_wrist_wrench_world": np.stack(
            [item.wrist_wrench_world.reshape(-1) for item in observations]
        ),
    }
    columns: dict[str, object] = {
        "index": np.asarray(source["index"], dtype=np.int64),
        "episode_index": np.asarray(source["episode_index"], dtype=np.int64),
        "frame_index": np.asarray(source["frame_index"], dtype=np.int64),
        "dataset_timestamp_s": np.asarray(source["timestamp"], dtype=np.float32),
        "sensor_timestamp_s": np.asarray(
            [item.timestamp_s for item in observations], dtype=np.float64
        ),
        "split": [str(split)] * row_count,
        "family_id": family_ids.tolist(),
    }
    for name, width in SENSOR_VECTOR_DIMS.items():
        columns[name] = _fixed_list(sensor_vectors[name], width)
    for output_name, geometry_name in GEOMETRY_VECTOR_MAP.items():
        columns[output_name] = _fixed_list(
            np.asarray(geometry[geometry_name]),
            TEACHER_VECTOR_DIMS[output_name],
        )
    for output_name, geometry_name in GEOMETRY_SCALAR_MAP.items():
        values = np.asarray(geometry[geometry_name])
        if values.shape != (row_count,):
            raise ValueError(f"{geometry_name} must have shape ({row_count},)")
        if output_name in {
            "teacher_peg_ok",
            "teacher_tray_ok",
            "teacher_insert_ok",
        }:
            columns[output_name] = values.astype(bool)
        elif output_name.endswith("_contact_count"):
            columns[output_name] = values.astype(np.int32)
        else:
            numeric = values.astype(np.float32)
            if not np.isfinite(numeric).all():
                raise ValueError(f"{geometry_name} contains non-finite values")
            columns[output_name] = numeric
    return pa.table(columns)


def write_estimation_episode(
    output_dir: Path,
    episode_index: int,
    source: Mapping[str, np.ndarray],
    observations: Sequence[CerebellumSensorObservation],
    geometry: Mapping[str, np.ndarray],
    *,
    split: str,
) -> Path:
    import pyarrow.parquet as parquet

    table = estimation_episode_table(source, observations, geometry, split=split)
    episode_dir = Path(output_dir) / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    path = episode_dir / f"episode_{int(episode_index):06d}.parquet"
    parquet.write_table(table, path)
    return path


@dataclass(frozen=True)
class SensorHistory:
    """A deployable history window that cannot expose teacher columns."""

    index: np.ndarray
    frame_index: np.ndarray
    dataset_timestamp_s: np.ndarray
    sensor: Mapping[str, np.ndarray]


def load_sensor_history(
    path: Path,
    *,
    end_row: int,
    window_size: int,
) -> SensorHistory:
    """Load one causal sensor-only window from an aligned episode shard."""
    import pyarrow.parquet as parquet

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    table = parquet.read_table(
        path,
        columns=[
            "index",
            "frame_index",
            "dataset_timestamp_s",
            *SENSOR_VECTOR_DIMS,
        ],
    )
    if end_row < 0 or end_row >= table.num_rows:
        raise IndexError(f"end_row {end_row} outside episode with {table.num_rows} rows")
    start_row = max(0, end_row - window_size + 1)
    window = table.slice(start_row, end_row - start_row + 1)
    sensor = {
        name: np.asarray(window[name].to_pylist(), dtype=np.float32)
        for name in SENSOR_VECTOR_DIMS
    }
    return SensorHistory(
        index=np.asarray(window["index"].to_numpy(), dtype=np.int64),
        frame_index=np.asarray(window["frame_index"].to_numpy(), dtype=np.int64),
        dataset_timestamp_s=np.asarray(
            window["dataset_timestamp_s"].to_numpy(), dtype=np.float32
        ),
        sensor=sensor,
    )
