import numpy as np
import pyarrow.parquet as parquet
import pytest

from retrieval_cerebellum.estimation_data import (
    SENSOR_VECTOR_DIMS,
    load_sensor_history,
    write_estimation_episode,
)
from retrieval_cerebellum.sensor_observation import CerebellumSensorObservation


def _source(length: int = 3) -> dict[str, np.ndarray]:
    return {
        "index": np.arange(10, 10 + length, dtype=np.int64),
        "episode_index": np.full(length, 2, dtype=np.int64),
        "frame_index": np.arange(4, 4 + length, dtype=np.int64),
        "timestamp": np.arange(length, dtype=np.float32) / 30.0,
    }


def _geometry(length: int = 3) -> dict[str, np.ndarray]:
    vectors = {
        name: np.full((length, 3), value, dtype=np.float32)
        for value, name in enumerate(
            (
                "peg_tip_in_hole_position",
                "peg_in_hole_rotvec",
                "peg_in_right_palm_position",
                "peg_in_right_palm_rotvec",
                "tray_in_left_palm_position",
                "tray_in_left_palm_rotvec",
            ),
            start=1,
        )
    }
    vectors.update(
        {
            "index": np.arange(10, 10 + length, dtype=np.int64),
            "episode_index": np.full(length, 2, dtype=np.int64),
            "frame_index": np.arange(4, 4 + length, dtype=np.int64),
            "family_id": np.full(length, "round_8mm", dtype=object),
            "lateral_error_m": np.linspace(0.01, 0.0, length),
            "axis_error_rad": np.linspace(0.1, 0.0, length),
            "approach_height_m": np.linspace(0.02, 0.0, length),
            "insertion_depth_m": np.linspace(0.0, 0.01, length),
            "target_depth_m": np.full(length, 0.04),
            "nominal_peg_size_m": np.full(length, 0.008),
            "peg_ok": np.ones(length, dtype=bool),
            "tray_ok": np.ones(length, dtype=bool),
            "insert_ok": np.zeros(length, dtype=bool),
            "peg_contact_count": np.full(length, 2, dtype=np.int32),
            "tray_contact_count": np.full(length, 3, dtype=np.int32),
        }
    )
    return vectors


def _observations(length: int = 3) -> list[CerebellumSensorObservation]:
    return [
        CerebellumSensorObservation(
            timestamp_s=0.5 + row / 30.0,
            state46=np.full(46, row, dtype=np.float32),
            arm_joint_torque=np.full((2, 7), row + 1, dtype=np.float32),
            fingertip_force_world=np.full((2, 4, 3), row + 2, dtype=np.float32),
            wrist_wrench_world=np.full((2, 6), row + 3, dtype=np.float32),
            images={},
            previous_action44=np.full(44, row + 4, dtype=np.float32),
        )
        for row in range(length)
    ]


def test_estimation_episode_has_explicit_sensor_and_teacher_schema(tmp_path):
    path = write_estimation_episode(
        tmp_path,
        2,
        _source(),
        _observations(),
        _geometry(),
        split="train",
    )

    table = parquet.read_table(path)

    assert table.num_rows == 3
    assert table["split"].to_pylist() == ["train"] * 3
    assert table["teacher_peg_ok"].to_pylist() == [True] * 3
    for name, width in SENSOR_VECTOR_DIMS.items():
        assert table.schema.field(name).type.list_size == width
    assert table.schema.field("teacher_peg_in_hole_position").type.list_size == 3


def test_sensor_history_never_loads_teacher_columns(tmp_path):
    path = write_estimation_episode(
        tmp_path,
        2,
        _source(),
        _observations(),
        _geometry(),
        split="validation",
    )

    history = load_sensor_history(path, end_row=2, window_size=2)

    np.testing.assert_array_equal(history.index, [11, 12])
    assert set(history.sensor) == set(SENSOR_VECTOR_DIMS)
    assert all(not name.startswith("teacher_") for name in history.sensor)
    assert history.sensor["sensor_state46"].shape == (2, 46)


def test_estimation_writer_rejects_misaligned_teacher_rows(tmp_path):
    geometry = _geometry()
    geometry["index"] = geometry["index"] + 1

    with pytest.raises(ValueError, match="index columns are not aligned"):
        write_estimation_episode(
            tmp_path,
            2,
            _source(),
            _observations(),
            geometry,
            split="test",
        )


def test_estimation_writer_requires_action_history(tmp_path):
    observations = _observations()
    observations[0] = CerebellumSensorObservation(
        timestamp_s=0.5,
        state46=np.zeros(46),
        arm_joint_torque=np.zeros((2, 7)),
        fingertip_force_world=np.zeros((2, 4, 3)),
        wrist_wrench_world=np.zeros((2, 6)),
        images={},
    )

    with pytest.raises(ValueError, match="previous_action44"):
        write_estimation_episode(
            tmp_path,
            2,
            _source(),
            observations,
            _geometry(),
            split="train",
        )
