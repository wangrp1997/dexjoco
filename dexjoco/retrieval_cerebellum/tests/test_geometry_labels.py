from dataclasses import replace

import numpy as np
import pyarrow.parquet as parquet
import pytest

from retrieval_cerebellum.geometry_labels import (
    GeometryPriorFrame,
    _hole_frame_rotation,
    relative_pose,
)
from retrieval_cerebellum.geometry_store import VECTOR_COLUMNS, write_geometry_episode


def test_relative_pose_is_invariant_to_shared_world_translation():
    identity = np.eye(3)
    position, rotvec = relative_pose(
        np.array([1.0, 2.0, 3.0]),
        identity,
        np.array([1.1, 2.2, 3.3]),
        identity,
    )
    np.testing.assert_allclose(position, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(rotvec, np.zeros(3))


def test_hole_frame_is_right_handed_and_uses_outward_axis():
    rotation = _hole_frame_rotation(np.eye(3), np.array([0.0, 0.0, -1.0]))

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, -1.0])


def test_hole_frame_accepts_read_only_mujoco_views():
    outward_axis = np.array([0.0, 0.0, -1.0])
    outward_axis.flags.writeable = False

    rotation = _hole_frame_rotation(np.eye(3), outward_axis)

    np.testing.assert_allclose(rotation[:, 2], outward_axis)


def _frame() -> GeometryPriorFrame:
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    return GeometryPriorFrame(
        family_id="round_8mm",
        peg_tip_world=vector,
        peg_axis_world=np.array([0.0, 0.0, 1.0]),
        hole_entry_world=np.zeros(3),
        hole_axis_world=np.array([0.0, 0.0, 1.0]),
        peg_tip_in_hole_position=vector,
        peg_in_hole_rotvec=np.zeros(3),
        peg_in_right_palm_position=vector,
        peg_in_right_palm_rotvec=np.zeros(3),
        tray_in_left_palm_position=vector,
        tray_in_left_palm_rotvec=np.zeros(3),
        lateral_error_m=0.1,
        axis_error_rad=0.2,
        approach_height_m=0.3,
        insertion_depth_m=0.0,
        target_depth_m=0.04,
        nominal_peg_size_m=0.008,
        peg_ok=True,
        tray_ok=True,
        insert_ok=False,
        peg_contact_count=2,
        tray_contact_count=3,
    )


def test_geometry_store_writes_explicit_vector_schema(tmp_path):
    first = _frame()
    second = replace(first, insert_ok=True, lateral_error_m=0.01)
    path = write_geometry_episode(
        tmp_path,
        3,
        global_index=np.array([10, 11]),
        frame_index=np.array([4, 5]),
        frames=[first, second],
    )

    table = parquet.read_table(path)
    assert table.num_rows == 2
    assert set(VECTOR_COLUMNS).issubset(table.column_names)
    assert table.schema.field("peg_tip_in_hole_position").type.list_size == 3
    assert table["insert_ok"].to_pylist() == [False, True]
