import numpy as np
from scipy.spatial.transform import Rotation

from retrieval_cerebellum.cad_geometry_vision import (
    solve_cad_pose_pnp,
    symmetric_assembly_state5,
)
from retrieval_cerebellum.spatial_visual_supervision import CameraCalibration


def test_cad_pnp_recovers_world_pose_from_mujoco_camera_projection():
    calibration = CameraCalibration.from_vertical_fov(
        width=640,
        height=640,
        vertical_fov_degrees=55.0,
        position_world=np.asarray([0.2, -0.1, 1.1]),
        rotation_world_from_camera=Rotation.from_euler(
            "xyz", [0.2, -0.1, 0.3]
        ).as_matrix(),
    )
    points_object = np.asarray(
        [
            [-0.08, -0.06, 0.00],
            [0.08, -0.06, 0.00],
            [0.08, 0.06, 0.00],
            [-0.08, 0.06, 0.00],
            [-0.05, -0.04, 0.08],
            [0.05, -0.04, 0.08],
            [0.05, 0.04, 0.08],
            [-0.05, 0.04, 0.08],
        ]
    )
    rotation_world = Rotation.from_euler("xyz", [0.15, 0.08, -0.2]).as_matrix()
    position_world = np.asarray([0.1, 0.2, 0.55])
    points_world = position_world + (rotation_world @ points_object.T).T
    points_uv, _, in_frame = calibration.project(points_world)
    assert in_frame.all()

    estimate = solve_cad_pose_pnp(points_object, points_uv, calibration)

    np.testing.assert_allclose(estimate.position_world, position_world, atol=1e-5)
    np.testing.assert_allclose(estimate.rotation_world, rotation_world, atol=1e-5)
    assert estimate.reprojection_rmse_px < 1e-4


def test_symmetric_state_ignores_peg_roll_and_keeps_metric_geometry():
    hole_rotation = Rotation.from_euler("z", 0.4).as_matrix()
    peg_axis = hole_rotation @ Rotation.from_rotvec([0.04, -0.03, 0.0]).apply(
        [0.0, 0.0, 1.0]
    )
    hole_position = np.asarray([0.2, -0.1, 0.5])
    relative = np.asarray([0.003, -0.002, -0.025])
    peg_tip = hole_position + hole_rotation @ relative

    state = symmetric_assembly_state5(
        peg_tip,
        peg_axis,
        hole_position,
        hole_rotation,
    )

    np.testing.assert_allclose(state[:2], relative[:2], atol=1e-10)
    np.testing.assert_allclose(state[4], 0.025, atol=1e-10)
    np.testing.assert_allclose(state[2:4], [0.04, -0.03], atol=5e-4)
