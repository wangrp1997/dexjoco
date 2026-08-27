import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from echo_insert.kinematic_estimator import (
    UnobservableTaskFrame,
    ego_depth_to_world,
    estimate_known_geometry_from_pads_and_points,
    fingertip_pads_in_palm,
    fingertip_pads_world,
)


def _transform(
    points: np.ndarray,
    rotation: Rotation,
    translation: np.ndarray,
) -> np.ndarray:
    return rotation.apply(points) + translation


def test_analytic_fk_and_fail_closed_grasp_frame() -> None:
    hand = np.asarray([0.1, 0.7, 0.8, 0.6] * 3 + [0.8, 0.4, 0.7, 0.6])
    expected_tip_origins = np.asarray(
        [
            [0.07272641, 0.05603497, 0.05800990],
            [0.07272641, 0.00729698, 0.06111779],
            [0.07272641, -0.04149655, 0.05928185],
            [0.06378843, 0.07411283, -0.04726742],
        ]
    )
    centers, axes = fingertip_pads_in_palm(hand, side="right")
    offsets = np.asarray([0.019, 0.019, 0.019, 0.035])[:, None]
    np.testing.assert_allclose(centers - axes * offsets, expected_tip_origins, atol=1e-8)

    state = np.zeros(46, dtype=np.float64)
    state[0:3] = [-0.2, 0.0, 0.7]
    state[7:10] = [0.2, 0.0, 0.7]
    aligned = Rotation.from_euler("y", -90.0, degrees=True)
    state[3:7] = state[10:14] = aligned.as_quat(scalar_first=True)
    state[14:30] = state[30:46] = hand
    pads_world, _ = fingertip_pads_world(state)
    assert pads_world.shape == (2, 4, 3)


def test_depth_ransac_recovers_coarse_tip_plane_and_hole_center() -> None:
    radius = 0.01785 + 0.012
    peg = np.asarray(
        [
            [radius, 0.0, 0.3435],
            [radius, 0.0, 0.3000],
            [radius, 0.0, 0.2565],
            [-radius, 0.0, 0.3000],
        ]
    )
    peg[:, 0] += 0.02
    left_pads = np.asarray(
        [
            [-0.08, -0.05, 0.06],
            [-0.08, 0.05, 0.06],
            [0.08, 0.05, 0.06],
            [0.08, -0.05, 0.06],
        ]
    )
    grid_x, grid_y = np.meshgrid(
        np.linspace(-0.09, 0.09, 21),
        np.linspace(-0.07, 0.07, 21),
    )
    tray_plane = np.column_stack(
        [grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, 0.126)]
    )
    lower_x, lower_y = np.meshgrid(
        np.linspace(-0.09, 0.09, 41),
        np.linspace(-0.07, 0.07, 41),
    )
    lower_plane = np.column_stack(
        [lower_x.ravel(), lower_y.ravel(), np.full(lower_x.size, 0.0135)]
    )
    rng = np.random.default_rng(7)
    outliers = rng.uniform(
        [-0.14, -0.14, 0.0],
        [0.14, 0.14, 0.24],
        size=(160, 3),
    )
    outliers[:, 2] = np.where(np.arange(outliers.shape[0]) % 2, 0.02, 0.22)
    scene_points = np.vstack([tray_plane, lower_plane, outliers])

    rotation = Rotation.from_euler("xyz", [23.0, -31.0, 17.0], degrees=True)
    translation = np.asarray([0.4, -0.2, 0.8])
    estimate = estimate_known_geometry_from_pads_and_points(
        _transform(peg, rotation, translation),
        _transform(left_pads, rotation, translation),
        _transform(scene_points, rotation, translation),
    )

    expected_tip = _transform(
        np.asarray([[0.02, 0.0, 0.2325]]), rotation, translation
    )[0]
    expected_entry = _transform(
        np.asarray([[0.0, 0.0, 0.1260]]), rotation, translation
    )[0]
    np.testing.assert_allclose(estimate.peg_insert_end_world, expected_tip, atol=1e-10)
    np.testing.assert_allclose(
        estimate.tray_entry_center_world,
        expected_entry,
        atol=2e-3,
    )
    expected_approach = expected_entry - expected_tip
    expected_distance = np.linalg.norm(expected_approach)
    expected_bearing = expected_approach / expected_distance
    expected_inward = rotation.apply([0.0, 0.0, -1.0])
    np.testing.assert_allclose(
        estimate.approach_bearing_world,
        expected_bearing,
        atol=2e-2,
    )
    np.testing.assert_allclose(estimate.basis_world[:, 2], expected_inward, atol=2e-2)
    assert estimate.approach_distance_m == pytest.approx(expected_distance, abs=2e-3)
    assert estimate.surface_distance_m == pytest.approx(0.1065, abs=2e-3)
    assert estimate.lateral_centering_distance_m == pytest.approx(0.02, abs=2e-3)
    assert estimate.axis_alignment_cosine == pytest.approx(1.0, abs=2e-3)
    assert estimate.maximum_advance_offset_m == pytest.approx(
        estimate.surface_distance_m + 0.0675
    )
    assert estimate.peg_radius_rms_m == pytest.approx(0.0, abs=1e-12)
    assert estimate.tray_plane_rms_m < 1e-8
    assert estimate.tray_plane_inliers >= 400
    assert estimate.tray_plane_radius_m > 0.08
    assert not estimate.basis_world.flags.writeable

    high_residual_peg = peg.copy()
    high_residual_peg[:3, 0] += 0.03
    high_residual_estimate = estimate_known_geometry_from_pads_and_points(
        _transform(high_residual_peg, rotation, translation),
        _transform(left_pads, rotation, translation),
        _transform(scene_points, rotation, translation),
    )
    assert high_residual_estimate.peg_radius_rms_m > 0.01

    far_peg = peg.copy()
    far_peg[:, 2] += 0.15
    far_estimate = estimate_known_geometry_from_pads_and_points(
        _transform(far_peg, rotation, translation),
        _transform(left_pads, rotation, translation),
        _transform(scene_points, rotation, translation),
    )
    assert far_estimate.approach_distance_m > 0.20
    assert far_estimate.maximum_advance_offset_m > 0.20

    sideways_peg = Rotation.from_euler("y", 90.0, degrees=True).apply(
        peg - np.asarray([0.02, 0.0, 0.3])
    ) + np.asarray([0.02, 0.0, 0.15])
    sideways_estimate = estimate_known_geometry_from_pads_and_points(
        _transform(sideways_peg, rotation, translation),
        _transform(left_pads, rotation, translation),
        _transform(scene_points, rotation, translation),
    )
    assert sideways_estimate.tray_alignment_cosine < 0.5
    assert sideways_estimate.surface_distance_m > 0.0

    depth = np.full((640, 640), np.inf, dtype=np.float32)
    depth[320, 320] = 1.0
    world_point = ego_depth_to_world(depth)
    assert world_point.shape == (1, 3)
    assert np.linalg.norm(world_point[0] - [-1.3, 0.0, 2.2]) == pytest.approx(
        1.0,
        abs=2e-3,
    )

    with pytest.raises(UnobservableTaskFrame, match="too few"):
        estimate_known_geometry_from_pads_and_points(
            _transform(peg, rotation, translation),
            _transform(left_pads, rotation, translation),
            _transform(scene_points[:10], rotation, translation),
        )
