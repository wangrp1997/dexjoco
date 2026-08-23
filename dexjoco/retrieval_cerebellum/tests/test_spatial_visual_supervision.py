import numpy as np

from retrieval_cerebellum.spatial_visual_supervision import (
    PEG_CLASS,
    SOCKET_CLASS,
    CameraCalibration,
    assembly_keypoints_world,
    downsample_semantic_mask,
    gaussian_keypoint_heatmaps,
    keypoint_visibility,
    resize_semantic_mask,
    semantic_mask_from_segmentation,
)


def test_camera_projection_uses_mujoco_negative_z_view_direction():
    calibration = CameraCalibration.from_vertical_fov(
        width=200,
        height=100,
        vertical_fov_degrees=90.0,
        position_world=np.zeros(3),
        rotation_world_from_camera=np.eye(3),
    )

    uv, depth, in_frame = calibration.project(
        np.asarray([[0.0, 0.0, -2.0], [1.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
    )

    np.testing.assert_allclose(uv[:2], [[100.0, 50.0], [125.0, 25.0]])
    np.testing.assert_allclose(depth, [2.0, 2.0, -1.0])
    np.testing.assert_array_equal(in_frame, [True, True, False])

    resized = calibration.rescaled(width=400, height=400)
    resized_uv, _, _ = resized.project(np.asarray([[1.0, 1.0, -2.0]]))
    np.testing.assert_allclose(resized_uv, [[250.0, 100.0]])


def test_semantic_mask_and_visibility_keep_object_identity():
    segmentation = np.full((9, 9, 2), -1, dtype=np.int32)
    segmentation[..., 1] = 5
    segmentation[2:5, 2:5, 0] = 11
    segmentation[5:8, 5:8, 0] = 22
    mask = semantic_mask_from_segmentation(
        segmentation,
        peg_geom_ids={11},
        socket_geom_ids={22},
    )

    visible = keypoint_visibility(
        mask,
        np.asarray([[3.0, 3.0], [4.0, 4.0], [6.0, 6.0], [0.0, 8.0]]),
        np.asarray([True, True, True, True]),
        support_radius_px=1,
    )

    assert mask[3, 3] == PEG_CLASS
    assert mask[6, 6] == SOCKET_CLASS
    np.testing.assert_array_equal(visible, [True, True, True, False])


def test_axis_keypoints_downsampling_and_heatmaps():
    points = assembly_keypoints_world(
        peg_tip_world=np.asarray([1.0, 2.0, 3.0]),
        peg_axis_world=np.asarray([0.0, 0.0, 2.0]),
        hole_entry_world=np.asarray([4.0, 5.0, 6.0]),
        hole_axis_world=np.asarray([0.0, 2.0, 0.0]),
        axis_length_m=0.02,
    )
    np.testing.assert_allclose(points[1], [1.0, 2.0, 3.02])
    np.testing.assert_allclose(points[3], [4.0, 5.02, 6.0])

    mask = np.arange(48, dtype=np.uint8).reshape(6, 8)
    np.testing.assert_array_equal(downsample_semantic_mask(mask, 2), mask[::2, ::2])
    resized_mask = resize_semantic_mask(mask, height=3, width=4)
    np.testing.assert_array_equal(resized_mask, mask[::2, ::2])

    heatmaps = gaussian_keypoint_heatmaps(
        np.asarray([[2.0, 3.0], [5.0, 1.0]]),
        np.asarray([True, False]),
        height=6,
        width=8,
        sigma_px=1.5,
    )
    assert heatmaps.shape == (2, 6, 8)
    assert heatmaps[0, 3, 2] == 1.0
    assert not np.any(heatmaps[1])
