from pathlib import Path

import numpy as np
import torch

from retrieval_cerebellum.spatial_visual_learning import (
    SpatialVisualSidecar,
    build_spatial_visual_model,
    gaussian_heatmaps_torch,
    heatmap_argmax_uv,
    prepare_episode_batch,
)


def test_sidecar_batch_flattens_frames_and_cameras(tmp_path: Path):
    path = tmp_path / "episode.npz"
    np.savez_compressed(
        path,
        episode_index=np.asarray(7),
        split=np.asarray("train"),
        frame_index=np.asarray([10, 11]),
        camera_keys=np.asarray(["ego", "wrist_left", "wrist_right"]),
        keypoint_names=np.asarray(
            ["peg_tip", "peg_axis_point", "hole_entry", "hole_axis_point"]
        ),
        image_size=np.asarray([8, 8]),
        mask_downsample=np.asarray(2),
        keypoints_uv=np.full((2, 3, 4, 2), 4.0, dtype=np.float32),
        keypoint_visible=np.ones((2, 3, 4), dtype=bool),
        semantic_masks=np.zeros((2, 3, 4, 4), dtype=np.uint8),
    )
    sidecar = SpatialVisualSidecar.load(path)
    camera_images = [
        [np.full((8, 8, 3), camera * 20 + frame, dtype=np.uint8) for frame in range(2)]
        for camera in range(3)
    ]

    batch = prepare_episode_batch(
        sidecar,
        camera_images,
        input_height=8,
        input_width=8,
    )

    assert batch.images.shape == (6, 3, 8, 8)
    np.testing.assert_array_equal(batch.camera_index, [0, 0, 1, 1, 2, 2])
    np.testing.assert_allclose(batch.keypoints_output_uv, 2.0)


def test_heatmap_targets_and_decoder_round_trip():
    points = torch.tensor([[[2.0, 3.0], [1.0, 1.0]]])
    visible = torch.tensor([[True, False]])
    heatmaps = gaussian_heatmaps_torch(
        points,
        visible,
        height=6,
        width=8,
        sigma_px=1.0,
    )

    decoded = heatmap_argmax_uv(heatmaps)

    torch.testing.assert_close(decoded[0, 0], points[0, 0])
    assert not torch.any(heatmaps[0, 1])


def test_spatial_visual_model_output_shapes():
    model = build_spatial_visual_model(base_channels=8)
    output = model(torch.zeros(2, 3, 64, 64), torch.tensor([0, 2]))

    assert output["segmentation_logits"].shape == (2, 3, 32, 32)
    assert output["heatmap_logits"].shape == (2, 4, 32, 32)
    assert output["visibility_logits"].shape == (2, 4)
