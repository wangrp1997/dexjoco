import numpy as np
import torch

from retrieval_cerebellum.ego_visual_state_estimation import (
    EGO_SPATIAL_FEATURE_DIM,
    EGO_VISUAL_PROPRIO_FEATURE_DIM,
    causal_feature_history,
    deployable_visual_proprio_features,
    spatial_features_from_logits,
)


def test_spatial_features_are_finite_and_rgb_observation_only():
    segmentation = torch.zeros(2, 3, 8, 8)
    segmentation[:, 1, 2:4, 1:3] = 5.0
    segmentation[:, 2, 4:7, 5:7] = 5.0
    heatmaps = torch.zeros(2, 4, 8, 8)
    heatmaps[:, 0, 2, 1] = 8.0
    heatmaps[:, 1, 3, 2] = 8.0
    heatmaps[:, 2, 4, 5] = 8.0
    heatmaps[:, 3, 6, 6] = 8.0

    features, reliability, diagnostics = spatial_features_from_logits(
        segmentation,
        heatmaps,
        torch.full((2, 4), 4.0),
    )

    assert features.shape == (2, EGO_SPATIAL_FEATURE_DIM)
    assert np.isfinite(features).all()
    assert np.all((reliability >= 0.0) & (reliability <= 1.0))
    np.testing.assert_array_equal(diagnostics["keypoints_uv"][0, 0], [1.0, 2.0])


def test_causal_history_repeats_first_frame_without_future_access():
    features = np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    history = causal_feature_history(features, 3)

    np.testing.assert_array_equal(history[0], [1.0, 10.0, 1.0, 10.0, 1.0, 10.0])
    np.testing.assert_array_equal(history[1], [1.0, 10.0, 1.0, 10.0, 2.0, 20.0])
    np.testing.assert_array_equal(history[2], [1.0, 10.0, 2.0, 20.0, 3.0, 30.0])


def test_deployable_features_only_join_rgb_proprio_and_action_history():
    combined = deployable_visual_proprio_features(
        np.zeros((2, EGO_SPATIAL_FEATURE_DIM), dtype=np.float32),
        np.ones((2, 46), dtype=np.float32),
        np.full((2, 44), 2.0, dtype=np.float32),
    )

    assert combined.shape == (2, EGO_VISUAL_PROPRIO_FEATURE_DIM)
    np.testing.assert_array_equal(combined[:, EGO_SPATIAL_FEATURE_DIM], 1.0)
    np.testing.assert_array_equal(combined[:, -1], 2.0)
