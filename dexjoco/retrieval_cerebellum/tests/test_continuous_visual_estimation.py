from pathlib import Path

import numpy as np

from retrieval_cerebellum.continuous_visual_estimation import (
    ContinuousVisualFeatureStore,
    ContinuousVisualStateModel,
    rotation_to_sixd,
)


def test_continuous_visual_store_returns_ordered_episode(tmp_path: Path):
    path = tmp_path / "cache.npz"
    np.savez_compressed(
        path,
        episode_index=np.asarray([2, 2, 1]),
        frame_index=np.asarray([8, 5, 3]),
        split=np.asarray(["test", "test", "train"]),
        projected_features=np.asarray([[8.0], [5.0], [3.0]]),
        camera_keys=np.asarray(["camera"]),
        model_name=np.asarray("model"),
    )

    frames, features, split = ContinuousVisualFeatureStore.load(path).episode(2)

    np.testing.assert_array_equal(frames, [5, 8])
    np.testing.assert_allclose(features[:, 0], [5.0, 8.0])
    assert split == "test"


def test_continuous_visual_model_round_trip(tmp_path: Path):
    rng = np.random.default_rng(8)
    features = rng.normal(size=(80, 6))
    targets = rng.normal(size=(80, 11))
    identity_sixd = rotation_to_sixd(np.eye(3))
    targets[:, 5:] = identity_sixd
    model = ContinuousVisualStateModel.fit(
        features[:60],
        targets[:60],
        alpha=1.0,
        calibration_features=features[60:],
        calibration_targets11=targets[60:],
    )
    path = tmp_path / "model.npz"
    model.save(path)

    estimate = ContinuousVisualStateModel.load(path).estimate(
        features[0],
        timestamp_s=0.5,
    )

    assert estimate.mean5.shape == (5,)
    assert estimate.covariance5.shape == (5, 5)
    np.testing.assert_allclose(estimate.hole_rotation_world, np.eye(3), atol=1e-8)
    assert 0.0 <= estimate.visual_reliability <= 1.0
