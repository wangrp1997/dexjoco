import numpy as np

from retrieval_cerebellum.belief_estimation import (
    BELIEF_DIM,
    RidgeObservationModel,
    SlidingWindowMHE,
    default_process_variance,
    stack_causal_history,
)
from retrieval_cerebellum.visual_initialization import (
    EpisodeVisualFeatureStore,
    TrainOnlyPCA,
    preprocess_clip_images,
)


def test_causal_history_replicates_left_edge():
    features = np.zeros((2, 143), dtype=np.float32)
    features[0, 0] = 1.0
    features[1, 0] = 2.0

    history = stack_causal_history(features, 3)

    assert history.shape == (2, 429)
    np.testing.assert_allclose(history[0, [0, 143, 286]], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(history[1, [0, 143, 286]], [1.0, 1.0, 2.0])


def test_ridge_observation_model_recovers_linear_targets(tmp_path):
    rng = np.random.default_rng(5)
    features = rng.normal(size=(200, 12))
    weights = rng.normal(size=(12, BELIEF_DIM))
    targets = features @ weights + 0.2

    model = RidgeObservationModel.fit(
        features,
        targets,
        history_size=1,
        alpha=1e-8,
    )
    prediction = model.predict(features)
    path = tmp_path / "model.npz"
    model.save(path)
    restored = RidgeObservationModel.load(path)

    np.testing.assert_allclose(prediction, targets, atol=1e-5)
    np.testing.assert_allclose(restored.predict(features), prediction)


def test_sliding_window_mhe_reduces_constant_signal_noise():
    rng = np.random.default_rng(7)
    truth = np.ones((100, BELIEF_DIM), dtype=np.float64)
    measurements = truth + rng.normal(0.0, 0.2, truth.shape)
    mhe = SlidingWindowMHE(
        np.full(BELIEF_DIM, 0.2**2),
        np.full(BELIEF_DIM, 0.02**2),
        window_size=8,
    )

    result = mhe.smooth(measurements)

    measurement_error = np.mean((measurements - truth) ** 2)
    estimate_error = np.mean((result.mean - truth) ** 2)
    assert estimate_error < measurement_error
    assert np.all(result.variance > 0.0)


def test_default_process_variance_uses_pose_units():
    variance = default_process_variance(
        position_std_m=0.002,
        rotation_std_rad=0.01,
    )

    for offset in range(0, BELIEF_DIM, 6):
        np.testing.assert_allclose(variance[offset : offset + 3], 0.002**2)
        np.testing.assert_allclose(variance[offset + 3 : offset + 6], 0.01**2)


def test_train_only_pca_projects_all_splits():
    train = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    pca = TrainOnlyPCA.fit(train, 1)

    projected = pca.transform(np.asarray([[3.0, 3.0], [4.0, 4.0]]))

    assert projected.shape == (2, 1)
    np.testing.assert_allclose(pca.mean, [1.0, 1.0])


def test_visual_feature_store_checks_split(tmp_path):
    path = tmp_path / "visual.npz"
    np.savez_compressed(
        path,
        episode_index=np.asarray([2, 7]),
        split=np.asarray(["train", "test"]),
        source_frame_index=np.asarray([10, 20]),
        camera_keys=np.asarray(["ego"]),
        projected_features=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
    )
    store = EpisodeVisualFeatureStore.load(path)

    np.testing.assert_allclose(store.feature_for(7, "test"), [3.0, 4.0])
    with np.testing.assert_raises(ValueError):
        store.feature_for(7, "train")


def test_clip_preprocessing_is_finite_and_center_cropped():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, 80:240] = 255

    tensor = preprocess_clip_images([image])

    assert tuple(tensor.shape) == (1, 3, 224, 224)
    assert bool(tensor.isfinite().all())
