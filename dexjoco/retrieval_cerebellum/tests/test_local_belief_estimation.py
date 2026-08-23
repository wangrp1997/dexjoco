from pathlib import Path

import numpy as np

from retrieval_cerebellum.local_belief_estimation import (
    LOCAL_TARGET_DIM,
    LocalAssemblyBeliefModel,
    local_target,
)


def test_local_target_appends_direct_five_dimensional_state():
    belief = np.arange(18, dtype=np.float64)[None, :]
    target = local_target(belief)

    assert target.shape == (1, LOCAL_TARGET_DIM)
    np.testing.assert_allclose(target[0, 18:], [0.0, 1.0, 3.0, 4.0, -2.0])


def test_local_belief_model_round_trip(tmp_path: Path):
    rng = np.random.default_rng(4)
    features = rng.normal(size=(40, 6))
    mapping = rng.normal(size=(6, LOCAL_TARGET_DIM))
    targets = features @ mapping
    model = LocalAssemblyBeliefModel.fit(
        features[:30],
        targets[:30],
        history_size=3,
        alpha=1e-3,
        calibration_features=features[30:],
        calibration_targets=targets[30:],
    )
    path = tmp_path / "model.npz"
    model.save(path)

    mean, covariance = LocalAssemblyBeliefModel.load(path).predict(features[30:31])

    assert mean.shape == (1, LOCAL_TARGET_DIM)
    assert covariance.shape == (1, LOCAL_TARGET_DIM, LOCAL_TARGET_DIM)
    assert np.linalg.eigvalsh(covariance[0]).min() >= -1e-9
