import inspect
import json
from pathlib import Path

import numpy as np

import retrieval_cerebellum.active_visual_observation as active_visual_module
from retrieval_cerebellum.active_visual_observation import (
    ActiveVisualReliabilityModel,
)
from retrieval_cerebellum.scripts.train_active_visual_observation import (
    _episode_pairs,
    _probe_pairs,
)
from retrieval_cerebellum.active_view_probe import (
    ActiveViewTransition,
    save_active_view_transitions,
)
from retrieval_cerebellum.v2_control import V2AssemblyEstimate


def _synthetic_dataset(seed: int = 4):
    rng = np.random.default_rng(seed)
    rows = 3000
    features = rng.normal(size=(rows, 3))
    controls = rng.normal(scale=0.01, size=(rows, 12))
    current = rng.uniform(0.25, 0.65, size=rows)
    context_effect = 0.01 + 0.04 * features[:, 0] - 0.03 * features[:, 1]
    gradient = np.column_stack(
        [
            1.4 + 0.2 * features[:, 2],
            -0.8 + 0.1 * features[:, 0],
            np.zeros((rows, 10)),
        ]
    )
    target = current + context_effect + np.sum(controls * gradient, axis=1)
    return features, controls, current, target


def test_active_visual_model_recovers_action_conditioned_reliability():
    features, controls, current, target = _synthetic_dataset()
    model = ActiveVisualReliabilityModel.fit(
        features,
        controls,
        current,
        target,
        alpha=1e-6,
    )
    prediction = np.asarray(
        [
            model.predict(feature, reliability, control)
            for feature, reliability, control in zip(
                features,
                current,
                controls,
                strict=True,
            )
        ]
    )
    assert np.mean(np.abs(prediction - target)) < 1e-6

    row = 17
    analytic = model.control_jacobian(features[row], current[row], controls[row])
    measured = np.empty(12)
    epsilon = 1e-6
    for column in range(12):
        perturbation = np.zeros(12)
        perturbation[column] = epsilon
        measured[column] = (
            model.predict_raw(
                features[row],
                current[row],
                controls[row] + perturbation,
            )
            - model.predict_raw(
                features[row],
                current[row],
                controls[row] - perturbation,
            )
        ) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, measured, atol=1e-9)


def test_active_visual_model_round_trip_and_sqp_bridge(tmp_path: Path):
    features, controls, current, target = _synthetic_dataset()
    model = ActiveVisualReliabilityModel.fit(
        features,
        controls,
        current,
        target,
        alpha=1e-4,
    )
    path = tmp_path / "active_visual.npz"
    model.save(path)
    restored = ActiveVisualReliabilityModel.load(path)
    estimate = V2AssemblyEstimate(
        timestamp_s=0.5,
        mean5=np.zeros(5),
        covariance5=np.diag([0.002, 0.002, 0.006, 0.006, 0.003]),
        hole_rotation_world=np.eye(3),
        visual_reliability=0.0,
    )
    row = 9
    observation_model = restored.as_sqp_observation_model(
        estimate,
        features[row],
        current[row],
        controls[row, :6],
        controls[row, 6:],
    )

    expected = restored.predict(features[row], current[row], controls[row])
    assert np.isclose(observation_model.reliability, expected)
    np.testing.assert_allclose(
        observation_model.control_reliability_jacobian,
        restored.control_jacobian(features[row], current[row], controls[row]),
    )
    np.testing.assert_allclose(
        observation_model.observation_covariance,
        estimate.covariance5,
    )


def test_active_visual_runtime_module_has_no_privileged_backdoor():
    source = inspect.getsource(active_visual_module)
    for forbidden in (
        "teacher_",
        "privileged",
        "object_pose",
        "raw_env",
        "._data",
        "site_xpos",
    ):
        assert forbidden not in source


def test_active_visual_training_pair_loader_requests_sensor_columns_only():
    source = inspect.getsource(_episode_pairs)
    assert "sensor_previous_action44" in source
    for forbidden in (
        "teacher_peg",
        "teacher_tray",
        "object_pose",
        "privileged",
    ):
        assert forbidden not in source


def test_unapproved_active_visual_artifact_is_rejected(tmp_path: Path):
    features, controls, current, target = _synthetic_dataset()
    model = ActiveVisualReliabilityModel.fit(
        features,
        controls,
        current,
        target,
        alpha=1.0,
    )
    model.save(tmp_path / "model.npz")
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "model": str(tmp_path / "model.npz"),
                "approved_for_active_control": False,
            }
        ),
        encoding="utf-8",
    )

    try:
        ActiveVisualReliabilityModel.load_approved(tmp_path)
    except RuntimeError as error:
        assert "not approved" in str(error)
    else:
        raise AssertionError("unapproved active visual model was loaded")


def test_probe_transition_file_feeds_sensor_only_training_pairs(tmp_path: Path):
    transition = ActiveViewTransition(
        feature_before=np.asarray([0.1, 0.2]),
        feature_after=np.asarray([0.15, 0.25]),
        reliability_before=0.4,
        reliability_after=0.5,
        control12=np.arange(12) * 1e-3,
        wrist_wrench_before=np.zeros((2, 6)),
        wrist_wrench_after=np.ones((2, 6)) * 0.1,
    )
    path = tmp_path / "probe.npz"
    save_active_view_transitions(
        path,
        (transition,),
        episode_index=17,
        frame_index=690,
    )

    episode, pairs = _probe_pairs(path)

    assert episode == 17
    np.testing.assert_allclose(pairs[0], [[0.1, 0.2]])
    np.testing.assert_allclose(pairs[1], [np.arange(12) * 1e-3])
    np.testing.assert_allclose(pairs[2], [0.4])
    np.testing.assert_allclose(pairs[3], [0.5])
