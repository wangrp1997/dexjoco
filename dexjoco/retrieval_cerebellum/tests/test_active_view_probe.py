import inspect
from pathlib import Path

import numpy as np

import retrieval_cerebellum.active_view_probe as active_view_probe_module
from retrieval_cerebellum.active_view_probe import (
    ActiveViewTransition,
    ActiveViewProbeConfig,
    SensorOnlyActiveViewProbe,
    load_active_view_transitions,
    save_active_view_transitions,
)
from retrieval_cerebellum.sensor_observation import CerebellumSensorObservation


def _observation(*, wrist_force: float = 0.0, fingertip_force: float = 1.0):
    wrench = np.zeros((2, 6), dtype=np.float32)
    wrench[:, 0] = wrist_force
    fingertip = np.zeros((2, 4, 3), dtype=np.float32)
    fingertip[:, :, 2] = fingertip_force
    return CerebellumSensorObservation(
        timestamp_s=0.0,
        state46=np.zeros(46, dtype=np.float32),
        arm_joint_torque=np.zeros((2, 7), dtype=np.float32),
        fingertip_force_world=fingertip,
        wrist_wrench_world=wrench,
        images={"ego": np.zeros((8, 8, 3), dtype=np.uint8)},
        previous_action44=np.zeros(44, dtype=np.float32),
    )


def test_active_view_probe_has_no_axial_insertion_component():
    probe = SensorOnlyActiveViewProbe()
    controls = probe.bimanual_controls(_observation())

    assert len(controls) == 8
    for control in controls:
        assert control.shape == (12,)
        assert control[2] == 0.0
        assert control[5] == 0.0
        assert control[8] == 0.0
        assert control[11] == 0.0


def test_active_view_probe_stops_when_wrench_or_grasp_is_unsafe():
    probe = SensorOnlyActiveViewProbe(
        ActiveViewProbeConfig(maximum_wrist_force_n=5.0)
    )

    assert probe.bimanual_controls(_observation(wrist_force=6.0)) == ()
    assert probe.bimanual_controls(_observation(fingertip_force=0.1)) == ()


def test_active_view_action_candidates_preserve_finger_commands():
    probe = SensorOnlyActiveViewProbe()
    current = np.linspace(-0.2, 0.2, 44)
    actions = probe.action_candidates(_observation(), current)

    assert len(actions) == 8
    for action in actions:
        np.testing.assert_allclose(action[6:22], current[6:22])
        np.testing.assert_allclose(action[28:44], current[28:44])


def test_active_view_probe_module_has_no_truth_access():
    source = inspect.getsource(active_view_probe_module)
    for forbidden in (
        "teacher_",
        "privileged",
        "object_pose",
        "raw_env",
        "._data",
        "site_xpos",
        "lateral_error_m",
    ):
        assert forbidden not in source


def test_active_view_transition_store_round_trip(tmp_path: Path):
    transition = ActiveViewTransition(
        feature_before=np.asarray([0.1, 0.2, 0.3]),
        feature_after=np.asarray([0.2, 0.1, 0.4]),
        reliability_before=0.3,
        reliability_after=0.45,
        control12=np.arange(12) * 1e-3,
        wrist_wrench_before=np.zeros((2, 6)),
        wrist_wrench_after=np.ones((2, 6)) * 0.1,
    )
    path = tmp_path / "transitions.npz"
    save_active_view_transitions(
        path,
        (transition,),
        episode_index=7,
        frame_index=42,
    )
    restored = load_active_view_transitions(path)

    assert len(restored) == 1
    np.testing.assert_allclose(restored[0].feature_before, transition.feature_before)
    np.testing.assert_allclose(restored[0].feature_after, transition.feature_after)
    np.testing.assert_allclose(restored[0].control12, transition.control12)
    assert np.isclose(
        restored[0].reliability_after,
        transition.reliability_after,
    )


def test_active_view_collection_runner_has_no_geometry_evaluator():
    source = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_active_view_collection.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "PrivilegedGeometryLabeler",
        "AssemblyEvaluator",
        "teacher_",
        "lateral_error_m",
        "axis_error_rad",
        "insert_ok",
        "site_xpos",
        ".xpos",
        ".xquat",
    ):
        assert forbidden not in source
