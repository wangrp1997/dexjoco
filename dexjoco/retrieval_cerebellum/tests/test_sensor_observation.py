import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

import retrieval_cerebellum.sensor_observation as sensor_observation_module
import retrieval_cerebellum.sim_sensor_adapter as sim_sensor_adapter_module
from retrieval_cerebellum.sensor_observation import (
    CerebellumSensorObservation,
    SensorTraceRecorder,
)
from retrieval_cerebellum.sim_sensor_adapter import SimCerebellumSensorAdapter


def _observation() -> CerebellumSensorObservation:
    return CerebellumSensorObservation(
        timestamp_s=0.25,
        state46=np.arange(46, dtype=np.float32),
        arm_joint_torque=np.arange(14, dtype=np.float32).reshape(2, 7),
        fingertip_force_world=np.arange(24, dtype=np.float32).reshape(2, 4, 3),
        wrist_wrench_world=np.arange(12, dtype=np.float32).reshape(2, 6),
        images={"ego": np.zeros((8, 9, 3), dtype=np.uint8)},
        previous_action44=np.arange(44, dtype=np.float32),
    )


def test_sensor_observation_is_read_only_and_contains_no_truth_fields():
    observation = _observation()

    with pytest.raises(ValueError):
        observation.state46[0] = 1.0
    with pytest.raises(ValueError):
        observation.images["ego"][0, 0, 0] = 1
    assert not hasattr(observation, "raw_env")
    assert not hasattr(observation, "object_pose")
    assert not hasattr(observation, "lateral_error_m")


def test_sensor_trace_records_shapes_instead_of_image_pixels(tmp_path):
    recorder = SensorTraceRecorder()
    recorder.append(_observation(), timestamp=7)
    path = tmp_path / "sensor_trace.jsonl"
    recorder.save(path)

    record = json.loads(path.read_text().strip())

    assert record["timestamp"] == 7
    assert record["images"]["ego"] == {"shape": [8, 9, 3], "dtype": "uint8"}
    assert "object_pose" not in record
    assert "lateral_error_m" not in record


class _ForbiddenObjectTruthData:
    def __init__(self) -> None:
        self.time = 0.5
        self.cfrc_ext = np.arange(48, dtype=np.float64).reshape(8, 6)
        self.site_xmat = np.stack([np.eye(3), np.eye(3)])
        self.qfrc_actuator = np.arange(14, dtype=np.float64)
        self._sensors = {
            "panda/wrist_force_right": np.array([1.0, 2.0, 3.0]),
            "panda/wrist_force_left": np.array([4.0, 5.0, 6.0]),
            "panda/wrist_torque_right": np.array([0.1, 0.2, 0.3]),
            "panda/wrist_torque_left": np.array([0.4, 0.5, 0.6]),
        }

    def sensor(self, name):
        return SimpleNamespace(data=self._sensors[name])

    def __getattr__(self, name):
        if name in {"xpos", "xquat", "xmat", "site_xpos", "subtree_com"}:
            raise AssertionError(f"forbidden object truth access: {name}")
        raise AttributeError(name)


class _FakeModel:
    def __init__(self) -> None:
        self._body_ids = {
            "ff_tip_right": 0,
            "mf_tip_right": 1,
            "rf_tip_right": 2,
            "th_tip_right": 3,
            "ff_tip_left": 4,
            "mf_tip_left": 5,
            "rf_tip_left": 6,
            "th_tip_left": 7,
        }
        self._site_ids = {"attachment_site_right": 0, "attachment_site_left": 1}
        self._joint_addresses = {
            f"joint{joint}_{side}": offset
            for offset, (side, joint) in enumerate(
                (side, joint)
                for side in ("right", "left")
                for joint in range(1, 8)
            )
        }

    def body(self, name):
        return SimpleNamespace(id=self._body_ids[name])

    def site(self, name):
        return SimpleNamespace(id=self._site_ids[name])

    def joint(self, name):
        return SimpleNamespace(dofadr=np.array([self._joint_addresses[name]]))


def test_sim_sensor_adapter_never_reads_object_pose_truth():
    data = _ForbiddenObjectTruthData()
    raw_env = SimpleNamespace(_model=_FakeModel(), _data=data)
    adapter = SimCerebellumSensorAdapter(raw_env)

    observation = adapter.capture(
        {
            "state": np.zeros(46, dtype=np.float32),
            "prompt": "insert",
            "ego": np.zeros((4, 5, 3), dtype=np.uint8),
        },
        previous_action44=np.zeros(44, dtype=np.float32),
    )

    np.testing.assert_allclose(observation.arm_joint_torque.reshape(-1), np.arange(14))
    np.testing.assert_allclose(observation.wrist_wrench_world[0], [1, 2, 3, 0.1, 0.2, 0.3])
    assert observation.fingertip_force_world.shape == (2, 4, 3)


def test_deployable_sensor_module_has_no_simulator_backdoor():
    source = inspect.getsource(sensor_observation_module)

    for forbidden in ("raw_env", "._data", ".xpos", ".xquat", ".xmat"):
        assert forbidden not in source

    adapter_source = inspect.getsource(sim_sensor_adapter_module)
    for forbidden in (".xpos", ".xquat", "peg_body", "socket_body", "hole_entry"):
        assert forbidden not in adapter_source
