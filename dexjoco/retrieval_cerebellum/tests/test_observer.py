from types import SimpleNamespace

import numpy as np
import pytest

from retrieval_cerebellum.observer import (
    PrivilegedCerebellumObserver,
    PrivilegedObserverConfig,
    RelativePoseSlipTracker,
)
from retrieval_cerebellum.primitives import AssemblyPrimitiveSet, PriorSource


def _z_rotation(angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def test_relative_pose_tracker_ignores_shared_world_motion():
    tracker = RelativePoseSlipTracker()
    identity = np.eye(3)

    first = tracker.update(
        hand_position_world=np.zeros(3),
        hand_rotation_world=identity,
        object_position_world=np.array([0.1, 0.0, 0.0]),
        object_rotation_world=identity,
        timestamp_s=0.0,
        active=True,
    )
    second = tracker.update(
        hand_position_world=np.array([0.2, 0.0, 0.0]),
        hand_rotation_world=identity,
        object_position_world=np.array([0.3, 0.0, 0.0]),
        object_rotation_world=identity,
        timestamp_s=0.1,
        active=True,
    )

    assert first.translation_mps == pytest.approx(0.0)
    assert second.translation_mps == pytest.approx(0.0)
    assert second.rotation_radps == pytest.approx(0.0)


def test_relative_pose_tracker_detects_translation_and_rotation_slip():
    tracker = RelativePoseSlipTracker()
    identity = np.eye(3)
    tracker.update(
        hand_position_world=np.zeros(3),
        hand_rotation_world=identity,
        object_position_world=np.array([0.1, 0.0, 0.0]),
        object_rotation_world=identity,
        timestamp_s=0.0,
        active=True,
    )

    estimate = tracker.update(
        hand_position_world=np.zeros(3),
        hand_rotation_world=identity,
        object_position_world=np.array([0.11, 0.0, 0.0]),
        object_rotation_world=_z_rotation(0.1),
        timestamp_s=0.1,
        active=True,
    )

    assert estimate.translation_mps == pytest.approx(0.1)
    assert estimate.rotation_radps == pytest.approx(1.0)


class _FakeModel:
    def __init__(self) -> None:
        self._body_ids = {
            "peg": 0,
            "tray": 1,
            "allegro_palm_right": 2,
            "allegro_palm_left": 3,
        }

    def body(self, name):
        return SimpleNamespace(id=self._body_ids[name])


class _FakeLabeler:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.reset_calls = 0

    def reset_reference(self, raw_env) -> None:
        self.reset_calls += 1

    def compute(self, raw_env):
        return self.outcome


class _FakeProvider:
    names = SimpleNamespace(peg_body="peg", socket_body="tray")

    def snapshot(self, raw_env):
        return AssemblyPrimitiveSet(
            family_id="round_8mm",
            section="round",
            peg_tip_world=np.array([0.0, 0.0, 0.05]),
            peg_axis_world=np.array([0.0, 0.0, 1.0]),
            hole_entry_world=np.zeros(3),
            hole_axis_world=np.array([0.0, 0.0, 1.0]),
            hole_bottom_world=np.array([0.0, 0.0, -0.04]),
            nominal_peg_size_m=0.008,
            source=PriorSource.PRIVILEGED,
        )


def _fake_raw():
    positions = np.array(
        [
            [0.00, 0.00, 0.10],
            [0.20, 0.00, 0.10],
            [0.00, 0.00, 0.12],
            [0.20, 0.00, 0.12],
        ],
        dtype=np.float64,
    )
    return SimpleNamespace(
        _model=_FakeModel(),
        _data=SimpleNamespace(
            xpos=positions,
            xmat=np.array([np.eye(3) for _ in range(4)]),
            time=0.0,
        ),
    )


def test_privileged_observer_confirms_stable_grasp_over_time():
    outcome = SimpleNamespace(
        peg_ok=True,
        tray_ok=True,
        insert_ok=False,
        peg_contact_count=3,
        tray_contact_count=2,
    )
    raw = _fake_raw()
    labeler = _FakeLabeler(outcome)
    observer = PrivilegedCerebellumObserver(
        raw,
        config=PrivilegedObserverConfig(stable_confirm_frames=2),
        labeler=labeler,
        primitive_provider=_FakeProvider(),
    )
    observer.reset(raw)

    first = observer.observe(raw, np.zeros(44))
    raw._data.time = 0.02
    second = observer.observe(raw, np.zeros(44))

    assert not first.peg_grasp_stable
    assert not first.tray_grasp_stable
    assert second.peg_grasp_stable
    assert second.tray_grasp_stable
    assert second.slip_speed_mps == pytest.approx(0.0)
    assert second.rotation_slip_radps == pytest.approx(0.0)
    assert labeler.reset_calls == 1


def test_privileged_observer_uses_hysteresis_for_single_slip_spike():
    outcome = SimpleNamespace(
        peg_ok=True,
        tray_ok=True,
        insert_ok=False,
        peg_contact_count=3,
        tray_contact_count=2,
    )
    raw = _fake_raw()
    observer = PrivilegedCerebellumObserver(
        raw,
        config=PrivilegedObserverConfig(
            stable_confirm_frames=1,
            unstable_confirm_frames=2,
        ),
        labeler=_FakeLabeler(outcome),
        primitive_provider=_FakeProvider(),
    )
    observer.reset(raw)
    assert observer.observe(raw, np.zeros(44)).peg_grasp_stable

    raw._data.time = 0.02
    raw._data.xpos[0, 0] += 0.002
    one_spike = observer.observe(raw, np.zeros(44))
    assert one_spike.peg_grasp_stable

    raw._data.time = 0.04
    raw._data.xpos[0, 0] += 0.002
    second_spike = observer.observe(raw, np.zeros(44))
    assert not second_spike.peg_grasp_stable


def test_privileged_observer_marks_nearby_ungrasped_objects_pregrasp_ready():
    outcome = SimpleNamespace(
        peg_ok=False,
        tray_ok=False,
        insert_ok=False,
        peg_contact_count=0,
        tray_contact_count=0,
    )
    raw = _fake_raw()
    observer = PrivilegedCerebellumObserver(
        raw,
        config=PrivilegedObserverConfig(pregrasp_distance_m=0.05),
        labeler=_FakeLabeler(outcome),
        primitive_provider=_FakeProvider(),
    )
    observer.reset(raw)

    observation = observer.observe(raw, np.zeros(44))

    assert observation.peg_pregrasp_ready
    assert observation.tray_pregrasp_ready
    assert observation.slip_speed_mps is None
