import numpy as np
import pytest

from retrieval_cerebellum.control import CerebellumMode, CerebellumObservation, HandoffDecision
from retrieval_cerebellum.primitives import AssemblyPrimitiveSet, ContactRegion, PriorSource


def _primitive_set() -> AssemblyPrimitiveSet:
    return AssemblyPrimitiveSet(
        family_id="round_8mm",
        section="round",
        peg_tip_world=np.array([0.003, 0.004, 0.020]),
        peg_axis_world=np.array([0.0, 0.0, -2.0]),
        hole_entry_world=np.zeros(3),
        hole_axis_world=np.array([0.0, 0.0, 1.0]),
        hole_bottom_world=np.array([0.0, 0.0, -0.04]),
        nominal_peg_size_m=0.008,
        source=PriorSource.PRIVILEGED,
    )


def test_assembly_primitive_geometry_features():
    primitives = _primitive_set()

    assert primitives.lateral_error_m == pytest.approx(0.005)
    assert primitives.axis_error_rad == pytest.approx(0.0)
    assert primitives.approach_height_m == pytest.approx(0.020)
    assert primitives.insertion_depth_m == pytest.approx(0.0)
    assert primitives.target_depth_m == pytest.approx(0.04)
    np.testing.assert_allclose(
        primitives.feature_vector(),
        [0.005, 0.0, 0.020, 0.0, 0.04, 0.008, -1.0],
    )


def test_contact_region_normalizes_and_freezes_vectors():
    region = ContactRegion(
        finger="thumb",
        center_object=np.array([0.01, 0.0, 0.0]),
        normal_object=np.array([0.0, 0.0, 3.0]),
        radius_m=0.005,
    )

    np.testing.assert_allclose(region.normal_object, [0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        region.center_object[0] = 0.0


def test_cerebellum_observation_validates_action_layout():
    observation = CerebellumObservation(
        state44=np.zeros(44),
        primitives=_primitive_set(),
        peg_grasped=True,
        tray_grasped=True,
    )
    assert observation.state44.dtype == np.float32

    with pytest.raises(ValueError, match=r"shape \(44,\)"):
        CerebellumObservation(
            state44=np.zeros(43),
            primitives=_primitive_set(),
            peg_grasped=True,
            tray_grasped=True,
        )


def test_handoff_decision_validates_confidence():
    decision = HandoffDecision(CerebellumMode.ALIGN, "geometry ready", 0.8)
    assert decision.mode is CerebellumMode.ALIGN

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        HandoffDecision(CerebellumMode.ALIGN, "bad confidence", 1.1)
