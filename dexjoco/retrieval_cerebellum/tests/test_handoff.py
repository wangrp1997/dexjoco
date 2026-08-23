from dataclasses import replace

import numpy as np

from retrieval_cerebellum.control import CerebellumMode, CerebellumObservation
from retrieval_cerebellum.handoff import RuleBasedHandoffPolicy
from retrieval_cerebellum.primitives import AssemblyPrimitiveSet, PriorSource


def _primitives(*, lateral_m: float = 0.0, height_m: float = 0.05) -> AssemblyPrimitiveSet:
    return AssemblyPrimitiveSet(
        family_id="round_8mm",
        section="round",
        peg_tip_world=np.array([lateral_m, 0.0, height_m]),
        peg_axis_world=np.array([0.0, 0.0, 1.0]),
        hole_entry_world=np.zeros(3),
        hole_axis_world=np.array([0.0, 0.0, 1.0]),
        hole_bottom_world=np.array([0.0, 0.0, -0.04]),
        nominal_peg_size_m=0.008,
        source=PriorSource.PRIVILEGED,
    )


def _observation(**changes) -> CerebellumObservation:
    base = CerebellumObservation(
        state44=np.zeros(44),
        primitives=_primitives(),
        peg_grasped=False,
        tray_grasped=False,
    )
    return replace(base, **changes)


def test_rule_policy_covers_grasp_pipeline():
    policy = RuleBasedHandoffPolicy()

    assert policy.decide(_observation()).mode is CerebellumMode.VLA_GRASP
    assert policy.decide(_observation(peg_pregrasp_ready=True)).mode is CerebellumMode.VLA_GRASP
    assert policy.decide(_observation(peg_contact_count=2)).mode is CerebellumMode.VLA_GRASP
    assert policy.decide(_observation(peg_grasped=True)).mode is CerebellumMode.VLA_GRASP


def test_rule_policy_requests_asymmetric_grasp_assist():
    policy = RuleBasedHandoffPolicy()

    peg_held = _observation(peg_grasped=True, peg_grasp_stable=True)
    assert policy.decide(peg_held).mode is CerebellumMode.GRASP_ASSIST

    policy.reset()
    tray_held = _observation(tray_grasped=True, tray_grasp_stable=True)
    assert policy.decide(tray_held).mode is CerebellumMode.GRASP_ASSIST


def test_rule_policy_transports_then_aligns_then_inserts():
    policy = RuleBasedHandoffPolicy()
    grasped = dict(
        peg_grasped=True,
        tray_grasped=True,
        peg_grasp_stable=True,
        tray_grasp_stable=True,
    )

    transport = _observation(primitives=_primitives(lateral_m=0.08, height_m=0.15), **grasped)
    assert policy.decide(transport).mode is CerebellumMode.TRANSPORT

    align = _observation(primitives=_primitives(lateral_m=0.02), **grasped)
    assert policy.decide(align).mode is CerebellumMode.ALIGN

    insert = _observation(primitives=_primitives(lateral_m=0.002), **grasped)
    assert policy.decide(insert).mode is CerebellumMode.INSERT


def test_rule_policy_remembers_grasp_loss():
    policy = RuleBasedHandoffPolicy()
    stable = _observation(
        peg_grasped=True,
        tray_grasped=True,
        peg_grasp_stable=True,
        tray_grasp_stable=True,
    )
    policy.decide(stable)

    lost = replace(stable, peg_grasped=False, peg_grasp_stable=False)
    for _ in range(policy.config.grasp_lost_confirm_frames - 1):
        decision = policy.decide(lost)
        assert decision.mode is CerebellumMode.VLA_GRASP
    decision = policy.decide(lost)

    assert decision.mode is CerebellumMode.VLA_REGRASP
    assert "peg" in decision.reason


def test_rule_policy_uses_slip_for_stabilize_and_regrasp():
    policy = RuleBasedHandoffPolicy()
    stable = _observation(
        peg_grasped=True,
        tray_grasped=True,
        peg_grasp_stable=True,
        tray_grasp_stable=True,
        slip_speed_mps=0.05,
    )
    assert policy.decide(stable).mode is CerebellumMode.GRASP_STABILIZE

    severe_slip = replace(stable, slip_speed_mps=0.10)
    for _ in range(policy.config.regrasp_slip_confirm_frames - 1):
        assert policy.decide(severe_slip).mode is CerebellumMode.GRASP_STABILIZE
    assert policy.decide(severe_slip).mode is CerebellumMode.VLA_REGRASP

    policy.reset()
    rotation_slip = replace(stable, slip_speed_mps=0.0, rotation_slip_radps=0.8)
    assert policy.decide(rotation_slip).mode is CerebellumMode.GRASP_STABILIZE

    severe_rotation_slip = replace(rotation_slip, rotation_slip_radps=2.0)
    for _ in range(policy.config.regrasp_slip_confirm_frames - 1):
        assert policy.decide(severe_rotation_slip).mode is CerebellumMode.GRASP_STABILIZE
    assert policy.decide(severe_rotation_slip).mode is CerebellumMode.VLA_REGRASP


def test_insert_contact_has_highest_priority():
    policy = RuleBasedHandoffPolicy()
    observation = _observation(insert_contact=True, slip_speed_mps=0.10)

    assert policy.decide(observation).mode is CerebellumMode.COMPLETE
