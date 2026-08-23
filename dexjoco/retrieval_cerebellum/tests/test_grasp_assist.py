from dataclasses import replace

import numpy as np

from retrieval_cerebellum.control import CerebellumObservation
from retrieval_cerebellum.grasp_assist import (
    AsymmetricGraspAssist,
    AsymmetricGraspAssistConfig,
)
from retrieval_cerebellum.primitives import AssemblyPrimitiveSet, PriorSource


def _observation(state44: np.ndarray | None = None, **changes) -> CerebellumObservation:
    primitives = AssemblyPrimitiveSet(
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
    base = CerebellumObservation(
        state44=np.zeros(44) if state44 is None else state44,
        primitives=primitives,
        peg_grasped=False,
        tray_grasped=False,
    )
    return replace(base, **changes)


def test_right_grasp_hold_preserves_both_policy_arms_and_assists_left_fingers():
    state = np.linspace(-0.1, 0.1, 44, dtype=np.float32)
    policy_action = state.copy()
    policy_action[22:28] = np.linspace(0.2, 0.3, 6, dtype=np.float32)
    observation = _observation(
        state,
        peg_grasped=True,
        peg_grasp_stable=True,
        tray_pregrasp_ready=True,
    )
    skill = AsymmetricGraspAssist(
        AsymmetricGraspAssistConfig(closure_step_rad=0.05)
    )

    merged = skill.step(observation, policy_action)

    np.testing.assert_allclose(merged[:6], policy_action[:6])
    np.testing.assert_allclose(merged[6:22], state[6:22])
    np.testing.assert_allclose(merged[22:28], policy_action[22:28])
    assert np.any(merged[28:44] > policy_action[28:44])
    assert skill.last_diagnostics.held_side == "right"
    assert skill.last_diagnostics.assisted_side == "left"
    assert skill.last_diagnostics.assisting_fingers


def test_assist_waits_for_pregrasp_before_changing_missing_hand():
    state = np.zeros(44, dtype=np.float32)
    policy_action = np.full(44, 0.1, dtype=np.float32)
    observation = _observation(
        state,
        tray_grasped=True,
        tray_grasp_stable=True,
        peg_pregrasp_ready=False,
    )
    skill = AsymmetricGraspAssist()

    merged = skill.step(observation, policy_action)

    np.testing.assert_allclose(merged[22:28], policy_action[22:28])
    np.testing.assert_allclose(merged[28:44], state[28:44])
    np.testing.assert_allclose(merged[:22], policy_action[:22])
    assert not skill.last_diagnostics.assisting_fingers


def test_assist_releases_after_both_grasps_become_stable():
    policy_action = np.full(44, 0.2, dtype=np.float32)
    skill = AsymmetricGraspAssist()
    active = _observation(
        peg_grasped=True,
        peg_grasp_stable=True,
        tray_pregrasp_ready=True,
    )
    skill.step(active, policy_action)
    completed = replace(
        active,
        tray_grasped=True,
        tray_grasp_stable=True,
    )

    merged = skill.step(completed, policy_action)

    np.testing.assert_allclose(merged, policy_action)
    assert not skill.active
    assert skill.last_diagnostics.outcome == "completed"
    assert "completed=1" in skill.episode_summary()

    lost_again = replace(
        active,
        peg_grasped=False,
        peg_grasp_stable=False,
        tray_grasped=True,
        tray_grasp_stable=True,
    )
    skill.step(lost_again, policy_action)
    assert not skill.active


def test_assist_aborts_when_held_grasp_is_lost():
    policy_action = np.full(44, 0.2, dtype=np.float32)
    skill = AsymmetricGraspAssist()
    active = _observation(
        peg_grasped=True,
        peg_grasp_stable=True,
        tray_pregrasp_ready=True,
    )
    skill.step(active, policy_action)

    merged = skill.step(
        replace(active, peg_grasped=False, peg_grasp_stable=False),
        policy_action,
    )

    np.testing.assert_allclose(merged, policy_action)
    assert not skill.active
    assert skill.last_diagnostics.outcome == "held_grasp_lost"


def test_assist_waits_for_cooldown_before_reactivation():
    policy_action = np.full(44, 0.2, dtype=np.float32)
    config = AsymmetricGraspAssistConfig(reactivation_cooldown_frames=2)
    skill = AsymmetricGraspAssist(config)
    active = _observation(
        peg_grasped=True,
        peg_grasp_stable=True,
        tray_pregrasp_ready=True,
    )
    skill.step(active, policy_action)
    skill.step(replace(active, peg_grasped=False, peg_grasp_stable=False), policy_action)

    merged = skill.step(active, policy_action)
    assert merged.shape == (44,)
    assert not skill.active
    skill.step(active, policy_action)
    assert skill.active
