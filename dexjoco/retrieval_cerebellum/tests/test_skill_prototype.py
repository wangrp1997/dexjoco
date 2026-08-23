import numpy as np

from retrieval_cerebellum.learning_data import state46_to_action44
from retrieval_cerebellum.skill_prototype import (
    RetrievalAugmentedSkillPrototype,
    SuccessfulSkillMemory,
    SuccessfulSkillTrajectory,
    belief_to_retrieval_descriptor,
)


def _trajectory(episode_index: int, offset: float) -> SuccessfulSkillTrajectory:
    state = np.zeros((6, 46), dtype=np.float32)
    state[:, 3] = 1.0
    state[:, 10] = 1.0
    state[:, 0] = offset
    state[:, 7] = -offset
    proprio = state46_to_action44(state)
    action = proprio.copy()
    action[:, 0] += np.linspace(0.0, 0.05, 6)
    action[:, 22] -= np.linspace(0.0, 0.03, 6)
    descriptor = np.zeros(14, dtype=np.float32)
    descriptor[0] = offset
    descriptor[-2:] = [0.0125, 0.008]
    return SuccessfulSkillTrajectory(
        episode_index=episode_index,
        family_id="round_8mm",
        split="train",
        descriptor=descriptor,
        state46=state,
        proprio_action44=proprio,
        demo_action44=action,
    )


def test_belief_descriptor_uses_hand_object_blocks():
    belief = np.arange(18, dtype=np.float32)

    descriptor = belief_to_retrieval_descriptor(
        belief,
        target_depth_m=0.0125,
        nominal_peg_size_m=0.008,
    )

    np.testing.assert_allclose(descriptor[:6], belief[6:12])
    np.testing.assert_allclose(descriptor[6:12], belief[12:18])


def test_prototype_retrieves_and_projects_motion_limits():
    memory = SuccessfulSkillMemory([_trajectory(1, 0.0), _trajectory(2, 1.0)])
    prototype = RetrievalAugmentedSkillPrototype(memory)
    state = _trajectory(9, 0.02).state46[0]
    belief = np.zeros(18, dtype=np.float32)
    belief[6] = 0.01

    plan = prototype.plan(belief, state, family_id="round_8mm", horizon=6)

    assert plan.source_episode_index == 1
    current = state46_to_action44(state)
    previous = np.concatenate([current[None], plan.adapted_actions44], axis=0)
    right_steps = np.linalg.norm(np.diff(previous[:, :3], axis=0), axis=1)
    left_steps = np.linalg.norm(np.diff(previous[:, 22:25], axis=0), axis=1)
    assert np.all(right_steps <= memory.limits.right_position_step_m + 1e-7)
    assert np.all(left_steps <= memory.limits.left_position_step_m + 1e-7)
    assert right_steps[0] <= memory.limits.right_position_step_m + 1e-7
    assert left_steps[0] <= memory.limits.left_position_step_m + 1e-7
