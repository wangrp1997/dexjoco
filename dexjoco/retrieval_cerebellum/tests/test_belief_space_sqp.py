import numpy as np

from retrieval_cerebellum.belief_space_sqp import (
    BeliefSpaceSQPConfig,
    BimanualInsertionBelief,
    InsertionGeometry,
    LinearizedVisualObservationModel,
    LocalBimanualInsertionModel,
    RetrievedInsertionCandidate,
    RetrievalConditionedBeliefSpaceSQP,
)
from retrieval_cerebellum.v2_control import V2AssemblyEstimate


def test_sqp_accepts_six_dimensional_wrist_twists():
    right_jacobian = np.zeros((5, 6), dtype=np.float64)
    left_jacobian = np.zeros((5, 6), dtype=np.float64)
    right_jacobian[:, :5] = np.eye(5)
    left_jacobian[:, :5] = -np.eye(5)
    model = LocalBimanualInsertionModel(
        right_state_jacobian=right_jacobian,
        left_state_jacobian=left_jacobian,
        right_wrench_jacobian=np.zeros((6, 6)),
        left_wrench_jacobian=np.zeros((6, 6)),
    )
    candidate = RetrievedInsertionCandidate(
        skill_id="episode_000001@0",
        retrieval_distance=0.0,
        nominal_states=np.array([[0.0] * 5, [0.0, 0.0, 0.0, 0.0, 0.01]]),
        nominal_actions44=np.zeros((2, 44)),
        nominal_right_controls=np.zeros((1, 6)),
        nominal_left_controls=np.zeros((1, 6)),
        nominal_right_wrenches=np.zeros((1, 6)),
        nominal_left_wrenches=np.zeros((1, 6)),
        right_wrench_capacity=np.full(1, 10.0),
        left_wrench_capacity=np.full(1, 10.0),
        right_capacity_std=np.zeros(1),
        left_capacity_std=np.zeros(1),
        mean_state_disturbance=np.zeros((1, 5)),
        contact_modes=("insertion",),
    )
    belief = BimanualInsertionBelief(
        mean=np.zeros(5),
        covariance=np.eye(5) * 1e-12,
        attachment_process_covariance=np.eye(12) * 1e-12,
        attachment_to_state=np.concatenate([right_jacobian, left_jacobian], axis=1),
    )
    solver = RetrievalConditionedBeliefSpaceSQP(
        model,
        InsertionGeometry(radial_clearance_m=0.002, target_depth_m=0.01),
        BeliefSpaceSQPConfig(max_iterations=20),
    )

    result = solver.solve_candidate(belief, candidate)

    assert result.success
    assert result.right_controls.shape == (1, 6)
    assert result.left_controls.shape == (1, 6)
    np.testing.assert_allclose(result.states[-1, 4], 0.01, atol=1e-7)


def test_short_horizon_terminal_depth_can_stop_before_hole_bottom():
    geometry = InsertionGeometry(
        radial_clearance_m=0.002,
        target_depth_m=0.01,
        terminal_depth_m=-0.002,
    )

    assert geometry.planning_terminal_depth_m == -0.002


def test_local_state_drift_is_present_in_sqp_rollout():
    right_jacobian = np.zeros((5, 6), dtype=np.float64)
    left_jacobian = np.zeros((5, 6), dtype=np.float64)
    model = LocalBimanualInsertionModel(
        right_state_jacobian=right_jacobian,
        left_state_jacobian=left_jacobian,
        right_wrench_jacobian=np.zeros((6, 6)),
        left_wrench_jacobian=np.zeros((6, 6)),
        state_drift=np.array([0.0, 0.0, 0.0, 0.0, 0.001]),
    )
    candidate = RetrievedInsertionCandidate(
        skill_id="drift",
        retrieval_distance=0.0,
        nominal_states=np.zeros((2, 5)),
        nominal_actions44=np.zeros((2, 44)),
        nominal_right_controls=np.zeros((1, 6)),
        nominal_left_controls=np.zeros((1, 6)),
        nominal_right_wrenches=np.zeros((1, 6)),
        nominal_left_wrenches=np.zeros((1, 6)),
        right_wrench_capacity=np.full(1, 10.0),
        left_wrench_capacity=np.full(1, 10.0),
        right_capacity_std=np.zeros(1),
        left_capacity_std=np.zeros(1),
        mean_state_disturbance=np.zeros((1, 5)),
        contact_modes=("insertion",),
    )
    belief = BimanualInsertionBelief(
        mean=np.zeros(5),
        covariance=np.eye(5) * 1e-12,
        attachment_process_covariance=np.eye(12) * 1e-12,
        attachment_to_state=np.zeros((5, 12)),
    )
    solver = RetrievalConditionedBeliefSpaceSQP(
        model,
        InsertionGeometry(
            radial_clearance_m=0.002,
            target_depth_m=0.01,
            terminal_depth_m=0.001,
        ),
        BeliefSpaceSQPConfig(max_iterations=20),
    )

    result = solver.solve_candidate(belief, candidate)

    assert result.success
    np.testing.assert_allclose(result.states[-1, 4], 0.001, atol=1e-8)


def test_preentry_alignment_uses_funnel_before_hole_wall_constraint():
    right_jacobian = np.zeros((5, 6), dtype=np.float64)
    left_jacobian = np.zeros((5, 6), dtype=np.float64)
    candidate = RetrievedInsertionCandidate(
        skill_id="preentry",
        retrieval_distance=0.0,
        nominal_states=np.array(
            [[0.003, 0.0, 0.02, 0.0, -0.02], [0.003, 0.0, 0.02, 0.0, -0.019]]
        ),
        nominal_actions44=np.zeros((2, 44)),
        nominal_right_controls=np.zeros((1, 6)),
        nominal_left_controls=np.zeros((1, 6)),
        nominal_right_wrenches=np.zeros((1, 6)),
        nominal_left_wrenches=np.zeros((1, 6)),
        right_wrench_capacity=np.full(1, 10.0),
        left_wrench_capacity=np.full(1, 10.0),
        right_capacity_std=np.zeros(1),
        left_capacity_std=np.zeros(1),
        mean_state_disturbance=np.zeros((1, 5)),
        contact_modes=("approach",),
    )
    belief = BimanualInsertionBelief(
        mean=candidate.nominal_states[0],
        covariance=np.eye(5) * 1e-12,
        attachment_process_covariance=np.eye(12) * 1e-12,
        attachment_to_state=np.zeros((5, 12)),
    )
    model = LocalBimanualInsertionModel(
        right_state_jacobian=right_jacobian,
        left_state_jacobian=left_jacobian,
        right_wrench_jacobian=np.zeros((6, 6)),
        left_wrench_jacobian=np.zeros((6, 6)),
        state_drift=np.array([0.0, 0.0, 0.0, 0.0, 0.001]),
    )
    result = RetrievalConditionedBeliefSpaceSQP(
        model,
        InsertionGeometry(
            radial_clearance_m=0.002,
            target_depth_m=0.01,
            terminal_depth_m=-0.019,
        ),
    ).solve_candidate(belief, candidate)

    assert result.success
    assert result.feasible


def test_visual_estimate_builds_information_update_for_sqp():
    estimate = V2AssemblyEstimate(
        timestamp_s=1.0,
        mean5=np.zeros(5),
        covariance5=np.diag([0.001, 0.001, 0.004, 0.004, 0.002]),
        hole_rotation_world=np.eye(3),
        visual_reliability=0.8,
    )
    observation_model = LinearizedVisualObservationModel.from_visual_estimate(
        estimate
    )
    belief = BimanualInsertionBelief.from_visual_estimate(
        estimate,
        attachment_process_covariance=np.eye(12) * 1e-6,
        attachment_to_state=np.zeros((5, 12)),
    )
    predicted = np.diag([0.01, 0.01, 0.02, 0.02, 0.01])
    posterior = observation_model.posterior_covariance(
        predicted,
        np.zeros(6),
        np.zeros(6),
    )

    np.testing.assert_allclose(belief.mean, estimate.mean5)
    np.testing.assert_allclose(belief.covariance, estimate.covariance5)
    assert np.linalg.eigvalsh(predicted - posterior).min() >= -1e-12
    assert np.trace(posterior) < np.trace(predicted)


def test_sqp_uses_control_dependent_visual_reliability():
    zero_jacobian = np.zeros((5, 6), dtype=np.float64)
    model = LocalBimanualInsertionModel(
        right_state_jacobian=zero_jacobian,
        left_state_jacobian=zero_jacobian,
        right_wrench_jacobian=np.zeros((6, 6)),
        left_wrench_jacobian=np.zeros((6, 6)),
        state_drift=np.asarray([0.0, 0.0, 0.0, 0.0, 0.001]),
    )
    candidate = RetrievedInsertionCandidate(
        skill_id="active_view",
        retrieval_distance=0.0,
        nominal_states=np.asarray(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.001]]
        ),
        nominal_actions44=np.zeros((2, 44)),
        nominal_right_controls=np.zeros((1, 6)),
        nominal_left_controls=np.zeros((1, 6)),
        nominal_right_wrenches=np.zeros((1, 6)),
        nominal_left_wrenches=np.zeros((1, 6)),
        right_wrench_capacity=np.full(1, 10.0),
        left_wrench_capacity=np.full(1, 10.0),
        right_capacity_std=np.zeros(1),
        left_capacity_std=np.zeros(1),
        mean_state_disturbance=np.zeros((1, 5)),
        contact_modes=("approach",),
    )
    belief = BimanualInsertionBelief(
        mean=np.zeros(5),
        covariance=np.diag([0.01, 0.01, 0.02, 0.02, 0.01]),
        attachment_process_covariance=np.eye(12) * 1e-8,
        attachment_to_state=np.zeros((5, 12)),
    )
    reliability_jacobian = np.zeros(12)
    reliability_jacobian[0] = 100.0
    observation_model = LinearizedVisualObservationModel(
        observation_covariance=np.diag([0.001, 0.001, 0.004, 0.004, 0.002]),
        reliability=0.1,
        control_reliability_jacobian=reliability_jacobian,
    )
    result = RetrievalConditionedBeliefSpaceSQP(
        model,
        InsertionGeometry(
            radial_clearance_m=1.0,
            target_depth_m=0.002,
            terminal_depth_m=0.001,
        ),
        BeliefSpaceSQPConfig(
            covariance_weight=1.0,
            correction_weight=0.01,
            max_iterations=100,
            maximum_preentry_tilt_rad=1.0,
            use_candidate_transition_residual=False,
        ),
        visual_observation_model=observation_model,
    ).solve_candidate(belief, candidate)

    assert result.success
    assert result.right_controls[0, 0] > 0.0, (
        result.message,
        result.iterations,
        result.objective,
        result.right_corrections,
    )
    achieved_reliability = observation_model.predicted_reliability(
        result.right_controls[0],
        result.left_controls[0],
    )
    assert achieved_reliability > observation_model.reliability
    assert np.trace(result.covariances[0]) < np.trace(belief.covariance)
