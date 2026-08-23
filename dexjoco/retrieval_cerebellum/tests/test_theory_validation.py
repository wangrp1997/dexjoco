import numpy as np

from retrieval_cerebellum.theory_validation import (
    covariance_quadratic_cost,
    contact_sdf_jacobian,
    ellipsoid_linear_support,
    grasp_internal_force_nullspace,
    information_form_posterior_covariance,
    invert_transform,
    pbvs_linear_rollout,
    pbvs_se3_rollout,
    propagate_relative_covariance,
    relative_right_error_jacobian,
    robust_pbvs_lyapunov_upper_bound,
    se3_exp,
    se3_log,
    visual_keypoint_jacobian,
)


def test_se3_exp_log_round_trip() -> None:
    twist = np.asarray([0.12, -0.05, 0.08, 0.2, -0.1, 0.15])
    assert np.allclose(se3_log(se3_exp(twist)), twist, atol=1e-10)


def test_relative_right_error_jacobian_matches_finite_difference() -> None:
    relative_transform = se3_exp(
        np.asarray([0.18, -0.06, 0.11, 0.25, -0.18, 0.09])
    )
    right_direction = np.asarray([0.3, -0.2, 0.1, -0.15, 0.12, 0.2])
    left_direction = np.asarray([-0.1, 0.25, -0.2, 0.18, 0.08, -0.12])
    perturbation_scale = 1e-6
    right_error = perturbation_scale * right_direction
    left_error = perturbation_scale * left_direction
    perturbed_relative = (
        se3_exp(-left_error) @ relative_transform @ se3_exp(right_error)
    )
    measured_right_error = se3_log(
        invert_transform(relative_transform) @ perturbed_relative
    )
    error_jacobian = relative_right_error_jacobian(relative_transform)
    predicted_right_error = error_jacobian @ np.concatenate(
        [right_error, left_error]
    )
    assert np.allclose(measured_right_error, predicted_right_error, atol=2e-12)


def test_perfect_state_pbvs_converges_monotonically() -> None:
    initial_error = np.asarray([0.03, -0.02, 0.01, 0.15, -0.1, 0.08])
    gain = np.asarray([2.0, 2.0, 1.5, 1.2, 1.2, 1.0])
    trace = pbvs_linear_rollout(initial_error, gain, steps=400, time_step=0.01)
    norms = np.linalg.norm(trace, axis=1)
    assert np.all(np.diff(norms) < 0.0)
    assert norms[-1] < 0.03 * norms[0]


def test_pbvs_converges_to_estimator_bias_and_slip_neighborhood() -> None:
    initial_error = np.asarray([0.02, -0.01, 0.015, 0.08, -0.04, 0.05])
    gain = np.asarray([2.0, 1.5, 1.8, 1.2, 1.0, 1.4])
    estimate_bias = np.asarray([0.003, -0.002, 0.001, 0.01, -0.008, 0.006])
    slip_disturbance = np.asarray([0.001, 0.0, -0.0005, 0.002, 0.0, -0.001])
    trace = pbvs_linear_rollout(
        initial_error,
        gain,
        steps=3000,
        time_step=0.01,
        estimate_bias6=estimate_bias,
        slip_disturbance6=slip_disturbance,
    )
    expected_equilibrium = -estimate_bias + slip_disturbance / gain
    assert np.allclose(trace[-1], expected_equilibrium, atol=1e-8)


def test_relative_covariance_propagation_matches_monte_carlo() -> None:
    relative_transform = se3_exp(np.asarray([0.1, -0.05, 0.08, 0.2, -0.1, 0.15]))
    pose_covariance = np.diag(
        np.asarray(
            [0.02, 0.03, 0.01, 0.04, 0.05, 0.02, 0.03, 0.02, 0.01, 0.06, 0.04, 0.03]
        )
    )
    propagated = propagate_relative_covariance(pose_covariance, relative_transform)

    rng = np.random.default_rng(0)
    samples = []
    for _ in range(5000):
        delta = rng.multivariate_normal(np.zeros(12), pose_covariance)
        right_error = delta[:6]
        left_error = delta[6:]
        perturbed_relative = (
            se3_exp(-left_error) @ relative_transform @ se3_exp(right_error)
        )
        samples.append(
            se3_log(invert_transform(relative_transform) @ perturbed_relative)
        )
    empirical = np.cov(np.asarray(samples).T)
    assert np.allclose(propagated, empirical, atol=0.01)


def test_visual_keypoint_jacobian_matches_finite_difference() -> None:
    camera_from_world = se3_exp(np.asarray([0.0, 0.0, 0.0, -0.2, 0.1, 0.0]))
    world_from_palm = se3_exp(np.asarray([0.4, -0.1, 0.2, 0.3, -0.2, 0.15]))
    in_hand_transform = se3_exp(np.asarray([0.05, 0.02, -0.01, 0.1, 0.05, -0.08]))
    keypoint_local = np.asarray([0.0, 0.0, 0.0135])

    analytic = visual_keypoint_jacobian(
        camera_from_world,
        world_from_palm,
        in_hand_transform,
        keypoint_local,
    )
    direction = np.asarray([0.2, -0.1, 0.05, 0.15, -0.1, 0.08])
    perturbation_scale = 1e-6
    perturbation = perturbation_scale * direction
    perturbed_in_hand = in_hand_transform @ se3_exp(perturbation)
    world_from_object = world_from_palm @ in_hand_transform
    world_from_object_perturbed = world_from_palm @ perturbed_in_hand
    point_world = world_from_object[:3, :3] @ keypoint_local + world_from_object[:3, 3]
    point_world_perturbed = (
        world_from_object_perturbed[:3, :3] @ keypoint_local
        + world_from_object_perturbed[:3, 3]
    )
    point = camera_from_world @ np.append(point_world, 1.0)
    point_perturbed = camera_from_world @ np.append(point_world_perturbed, 1.0)
    projected = point[:2] / point[2]
    projected_perturbed = point_perturbed[:2] / point_perturbed[2]
    measured = (projected_perturbed - projected) / perturbation_scale
    assert np.allclose(analytic @ direction, measured, atol=5e-5)


def test_contact_sdf_jacobian_matches_finite_difference() -> None:
    world_from_object = se3_exp(np.asarray([0.2, -0.05, 0.1, 0.25, -0.1, 0.12]))
    fingertip_world = world_from_object[:3, :3] @ np.asarray([0.0, 0.0, 0.02]) + world_from_object[:3, 3]
    surface_gradient = np.asarray([0.0, 0.0, 1.0])

    analytic = contact_sdf_jacobian(
        world_from_object,
        fingertip_world,
        surface_gradient,
    )
    direction = np.asarray([0.1, -0.05, 0.02, 0.12, -0.08, 0.06])
    perturbation_scale = 1e-6
    perturbation = perturbation_scale * direction
    perturbed_object = world_from_object @ se3_exp(perturbation)

    def sdf_value(transform4: np.ndarray) -> float:
        object_from_world_local = invert_transform(transform4)
        point_object_local = (
            object_from_world_local[:3, :3] @ fingertip_world
            + object_from_world_local[:3, 3]
        )
        return float(surface_gradient @ point_object_local)

    value = sdf_value(world_from_object)
    value_perturbed = sdf_value(perturbed_object)
    measured = (value_perturbed - value) / perturbation_scale
    assert np.allclose(analytic @ direction, measured, atol=5e-5)


def test_grasp_internal_force_nullspace_annihilates_object_wrench() -> None:
    grasp_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.1, 0.0, 0.0, -0.1, 0.0],
            [0.1, 0.0, 0.0, -0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    nullspace = grasp_internal_force_nullspace(grasp_matrix)
    assert nullspace.shape[1] >= 1
    internal_force = nullspace[:, 0]
    assert np.allclose(grasp_matrix @ internal_force, np.zeros(6), atol=1e-10)


def test_pbvs_se3_rollout_reaches_bias_slip_equilibrium() -> None:
    desired = np.eye(4)
    initial = se3_exp(np.asarray([0.03, -0.02, 0.01, 0.12, -0.08, 0.06]))
    relative_jacobian = np.eye(6)
    estimate_bias = np.asarray([0.004, -0.002, 0.001, 0.01, -0.006, 0.005])
    slip_disturbance = np.asarray([0.001, 0.0, -0.0005, 0.002, 0.0, -0.001])
    gain = np.asarray([2.0, 1.8, 1.5, 1.2, 1.0, 1.4])
    error_trace, _ = pbvs_se3_rollout(
        desired,
        initial,
        relative_jacobian,
        steps=4000,
        time_step=0.01,
        gain6=gain,
        estimate_bias6=estimate_bias,
        slip_disturbance6=slip_disturbance,
    )
    expected = -estimate_bias + slip_disturbance / gain
    assert np.allclose(error_trace[-1], expected, atol=5e-3)


def test_multifinger_information_monotonically_reduces_task_covariance() -> None:
    prior = np.diag(np.asarray([0.08, 0.05, 0.04, 0.07, 0.06, 0.03]))
    visual_jacobian = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.2, 0.0],
            [0.0, 1.0, 0.0, -0.2, 0.0, 0.0],
        ]
    )
    finger_jacobians = [
        np.asarray([[0.0, 0.0, 1.0, 0.0, -0.03, 0.02]]),
        np.asarray([[0.0, 1.0, 0.0, 0.04, 0.0, -0.03]]),
        np.asarray([[1.0, 0.0, 0.0, 0.0, 0.05, -0.02]]),
    ]
    visual_only = information_form_posterior_covariance(
        prior,
        [visual_jacobian],
        [np.eye(2) * 0.02],
    )
    multifinger = information_form_posterior_covariance(
        prior,
        [visual_jacobian, *finger_jacobians],
        [np.eye(2) * 0.02, *[np.eye(1) * 0.005 for _ in finger_jacobians]],
    )
    covariance_reduction = visual_only - multifinger
    assert np.linalg.eigvalsh(covariance_reduction).min() >= -1e-12
    task_map = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    assert np.trace(task_map @ multifinger @ task_map.T) < np.trace(
        task_map @ visual_only @ task_map.T
    )


def test_pbvs_covariance_cost_matches_monte_carlo() -> None:
    covariance = np.diag(np.asarray([0.004, 0.002, 0.003, 0.01, 0.008, 0.006]))
    gain = np.diag(np.asarray([2.0, 1.8, 1.5, 1.2, 1.0, 1.4]))
    task_weight = np.diag(np.asarray([4.0, 4.0, 2.0, 1.5, 1.5, 1.0]))
    time_step = 0.01
    closed_loop_weight = time_step**2 * gain.T @ task_weight @ gain
    predicted = covariance_quadratic_cost(covariance, closed_loop_weight)

    rng = np.random.default_rng(3)
    errors = rng.multivariate_normal(np.zeros(6), covariance, size=200000)
    injected = -time_step * (errors @ gain.T)
    empirical = np.mean(np.einsum("bi,ij,bj->b", injected, task_weight, injected))
    assert np.isclose(empirical, predicted, rtol=0.01)


def test_ellipsoid_support_formula_has_constructive_maximizer() -> None:
    covariance = np.asarray(
        [
            [0.04, 0.01, 0.0],
            [0.01, 0.03, 0.005],
            [0.0, 0.005, 0.02],
        ]
    )
    direction = np.asarray([0.7, -0.2, 0.5])
    confidence_radius_squared = 7.815
    support = ellipsoid_linear_support(
        direction,
        covariance,
        confidence_radius_squared,
    )
    maximizer = (
        np.sqrt(confidence_radius_squared)
        * covariance
        @ direction
        / np.sqrt(direction @ covariance @ direction)
    )
    ellipsoid_value = maximizer @ np.linalg.solve(covariance, maximizer)
    assert np.isclose(ellipsoid_value, confidence_radius_squared)
    assert np.isclose(direction @ maximizer, support)


def test_robust_lyapunov_bound_dominates_sampled_uncertainty() -> None:
    estimated_error = np.asarray([0.02, -0.01, 0.015, 0.04, -0.03])
    relative_error_rate = np.asarray([-0.03, 0.012, -0.02, -0.025, 0.018])
    task_weight = np.diag(np.asarray([5.0, 5.0, 3.0, 2.0, 2.0]))
    covariance = np.diag(np.asarray([0.001, 0.0008, 0.0012, 0.003, 0.002]))
    confidence_radius_squared = 11.07
    slip_bound = 0.004
    upper_bound = robust_pbvs_lyapunov_upper_bound(
        estimated_error,
        relative_error_rate,
        task_weight,
        covariance,
        confidence_radius_squared=confidence_radius_squared,
        slip_bound=slip_bound,
    )

    rng = np.random.default_rng(5)
    covariance_root = np.linalg.cholesky(covariance)
    sampled_rates = []
    for _ in range(100000):
        estimation_direction = rng.normal(size=5)
        estimation_direction /= np.linalg.norm(estimation_direction)
        estimation_radius = rng.random() ** (1.0 / 5.0)
        estimation_error = (
            np.sqrt(confidence_radius_squared)
            * covariance_root
            @ (estimation_radius * estimation_direction)
        )
        slip_direction = rng.normal(size=5)
        slip_direction /= np.linalg.norm(slip_direction)
        slip = slip_bound * rng.random() ** (1.0 / 5.0) * slip_direction
        true_error = estimated_error - estimation_error
        sampled_rates.append(
            true_error @ task_weight @ (relative_error_rate + slip)
        )
    assert max(sampled_rates) <= upper_bound + 1e-12
