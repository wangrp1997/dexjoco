"""Numerical checks for the SE(3) IEKF-PBVS methodology derivation."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def skew(vector3: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector3, dtype=np.float64).reshape(3)
    return np.asarray(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )


def se3_exp(twist6: np.ndarray) -> np.ndarray:
    twist = np.asarray(twist6, dtype=np.float64).reshape(6)
    translation = twist[:3]
    rotation_vector = twist[3:]
    angle = float(np.linalg.norm(rotation_vector))
    rotation_skew = skew(rotation_vector)
    if angle < 1e-8:
        left_jacobian = (
            np.eye(3)
            + 0.5 * rotation_skew
            + (1.0 / 6.0) * rotation_skew @ rotation_skew
        )
    else:
        left_jacobian = (
            np.eye(3)
            + (1.0 - np.cos(angle)) / angle**2 * rotation_skew
            + (angle - np.sin(angle)) / angle**3 * rotation_skew @ rotation_skew
        )
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
    transform[:3, 3] = left_jacobian @ translation
    return transform


def se3_log(transform4: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform4, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"transform4 must have shape (4, 4), got {transform.shape}")
    rotation_vector = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    angle = float(np.linalg.norm(rotation_vector))
    rotation_skew = skew(rotation_vector)
    if angle < 1e-8:
        inverse_left_jacobian = (
            np.eye(3)
            - 0.5 * rotation_skew
            + (1.0 / 12.0) * rotation_skew @ rotation_skew
        )
    else:
        coefficient = (
            1.0 / angle**2
            - (1.0 + np.cos(angle)) / (2.0 * angle * np.sin(angle))
        )
        inverse_left_jacobian = (
            np.eye(3)
            - 0.5 * rotation_skew
            + coefficient * rotation_skew @ rotation_skew
        )
    translation = inverse_left_jacobian @ transform[:3, 3]
    return np.concatenate([translation, rotation_vector])


def invert_transform(transform4: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform4, dtype=np.float64)
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def adjoint(transform4: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3:] = skew(translation) @ rotation
    result[3:, 3:] = rotation
    return result


def relative_right_error_jacobian(relative_transform: np.ndarray) -> np.ndarray:
    """Map right/left in-hand right errors to relative-pose right error."""
    return np.concatenate(
        [
            np.eye(6),
            -adjoint(invert_transform(relative_transform)),
        ],
        axis=1,
    )


def propagate_relative_covariance(
    pose_covariance12x12: np.ndarray,
    relative_transform: np.ndarray,
) -> np.ndarray:
    """Propagate bilateral in-hand covariance to relative-pose covariance."""
    covariance = np.asarray(pose_covariance12x12, dtype=np.float64)
    if covariance.shape != (12, 12):
        raise ValueError(f"pose_covariance12x12 must be (12, 12), got {covariance.shape}")
    error_jacobian = relative_right_error_jacobian(relative_transform)
    return error_jacobian @ covariance @ error_jacobian.T


def pinhole_jacobian(point_camera3: np.ndarray, focal_length: float = 1.0) -> np.ndarray:
    """Jacobian of the pinhole projection pi([x,y,z]) = f [x/z, y/z]."""
    point = np.asarray(point_camera3, dtype=np.float64).reshape(3)
    x, y, z = point
    if abs(z) < 1e-8:
        raise ValueError("camera depth must be non-zero for projection Jacobian")
    return focal_length * np.asarray(
        [
            [1.0 / z, 0.0, -x / z**2],
            [0.0, 1.0 / z, -y / z**2],
        ]
    )


def visual_keypoint_jacobian(
    camera_from_world: np.ndarray,
    world_from_palm: np.ndarray,
    in_hand_transform: np.ndarray,
    keypoint_local3: np.ndarray,
    *,
    focal_length: float = 1.0,
) -> np.ndarray:
    """Return dh/d(delta_xi) for pinhole projection h(pi(T_C^W T_W^O p)).

    IEKF with residual r = z - h should use H = -visual_keypoint_jacobian(...).
    """
    camera_from_world = np.asarray(camera_from_world, dtype=np.float64)
    world_from_palm = np.asarray(world_from_palm, dtype=np.float64)
    in_hand_transform = np.asarray(in_hand_transform, dtype=np.float64)
    keypoint_local = np.asarray(keypoint_local3, dtype=np.float64).reshape(3)

    world_from_object = world_from_palm @ in_hand_transform
    point_world = world_from_object[:3, :3] @ keypoint_local + world_from_object[:3, 3]
    point_camera = (
        camera_from_world[:3, :3] @ point_world + camera_from_world[:3, 3]
    )

    jacobian_world = (
        world_from_palm[:3, :3]
        @ in_hand_transform[:3, :3]
        @ np.hstack([np.eye(3), -skew(keypoint_local)])
    )
    jacobian_camera = camera_from_world[:3, :3] @ jacobian_world
    return pinhole_jacobian(point_camera, focal_length=focal_length) @ jacobian_camera


def contact_sdf_jacobian(
    world_from_object: np.ndarray,
    fingertip_world3: np.ndarray,
    surface_gradient_object3: np.ndarray,
) -> np.ndarray:
    """Jacobian of phi((T_W^O)^-1 p_F) w.r.t. right-multiplicative object error."""
    world_from_object = np.asarray(world_from_object, dtype=np.float64)
    fingertip_world = np.asarray(fingertip_world3, dtype=np.float64).reshape(3)
    surface_gradient = np.asarray(surface_gradient_object3, dtype=np.float64).reshape(3)

    object_from_world = invert_transform(world_from_object)
    point_object = (
        object_from_world[:3, :3] @ fingertip_world + object_from_world[:3, 3]
    )
    gradient = surface_gradient
    point_jacobian = np.hstack([-np.eye(3), -skew(point_object)])
    return gradient @ point_jacobian


def grasp_internal_force_nullspace(
    grasp_matrix6xn: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Return an orthonormal basis for ker(G), i.e. internal grasp forces."""
    grasp_matrix = np.asarray(grasp_matrix6xn, dtype=np.float64)
    if grasp_matrix.ndim != 2 or grasp_matrix.shape[0] != 6:
        raise ValueError(
            f"grasp_matrix6xn must have shape (6, n), got {grasp_matrix.shape}"
        )
    _, singular_values, vh = np.linalg.svd(grasp_matrix, full_matrices=True)
    rank = int(np.sum(singular_values > tolerance))
    nullspace = vh[rank:].T
    return nullspace


def pbvs_se3_rollout(
    desired_relative_transform: np.ndarray,
    initial_relative_transform: np.ndarray,
    relative_jacobian6xn: np.ndarray,
    *,
    steps: int,
    time_step: float,
    gain6: np.ndarray | None = None,
    estimate_bias6: np.ndarray | None = None,
    slip_disturbance6: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate SE(3) PBVS error e = Log(T_des^-1 T) with linearized e_dot = J u."""
    desired = np.asarray(desired_relative_transform, dtype=np.float64)
    current = np.asarray(initial_relative_transform, dtype=np.float64).copy()
    jacobian = np.asarray(relative_jacobian6xn, dtype=np.float64)
    gain = np.ones(6) if gain6 is None else np.asarray(gain6, dtype=np.float64).reshape(6)
    estimate_bias = (
        np.zeros(6)
        if estimate_bias6 is None
        else np.asarray(estimate_bias6, dtype=np.float64).reshape(6)
    )
    slip_disturbance = (
        np.zeros(6)
        if slip_disturbance6 is None
        else np.asarray(slip_disturbance6, dtype=np.float64).reshape(6)
    )

    error_trace = []
    transform_trace = []
    for _ in range(steps):
        error = se3_log(invert_transform(desired) @ current)
        error_trace.append(error.copy())
        transform_trace.append(current.copy())
        estimated_error = error + estimate_bias
        control = -np.linalg.pinv(jacobian) @ (gain * estimated_error)
        error_rate = jacobian @ control + slip_disturbance
        current = current @ se3_exp(time_step * error_rate)
    error_trace.append(se3_log(invert_transform(desired) @ current))
    transform_trace.append(current.copy())
    return np.asarray(error_trace), np.asarray(transform_trace)


def pbvs_linear_rollout(
    initial_error6: np.ndarray,
    gain6: np.ndarray,
    *,
    steps: int,
    time_step: float,
    estimate_bias6: np.ndarray | None = None,
    slip_disturbance6: np.ndarray | None = None,
) -> np.ndarray:
    error = np.asarray(initial_error6, dtype=np.float64).reshape(6).copy()
    gain = np.asarray(gain6, dtype=np.float64).reshape(6)
    estimate_bias = (
        np.zeros(6)
        if estimate_bias6 is None
        else np.asarray(estimate_bias6, dtype=np.float64).reshape(6)
    )
    slip_disturbance = (
        np.zeros(6)
        if slip_disturbance6 is None
        else np.asarray(slip_disturbance6, dtype=np.float64).reshape(6)
    )
    trace = [error.copy()]
    for _ in range(steps):
        estimated_error = error + estimate_bias
        error += time_step * (-gain * estimated_error + slip_disturbance)
        trace.append(error.copy())
    return np.asarray(trace)


def information_form_posterior_covariance(
    prior_covariance: np.ndarray,
    observation_jacobians: list[np.ndarray],
    observation_covariances: list[np.ndarray],
    *,
    information_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the local Gaussian posterior covariance in information form."""
    prior = np.asarray(prior_covariance, dtype=np.float64)
    if len(observation_jacobians) != len(observation_covariances):
        raise ValueError("observation Jacobian and covariance counts must match")
    weights = (
        np.ones(len(observation_jacobians), dtype=np.float64)
        if information_weights is None
        else np.asarray(information_weights, dtype=np.float64)
    )
    if weights.shape != (len(observation_jacobians),):
        raise ValueError("information_weights must contain one value per observation")
    if np.any(weights < 0.0):
        raise ValueError("information_weights must be nonnegative")

    information = np.linalg.inv(prior)
    for weight, jacobian, covariance in zip(
        weights,
        observation_jacobians,
        observation_covariances,
        strict=True,
    ):
        observation_jacobian = np.asarray(jacobian, dtype=np.float64)
        observation_covariance = np.asarray(covariance, dtype=np.float64)
        information += weight * (
            observation_jacobian.T
            @ np.linalg.solve(observation_covariance, observation_jacobian)
        )
    posterior = np.linalg.inv(information)
    return 0.5 * (posterior + posterior.T)


def covariance_quadratic_cost(
    covariance: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Return E[x^T W x] = tr(W Sigma) for a zero-mean error."""
    covariance = np.asarray(covariance, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    return float(np.trace(weight @ covariance))


def ellipsoid_linear_support(
    direction: np.ndarray,
    covariance: np.ndarray,
    confidence_radius_squared: float,
) -> float:
    """Maximize direction^T x over x^T covariance^-1 x <= radius^2."""
    direction = np.asarray(direction, dtype=np.float64).reshape(-1)
    covariance = np.asarray(covariance, dtype=np.float64)
    if confidence_radius_squared < 0.0:
        raise ValueError("confidence_radius_squared must be nonnegative")
    value = confidence_radius_squared * float(direction @ covariance @ direction)
    return float(np.sqrt(max(0.0, value)))


def robust_pbvs_lyapunov_upper_bound(
    estimated_error: np.ndarray,
    relative_error_rate: np.ndarray,
    task_weight: np.ndarray,
    estimation_covariance: np.ndarray,
    *,
    confidence_radius_squared: float,
    slip_bound: float,
) -> float:
    """Upper-bound Vdot for ellipsoidal estimation error and bounded slip."""
    estimated_error = np.asarray(estimated_error, dtype=np.float64).reshape(-1)
    relative_error_rate = np.asarray(relative_error_rate, dtype=np.float64).reshape(-1)
    task_weight = np.asarray(task_weight, dtype=np.float64)
    estimation_covariance = np.asarray(estimation_covariance, dtype=np.float64)
    if slip_bound < 0.0:
        raise ValueError("slip_bound must be nonnegative")

    weighted_rate = task_weight @ relative_error_rate
    estimation_support = ellipsoid_linear_support(
        weighted_rate,
        estimation_covariance,
        confidence_radius_squared,
    )
    weighted_covariance = task_weight @ estimation_covariance @ task_weight
    covariance_radius = np.sqrt(
        max(
            0.0,
            confidence_radius_squared
            * float(np.linalg.eigvalsh(weighted_covariance).max()),
        )
    )
    disturbance_bound = slip_bound * (
        np.linalg.norm(task_weight @ estimated_error) + covariance_radius
    )
    nominal_rate = float(estimated_error @ weighted_rate)
    return nominal_rate + estimation_support + disturbance_bound
