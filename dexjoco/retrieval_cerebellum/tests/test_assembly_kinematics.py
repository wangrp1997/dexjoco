import numpy as np
from scipy.spatial.transform import Rotation

from retrieval_cerebellum.assembly_kinematics import (
    BimanualAssemblyKinematics,
    align_and_project_finger_references,
    apply_bimanual_wrist_twists,
    bimanual_controls_to_action44,
    palm_pose_from_action44,
    pose_matrix,
)


def test_reconstructs_bilateral_attachment_model_from_observation():
    right = pose_matrix([0.4, -0.2, 0.6], [0.1, -0.2, 0.05])
    left = pose_matrix([-0.3, 0.1, 0.5], [-0.1, 0.05, 0.2])
    peg_in_right = pose_matrix([0.02, -0.01, 0.08], [0.05, 0.02, -0.03])
    hole_in_left = pose_matrix([0.03, 0.04, 0.12], [-0.04, 0.03, 0.01])
    peg_tip_in_peg = np.array([0.0, 0.0, -0.06])
    original = BimanualAssemblyKinematics(
        peg_in_right_palm=peg_in_right,
        hole_in_left_palm=hole_in_left,
        peg_tip_in_peg=peg_tip_in_peg,
    )
    peg_world = right @ peg_in_right
    hole_world = left @ hole_in_left
    tip_world = peg_world[:3, :3] @ peg_tip_in_peg + peg_world[:3, 3]
    tip_in_hole = hole_world[:3, :3].T @ (tip_world - hole_world[:3, 3])
    peg_in_hole_rotvec = Rotation.from_matrix(
        hole_world[:3, :3].T @ peg_world[:3, :3]
    ).as_rotvec()

    reconstructed = BimanualAssemblyKinematics.from_observation(
        right_palm_world=right,
        left_palm_world=left,
        peg_in_right_palm=peg_in_right,
        peg_tip_in_hole_position=tip_in_hole,
        peg_in_hole_rotvec=peg_in_hole_rotvec,
        peg_tip_in_peg=peg_tip_in_peg,
    )

    np.testing.assert_allclose(
        reconstructed.assembly_state(right, left),
        original.assembly_state(right, left),
        atol=1e-10,
    )


def test_aligned_state_jacobians_have_expected_bilateral_signs():
    identity = np.eye(4)
    kinematics = BimanualAssemblyKinematics(
        peg_in_right_palm=identity,
        hole_in_left_palm=identity,
        peg_tip_in_peg=np.zeros(3),
    )

    right, left = kinematics.state_jacobians(identity, identity)

    np.testing.assert_allclose(right[0, 0], 1.0, atol=1e-8)
    np.testing.assert_allclose(right[1, 1], 1.0, atol=1e-8)
    np.testing.assert_allclose(right[4, 2], -1.0, atol=1e-8)
    np.testing.assert_allclose(right[2, 3], 1.0, atol=1e-8)
    np.testing.assert_allclose(right[3, 4], 1.0, atol=1e-8)
    np.testing.assert_allclose(left[0, 0], -1.0, atol=1e-8)
    np.testing.assert_allclose(left[1, 1], -1.0, atol=1e-8)
    np.testing.assert_allclose(left[4, 2], 1.0, atol=1e-8)
    np.testing.assert_allclose(left[2, 3], -1.0, atol=1e-8)
    np.testing.assert_allclose(left[3, 4], -1.0, atol=1e-8)


def test_bimanual_twists_map_to_absolute_action44_and_preserve_finger_reference():
    current = np.zeros(44, dtype=np.float64)
    reference = np.arange(44, dtype=np.float64)
    right_twist = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.1])
    left_twist = np.array([0.0, -0.02, 0.0, 0.0, -0.2, 0.0])

    command = apply_bimanual_wrist_twists(
        current,
        right_twist,
        left_twist,
        finger_reference44=reference,
    )

    np.testing.assert_allclose(command[:6], right_twist, atol=1e-7)
    np.testing.assert_allclose(command[22:28], left_twist, atol=1e-7)
    np.testing.assert_allclose(command[6:22], reference[6:22])
    np.testing.assert_allclose(command[28:44], reference[28:44])

    controls = np.stack([right_twist, right_twist])
    plan = bimanual_controls_to_action44(
        current,
        controls,
        np.stack([left_twist, left_twist]),
    )
    np.testing.assert_allclose(plan[-1, :3], 2.0 * right_twist[:3], atol=1e-7)
    np.testing.assert_allclose(plan[-1, 22:25], 2.0 * left_twist[:3], atol=1e-7)


def test_retrieved_fingers_are_aligned_to_current_grasp_and_step_limited():
    current = np.zeros(44, dtype=np.float64)
    current[6:22] = 0.5
    current[28:44] = -0.25
    nominal = np.zeros((3, 44), dtype=np.float64)
    nominal[0, 6:22] = -0.8
    nominal[0, 28:44] = 0.9
    nominal[1:, 6:22] = 0.2
    nominal[1:, 28:44] = -0.1
    limits = np.full(32, 0.03)

    references = align_and_project_finger_references(current, nominal, limits)

    np.testing.assert_allclose(references[0, 6:22], 0.53)
    np.testing.assert_allclose(references[0, 28:44], -0.28)
    assert np.max(np.abs(references[1, 6:22] - references[0, 6:22])) <= 0.030001
    assert np.max(np.abs(references[1, 28:44] - references[0, 28:44])) <= 0.030001


def test_policy_tcp_pose_is_converted_to_allegro_palm():
    palm = palm_pose_from_action44(np.zeros(44), side="right")

    np.testing.assert_allclose(palm[:3, 3], [0.05, 0.0, 0.03])
    np.testing.assert_allclose(
        palm[:3, :3],
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
    )
