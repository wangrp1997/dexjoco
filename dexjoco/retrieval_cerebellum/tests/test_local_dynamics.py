import numpy as np

from retrieval_cerebellum.local_dynamics import identify_one_step_dynamics


def test_paired_rollouts_separate_drift_linear_response_and_even_nonlinearity():
    drift = np.array([0.1, -0.2, 0.03, 0.04, -0.05])
    right = np.arange(30, dtype=np.float64).reshape(5, 6) / 10.0
    left = -np.flip(right, axis=1)

    def rollout(right_twist, left_twist):
        quadratic = np.full(5, 3.0 * np.sum(right_twist**2 + left_twist**2))
        return drift + right @ right_twist + left @ left_twist + quadratic

    result = identify_one_step_dynamics(
        rollout,
        translation_step_m=0.01,
        rotation_step_rad=0.02,
        rollout_steps=4,
    )

    np.testing.assert_allclose(result.drift, drift)
    np.testing.assert_allclose(result.right_state_jacobian, right)
    np.testing.assert_allclose(result.left_state_jacobian, left)
    np.testing.assert_allclose(result.right_even_residual[:, :3], 3e-4)
    np.testing.assert_allclose(result.right_even_residual[:, 3:], 1.2e-3)
    assert result.rollout_steps == 4


def test_rollout_output_must_be_five_dimensional():
    def rollout(_right_twist, _left_twist):
        return np.zeros(4)

    try:
        identify_one_step_dynamics(rollout)
    except ValueError as error:
        assert "shape (5,)" in str(error)
    else:
        raise AssertionError("invalid rollout shape was accepted")
