import numpy as np
import pytest

from retrieval_cerebellum.image_space_servo import (
    damped_servo_command,
    identify_central_difference_jacobian,
    image_alignment_feature,
)


def test_image_alignment_feature_is_zero_for_matching_primitives():
    points = np.asarray([[10, 20], [12, 25], [10, 20], [12, 25]])

    feature = image_alignment_feature(points, image_size=160)

    np.testing.assert_allclose(feature, 0.0)


def test_central_difference_identifies_known_jacobian():
    matrix = np.asarray(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.1, 1.1, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.1],
            [0.0, 0.0, 0.2, 0.9],
        ]
    )
    amplitudes = np.asarray([0.01, 0.01, 0.02, 0.02])
    positive = np.stack([matrix[:, index] * value for index, value in enumerate(amplitudes)])
    negative = -positive

    identified = identify_central_difference_jacobian(positive, negative, amplitudes)

    np.testing.assert_allclose(identified.matrix, matrix)
    assert identified.rank == 4


def test_damped_command_reduces_linearized_error():
    matrix = np.asarray(
        [
            [1.0, 0.0, 0.1, 0.0],
            [0.0, 1.0, 0.0, 0.1],
            [0.2, 0.0, 1.0, 0.0],
            [0.0, 0.2, 0.0, 1.0],
        ]
    )
    amplitudes = np.full(4, 0.01)
    positive = np.stack([matrix[:, index] * value for index, value in enumerate(amplitudes)])
    identified = identify_central_difference_jacobian(positive, -positive, amplitudes)
    feature = np.asarray([0.2, -0.1, 0.08, -0.04])

    command = damped_servo_command(feature, identified, damping=1e-3)

    assert np.linalg.norm(feature + matrix @ command) < np.linalg.norm(feature)


def test_rank_deficient_jacobian_is_rejected():
    positive = np.zeros((4, 4))
    identified = identify_central_difference_jacobian(
        positive,
        positive,
        np.ones(4),
    )

    with pytest.raises(ValueError, match="not full row rank"):
        damped_servo_command(np.ones(4), identified)
