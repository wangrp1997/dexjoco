import numpy as np

from retrieval_cerebellum.finger_kinematics import AllegroFingertipKinematics


def _state46() -> np.ndarray:
    state = np.zeros(46, dtype=np.float32)
    state[3] = 1.0
    state[10] = 1.0
    return state


def test_fingertip_kinematics_is_finite_and_palm_relative():
    kinematics = AllegroFingertipKinematics()

    positions = kinematics.positions_in_palm(_state46())

    assert positions.shape == (2, 4, 3)
    assert np.isfinite(positions).all()
    assert np.max(np.linalg.norm(positions, axis=-1)) < 0.3


def test_fingertip_kinematics_changes_with_encoder_state():
    kinematics = AllegroFingertipKinematics()
    first = kinematics.positions_in_palm(_state46())
    state = _state46()
    state[15] = 0.8

    second = kinematics.positions_in_palm(state)

    assert np.linalg.norm(second[0, 0] - first[0, 0]) > 1e-4
    np.testing.assert_allclose(second[0, 1:], first[0, 1:], atol=1e-6)
