import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from echo_insert.public_io import (
    PublicObservation,
    apply_right_micro_action,
    apply_right_tip_pivot_action,
    checked_task_basis,
    state46_to_action44,
    wrist_wrench_task,
)
from echo_insert.sim_depth import read_ego_depth_m


def _state46() -> np.ndarray:
    state = np.zeros(46, dtype=np.float64)
    state[3] = 1.0
    state[10] = 1.0
    state[14:30] = np.arange(16)
    state[30:46] = np.arange(16) + 20.0
    return state


def test_public_observation_copies_locks_and_rejects_private_shapes() -> None:
    state = _state46()
    previous = np.zeros(44)
    wrench = np.arange(12, dtype=np.float64).reshape(2, 6)
    tactile = np.arange(8, dtype=np.float64).reshape(2, 4)
    depth = np.ones((640, 640), dtype=np.float32)
    observation = PublicObservation(
        state,
        previous,
        wrench,
        tactile,
        ego_depth_m=depth,
    )

    state[0] = previous[0] = wrench[0, 0] = tactile[0, 0] = depth[0, 0] = 999.0
    assert observation.state46[0] == 0.0
    assert observation.previous_action44[0] == 0.0
    assert observation.wrist_wrench_local[0, 0] == 0.0
    assert observation.fingertip_load is not None
    assert observation.fingertip_load[0, 0] == 0.0
    assert observation.ego_depth_m is not None
    assert observation.ego_depth_m[0, 0] == 1.0
    assert all(
        not array.flags.writeable
        for array in (
            observation.state46,
            observation.previous_action44,
            observation.wrist_wrench_local,
            observation.fingertip_load,
            observation.ego_depth_m,
        )
    )

    with pytest.raises(ValueError):
        PublicObservation(np.zeros(61), np.zeros(44), np.zeros((2, 6)))
    bad = _state46()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        PublicObservation(bad, np.zeros(44), np.zeros((2, 6)))
    with pytest.raises(ValueError):
        PublicObservation(
            _state46(),
            np.zeros(44),
            np.zeros((2, 6)),
            ego_depth_m=np.zeros((640, 640)),
        )


def test_state_layout_and_world_task_basis_validation() -> None:
    state = _state46()
    state[0:3] = [0.1, 0.2, 0.3]
    state[7:10] = [-0.1, -0.2, -0.3]
    right = Rotation.from_euler("z", 30, degrees=True)
    left = Rotation.from_euler("y", -40, degrees=True)
    state[3:7] = right.as_quat(scalar_first=True)
    state[10:14] = left.as_quat(scalar_first=True)

    action = state46_to_action44(state)
    np.testing.assert_allclose(action[0:3], state[0:3])
    np.testing.assert_allclose(action[3:6], right.as_rotvec())
    np.testing.assert_allclose(action[6:22], state[14:30])
    np.testing.assert_allclose(action[22:25], state[7:10])
    np.testing.assert_allclose(action[25:28], left.as_rotvec())
    np.testing.assert_allclose(action[28:44], state[30:46])

    fixed = Rotation.from_euler("x", 20, degrees=True).as_matrix()
    basis = checked_task_basis(fixed)
    np.testing.assert_allclose(basis, fixed)
    assert not action.flags.writeable
    assert not basis.flags.writeable
    with pytest.raises(ValueError):
        checked_task_basis(np.diag([1.0, 1.0, 2.0]))


def test_wrench_rotation_and_micro_action_are_sensor_only_and_frozen() -> None:
    state = _state46()
    right = Rotation.from_euler("z", 90, degrees=True)
    left = Rotation.from_euler("x", 90, degrees=True)
    state[3:7] = right.as_quat(scalar_first=True)
    state[10:14] = left.as_quat(scalar_first=True)
    local = np.asarray(
        [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
    )
    observation = PublicObservation(state, np.zeros(44), local)
    basis = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    task_wrench = wrist_wrench_task(observation, basis)
    for side, rotation in enumerate((right, left)):
        np.testing.assert_allclose(
            task_wrench[side, :3],
            basis.T @ rotation.apply(local[side, :3]),
        )
        np.testing.assert_allclose(
            task_wrench[side, 3:],
            basis.T @ rotation.apply(local[side, 3:]),
        )

    previous = np.zeros(44)
    previous[0:3] = [0.4, -0.2, 0.7]
    previous_rotation = Rotation.from_euler("x", 10, degrees=True)
    previous[3:6] = previous_rotation.as_rotvec()
    frozen = np.linspace(-1.0, 1.0, 44)
    micro = np.asarray([0.001, -0.002, 0.003, 0.01, -0.02])
    command = apply_right_micro_action(previous, frozen, micro, basis)

    np.testing.assert_allclose(command[:3], previous[:3] + basis @ micro[:3])
    expected_rotation = Rotation.from_rotvec(
        basis @ np.asarray([micro[3], micro[4], 0.0])
    ) * previous_rotation
    np.testing.assert_allclose(command[3:6], expected_rotation.as_rotvec())
    np.testing.assert_allclose(command[6:44], frozen[6:44])
    assert not task_wrench.flags.writeable
    assert not command.flags.writeable

    with pytest.raises(ValueError):
        apply_right_micro_action(previous, frozen, np.zeros(6), basis)



def test_tip_pivot_action_holds_the_estimated_insert_end_fixed() -> None:
    state = _state46()
    state[0:3] = [0.3, -0.2, 0.6]
    current_rotation = Rotation.from_euler("xyz", [10.0, -20.0, 5.0], degrees=True)
    state[3:7] = current_rotation.as_quat(scalar_first=True)
    frozen = state46_to_action44(state)
    pivot = np.asarray([0.36, -0.18, 0.64])
    rotation_step = np.asarray([0.0, 0.03, -0.02])
    command = apply_right_tip_pivot_action(
        state,
        frozen,
        rotation_step,
        pivot,
    )

    pivot_in_tcp = current_rotation.inv().apply(pivot - state[:3])
    commanded_rotation = Rotation.from_rotvec(np.asarray(command[3:6]).copy())
    np.testing.assert_allclose(
        command[:3] + commanded_rotation.apply(pivot_in_tcp),
        pivot,
    )
    np.testing.assert_allclose(command[6:], frozen[6:])

    target_pivot = pivot + np.asarray([0.01, -0.02, 0.03])
    retargeted = apply_right_tip_pivot_action(
        state,
        frozen,
        rotation_step,
        pivot,
        target_pivot_world=target_pivot,
    )
    retargeted_rotation = Rotation.from_rotvec(
        np.asarray(retargeted[3:6]).copy()
    )
    np.testing.assert_allclose(
        retargeted[:3] + retargeted_rotation.apply(pivot_in_tcp),
        target_pivot,
    )

def test_fixed_ego_depth_sensor_linearises_the_z_buffer() -> None:
    class Viewer:
        def render(self, *, render_mode: str, camera_id: int) -> np.ndarray:
            assert render_mode == "depth_array"
            assert camera_id == 3
            return np.full((640, 640), 0.5, dtype=np.float32)

    camera_map = type("Map", (), {"znear": 0.01, "zfar": 10.0})()
    model = type(
        "Model",
        (),
        {
            "stat": type("Stat", (), {"extent": 1.0})(),
            "vis": type("Vis", (), {"map": camera_map})(),
        },
    )()
    raw = type(
        "Raw",
        (),
        {
            "_viewer": Viewer(),
            "_front_camera_id": 3,
            "camera_id": (3, 4, 5),
            "_model": model,
        },
    )()

    depth = read_ego_depth_m(raw)
    expected = 0.01 * 10.0 / (10.0 - 0.5 * (10.0 - 0.01))
    assert depth.shape == (640, 640)
    assert depth.dtype == np.float32
    np.testing.assert_allclose(depth, expected)
