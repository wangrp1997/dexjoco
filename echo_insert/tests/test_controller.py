import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from echo_insert.controller import EchoConfig, EchoController
from echo_insert.optimizer import (
    EnergyInformationOptimizer,
    MultiOutputRidgeRLS,
    OptimizerConfig,
)
from echo_insert.public_io import PublicObservation, state46_to_action44


def _state46() -> np.ndarray:
    state = np.zeros(46, dtype=np.float64)
    state[0:3] = [0.2, -0.1, 0.7]
    state[3] = 1.0
    state[7:10] = [-0.2, 0.1, 0.7]
    state[10] = 1.0
    state[14:30] = np.linspace(0.1, 0.4, 16)
    state[30:46] = np.linspace(-0.2, 0.2, 16)
    return state


def _observation(
    state: np.ndarray,
    *,
    previous_action: np.ndarray | None = None,
    right_wrench: np.ndarray | None = None,
) -> PublicObservation:
    wrench = np.zeros((2, 6), dtype=np.float64)
    if right_wrench is not None:
        wrench[0] = right_wrench
    return PublicObservation(
        state46=state,
        previous_action44=(
            state46_to_action44(state)
            if previous_action is None
            else previous_action
        ),
        wrist_wrench_local=wrench,
    )


def _controller(
    state: np.ndarray,
    config: EchoConfig | None = None,
) -> EchoController:
    basis = np.eye(3)
    peg_tip = state[:3] + np.asarray([0.0, 0.0, 0.05])
    return EchoController(
        basis,
        basis[:, 2],
        peg_tip,
        peg_tip + basis[:, 2] * 0.04,
        config,
    )


def _state_from_action(state: np.ndarray, action44: np.ndarray) -> np.ndarray:
    result = state.copy()
    result[:3] = action44[:3]
    rotation = Rotation.from_rotvec(np.asarray(action44[3:6]).copy())
    result[3:7] = rotation.as_quat(scalar_first=True)
    return result

def _finish_baseline(

    controller: EchoController,
    state: np.ndarray,
) -> tuple[np.ndarray, object]:
    action = state46_to_action44(state)
    diagnostics = None
    for _ in range(controller.config.baseline_steps):
        action, diagnostics = controller.step(
            _observation(state, previous_action=action)
        )
    assert diagnostics is not None
    return action, diagnostics


def test_task_frame_follows_public_left_wrist_pose() -> None:
    state = _state46()
    controller = _controller(state)
    controller.reset(_observation(state))
    initial_basis, _, _, _, initial_target, _ = controller._precontact_geometry(state)
    shift = np.asarray([0.012, -0.007, 0.003])

    left_moved = state.copy()
    left_moved[7:10] += shift
    moved_basis, _, _, _, moved_target, _ = controller._precontact_geometry(left_moved)
    np.testing.assert_allclose(moved_basis, initial_basis)
    np.testing.assert_allclose(moved_target, initial_target + shift)

    both_moved = left_moved.copy()
    both_moved[:3] += shift
    _, _, _, _, comoving_target, _ = controller._precontact_geometry(both_moved)
    np.testing.assert_allclose(comoving_target, initial_target)



def test_baseline_holds_then_free_space_advances_without_rls_updates() -> None:
    state = _state46()
    initial = _observation(state)
    controller = _controller(state)
    controller.reset(initial)

    action, diagnostics = _finish_baseline(controller, state)
    frozen = state46_to_action44(state)
    assert diagnostics.status == "baseline"
    assert diagnostics.baseline_remaining == 0
    np.testing.assert_allclose(action, frozen)

    action, diagnostics = controller.step(
        _observation(state, previous_action=action)
    )
    assert diagnostics.status == "approach"
    assert diagnostics.selected_candidate == "advance"
    assert diagnostics.selected_u5[2] > 0.0
    np.testing.assert_allclose(action[6:22], frozen[6:22])
    np.testing.assert_allclose(action[22:], frozen[22:])
    assert action[2] > frozen[2]

    initial_covariance = controller.optimizer.model.covariance.copy()
    for _ in range(20):
        action, diagnostics = controller.step(
            _observation(state, previous_action=action)
        )
        assert diagnostics.status == "approach"
        assert diagnostics.selected_u5[:2] == (0.0, 0.0)
        assert diagnostics.selected_u5[2] > 0.0
        assert diagnostics.selected_u5[3:] == (0.0, 0.0)

    assert diagnostics.rls_updates == 0
    np.testing.assert_allclose(
        controller.optimizer.model.covariance,
        initial_covariance,
    )


def test_reset_preserves_commanded_grasp_targets() -> None:
    state = _state46()
    previous = state46_to_action44(state).copy()
    previous[6:22] += 0.2
    previous[28:44] -= 0.2
    controller = _controller(state)
    controller.reset(_observation(state, previous_action=previous))

    action, _ = controller.step(_observation(state, previous_action=previous))

    np.testing.assert_allclose(action[6:22], previous[6:22])
    np.testing.assert_allclose(action[28:44], previous[28:44])


def test_interaction_requires_three_samples_and_learns_only_onset() -> None:
    state = _state46()
    controller = _controller(state)
    controller.reset(_observation(state))
    action, _ = _finish_baseline(controller, state)
    action, _ = controller.step(_observation(state, previous_action=action))
    contact_wrench = np.asarray([0.0, 0.0, -2.0, 0.0, 0.0, 0.0])

    action, diagnostics = controller.step(
        _observation(state, previous_action=action, right_wrench=contact_wrench)
    )
    assert diagnostics.status == "interaction_confirm"
    assert diagnostics.selected_u5 == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert diagnostics.rls_updates == 0

    action, diagnostics = controller.step(
        _observation(state, previous_action=action)
    )
    assert diagnostics.status == "approach"
    assert diagnostics.rls_updates == 0

    for expected_status in ("interaction_confirm", "interaction_confirm", "optimize"):
        action, diagnostics = controller.step(
            _observation(state, previous_action=action, right_wrench=contact_wrench)
        )
        assert diagnostics.status == expected_status

    assert diagnostics.rls_updates == 1
    assert diagnostics.selected_candidate != "advance"


def test_low_axial_contact_keeps_continuous_spiral_active() -> None:
    state = _state46()
    controller = _controller(state)
    controller.reset(_observation(state))
    action, _ = _finish_baseline(controller, state)
    action, _ = controller.step(_observation(state, previous_action=action))
    state = _state_from_action(state, action)
    low_contact = np.asarray([0.8, 0.0, 0.2, 0.0, 0.0, 0.0])
    saw_xy = False
    saw_z = False

    for _ in range(33):
        action, diagnostics = controller.step(
            _observation(state, previous_action=action, right_wrench=low_contact)
        )
        state = _state_from_action(state, action)
        if diagnostics.status == "optimize":
            assert diagnostics.selected_candidate == "spiral"
            saw_xy = saw_xy or bool(np.linalg.norm(diagnostics.selected_u5[:2]))
            saw_z = saw_z or diagnostics.selected_u5[2] > 0.0

    assert saw_xy and saw_z
    assert not diagnostics.entry_mode
    assert not controller.optimizer.probe_pending
    assert controller._repeat_steps_remaining == 0


def test_brief_contact_loss_keeps_continuous_spiral_active() -> None:
    state = _state46()
    controller = _controller(state)
    controller.reset(_observation(state))
    action, _ = _finish_baseline(controller, state)
    action, _ = controller.step(_observation(state, previous_action=action))
    state = _state_from_action(state, action)
    contact = np.asarray([0.8, 0.0, 0.2, 0.0, 0.0, 0.0])
    for _ in range(3):
        action, _ = controller.step(
            _observation(state, previous_action=action, right_wrench=contact)
        )
        state = _state_from_action(state, action)

    _, diagnostics = controller.step(_observation(state, previous_action=action))

    assert diagnostics.status == "optimize"
    assert diagnostics.selected_candidate == "spiral"
    assert not controller.optimizer.probe_pending
    assert diagnostics.selected_u5[2] > 0.0


def test_high_axial_contact_spiral_unloads() -> None:
    state = _state46()
    controller = _controller(state)
    controller.reset(_observation(state))
    action, _ = _finish_baseline(controller, state)
    action, _ = controller.step(_observation(state, previous_action=action))
    high_load = controller.config.optimizer.axial_probe_force_limit_n + 1.0
    contact = np.asarray([0.0, 0.0, -high_load, 0.0, 0.0, 0.0])

    for _ in range(3):
        action, diagnostics = controller.step(
            _observation(state, previous_action=action, right_wrench=contact)
        )
    assert diagnostics.status == "optimize"
    assert diagnostics.selected_candidate == "spiral"
    assert diagnostics.selected_u5[2] < 0.0
    assert diagnostics.selected_u5[:2] == (0.0, 0.0)


def test_rls_update_is_finite() -> None:
    config = OptimizerConfig()
    model = MultiOutputRidgeRLS(config)
    scale = config.action_scale
    wrench_scale = config.wrench_scale_array
    response = np.diag([-0.3, 0.2, 0.1, -0.15, 0.25])

    for _ in range(6):
        for axis in range(5):
            for sign in (-1.0, 1.0):
                phi = np.eye(5)[axis] * sign
                u5 = scale * phi
                model.update(u5, (response @ phi) * wrench_scale, u5)

    prediction = model.predict(scale * np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert model.updates == 60
    assert np.isfinite(model.weights).all()
    assert np.isfinite(model.covariance).all()
    assert np.isfinite(prediction.wrench_delta5).all()
    np.testing.assert_allclose(prediction.motion5, scale * [1, 0, 0, 0, 0], atol=1e-6)


def test_blocked_model_prefers_lower_predicted_energy_than_advance() -> None:
    config = OptimizerConfig()
    optimizer = EnergyInformationOptimizer(config)
    response = np.zeros((5, 5), dtype=np.float64)
    response[0, 0] = -0.5

    for _ in range(8):
        for axis in range(5):
            for sign in (-1.0, 1.0):
                phi = np.eye(5)[axis] * sign
                u5 = config.action_scale * phi
                optimizer.update(
                    u5,
                    (response @ phi) * config.wrench_scale_array,
                    u5,
                )

    selection = optimizer.select(
        np.asarray([4.0, 0.0, 0.0, 0.0, 0.0]),
        np.zeros(5),
        cumulative_positive_work_j=0.0,
    )
    advance = next(
        candidate
        for candidate in selection.candidates
        if candidate.name == "advance"
    )
    assert selection.best.name == "tangent_x_pos"
    assert selection.best.score < advance.score
    assert np.linalg.norm(selection.best.predicted_wrench5[:2]) < np.linalg.norm(
        advance.predicted_wrench5[:2]
    )


def test_slow_productive_response_advances_instead_of_deadlocking_on_hold() -> None:
    config = OptimizerConfig()
    optimizer = EnergyInformationOptimizer(config)

    for _ in range(8):
        for axis in range(5):
            for sign in (-1.0, 1.0):
                phi = np.eye(5)[axis] * sign
                u5 = config.action_scale * phi
                optimizer.update(
                    u5,
                    np.zeros(5),
                    0.05 * u5,
                )

    selection = optimizer.select(
        np.zeros(5),
        np.zeros(5),
        cumulative_positive_work_j=0.0,
    )
    assert selection.best.name == "advance"


def test_hard_load_unloads_and_positive_work_budget_recovers() -> None:
    state = _state46()
    controller = _controller(state)
    controller.reset(_observation(state))
    action, _ = _finish_baseline(controller, state)

    action, approach_diagnostics = controller.step(
        _observation(state, previous_action=action)
    )
    assert approach_diagnostics.status == "approach"
    high_force = np.asarray(
        [controller.config.optimizer.hard_force_n + 1.0, 0, 0, 0, 0, 0]
    )
    unloaded, diagnostics = controller.step(
        _observation(
            state,
            previous_action=action,
            right_wrench=high_force,
        )
    )
    assert diagnostics.status == "safety"
    assert diagnostics.safety_reason == "force_or_torque_limit"
    assert diagnostics.selected_u5[2] < 0.0
    assert unloaded[2] < action[2]

    work_config = EchoConfig(
        optimizer=OptimizerConfig(
            hard_force_n=100.0,
            hard_positive_work_j=1e-6,
        )
    )
    work_controller = _controller(state, work_config)
    work_controller.reset(_observation(state))
    action, _ = _finish_baseline(work_controller, state)
    action, advance_diagnostics = work_controller.step(
        _observation(state, previous_action=action)
    )
    assert advance_diagnostics.status == "approach"
    assert advance_diagnostics.selected_candidate == "advance"

    moved = state.copy()
    moved[2] += work_config.optimizer.advance_step_m
    action, diagnostics = work_controller.step(
        _observation(
            moved,
            previous_action=action,
            right_wrench=np.asarray([0.0, 0.0, -2.0, 0.0, 0.0, 0.0]),
        )
    )
    assert diagnostics.status == "safety"
    assert diagnostics.safety_reason == "positive_work_limit"
    assert diagnostics.selected_u5 == (0.0, 0.0, 0.0, 0.0, 0.0)

    for _ in range(work_config.positive_work_window_steps - 1):
        action, diagnostics = work_controller.step(
            _observation(moved, previous_action=action)
        )
        assert diagnostics.status == "safety"
        assert diagnostics.safety_reason == "positive_work_limit"

    _, diagnostics = work_controller.step(
        _observation(moved, previous_action=action)
    )
    assert diagnostics.status == "approach"
    assert diagnostics.safety_reason == ""
    assert diagnostics.cumulative_positive_work_j == pytest.approx(0.0)


def test_optimizer_rejects_candidates_outside_cumulative_workspace() -> None:
    config = OptimizerConfig(maximum_tangent_offset_m=0.001)
    optimizer = EnergyInformationOptimizer(config)

    selection = optimizer.select(
        np.zeros(5),
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=np.asarray([
            config.maximum_tangent_offset_m - 0.5 * config.tangent_step_m,
            0.0, 0.0, 0.0, 0.0,
        ]),
    )

    positive_x = next(
        candidate
        for candidate in selection.candidates
        if candidate.name == "tangent_x_pos"
    )
    assert not positive_x.safe
    assert selection.best.safe


def test_spatial_energy_search_persists_across_contact_model_reset() -> None:
    config = OptimizerConfig(
        axial_progress_weight=0.0,
        lateral_weight=0.0,
        positive_work_weight=0.0,
        effort_weight=0.0,
        slew_weight=0.0,
        tactile_weight=0.0,
        information_weight=0.0,
        spatial_energy_weight=0.0,
        search_novelty_weight=0.0,
        axial_preload_weight=0.0,
        axial_probe_weight=0.0,
        tilt_offset_weight=0.0,
        revisit_weight=1.0,
    )
    optimizer = EnergyInformationOptimizer(config)
    tx, rr = config.tangent_step_m, config.rotation_step_rad
    visited = (
        np.zeros(5),
        np.asarray([tx, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([-tx, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([0.0, tx, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0, rr, 0.0]),
        np.asarray([0.0, 0.0, 0.0, -rr, 0.0]),
        np.asarray([0.0, 0.0, 0.0, 0.0, rr]),
        np.asarray([0.0, 0.0, 0.0, 0.0, -rr]),
    )
    for offset in visited:
        optimizer.select(
            np.zeros(5),
            np.zeros(5),
            cumulative_positive_work_j=0.0,
            command_offset5=offset,
        )

    selection = optimizer.select(
        np.zeros(5),
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=np.zeros(5),
    )

    assert selection.best.name == "tangent_y_neg"
    optimizer.model.reset()
    assert optimizer.search_cells == 4
    optimizer.reset()
    assert optimizer.search_cells == 0


def test_power_sign_is_an_explicit_sensor_convention() -> None:
    with pytest.raises(ValueError):
        OptimizerConfig(power_sign=0.0)


def test_low_load_approach_stops_at_axial_workspace_limit() -> None:
    state = _state46()
    config = EchoConfig(
        precontact_approach_step_m=0.0008,
        optimizer=OptimizerConfig(maximum_advance_offset_m=0.0016),
    )
    controller = _controller(state, config)
    controller.reset(_observation(state))
    action, _ = _finish_baseline(controller, state)

    for _ in range(2):
        action, diagnostics = controller.step(
            _observation(state, previous_action=action)
        )
        assert diagnostics.status == "approach"
        state = _state_from_action(state, action)

    _, diagnostics = controller.step(_observation(state, previous_action=action))
    assert diagnostics.status == "safety"
    assert diagnostics.safety_reason == "advance_workspace_limit"
    assert diagnostics.command_offset5[2] == pytest.approx(0.0)


def test_approach_keeps_last_commanded_orientation_when_measured_pose_drifts() -> None:
    state = _state46()
    controller = _controller(state)
    controller.reset(_observation(state))
    action, _ = _finish_baseline(controller, state)
    action, diagnostics = controller.step(
        _observation(state, previous_action=action)
    )
    assert diagnostics.status == "approach"

    drifted = state.copy()
    drifted[3:7] = Rotation.from_rotvec([0.3, 0.0, 0.0]).as_quat(
        scalar_first=True
    )
    held, diagnostics = controller.step(
        _observation(drifted, previous_action=action)
    )

    assert diagnostics.status == "approach"
    np.testing.assert_allclose(held[3:6], action[3:6])


def test_precontact_aligns_about_tip_then_centers_before_approach() -> None:
    state = _state46()
    basis = np.eye(3)
    peg_axis = np.asarray([1.0, 0.0, 0.0])
    peg_tip = state[:3] + np.asarray([0.1, 0.0, 0.0])
    tray_entry = peg_tip + np.asarray([-0.04, 0.0, 0.08])
    config = EchoConfig(
        baseline_steps=1,
        alignment_tolerance_rad=0.01,
        alignment_step_rad=0.4,
        maximum_alignment_translation_step_m=0.05,
        centering_tolerance_m=0.001,
        centering_step_m=0.02,
        optimizer=OptimizerConfig(maximum_advance_offset_m=0.1),
    )
    controller = EchoController(
        basis,
        peg_axis,
        peg_tip,
        tray_entry,
        config,
    )
    controller.reset(_observation(state))
    previous_action = state46_to_action44(state)
    pivot_in_initial_tcp = peg_tip - state[:3]
    statuses: list[str] = []

    for _ in range(40):
        action, diagnostics = controller.step(
            _observation(state, previous_action=previous_action)
        )
        statuses.append(diagnostics.status)
        if diagnostics.status == "align":
            commanded_rotation = Rotation.from_rotvec(np.asarray(action[3:6]).copy())
            commanded_tip = action[:3] + commanded_rotation.apply(
                pivot_in_initial_tcp
            )
            np.testing.assert_allclose(commanded_tip, peg_tip, atol=1e-12)
        state = _state_from_action(state, action)
        previous_action = action
        if diagnostics.status == "approach":
            break

    expected_order = [
        "align",
        "alignment_settle",
        "aligned_baseline",
        "center",
        "approach",
    ]
    assert [statuses.index(name) for name in expected_order] == sorted(
        statuses.index(name) for name in expected_order
    )
    assert diagnostics.axis_error_rad <= config.alignment_tolerance_rad
    assert diagnostics.lateral_error_m <= config.centering_tolerance_m


def test_alignment_translation_is_saturated_instead_of_terminal() -> None:
    state = _state46()
    peg_tip = state[:3] + np.asarray([0.1, 0.0, 0.0])
    config = EchoConfig(
        baseline_steps=1,
        alignment_step_rad=0.4,
        maximum_alignment_translation_step_m=0.001,
    )
    controller = EchoController(
        np.eye(3),
        np.asarray([1.0, 0.0, 0.0]),
        peg_tip,
        peg_tip + np.asarray([0.0, 0.0, 0.08]),
        config,
    )
    controller.reset(_observation(state))
    action, _ = controller.step(_observation(state))
    action, diagnostics = controller.step(
        _observation(state, previous_action=action)
    )

    assert diagnostics.status == "align"
    assert np.linalg.norm(action[:3] - state[:3]) == pytest.approx(0.001)


def test_alignment_continues_under_moderate_load_but_respects_hard_limit() -> None:
    state = _state46()
    basis = np.eye(3)
    peg_tip = state[:3] + np.asarray([0.1, 0.0, 0.0])
    config = EchoConfig(baseline_steps=1)
    controller = EchoController(
        basis,
        np.asarray([1.0, 0.0, 0.0]),
        peg_tip,
        peg_tip + np.asarray([0.0, 0.0, 0.08]),
        config,
    )
    controller.reset(_observation(state))
    action, _ = controller.step(_observation(state))
    moderate_force = np.asarray([5.0, 0, 0, 0, 0, 0])

    action, diagnostics = controller.step(
        _observation(
            state,
            previous_action=action,
            right_wrench=moderate_force,
        )
    )
    assert diagnostics.status == "safety"
    assert diagnostics.selected_candidate == "precontact_unload"
    assert diagnostics.safety_reason == "unexpected_precontact_load"

    action, diagnostics = controller.step(
        _observation(state, previous_action=action)
    )
    assert diagnostics.status == "align"
    assert diagnostics.safety_reason == ""

    hard_force = np.asarray([config.optimizer.hard_force_n + 1.0, 0, 0, 0, 0, 0])
    _, diagnostics = controller.step(
        _observation(state, previous_action=action, right_wrench=hard_force)
    )
    assert diagnostics.status == "safety"
    assert diagnostics.selected_candidate == "safety_unload"
    assert diagnostics.safety_reason == "force_or_torque_limit"

def test_measured_entry_progress_overrides_revisit_energy() -> None:
    config = OptimizerConfig(
        entry_progress_m=0.003,
        axial_progress_weight=0.0,
        entry_progress_weight=1.0,
        lateral_weight=0.0,
        positive_work_weight=0.0,
        effort_weight=0.0,
        slew_weight=0.0,
        tactile_weight=0.0,
        information_weight=0.0,
        spatial_energy_weight=0.0,
        search_novelty_weight=0.0,
        axial_preload_weight=0.0,
        axial_probe_weight=0.0,
        tilt_offset_weight=0.0,
        revisit_weight=1.0,
    )
    optimizer = EnergyInformationOptimizer(config)
    wrench = np.asarray([0.0, 0.0, 0.2, 0.0, 0.0])
    selection = optimizer.select(
        wrench,
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        forced_name="advance",
    )
    assert selection.best.name == "advance"
    assert optimizer.probe_pending
    assert optimizer.axial_probe_cells == 0

    selection = optimizer.select(
        wrench,
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=np.asarray([0.0, 0.0, 0.001, 0.0, 0.0]),
    )
    assert selection.best.name == "advance"
    assert optimizer.probe_pending
    assert optimizer.axial_probe_cells == 0

    selection = optimizer.select(
        wrench,
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=np.asarray([0.0, 0.0, 0.0031, 0.0, 0.0]),
    )
    assert optimizer.entry_mode
    assert optimizer.axial_probe_cells == 1
    assert not optimizer.probe_pending
    assert selection.best.name == "advance"
    assert all(
        candidate.safe
        for candidate in selection.candidates
        if candidate.name.startswith("tangent_")
    )

    drifted_selection = optimizer.select(
        wrench,
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=np.asarray(
            [4 * config.tangent_step_m, 0.0, 0.0031, 0.0, 0.0]
        ),
    )
    tangent_neg = next(
        candidate
        for candidate in drifted_selection.candidates
        if candidate.name == "tangent_x_neg"
    )
    tangent_pos = next(
        candidate
        for candidate in drifted_selection.candidates
        if candidate.name == "tangent_x_pos"
    )
    assert tangent_neg.safe
    assert not tangent_pos.safe


    probe_high = wrench.copy()
    probe_high[2] = config.axial_probe_force_limit_n + 0.1
    optimizer.select(
        probe_high,
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=np.asarray([0.0, 0.0, 0.0031, 0.0, 0.0]),
    )
    assert optimizer.entry_mode

    entry_high = wrench.copy()
    entry_high[2] = config.entry_force_limit_n + 0.1
    optimizer.select(
        entry_high,
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=np.asarray([0.0, 0.0, 0.0031, 0.0, 0.0]),
    )
    assert not optimizer.entry_mode
    optimizer.select(
        wrench,
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=np.asarray([0.0, 0.0, 0.0031, 0.0, 0.0]),
    )
    assert not optimizer.entry_mode

    x_offset = 4 * config.tangent_step_m
    optimizer.select(
        wrench,
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=np.asarray([x_offset, 0.0, 0.0031, 0.0, 0.0]),
        forced_name="advance",
    )
    optimizer.select(
        wrench,
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=np.asarray([x_offset, 0.0, 0.0062, 0.0, 0.0]),
    )
    assert optimizer.entry_mode

    optimizer.reset_interaction()
    assert not optimizer.entry_mode
    assert not optimizer.probe_pending
    assert optimizer.search_cells > 0

def test_stalled_entry_releases_original_cell() -> None:
    config = OptimizerConfig(entry_stall_steps=3)
    optimizer = EnergyInformationOptimizer(config)
    wrench = np.asarray([0.0, 0.0, 0.2, 0.0, 0.0])
    optimizer.select(
        wrench,
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        forced_name="advance",
    )
    optimizer.select(
        wrench,
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=np.asarray([0.0, 0.0, 0.0031, 0.0, 0.0]),
    )
    assert optimizer.entry_mode

    for _ in range(config.entry_stall_steps - 1):
        optimizer.select(
            wrench,
            np.zeros(5),
            cumulative_positive_work_j=0.0,
            command_offset5=np.asarray([0.0, 0.0, 0.0031, 0.0, 0.0]),
        )

    assert not optimizer.entry_mode
    assert (0, 0) in optimizer._failed_entry_cells



def test_recovery_uses_monotonic_workspace_correction() -> None:
    state = _state46()
    controller = _controller(state)
    controller.reset(_observation(state))
    action, _ = _finish_baseline(controller, state)
    action, _ = controller.step(_observation(state, previous_action=action))
    contact = np.asarray([0.0, 0.0, -2.0, 0.0, 0.0, 0.0])
    for _ in range(3):
        action, _ = controller.step(
            _observation(state, previous_action=action, right_wrench=contact)
        )
    controller._recovering_interaction = True
    controller._repeat_steps_remaining = 0
    outside = state.copy()
    outside[0] += controller.config.optimizer.maximum_tangent_offset_m + 0.01
    outside[3:7] = Rotation.from_rotvec([0.1, 0.0, 0.0]).as_quat(
        scalar_first=True
    )

    _, diagnostics = controller.step(
        _observation(outside, previous_action=action, right_wrench=contact)
    )

    assert diagnostics.status == "optimize"
    assert diagnostics.selected_candidate in {"tangent_x_neg", "roll_neg"}
    assert diagnostics.safety_reason == ""

def test_optimizer_allows_monotonic_workspace_correction() -> None:
    config = OptimizerConfig()
    optimizer = EnergyInformationOptimizer(config)
    offset = np.asarray(
        [
            config.maximum_tangent_offset_m + 0.0002,
            0.0,
            0.0,
            config.maximum_tilt_offset_rad + 0.0002,
            0.0,
        ]
    )

    x_selection = optimizer.select(
        np.zeros(5),
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=offset,
        forced_name="tangent_x_neg",
    )
    hold = next(
        candidate for candidate in x_selection.candidates if candidate.name == "hold"
    )
    assert x_selection.best.safe
    assert not hold.safe

    roll_selection = optimizer.select(
        np.zeros(5),
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=offset,
        forced_name="roll_neg",
    )
    assert roll_selection.best.safe

def test_optimize_translation_rejects_measured_orientation_drift() -> None:
    state = _state46()
    controller = _controller(state)
    controller.reset(_observation(state))
    action, _ = _finish_baseline(controller, state)
    action, _ = controller.step(_observation(state, previous_action=action))
    state = _state_from_action(state, action)
    contact = np.asarray([0.8, 0.0, 0.2, 0.0, 0.0, 0.0])
    for _ in range(3):
        action, diagnostics = controller.step(
            _observation(state, previous_action=action, right_wrench=contact)
        )
        state = _state_from_action(state, action)
    assert diagnostics.selected_candidate == "spiral"
    commanded_rotation = action[3:6].copy()
    drifted = state.copy()
    drifted[3:7] = Rotation.from_rotvec(
        commanded_rotation + np.asarray([0.03, 0.0, 0.0])
    ).as_quat(scalar_first=True)

    next_action, diagnostics = controller.step(
        _observation(
            drifted,
            previous_action=action,
            right_wrench=contact,
        )
    )

    assert diagnostics.selected_candidate == "spiral"
    np.testing.assert_allclose(next_action[3:6], commanded_rotation)

def test_continuous_spiral_preserves_advance_accumulation() -> None:
    state = _state46()
    controller = _controller(state)
    controller.reset(_observation(state))
    action, _ = _finish_baseline(controller, state)
    action, _ = controller.step(_observation(state, previous_action=action))
    state = _state_from_action(state, action)
    contact = np.asarray([0.8, 0.0, 0.2, 0.0, 0.0, 0.0])
    for _ in range(3):
        action, diagnostics = controller.step(
            _observation(state, previous_action=action, right_wrench=contact)
        )
        state = _state_from_action(state, action)
    assert diagnostics.selected_candidate == "spiral"

    previous_z = float(action[2])
    next_action, diagnostics = controller.step(
        _observation(state, previous_action=action, right_wrench=contact)
    )

    assert diagnostics.selected_candidate == "spiral"
    assert next_action[2] == pytest.approx(previous_z + diagnostics.selected_u5[2])
    assert not diagnostics.entry_mode


def test_frontier_energy_routes_through_visited_cells() -> None:
    config = OptimizerConfig(
        axial_progress_weight=0.0,
        entry_progress_weight=0.0,
        lateral_weight=0.0,
        positive_work_weight=0.0,
        effort_weight=0.0,
        slew_weight=0.0,
        tactile_weight=0.0,
        information_weight=0.0,
        frontier_weight=1.0,
        spatial_energy_weight=0.0,
        search_novelty_weight=0.0,
        axial_preload_weight=0.0,
        axial_probe_weight=0.0,
        tilt_offset_weight=0.0,
        revisit_weight=0.0,
    )
    optimizer = EnergyInformationOptimizer(config)
    target = (2, 0)
    optimizer._spatial_energy.update(
        {cell: 0.0 for cell in optimizer._search_grid if cell != target}
    )
    optimizer._axial_probed_cells.add((0, 0))

    selection = optimizer.select(
        np.zeros(5),
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=np.zeros(5),
    )

    assert optimizer.frontier_cells_remaining == 1
    assert selection.best.name == "tangent_x_pos"
    optimizer.select(
        np.zeros(5),
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=np.asarray(
            [2 * config.tangent_step_m, 0.0, 0.0, 0.0, 0.0]
        ),
        forced_name="hold",
    )
    assert optimizer.frontier_cells_remaining == 0


def test_xy_spiral_is_energy_scored_without_starting_entry_probe() -> None:
    config = OptimizerConfig()
    optimizer = EnergyInformationOptimizer(config)
    spiral = np.asarray([0.0003, 0.0002, 0.0004, 0.0, 0.0])

    selection = optimizer.select(
        np.asarray([0.0, 0.0, config.axial_preload_n, 0.0, 0.0]),
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        forced_name="spiral",
        extra_candidate=("spiral", spiral),
    )

    assert selection.best.name == "spiral"
    np.testing.assert_allclose(selection.best.u5, spiral)
    assert np.isfinite(selection.best.score)
    assert not optimizer.probe_pending


def test_spiral_combines_xy_with_proportional_preload() -> None:
    controller = _controller(_state46())
    wrench5 = np.asarray(
        [0.0, 0.0, 0.75 * controller.config.optimizer.axial_preload_n, 0.0, 0.0]
    )

    u5 = controller._spiral_micro_action(wrench5)

    assert np.linalg.norm(u5[:2]) > 0.0
    assert 0.0 < u5[2] < 0.5 * controller.config.optimizer.advance_step_m


def test_spiral_target_is_held_for_actuator_tracking() -> None:
    controller = _controller(_state46())
    wrench5 = np.asarray(
        [0.0, 0.0, controller.config.optimizer.axial_preload_n, 0.0, 0.0]
    )

    actions = [controller._spiral_micro_action(wrench5) for _ in range(4)]

    assert np.linalg.norm(actions[0][:2]) > 0.0
    np.testing.assert_allclose(actions[1], np.zeros(5))
    assert np.linalg.norm(actions[2][:2]) > 0.0
    np.testing.assert_allclose(actions[3], np.zeros(5))


def test_local_probe_clears_when_preload_recovers() -> None:
    config = OptimizerConfig()
    optimizer = EnergyInformationOptimizer(config)
    low_wrench = np.asarray([0.0, 0.0, 0.2, 0.0, 0.0])
    optimizer.select(
        low_wrench,
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        forced_name="advance",
    )
    assert optimizer.probe_pending

    supported_wrench = low_wrench.copy()
    supported_wrench[2] = config.axial_preload_n
    optimizer.select(
        supported_wrench,
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        command_offset5=np.asarray([0.0, 0.0, 0.001, 0.0, 0.0]),
    )

    assert not optimizer.probe_pending
    assert optimizer.axial_probe_cells == 1
    assert not optimizer.entry_mode


def test_spiral_probe_with_xy_starts_local_probe() -> None:
    config = OptimizerConfig()
    optimizer = EnergyInformationOptimizer(config)
    probe = np.asarray([0.0004, 0.0002, 0.0004, 0.0, 0.0])

    selection = optimizer.select(
        np.asarray([0.0, 0.0, 0.2, 0.0, 0.0]),
        np.zeros(5),
        cumulative_positive_work_j=0.0,
        forced_name="spiral_probe",
        extra_candidate=("spiral_probe", probe),
    )

    assert selection.best.name == "spiral_probe"
    assert optimizer.probe_pending
