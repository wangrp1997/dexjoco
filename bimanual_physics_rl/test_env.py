import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from .env import BimanualPhysicsRLEnv
from .causal import CausalAssemblyController
from .keypose import _peg_visual_features
from .student import (
    StudentNet,
    RecurrentStudentNet,
    _loss,
    _normalization,
    absolute_to_action,
    action_to_absolute,
    action_to_residual,
    public_observation,
    residual_to_action,
)


class EnvironmentSmokeTest(unittest.TestCase):
    def test_pose_only_loss_ignores_hand_error(self):
        prediction = torch.zeros(1, 1, 44)
        target = prediction.clone()
        target[..., 6:22] = 100.0
        target[..., 28:44] = 100.0
        self.assertEqual(float(_loss(prediction, target, hand_weight=0.0)), 0.0)

    def test_training_normalization_respects_start_step(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.npz"
            np.savez(path, state=np.arange(4, dtype=np.float32)[:, None])
            mean, std = _normalization([path], "state", 1e-4, start_step=2)
            sliced = Path(directory) / "sliced.npz"
            np.savez(
                sliced,
                state=np.arange(2, 4, dtype=np.float32)[:, None],
                first_step=np.asarray(2),
            )
            sliced_mean, sliced_std = _normalization(
                [sliced], "state", 1e-4, start_step=2
            )
        np.testing.assert_allclose(mean, [2.5])
        np.testing.assert_allclose(std, [0.5])
        np.testing.assert_allclose(sliced_mean, mean)
        np.testing.assert_allclose(sliced_std, std)

    def test_peg_visual_features_use_rgb_only(self):
        images = np.zeros((3, 32, 32, 3), dtype=np.uint8)
        images[0, 8:16, 12:20] = (255, 200, 0)
        features = _peg_visual_features(images)
        self.assertEqual(features.shape, (60,))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features[0], 1.0)
        self.assertEqual(features[20], 0.0)

    def test_student_observation_drops_privileged_state_tail(self):
        obs = {
            "state": np.arange(61, dtype=np.float32),
            "ego": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist_left": np.ones((8, 8, 3), dtype=np.uint8),
            "wrist_right": np.full((8, 8, 3), 2, dtype=np.uint8),
        }
        images, state = public_observation(obs)
        self.assertEqual(images.shape, (3, 8, 8, 3))
        np.testing.assert_array_equal(state, np.arange(46, dtype=np.float32))

    def test_student_relative_action_round_trip(self):
        state = np.zeros(46, dtype=np.float32)
        state[[3, 10]] = 1.0
        residual = np.linspace(-0.02, 0.02, 44, dtype=np.float32)
        action = residual_to_action(state, residual)
        np.testing.assert_allclose(action_to_residual(state, action), residual, atol=1e-7)

    def test_student_action_chunk_shape(self):
        model = StudentNet(action_horizon=4).eval()
        with torch.inference_mode():
            output = model(
                torch.zeros(1, 3, 3, 64, 64),
                torch.zeros(1, 46),
                torch.zeros(1, 44),
            )
        self.assertEqual(tuple(output.shape), (1, 4, 44))

    def test_student_absolute_action_round_trip(self):
        state = np.zeros(46, dtype=np.float32)
        state[[3, 10]] = 1.0
        residual = np.linspace(-0.02, 0.02, 44, dtype=np.float32)
        action = residual_to_action(state, residual)
        np.testing.assert_allclose(
            absolute_to_action(action_to_absolute(action)), action, atol=1e-7
        )

    def test_recurrent_student_shape(self):
        model = RecurrentStudentNet().eval()
        with torch.inference_mode():
            output, hidden = model(
                torch.zeros(1, 2, 3, 3, 64, 64),
                torch.zeros(1, 2, 46),
                torch.zeros(1, 2, 44),
            )
        self.assertEqual(tuple(output.shape), (1, 2, 44))
        self.assertEqual(tuple(hidden.shape), (1, 1, 256))

    def test_causal_mode_exposes_only_three_physical_gains(self):
        env = BimanualPhysicsRLEnv(seed=5, causal_templates="unused")
        try:
            self.assertEqual(env.action_space.shape, (3,))
        finally:
            env.close()

    def test_precision_phases_require_stable_geometric_alignment(self):
        data = SimpleNamespace(
            xmat=np.stack([np.eye(3).ravel(), np.eye(3).ravel()]),
            xpos=np.zeros((2, 3)),
            site_xpos=np.asarray([[0.0005, 0.0, 0.2]]),
        )
        raw = SimpleNamespace(
            _data=data,
            _peg_body_id=0,
            _socket_body_id=1,
            _peg_tip_site_id=0,
        )
        controller = object.__new__(CausalAssemblyController)
        controller.raw = raw
        controller._fine_phase = "align"
        controller._fine_step = 0
        controller._fine_stable = 0

        for _ in range(9):
            controller._advance_precision_phase()
        self.assertEqual(controller._fine_phase, "align")
        controller._advance_precision_phase()
        self.assertEqual(controller._fine_phase, "orient")
        for _ in range(10):
            controller._advance_precision_phase()
        self.assertEqual(controller._fine_phase, "insert")

    def test_step_preserves_native_success_contract(self):
        env = BimanualPhysicsRLEnv(seed=7)
        try:
            observation, info = env.reset()
            self.assertEqual(observation.shape, env.observation_space.shape)
            self.assertEqual(info["is_success"], info["succeed"])
            self.assertEqual(env.raw.hz, 0)

            observation, reward, _, _, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            self.assertTrue(np.isfinite(observation).all())
            self.assertTrue(np.isfinite(reward))
            self.assertEqual(info["is_success"], info["succeed"])
        finally:
            env.close()

    def test_reference_root_bank_restores_and_steps(self):
        source = BimanualPhysicsRLEnv(seed=11)
        try:
            source.reset()
            raw = source.raw
            state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(raw._model, state_spec))
            mujoco.mj_getState(raw._model, raw._data, state, state_spec)
            reference = np.concatenate(
                [
                    raw._data.mocap_pos[raw._mocap_right_id],
                    raw._data.mocap_quat[raw._mocap_right_id],
                    raw._data.mocap_pos[raw._mocap_left_id],
                    raw._data.mocap_quat[raw._mocap_left_id],
                    raw._data.ctrl[raw._allegro_ctrl_ids],
                ]
            ).astype(np.float32)
            next_reference = reference.copy()
            next_reference[[0, 7]] += 0.001
            payload = {
                "state": state[None],
                "table_pos": raw._model.body_pos[raw._table_body_id][None],
                "leg_sizes": raw._model.geom_size[raw._table_leg_geom_ids][None],
                "delta_h": np.asarray([raw.delta_h]),
                "offset": np.asarray([2], dtype=np.int32),
                "source_episode": np.asarray([3], dtype=np.int32),
                "source_frame": np.asarray([9], dtype=np.int32),
                "reference_action": np.stack([reference, next_reference])[None],
                "reference_length": np.asarray([2], dtype=np.int32),
                "state_spec": np.asarray(int(state_spec), dtype=np.int64),
                "version": np.asarray(2, dtype=np.int32),
            }
        finally:
            source.close()

        with tempfile.TemporaryDirectory() as directory:
            root_path = Path(directory) / "roots.npz"
            np.savez_compressed(root_path, **payload)
            env = BimanualPhysicsRLEnv(
                seed=11,
                root_bank=root_path,
                episode_steps=3,
                root_noise=0.002,
                online_bias_correction=True,
            )
            try:
                observation, info = env.reset()
                self.assertEqual(info["root_source_episode"], 3)
                np.testing.assert_allclose(observation[-15:-1], reference[:14])
                self.assertEqual(observation[-1], 0.0)
                observation, reward, _, _, _ = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                np.testing.assert_allclose(observation[-15:-1], next_reference[:14])
                self.assertEqual(observation[-1], 1.0)
                env.step(np.zeros(env.action_space.shape, dtype=np.float32))
                np.testing.assert_allclose(
                    env._estimated_reference_bias, env._reference_pos_bias, atol=1e-7
                )
                mocap_ids = [env.raw._mocap_right_id, env.raw._mocap_left_id]
                np.testing.assert_allclose(
                    env.raw._data.mocap_pos[mocap_ids],
                    next_reference[[0, 1, 2, 7, 8, 9]].reshape(2, 3),
                    atol=1e-7,
                )
                self.assertTrue(np.isfinite(observation).all())
                self.assertTrue(np.isfinite(reward))
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
