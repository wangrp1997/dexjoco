import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from .env import BimanualPhysicsRLEnv
from .causal import CausalAssemblyController


class EnvironmentSmokeTest(unittest.TestCase):
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
