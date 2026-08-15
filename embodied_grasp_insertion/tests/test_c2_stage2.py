"""Unit tests for C2 Stage-2 helpers (no MuJoCo)."""

from __future__ import annotations

import unittest

import numpy as np

from embodied_grasp_insertion.observability.c2_stage2_action_conditioned import (
    _action_finger_flat,
    episode_equal_mae,
    paired_bootstrap_mae_diff,
)


class TestC2Stage2(unittest.TestCase):
    def test_action_flat(self):
        a = np.zeros((16, 44))
        a[:, 6:22] = 1.0
        self.assertEqual(_action_finger_flat(a).shape, (256,))
        self.assertTrue(np.allclose(_action_finger_flat(a), 1.0))

    def test_paired_better(self):
        eps = np.array([0, 0, 1, 1])
        err_a = np.array([0.1, 0.1, 0.2, 0.2])
        err_b = np.array([0.5, 0.5, 0.6, 0.6])
        d = paired_bootstrap_mae_diff(eps, err_a, err_b, seed=0, n_boot=200)
        self.assertTrue(d["a_significantly_better"])
        self.assertAlmostEqual(episode_equal_mae(eps, np.zeros(4), err_a), 0.15)


if __name__ == "__main__":
    unittest.main()
