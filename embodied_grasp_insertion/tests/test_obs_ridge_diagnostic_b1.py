"""Unit tests for P0-Obs-B1 helpers (no MuJoCo)."""

from __future__ import annotations

import unittest

import numpy as np

from embodied_grasp_insertion.observability.ridge_diagnostic_b1 import (
    ORACLE_CONDITION,
    SampleRow,
    T_OBS,
    build_features,
    drift_targets,
    paired_bootstrap_diff,
    per_episode_metrics,
)


def _row() -> SampleRow:
    o2h_t = np.linspace(0, 0.1, 16 * 3).reshape(16, 3)
    o2h_r = np.linspace(0, 0.2, 16 * 3).reshape(16, 3)
    return SampleRow(
        path="x.npz",
        episode_index=1,
        root_id="1:10:transport",
        root_phase="transport",
        root_frame=10,
        split="train",
        act44=np.arange(16 * 44, dtype=np.float64).reshape(16, 44),
        ft12=np.arange(16 * 12, dtype=np.float64).reshape(16, 12) * 0.01,
        o2h_t=o2h_t,
        o2h_r=o2h_r,
    )


class TestRidgeB1(unittest.TestCase):
    def test_drift_and_oracle_excludes_target(self):
        row = _row()
        dt, dr = drift_targets(row, 8)
        np.testing.assert_allclose(dt, row.o2h_t[T_OBS + 8] - row.o2h_t[T_OBS])
        feat = build_features(row, ORACLE_CONDITION, 8)
        # frames 0..T_OBS inclusive → (T_OBS+1)*6
        self.assertEqual(feat.shape, ((T_OBS + 1) * 6,))
        # Must not equal stacking through target frame
        leak = np.concatenate(
            [
                row.o2h_t[: T_OBS + 8 + 1].reshape(-1),
                row.o2h_r[: T_OBS + 8 + 1].reshape(-1),
            ]
        )
        self.assertNotEqual(feat.shape[0], leak.shape[0])

    def test_paired_bootstrap_sign(self):
        ep_a = {0: {"translation_mae_m": 0.1}, 1: {"translation_mae_m": 0.2}}
        ep_b = {0: {"translation_mae_m": 0.3}, 1: {"translation_mae_m": 0.4}}
        d = paired_bootstrap_diff(ep_a, ep_b, metric="translation_mae_m", n_boot=200)
        self.assertTrue(d["significantly_better"])
        self.assertLess(d["ci95_hi"], 0.0)

    def test_per_episode(self):
        eps = np.array([0, 0, 1])
        gt = np.zeros((3, 3))
        pred = np.array([[0.1, 0, 0], [0.3, 0, 0], [0.2, 0, 0]])
        m = per_episode_metrics(eps, pred, gt, np.zeros((3, 3)), np.zeros((3, 3)))
        self.assertAlmostEqual(m[0]["translation_mae_m"], 0.2)


if __name__ == "__main__":
    unittest.main()
