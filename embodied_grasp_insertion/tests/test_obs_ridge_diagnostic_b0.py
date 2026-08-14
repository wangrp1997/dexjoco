"""Unit tests for P0-Obs-B0 Ridge diagnostic helpers (no MuJoCo, no pack I/O)."""

from __future__ import annotations

import unittest

import numpy as np

from embodied_grasp_insertion.observability.ridge_diagnostic_b0 import (
    CONDITION_ORDER,
    SampleRow,
    TARGET_FRAME_IDX,
    _better,
    build_features,
    episode_equal_metrics,
    judge_verdict,
    rotation_geodesic_deg,
)


def _fake_row(ep: int = 0, split: str = "train") -> SampleRow:
    return SampleRow(
        path="x.npz",
        episode_index=ep,
        root_id=f"{ep}:10:transport",
        split=split,
        act44=np.arange(16 * 44, dtype=np.float64).reshape(16, 44),
        ft12=np.arange(16 * 12, dtype=np.float64).reshape(16, 12) * 0.01,
        o2h_t=np.linspace(0, 1, 16 * 3).reshape(16, 3),
        o2h_r=np.linspace(0, 0.5, 16 * 3).reshape(16, 3),
    )


class TestRidgeB0(unittest.TestCase):
    def test_feature_shapes_and_fixed_target(self):
        row = _fake_row()
        self.assertEqual(build_features(row, "A_H1").shape, (44,))
        self.assertEqual(build_features(row, "A_H8").shape, (352,))
        self.assertEqual(build_features(row, "B_H1").shape, (56,))
        self.assertEqual(build_features(row, "B_H8").shape, (448,))
        self.assertEqual(build_features(row, "B_H8_shuffled_FT").shape, (448,))
        self.assertEqual(build_features(row, "privileged_o2h_ceiling").shape, (48,))
        self.assertIsNone(build_features(row, "train_mean"))
        np.testing.assert_array_equal(row.y_t, row.o2h_t[TARGET_FRAME_IDX])

    def test_geodesic_identity_zero(self):
        r = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]])
        d = rotation_geodesic_deg(r, r)
        np.testing.assert_allclose(d, 0.0, atol=1e-8)

    def test_episode_equal_and_judge(self):
        eps = np.array([0, 0, 1, 1])
        gt_t = np.zeros((4, 3))
        pred_t = np.array([[0.1, 0, 0], [0.3, 0, 0], [0.2, 0, 0], [0.2, 0, 0]])
        gt_r = np.zeros((4, 3))
        pred_r = np.zeros((4, 3))
        m = episode_equal_metrics(eps, pred_t, gt_t, pred_r, gt_r)
        # ep0 mean norms 0.2; ep1 mean 0.2 → overall 0.2
        self.assertAlmostEqual(m["translation_mae_m"], 0.2, places=6)

        def pack(t_mae: float, r_mae: float) -> dict:
            return {
                "splits": {
                    "val": {
                        "metrics": {
                            "translation_mae_m": t_mae,
                            "rotation_geodesic_mae_deg": r_mae,
                        }
                    },
                    "test": {
                        "metrics": {
                            "translation_mae_m": t_mae,
                            "rotation_geodesic_mae_deg": r_mae,
                        }
                    },
                }
            }

        by = {name: pack(1.0, 10.0) for name in CONDITION_ORDER}
        by["train_mean"] = pack(1.0, 10.0)
        by["A_H8"] = pack(0.5, 5.0)
        v = judge_verdict(by)
        self.assertEqual(v["overall_verdict"], "diagnostic_signal")
        self.assertIn("A_H8", v["stable_better_than_train_mean"])
        self.assertFalse(v["ft_helps_claim"])
        self.assertFalse(v["claims_observability_p0_pass"])
        self.assertFalse(v["allow_policy_training"])

        by2 = {name: pack(1.0, 10.0) for name in CONDITION_ORDER}
        v2 = judge_verdict(by2)
        self.assertEqual(v2["research_decision"], "stop_sensing_insufficient")
        self.assertTrue(_better(pack(0.1, 1.0)["splits"]["val"]["metrics"], pack(1, 2)["splits"]["val"]["metrics"]))


if __name__ == "__main__":
    unittest.main()
