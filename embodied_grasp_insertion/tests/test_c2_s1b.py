"""Unit tests for C2-S1b helpers (no MuJoCo)."""

from __future__ import annotations

import unittest

import numpy as np

from embodied_grasp_insertion.scripts.run_p0_c2_s1b import existence_fork, judge_s1b
from embodied_grasp_insertion.simulation.calibrated_interventions import (
    max_feasible_scale_for_offset,
)


class TestC2S1b(unittest.TestCase):
    def test_existence_heterogeneous_vs_direction(self):
        hold = {
            "trans_drift_max_m": 0.02,
            "rot_drift_max_rad": 0.1,
            "contact_retention_vs_root_mean": 1.0,
            "terminal_peg_ok": True,
            "object_dropped_proxy": False,
        }
        better = dict(hold, trans_drift_max_m=0.006, rot_drift_max_rad=0.03)
        ex = existence_fork(
            hold,
            better,
            hold_spreads={k: 0.0 for k in hold},
            interv_spreads={k: 0.0 for k in hold},
            gates={
                "existence_trans_drift_m": 0.001,
                "existence_rot_drift_rad": 0.01,
                "existence_retention_abs": 0.05,
            },
            k=3.0,
        )
        self.assertTrue(ex["exists"])

    def test_judge_three_ways(self):
        a = judge_s1b(
            actuation_moved=False,
            existence_on_frozen=True,
            existence_on_heldout=True,
            directional=False,
        )
        self.assertEqual(a["overall_verdict"], "h2_untested_actuation_dead")
        b = judge_s1b(
            actuation_moved=True,
            existence_on_frozen=False,
            existence_on_heldout=False,
            directional=False,
        )
        self.assertIn("h2_failed", b["overall_verdict"])

    def test_independent_scale_helper_math(self):
        off = np.ones(16) * 0.02
        pos = np.ones(16) * 0.01
        neg = np.ones(16) * 0.01
        s = max_feasible_scale_for_offset(off, pos, neg)
        self.assertAlmostEqual(s, 0.5)


if __name__ == "__main__":
    unittest.main()
