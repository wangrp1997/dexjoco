"""Unit tests for frozen P0-C2 root criteria (no MuJoCo)."""

from __future__ import annotations

import unittest

from embodied_grasp_insertion.physics.c2_root_criteria import (
    accept_screened_root,
    select_ranked_roots,
)


class TestC2RootCriteria(unittest.TestCase):
    def test_accept_excited_intervenable(self):
        hold = {
            "trans_drift_max_m": 0.005,
            "contact_retention_vs_root_mean": 1.0,
            "peg_contact_absent_steps": 0,
            "terminal_peg_ok": True,
            "object_dropped_proxy": False,
        }
        d = accept_screened_root(
            root_contact_total=3,
            root_peg_ok=True,
            root_insert_ok=False,
            hold_metrics=hold,
        )
        self.assertTrue(d["accepted"])
        self.assertIn("hold_trans_drift", d["excited_reasons"])

    def test_reject_stable_ceiling(self):
        hold = {
            "trans_drift_max_m": 0.0005,
            "contact_retention_vs_root_mean": 1.0,
            "peg_contact_absent_steps": 0,
            "terminal_peg_ok": True,
            "object_dropped_proxy": False,
        }
        d = accept_screened_root(
            root_contact_total=3,
            root_peg_ok=True,
            root_insert_ok=False,
            hold_metrics=hold,
        )
        self.assertFalse(d["accepted"])

    def test_rank_caps(self):
        cands = []
        for ep in (0, 0, 0, 1, 2):
            cands.append(
                {
                    "accepted": True,
                    "episode_index": ep,
                    "hold_metrics": {"trans_drift_max_m": 0.01 - 0.001 * len(cands)},
                }
            )
        picked = select_ranked_roots(cands, max_total=3, max_per_episode=1)
        self.assertEqual(len(picked), 3)
        self.assertEqual(len({p["episode_index"] for p in picked}), 3)


if __name__ == "__main__":
    unittest.main()
