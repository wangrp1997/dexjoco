"""Unit tests for P0-Obs-D0 feasibility helpers (no MuJoCo)."""

from __future__ import annotations

import unittest

from embodied_grasp_insertion.observability.feasibility import (
    FrameRec,
    RootRec,
    assign_roots_to_splits,
    atomic_episode_split,
    check_split_leakage,
    count_phase_contiguous_windows,
    count_root_anchored_windows,
    derive_roots_from_history,
)


def _h(frame, peg_ok=True, insert_ok=False, tip=0.1, contact=2, gap=False):
    return FrameRec(
        frame=frame,
        peg_ok=peg_ok,
        insert_ok=insert_ok,
        tip_dist_m=tip,
        tray_ok=False,
        contact_total=contact,
        terminated=False,
        truncated=False,
        gap_from_prev=gap,
    )


class TestFeasibilityHelpers(unittest.TestCase):
    def test_derive_roots_and_windows(self):
        hist = []
        # frames 1..40: peg_ok; tip large after 25
        for f in range(1, 41):
            tip = 0.02 if f < 25 else 0.12
            hist.append(_h(f, tip=tip))
        roots = derive_roots_from_history(0, hist)
        phases = {r.phase for r in roots}
        self.assertIn("early_grasp", phases)
        self.assertIn("transport", phases)
        last = hist[-1].frame
        win = count_root_anchored_windows(roots, last_frame=last, terminated_at=None)
        self.assertGreaterEqual(win["H1"], 1)
        self.assertGreaterEqual(win["H8"], 1)

    def test_phase_windows_require_contiguity(self):
        hist = [_h(1), _h(2), _h(3), _h(4), _h(6, gap=True), _h(7)]
        w = count_phase_contiguous_windows(hist)
        self.assertEqual(w["H4"], 1)  # only frames 1-4
        self.assertEqual(w["H1"], 6)
        self.assertEqual(w["H8"], 0)

    def test_atomic_split_no_leak(self):
        split = atomic_episode_split(list(range(100)))
        self.assertEqual(sum(len(v) for v in split.values()), 100)
        roots = [
            RootRec(0, 10, "transport", 0.1, True, False, 2),
            RootRec(50, 20, "early_grasp", 0.1, True, False, 1),
        ]
        # ensure episodes exist in split
        rs = assign_roots_to_splits(roots, split)
        leak = check_split_leakage(split, rs)
        self.assertTrue(leak["ok"], leak)


if __name__ == "__main__":
    unittest.main()
