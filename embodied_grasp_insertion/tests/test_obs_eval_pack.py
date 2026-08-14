"""Unit tests for P0-Obs-D1 eval pack helpers (no MuJoCo)."""

from __future__ import annotations

import unittest

import numpy as np

from embodied_grasp_insertion.observability.eval_pack import (
    PRIMARY_H,
    STORE_H,
    build_fixed_split,
    sample_meta,
    slice_view,
    validate_sample_arrays,
)


class TestEvalPack(unittest.TestCase):
    def test_slice_views(self):
        a = np.arange(STORE_H * 44, dtype=np.float64).reshape(STORE_H, 44)
        self.assertEqual(slice_view(a, 1).shape, (1, 44))
        self.assertEqual(slice_view(a, PRIMARY_H).shape, (PRIMARY_H, 44))
        self.assertEqual(slice_view(a, STORE_H).shape, (STORE_H, 44))
        np.testing.assert_array_equal(slice_view(a, 4), a[:4])

    def test_validate_ok_and_forbidden(self):
        frames = np.arange(100, 100 + STORE_H, dtype=np.int64)
        data = {
            "frames": frames,
            "act44": np.zeros((STORE_H, 44)),
            "ft12": np.zeros((STORE_H, 12)),
            "o2h_translation_m": np.zeros((STORE_H, 3)),
            "o2h_rotvec_rad": np.zeros((STORE_H, 3)),
            "finger_force_norm_N": np.zeros((STORE_H, 4)),
            "contact_active": np.zeros((STORE_H, 4), dtype=bool),
            "o2h_vel_available": np.array([False] + [True] * (STORE_H - 1)),
        }
        meta = sample_meta(
            episode_index=0,
            root_id="0:100:transport",
            root_phase="transport",
            root_frame=100,
            split="train",
            geometry_family_id="round_8mm",
            target_instance_id="socket_site",
            socket_site="socket_site",
        )
        self.assertEqual(validate_sample_arrays(data, meta), [])
        bad = dict(data)
        bad["peg7"] = np.zeros(7)
        self.assertTrue(any("forbidden" in x for x in validate_sample_arrays(bad, meta)))

    def test_split_digest_stable(self):
        a = build_fixed_split()
        b = build_fixed_split()
        self.assertEqual(a["digest"], b["digest"])
        self.assertEqual(a["counts"]["train"], 70)


if __name__ == "__main__":
    unittest.main()
