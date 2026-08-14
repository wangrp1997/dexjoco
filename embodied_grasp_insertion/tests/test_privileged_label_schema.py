"""Unit tests for P0-L1 privileged label velocity contract (no MuJoCo)."""

from __future__ import annotations

import unittest

import numpy as np

from embodied_grasp_insertion.labels.privileged_schema import (
    EXCLUDED_FIELDS,
    SCHEMA_VERSION,
    o2h_velocity_from_poses,
    schema_document,
)
from embodied_grasp_insertion.physics.grasp_metrics import (
    REFERENCE_BODY,
    ObjectInHandPose,
)


class TestVelocityContract(unittest.TestCase):
    def test_finite_diff_linear_and_relative_rot(self):
        a = ObjectInHandPose(
            reference_body=REFERENCE_BODY,
            translation=np.array([0.0, 0.0, 0.0]),
            rotvec=np.array([0.0, 0.0, 0.0]),
        )
        b = ObjectInHandPose(
            reference_body=REFERENCE_BODY,
            translation=np.array([0.02, 0.0, 0.0]),
            rotvec=np.array([0.0, 0.0, 0.1]),
        )
        dt = 0.02
        vel = o2h_velocity_from_poses(a, b, dt)
        self.assertTrue(vel.available)
        self.assertAlmostEqual(vel.linear_mps[0], 1.0, places=12)
        self.assertAlmostEqual(vel.angular_radps[2], 5.0, places=12)

    def test_schema_excludes_slip_and_fine_modes(self):
        doc = schema_document()
        self.assertEqual(doc["schema_version"], SCHEMA_VERSION)
        for name in (
            "slip_truth",
            "slip",
            "contact_mode_jam",
            "contact_mode_capture",
        ):
            self.assertIn(name, EXCLUDED_FIELDS)


if __name__ == "__main__":
    unittest.main()
