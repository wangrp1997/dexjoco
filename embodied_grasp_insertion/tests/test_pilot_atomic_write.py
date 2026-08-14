"""Mock/unit tests for pilot atomic write (no MuJoCo, no formal out_root)."""

from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np

from embodied_grasp_insertion.pilot import ALLOWED_OUT_ROOT, WRITE_IMPLEMENTATION_ENABLED
from embodied_grasp_insertion.pilot.atomic_write import (
    PilotWriteError,
    PilotWriteRefused,
    commit_trajectory,
    commit_trajectory_mock,
    simulate_mid_write_failure_then_cleanup,
)
from embodied_grasp_insertion.pilot.paths import PilotPathError
from embodied_grasp_insertion.pilot.traj_schema import (
    PilotSchemaError,
    validate_meta,
    validate_run_manifest,
)


def _meta(**over):
    m = {
        "traj_id": str(uuid.uuid4()),
        "pilot_tag": "micro_demo_pilot_v0",
        "training_forbidden": True,
        "dry_run": False,
        "geometry_family_id": "round_8mm",
        "target_instance_id": "site_a",
        "socket_site": "site_a",
        "root_source": "demo_transport",
        "matched_snapshot_branch": True,
        "snap_call_count_after_establish": 0,
        "is_insertion_demo": False,
        "created_at": "2026-08-14T00:00:00Z",
        "horizon_steps_used": 80,
        "horizon_budget_max": 80,
        "oracle_usage": {"note": "test"},
        "episode_index": 0,
        "root_frame": 10,
    }
    m.update(over)
    return m


def _labels(**over):
    lab = {
        "gates": [
            {"name": "physical_grasp", "passed": True},
            {"name": "target_hole_semantics", "passed": True},
            {"name": "insert_label_consistency", "passed": True},
        ],
        "all_gates_passed": True,
        "insert_phase": "skipped",
        "insert_ok": False,
        "is_insertion_demo": False,
        "stop_reason": None,
    }
    lab.update(over)
    return lab


def _states(t: int = 4):
    return {
        "t": np.arange(t, dtype=np.int64),
        "dummy": np.zeros((t, 1), dtype=np.float64),
    }


class TestTrajSchema(unittest.TestCase):
    def test_meta_ok(self):
        validate_meta(_meta())

    def test_meta_rejects_dry_run_true(self):
        with self.assertRaises(PilotSchemaError):
            validate_meta(_meta(dry_run=True))

    def test_meta_rejects_bad_uuid(self):
        with self.assertRaises(PilotSchemaError):
            validate_meta(_meta(traj_id="not-a-uuid"))

    def test_meta_rejects_insert_demo(self):
        with self.assertRaises(PilotSchemaError):
            validate_meta(_meta(is_insertion_demo=True))

    def test_meta_rejects_used_gt_budget(self):
        with self.assertRaises(PilotSchemaError):
            validate_meta(_meta(horizon_steps_used=81, horizon_budget_max=80))

    def test_manifest_rejects_write_ok_empty_traj(self):
        with self.assertRaises(PilotSchemaError):
            validate_run_manifest(
                {
                    "protocol": "micro_demo_pilot_v0",
                    "run_id": str(uuid.uuid4()),
                    "created_at": "2026-08-14T00:00:00Z",
                    "dry_run": False,
                    "WRITE_IMPLEMENTATION_ENABLED": False,
                    "trajectories": [],
                    "verdict": "write_ok",
                }
            )


class TestAtomicWriteMock(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(dir="/tmp")
        self.root = Path(self._td.name) / "pilot_mock"
        self.addCleanup(self._td.cleanup)

    def test_write_impl_still_false(self):
        self.assertFalse(WRITE_IMPLEMENTATION_ENABLED)

    def test_production_commit_refused(self):
        with self.assertRaises(PilotWriteRefused):
            commit_trajectory(meta=_meta(), labels=_labels(), states=_states())
        self.assertFalse(ALLOWED_OUT_ROOT.exists())

    def test_mock_success(self):
        meta = _meta()
        res = commit_trajectory_mock(
            out_root=self.root, meta=meta, labels=_labels(), states=_states()
        )
        self.assertTrue((res.traj_dir / "COMMITTED").is_file())
        self.assertTrue((res.traj_dir / "meta.json").is_file())
        self.assertTrue((res.traj_dir / "labels.json").is_file())
        self.assertTrue((res.traj_dir / "states.npz").is_file())
        self.assertTrue(res.manifest_path.is_file())
        tmp = self.root / ".tmp"
        if tmp.exists():
            self.assertEqual(list(tmp.iterdir()), [])
        man = json.loads(res.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(man["verdict"], "write_ok")

    def test_refuse_overwrite(self):
        meta = _meta()
        commit_trajectory_mock(
            out_root=self.root, meta=meta, labels=_labels(), states=_states()
        )
        with self.assertRaises(PilotWriteError):
            commit_trajectory_mock(
                out_root=self.root, meta=meta, labels=_labels(), states=_states()
            )

    def test_mid_failure_cleanup(self):
        meta = _meta()
        simulate_mid_write_failure_then_cleanup(
            out_root=self.root, meta=meta, labels=_labels()
        )
        self.assertFalse((self.root / "trajectories" / meta["traj_id"]).exists())

    def test_object_dtype_rejected(self):
        bad = {"t": np.array([1, 2], dtype=object)}
        with self.assertRaises(PilotSchemaError):
            commit_trajectory_mock(
                out_root=self.root, meta=_meta(), labels=_labels(), states=bad
            )

    def test_mock_rejects_formal_out_root(self):
        with self.assertRaises(PilotWriteError):
            commit_trajectory_mock(
                out_root=ALLOWED_OUT_ROOT,
                meta=_meta(),
                labels=_labels(),
                states=_states(),
            )

    def test_insert_ok_true_rejected(self):
        with self.assertRaises(PilotSchemaError):
            commit_trajectory_mock(
                out_root=self.root,
                meta=_meta(),
                labels=_labels(insert_ok=True),
                states=_states(),
            )

    def test_manifest_fail_rolls_back_traj(self):
        meta = _meta()
        with self.assertRaises(PilotWriteError) as ctx:
            commit_trajectory_mock(
                out_root=self.root,
                meta=meta,
                labels=_labels(),
                states=_states(),
                inject_fail_after="manifest_write",
            )
        self.assertIn("rolled_back=True", str(ctx.exception))
        self.assertFalse((self.root / "trajectories" / meta["traj_id"]).exists())
        incomplete = list((self.root / "manifests").glob("incomplete_run_*.json"))
        self.assertEqual(len(incomplete), 1)
        man = json.loads(incomplete[0].read_text(encoding="utf-8"))
        self.assertEqual(man["verdict"], "incomplete")

    def test_traj_rename_inject_rolls_back(self):
        meta = _meta()
        with self.assertRaises(PilotWriteError):
            commit_trajectory_mock(
                out_root=self.root,
                meta=meta,
                labels=_labels(),
                states=_states(),
                inject_fail_after="traj_rename",
            )
        self.assertFalse((self.root / "trajectories" / meta["traj_id"]).exists())

    def test_reject_trajectories_symlink(self):
        self.root.mkdir(parents=True)
        (self.root / "manifests").mkdir()
        (self.root / ".tmp").mkdir()
        real = Path(self._td.name) / "evil_trajs"
        real.mkdir()
        (self.root / "trajectories").symlink_to(real)
        with self.assertRaises((PilotPathError, PilotWriteError)):
            commit_trajectory_mock(
                out_root=self.root, meta=_meta(), labels=_labels(), states=_states()
            )

    def test_reject_dotdot_mock_path(self):
        with self.assertRaises((PilotPathError, PilotWriteError)):
            commit_trajectory_mock(
                out_root="/tmp/../etc/passwd",
                meta=_meta(),
                labels=_labels(),
                states=_states(),
            )


if __name__ == "__main__":
    unittest.main()
