"""Unit tests for pilot dry-run guards (no MuJoCo)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from embodied_grasp_insertion.pilot import ALLOWED_OUT_ROOT, MAX_HORIZON_STEPS, MIN_HORIZON_STEPS
from embodied_grasp_insertion.pilot.config_schema import (
    PilotConfigError,
    plan_physical_horizon,
    validate_pilot_config,
)
from embodied_grasp_insertion.pilot.paths import (
    PilotPathError,
    assert_not_pilot_path_for_training,
    assert_under_allowlisted_out_root,
    reject_symlinks_along_path,
    resolve_strict,
    write_dry_run_report_atomic,
)


def _valid_cfg(**overrides):
    cfg = {
        "dry_run": True,
        "training_forbidden": True,
        "stop_on_first_gate_failure": True,
        "max_families": 1,
        "families": ["round_8mm"],
        "max_episodes_per_family": 2,
        "max_trajectories_per_episode": 1,
        "max_total_trajectories": 1,
        "max_horizon_steps": 80,
        "out_root": "data/pilot_micro_demo_v0",
        "gates": {
            "physical_grasp": True,
            "target_hole_semantics": True,
            "insert_label_consistency": True,
            "require_matched_snapshot": True,
            "require_snap_after_establish_eq_0": True,
            "require_transport_lateral_fields": True,
        },
    }
    cfg.update(overrides)
    return cfg


class TestConfigSchema(unittest.TestCase):
    def test_valid(self):
        v = validate_pilot_config(_valid_cfg())
        self.assertEqual(v.max_total_trajectories, 1)
        self.assertEqual(v.families, ("round_8mm",))

    def test_reject_zero_total(self):
        with self.assertRaises(PilotConfigError):
            validate_pilot_config(_valid_cfg(max_total_trajectories=0))

    def test_reject_negative_horizon(self):
        with self.assertRaises(PilotConfigError):
            validate_pilot_config(_valid_cfg(max_horizon_steps=-1))

    def test_horizon_boundary_1_rejected(self):
        with self.assertRaises(PilotConfigError):
            validate_pilot_config(_valid_cfg(max_horizon_steps=1))

    def test_horizon_boundary_4_rejected(self):
        with self.assertRaises(PilotConfigError):
            validate_pilot_config(_valid_cfg(max_horizon_steps=4))

    def test_horizon_boundary_5_accepted(self):
        v = validate_pilot_config(_valid_cfg(max_horizon_steps=5))
        self.assertEqual(v.max_horizon_steps, MIN_HORIZON_STEPS)
        plan = plan_physical_horizon(5)
        self.assertLessEqual(plan["planned_total_steps"], 5)

    def test_horizon_boundary_80_accepted(self):
        v = validate_pilot_config(_valid_cfg(max_horizon_steps=80))
        self.assertEqual(v.max_horizon_steps, MAX_HORIZON_STEPS)
        plan = plan_physical_horizon(80)
        self.assertLessEqual(plan["planned_total_steps"], 80)

    def test_reject_horizon_above_cap(self):
        with self.assertRaises(PilotConfigError):
            validate_pilot_config(_valid_cfg(max_horizon_steps=MAX_HORIZON_STEPS + 1))

    def test_reject_gate_false(self):
        g = _valid_cfg()["gates"].copy()
        g["physical_grasp"] = False
        with self.assertRaises(PilotConfigError):
            validate_pilot_config(_valid_cfg(gates=g))

    def test_reject_unknown_key(self):
        cfg = _valid_cfg()
        cfg["extra_field"] = 1
        with self.assertRaises(PilotConfigError):
            validate_pilot_config(cfg)

    def test_reject_wrong_type(self):
        with self.assertRaises(PilotConfigError):
            validate_pilot_config(_valid_cfg(max_families="1"))

    def test_reject_too_many_families(self):
        with self.assertRaises(PilotConfigError):
            validate_pilot_config(
                _valid_cfg(families=["round_8mm", "round_16mm"], max_families=1)
            )

    def test_horizon_plan_respects_budget(self):
        plan = plan_physical_horizon(80)
        self.assertLessEqual(plan["planned_total_steps"], 80)
        self.assertGreaterEqual(plan["transport_steps"], 1)


class TestPaths(unittest.TestCase):
    def test_reject_dotdot(self):
        with self.assertRaises(PilotPathError):
            resolve_strict("/tmp/../etc/passwd")

    def test_symlink_rejected_before_resolve(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as td:
            real = Path(td) / "realdir"
            real.mkdir()
            link = Path(td) / "linkdir"
            link.symlink_to(real)
            with self.assertRaises(PilotPathError):
                reject_symlinks_along_path(link / "file.txt")

    def test_allowlist_out_root(self):
        p = assert_under_allowlisted_out_root(ALLOWED_OUT_ROOT / "trajectories" / "x")
        self.assertTrue(str(p).startswith(str(ALLOWED_OUT_ROOT)))

    def test_allowlist_rejects_tmp(self):
        with self.assertRaises(PilotPathError):
            assert_under_allowlisted_out_root("/tmp/not_pilot")

    def test_training_guard_string(self):
        with self.assertRaises(PilotPathError):
            assert_not_pilot_path_for_training("/data/pilot_micro_demo_v0/foo")

    def test_training_guard_symlink_alias(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as td:
            fake_pilot = Path(td) / "pilot_micro_demo_v0"
            fake_pilot.mkdir()
            alias = Path(td) / "alias_pilot"
            alias.symlink_to(fake_pilot)
            # String match on alias name alone may not fire; resolve into a path
            # containing pilot dir name via the symlink target name.
            # Also cover resolved-into-ALLOWED_OUT_ROOT when root exists.
            from embodied_grasp_insertion.pilot.paths import path_mentions_pilot

            self.assertTrue(path_mentions_pilot(fake_pilot))
            with self.assertRaises(PilotPathError):
                assert_not_pilot_path_for_training(fake_pilot)
            # Symlink whose name lacks pilot token but resolves into ALLOWED_OUT_ROOT
            created = False
            if not ALLOWED_OUT_ROOT.exists():
                ALLOWED_OUT_ROOT.mkdir(parents=True, exist_ok=True)
                created = True
            try:
                alias2 = Path(td) / "harmless_name"
                if alias2.exists() or alias2.is_symlink():
                    alias2.unlink()
                alias2.symlink_to(ALLOWED_OUT_ROOT)
                with self.assertRaises(PilotPathError):
                    assert_not_pilot_path_for_training(alias2)
            finally:
                if created:
                    # remove only if we created an empty tree
                    try:
                        if ALLOWED_OUT_ROOT.exists() and not any(ALLOWED_OUT_ROOT.iterdir()):
                            ALLOWED_OUT_ROOT.rmdir()
                            parent = ALLOWED_OUT_ROOT.parent
                            if parent.name == "data" and parent.exists() and not any(parent.iterdir()):
                                pass  # keep data/
                    except OSError:
                        pass

    def test_report_no_overwrite(self):
        path = Path(f"/tmp/pilot_report_test_{os.getpid()}.json")
        if path.exists():
            path.unlink()
        write_dry_run_report_atomic(path, '{"ok":true}\n')
        with self.assertRaises(PilotPathError):
            write_dry_run_report_atomic(path, '{"ok":false}\n')
        path.unlink(missing_ok=True)

    def test_report_rejects_symlink(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as td:
            real = Path(td) / "real.json"
            real.write_text("x\n", encoding="utf-8")
            link = Path(td) / "link.json"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(real)
            with self.assertRaises(PilotPathError):
                write_dry_run_report_atomic(link, '{"x":1}\n')


class TestWriteFlagsRefusedLogic(unittest.TestCase):
    def test_write_implementation_disabled(self):
        from embodied_grasp_insertion.pilot import WRITE_IMPLEMENTATION_ENABLED

        self.assertFalse(WRITE_IMPLEMENTATION_ENABLED)


if __name__ == "__main__":
    unittest.main()
