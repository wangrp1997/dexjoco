"""Tests for audit path helpers (/tmp-safe)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from embodied_grasp_insertion.io_paths import path_for_manifest


class TestPathForManifest(unittest.TestCase):
    def test_under_project_is_relative(self):
        root = Path("/home/wangrenpeng/dexjoco/embodied_grasp_insertion")
        p = root / "docs" / "OBSERVABILITY_PRIVILEGED_LABEL_SMOKE.md"
        out = path_for_manifest(p, project_root=root)
        self.assertEqual(out, "docs/OBSERVABILITY_PRIVILEGED_LABEL_SMOKE.md")
        self.assertFalse(out.startswith("/"))

    def test_tmp_falls_back_to_absolute(self):
        root = Path("/home/wangrenpeng/dexjoco/embodied_grasp_insertion")
        with tempfile.TemporaryDirectory(dir="/tmp") as td:
            p = Path(td) / "out" / "manifest.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}", encoding="utf-8")
            out = path_for_manifest(p, project_root=root)
            self.assertTrue(out.startswith("/tmp"), out)
            # Must not raise; must be absolute when outside root
            self.assertEqual(Path(out), p.resolve())


if __name__ == "__main__":
    unittest.main()
