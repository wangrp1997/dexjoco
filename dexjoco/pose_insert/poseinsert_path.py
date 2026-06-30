"""Add forked PoseInsert repo to ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

_POSEINSERT_ROOT: Path | None = None


def poseinsert_root() -> Path:
    global _POSEINSERT_ROOT
    if _POSEINSERT_ROOT is None:
        _POSEINSERT_ROOT = Path(__file__).resolve().parents[2] / "PoseInsert"
    return _POSEINSERT_ROOT


def ensure_poseinsert_on_path() -> Path:
    root = poseinsert_root()
    if not root.is_dir():
        raise FileNotFoundError(f"PoseInsert not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
