"""Add forked PoseInsert repo to ``sys.path``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_POSEINSERT_ROOT: Path | None = None


def _candidate_poseinsert_roots(repo_root: Path) -> tuple[Path, ...]:
    return (
        repo_root / "PoseInsert",
        repo_root / "interaction_retarget" / "PoseInsert",
        repo_root / "refs" / "PoseInsert",
    )


def poseinsert_root() -> Path:
    global _POSEINSERT_ROOT
    if _POSEINSERT_ROOT is None:
        env_root = os.environ.get("DEXJOECO_POSEINSERT_ROOT", "").strip()
        if env_root:
            _POSEINSERT_ROOT = Path(env_root).expanduser().resolve()
        else:
            repo_root = Path(__file__).resolve().parents[2]
            for candidate in _candidate_poseinsert_roots(repo_root):
                if (candidate / "policy").is_dir():
                    _POSEINSERT_ROOT = candidate.resolve()
                    break
            else:
                _POSEINSERT_ROOT = repo_root / "PoseInsert"
    return _POSEINSERT_ROOT


def ensure_poseinsert_on_path() -> Path:
    root = poseinsert_root()
    if not root.is_dir():
        raise FileNotFoundError(f"PoseInsert not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
