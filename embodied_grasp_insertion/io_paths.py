"""Path helpers for audit manifests/reports (allow /tmp outputs)."""

from __future__ import annotations

from pathlib import Path


def path_for_manifest(path: Path | str, *, project_root: Path) -> str:
    """Prefer path relative to project_root; fall back to absolute (e.g. /tmp)."""
    p = Path(path).expanduser()
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    root = project_root.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)
