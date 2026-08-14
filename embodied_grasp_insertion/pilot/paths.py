"""Path allowlist / training guards for micro-demo pilot v0."""

from __future__ import annotations

import os
import stat
import uuid
from pathlib import Path

from embodied_grasp_insertion.pilot import (
    ALLOWED_OUT_ROOT,
    PILOT_DIR_NAME,
    PROJECT_ROOT,
)


class PilotPathError(ValueError):
    """Raised when a path violates pilot allowlist or training ban."""


def is_uuid_traj_id(traj_id: str) -> bool:
    try:
        uuid.UUID(str(traj_id))
        return True
    except ValueError:
        return False


def new_traj_id() -> str:
    return str(uuid.uuid4())


def _lstat_reject_symlink(path: Path) -> None:
    """Reject if this exact path is a symlink (does not follow)."""
    try:
        st = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(st.st_mode):
        raise PilotPathError(f"symlink not allowed: {path}")


def reject_symlinks_along_path(path: Path | str) -> Path:
    """Walk original path components with lstat; reject any symlink before resolve."""
    p = Path(path)
    if ".." in p.parts:
        raise PilotPathError(f"path must not contain '..': {path}")

    # Absolute: walk from root; relative: walk from cwd without resolving yet.
    if p.is_absolute():
        cur = Path(p.anchor)
        parts = p.parts[1:]
    else:
        cur = Path(".")
        parts = p.parts

    for part in parts:
        cur = cur / part
        _lstat_reject_symlink(cur)

    # Also reject if final exists as symlink (already covered) and parents.
    return p


def resolve_strict(path: Path | str) -> Path:
    """lstat-reject symlinks on the original path, then resolve."""
    raw = reject_symlinks_along_path(path)
    # resolve(strict=False) may still traverse if we missed a component; re-check
    # each resolved prefix via lstat on the unresolved rebuild when possible.
    resolved = raw if raw.is_absolute() else (Path.cwd() / raw)
    # Rebuild absolute without following: use os.path.abspath which does not
    # resolve symlinks the same as Path.resolve — still call resolve after checks.
    # Final resolve for allowlist compare:
    out = Path(os.path.abspath(str(raw)))
    # Walk abspath components again with lstat.
    reject_symlinks_along_path(out)
    # Path.resolve follows symlinks — only call after ensuring no symlinks on path.
    return out.resolve(strict=False)


def _under_root(resolved: Path, root: Path) -> bool:
    try:
        resolved.relative_to(root)
    except ValueError:
        return False
    root_s = str(root)
    res_s = str(resolved)
    return res_s == root_s or res_s.startswith(root_s + os.sep)


def assert_under_allowlisted_out_root(path: Path | str) -> Path:
    """Require path to resolve strictly under ALLOWED_OUT_ROOT (allowlist)."""
    resolved = resolve_strict(path)
    root = ALLOWED_OUT_ROOT
    if not _under_root(resolved, root):
        raise PilotPathError(f"path {resolved} is outside allowlisted pilot root {root}")
    return resolved


def assert_out_root_empty_or_absent(out_root: Path | str | None = None) -> Path:
    root = assert_under_allowlisted_out_root(out_root or ALLOWED_OUT_ROOT)
    if not root.exists():
        return root
    if any(root.iterdir()):
        raise PilotPathError(f"out_root not empty; refuse overwrite: {root}")
    return root


def assert_traj_slot_absent(traj_id: str, *, out_root: Path | str | None = None) -> Path:
    if not is_uuid_traj_id(traj_id):
        raise PilotPathError(f"traj_id must be UUID, got {traj_id!r}")
    root = assert_under_allowlisted_out_root(out_root or ALLOWED_OUT_ROOT)
    dest = assert_under_allowlisted_out_root(root / "trajectories" / traj_id)
    if dest.exists():
        raise PilotPathError(f"traj already exists; refuse overwrite: {dest}")
    return dest


def path_mentions_pilot(path: Path | str) -> bool:
    s = str(path).replace("\\", "/")
    return PILOT_DIR_NAME in s.split("/")


def assert_not_pilot_path_for_training(path: Path | str, *, context: str = "") -> None:
    """Refuse pilot data as training input (string match + resolved allowlist root)."""
    ctx = f" ({context})" if context else ""
    if path_mentions_pilot(path):
        raise PilotPathError(
            f"training/dataset path must not include {PILOT_DIR_NAME}{ctx}: {path}"
        )
    # Symlink alias: resolve without requiring path to be under pilot for exists;
    # if resolved lands in ALLOWED_OUT_ROOT, ban.
    try:
        # For training ban we DO want to detect symlink-into-pilot; so follow links
        # carefully: lstat chain for the given path; if any link, resolve target.
        p = Path(path)
        resolved = p.resolve(strict=False)
    except Exception:
        return
    if _under_root(resolved, ALLOWED_OUT_ROOT):
        raise PilotPathError(
            f"training/dataset path resolves into pilot root{ctx}: {path} -> {resolved}"
        )


def assert_dry_run_report_path(path: Path | str) -> Path:
    """Validate /tmp report destination (no overwrite, no symlink). Does not create."""
    raw = Path(path)
    if ".." in raw.parts:
        raise PilotPathError(f"path must not contain '..': {path}")
    if not str(raw).startswith("/tmp/") and str(raw) != "/tmp":
        # Require literal /tmp prefix before resolve to avoid alias tricks.
        raise PilotPathError(f"dry-run report must be under /tmp/, got {path}")

    parent = raw.parent
    reject_symlinks_along_path(parent)
    _lstat_reject_symlink(raw)  # refuse if report path itself is symlink

    parent_abs = Path(os.path.abspath(str(parent))).resolve(strict=False)
    tmp = Path("/tmp").resolve()
    if not _under_root(parent_abs, tmp):
        raise PilotPathError(f"dry-run report parent must be under /tmp, got {parent_abs}")
    try:
        parent_abs.relative_to(PROJECT_ROOT.resolve())
        raise PilotPathError(f"dry-run report must not be under repo: {parent_abs}")
    except ValueError:
        pass

    if not parent.exists():
        raise PilotPathError(f"dry-run report parent must exist: {parent}")
    if raw.exists():
        raise PilotPathError(f"dry-run report already exists; refuse overwrite: {raw}")
    return raw


def write_dry_run_report_atomic(path: Path | str, payload: str) -> Path:
    """Create report with O_CREAT|O_EXCL|O_NOFOLLOW (no overwrite, no symlink follow)."""
    dest = assert_dry_run_report_path(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(str(dest), flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return dest


def safe_delete_under_pilot_root(path: Path | str) -> Path:
    resolved = assert_under_allowlisted_out_root(path)
    if not resolved.exists():
        return resolved
    if resolved.is_dir():
        import shutil

        shutil.rmtree(resolved)
    else:
        resolved.unlink()
    return resolved
