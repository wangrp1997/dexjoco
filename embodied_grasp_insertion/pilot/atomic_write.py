"""Atomic trajectory commit for micro-demo pilot (design + mock tests).

Production entry ``commit_trajectory`` refuses while WRITE_IMPLEMENTATION_ENABLED=False.
Mock entry ``commit_trajectory_mock`` only accepts out_root under /tmp for unit tests.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from embodied_grasp_insertion.pilot import (
    ALLOWED_OUT_ROOT,
    PILOT_TAG,
    WRITE_IMPLEMENTATION_ENABLED,
)
from embodied_grasp_insertion.pilot.paths import (
    PilotPathError,
    assert_under_allowlisted_out_root,
    is_uuid_traj_id,
    reject_symlinks_along_path,
    resolve_strict,
)
from embodied_grasp_insertion.pilot.traj_schema import (
    PilotSchemaError,
    dumps_json,
    validate_labels,
    validate_meta,
    validate_run_manifest,
    validate_states_arrays,
)


class PilotWriteRefused(RuntimeError):
    """Raised when production write is disabled or policy refuses."""


class PilotWriteError(RuntimeError):
    """Raised on write/rollback failures."""


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _under_tmp(path: Path) -> bool:
    tmp = Path("/tmp").resolve()
    try:
        path.resolve().relative_to(tmp)
        return True
    except ValueError:
        return False


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_file_fsync(path: Path, data: bytes | str) -> None:
    if isinstance(data, str):
        payload = data.encode("utf-8")
    else:
        payload = data
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _atomic_rename(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        raise PilotWriteError(f"refuse overwrite existing dest: {dst}")
    os.rename(str(src), str(dst))
    _fsync_dir(dst.parent)


@dataclass(frozen=True)
class CommitResult:
    traj_id: str
    traj_dir: Path
    run_id: str
    manifest_path: Path


def _assert_real_dir(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise PilotPathError(f"{label} must not be a symlink: {path}")
    if not path.is_dir():
        raise PilotWriteError(f"{label} must be a directory: {path}")


def _assert_real_file_if_exists(path: Path, *, label: str) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        raise PilotPathError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise PilotWriteError(f"{label} must be a regular file: {path}")


def _ensure_scaffold_dirs(out_root: Path) -> None:
    """Validate/create trajectories, manifests, .tmp as real directories (no symlinks)."""
    for name in ("trajectories", "manifests", ".tmp"):
        p = out_root / name
        if p.exists() or p.is_symlink():
            _assert_real_dir(p, label=name)
        else:
            p.mkdir(parents=False, exist_ok=False)
            _fsync_dir(out_root)


def _prepare_out_root(out_root: Path) -> None:
    """Create empty out_root scaffold if absent; refuse dirty/symlink scaffolds."""
    reject_symlinks_along_path(out_root)
    if out_root.exists():
        if out_root.is_symlink():
            raise PilotPathError(f"out_root symlink forbidden: {out_root}")
        if not out_root.is_dir():
            raise PilotWriteError(f"out_root must be a directory: {out_root}")
        kids = list(out_root.iterdir())
        allowed_names = {".tmp", "trajectories", "manifests", "README.md", "PILOT_BANNER.json"}
        for k in kids:
            if k.name not in allowed_names:
                raise PilotWriteError(f"out_root has unexpected entry: {k}")
            if k.name in {".tmp", "trajectories", "manifests"}:
                _assert_real_dir(k, label=k.name)
            if k.name in {"README.md", "PILOT_BANNER.json"}:
                _assert_real_file_if_exists(k, label=k.name)
        _ensure_scaffold_dirs(out_root)
        return

    out_root.mkdir(parents=True, exist_ok=False)
    (out_root / "trajectories").mkdir()
    (out_root / "manifests").mkdir()
    (out_root / ".tmp").mkdir()
    _write_file_fsync(
        out_root / "PILOT_BANNER.json",
        dumps_json(
            {
                "training_forbidden": True,
                "revocable": True,
                "pilot_tag": PILOT_TAG,
            }
        ),
    )
    _write_file_fsync(
        out_root / "README.md",
        "# pilot_micro_demo_v0\n\nNOT a training dataset. Revocable. training_forbidden=true.\n",
    )
    _fsync_dir(out_root)


def _cleanup_tree(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _record_incomplete_and_rollback(
    *,
    out_root: Path,
    run_id: str,
    traj_id: str,
    traj_final: Path,
    error: BaseException,
) -> Path | None:
    """Best-effort incomplete manifest, then remove published traj (consistency)."""
    incomplete_path: Path | None = None
    try:
        incomplete = {
            "protocol": PILOT_TAG,
            "run_id": run_id,
            "created_at": _utc(),
            "dry_run": False,
            "WRITE_IMPLEMENTATION_ENABLED": WRITE_IMPLEMENTATION_ENABLED,
            "trajectories": [
                {
                    "traj_id": traj_id,
                    "path": str(traj_final),
                    "gates_ok": True,
                }
            ],
            "verdict": "incomplete",
            "error": f"{type(error).__name__}: {error}",
            "action": "rolled_back_traj_after_manifest_failure",
            "training_forbidden": True,
            "rollback": {"traj_path": str(traj_final)},
        }
        validate_run_manifest(incomplete)
        inc_dir = out_root / "manifests"
        _assert_real_dir(inc_dir, label="manifests")
        name = f"incomplete_run_{run_id}.json"
        dest = inc_dir / name
        if dest.exists() or dest.is_symlink():
            raise PilotWriteError(f"incomplete record already exists: {dest}")
        partial = out_root / ".tmp" / f"incomplete_{run_id}.partial"
        if partial.exists() or partial.is_symlink():
            _cleanup_tree(partial)
        _write_file_fsync(partial, dumps_json(incomplete))
        _atomic_rename(partial, dest)
        incomplete_path = dest
    except Exception:
        incomplete_path = None
    # Always attempt to remove orphan COMMITTED traj.
    try:
        if traj_final.exists() or traj_final.is_symlink():
            _cleanup_tree(traj_final)
    except Exception as rb_err:
        raise PilotWriteError(
            f"manifest failed ({error}); traj rollback also failed ({rb_err}); "
            f"incomplete_record={incomplete_path}"
        ) from rb_err
    return incomplete_path


def _commit_into_root(
    *,
    out_root: Path,
    meta: dict[str, Any],
    labels: dict[str, Any],
    states: dict[str, np.ndarray],
    run_extra: dict[str, Any] | None = None,
    inject_fail_after: str | None = None,
) -> CommitResult:
    validate_meta(meta, require_dry_run_false=True)
    validate_labels(labels)
    validate_states_arrays(states)
    traj_id = str(meta["traj_id"])
    if not is_uuid_traj_id(traj_id):
        raise PilotSchemaError("traj_id invalid")

    _prepare_out_root(out_root)
    traj_final = out_root / "trajectories" / traj_id
    if traj_final.exists() or traj_final.is_symlink():
        raise PilotWriteError(f"traj already exists; refuse overwrite: {traj_final}")

    run_id = str(uuid.uuid4())
    tmp_run = out_root / ".tmp" / run_id
    tmp_traj = tmp_run / f"traj_{traj_id}"
    if tmp_run.exists() or tmp_run.is_symlink():
        raise PilotWriteError(f"tmp run exists: {tmp_run}")
    tmp_run.mkdir(parents=True)
    tmp_traj.mkdir()

    traj_published = False
    try:
        _write_file_fsync(tmp_traj / "meta.json", dumps_json(meta))
        _write_file_fsync(tmp_traj / "labels.json", dumps_json(labels))
        npz_path = tmp_traj / "states.npz"
        with tempfile.NamedTemporaryFile(dir=tmp_traj, delete=False, suffix=".npz") as tf:
            tmp_npz = Path(tf.name)
        try:
            np.savez(tmp_npz, **states)
            loaded = dict(np.load(tmp_npz, allow_pickle=False))
            validate_states_arrays({k: np.asarray(v) for k, v in loaded.items()})
            os.rename(str(tmp_npz), str(npz_path))
            fd = os.open(str(npz_path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            if tmp_npz.exists():
                tmp_npz.unlink(missing_ok=True)

        _write_file_fsync(tmp_traj / "COMMITTED", "")
        _fsync_dir(tmp_traj)

        _assert_real_dir(out_root / "trajectories", label="trajectories")
        _atomic_rename(tmp_traj, traj_final)
        traj_published = True

        if inject_fail_after == "traj_rename":
            raise PilotWriteError("injected failure after traj rename")

        man = {
            "protocol": PILOT_TAG,
            "run_id": run_id,
            "created_at": _utc(),
            "dry_run": False,
            "WRITE_IMPLEMENTATION_ENABLED": WRITE_IMPLEMENTATION_ENABLED,
            "training_forbidden": True,
            "trajectories": [
                {
                    "traj_id": traj_id,
                    "path": str(traj_final),
                    "gates_ok": True,
                }
            ],
            "verdict": "write_ok",
            "rollback": {
                "command": (
                    "python -m embodied_grasp_insertion.scripts.rollback_micro_demo_pilot "
                    "--target <path> --yes"
                ),
                "traj_path": str(traj_final),
            },
        }
        if run_extra:
            man.update(run_extra)
        validate_run_manifest(man)

        if inject_fail_after == "manifest_write":
            raise PilotWriteError("injected failure before manifest write")

        man_name = f"run_{man['created_at'].replace(':', '').replace('-', '')}_{run_id}.json"
        man_final = out_root / "manifests" / man_name
        _assert_real_dir(out_root / "manifests", label="manifests")
        man_partial = out_root / ".tmp" / run_id / "manifest.json.partial"
        (out_root / ".tmp" / run_id).mkdir(parents=True, exist_ok=True)
        _write_file_fsync(man_partial, dumps_json(man))
        _atomic_rename(man_partial, man_final)

        _cleanup_tree(tmp_run)
        return CommitResult(
            traj_id=traj_id,
            traj_dir=traj_final,
            run_id=run_id,
            manifest_path=man_final,
        )
    except Exception as e:
        _cleanup_tree(tmp_run)
        if traj_published:
            inc = _record_incomplete_and_rollback(
                out_root=out_root,
                run_id=run_id,
                traj_id=traj_id,
                traj_final=traj_final,
                error=e,
            )
            raise PilotWriteError(
                f"commit aborted after traj publish; rolled_back=True; "
                f"incomplete_record={inc}; cause={type(e).__name__}: {e}"
            ) from e
        if traj_final.exists() and not (traj_final / "COMMITTED").exists():
            _cleanup_tree(traj_final)
        raise


def commit_trajectory(
    *,
    meta: dict[str, Any],
    labels: dict[str, Any],
    states: dict[str, np.ndarray],
    run_extra: dict[str, Any] | None = None,
) -> CommitResult:
    """Production commit — refused while WRITE_IMPLEMENTATION_ENABLED is False."""
    if not WRITE_IMPLEMENTATION_ENABLED:
        raise PilotWriteRefused(
            "WRITE_IMPLEMENTATION_ENABLED=False; production trajectory write refused"
        )
    out_root = assert_under_allowlisted_out_root(ALLOWED_OUT_ROOT)
    return _commit_into_root(
        out_root=out_root,
        meta=meta,
        labels=labels,
        states=states,
        run_extra=run_extra,
    )


def commit_trajectory_mock(
    *,
    out_root: Path | str,
    meta: dict[str, Any],
    labels: dict[str, Any],
    states: dict[str, np.ndarray],
    run_extra: dict[str, Any] | None = None,
    inject_fail_after: str | None = None,
) -> CommitResult:
    """Unit-test only: write under /tmp mock root. Never targets formal ALLOWED_OUT_ROOT."""
    if inject_fail_after is not None and inject_fail_after not in (
        "traj_rename",
        "manifest_write",
    ):
        raise PilotWriteError(f"unknown inject_fail_after={inject_fail_after!r}")
    root = resolve_strict(out_root)
    if not _under_tmp(root):
        raise PilotWriteError(f"mock out_root must be under /tmp, got {root}")
    formal = ALLOWED_OUT_ROOT.resolve()
    try:
        root.resolve().relative_to(formal)
        raise PilotWriteError("mock out_root must not be formal ALLOWED_OUT_ROOT")
    except ValueError:
        pass
    return _commit_into_root(
        out_root=root,
        meta=meta,
        labels=labels,
        states=states,
        run_extra=run_extra,
        inject_fail_after=inject_fail_after,
    )


def simulate_mid_write_failure_then_cleanup(
    *,
    out_root: Path,
    meta: dict[str, Any],
    labels: dict[str, Any],
) -> None:
    """Test helper: create tmp traj then abort and cleanup (no final traj)."""
    validate_meta(meta, require_dry_run_false=True)
    validate_labels(labels)
    _prepare_out_root(out_root)
    traj_id = str(meta["traj_id"])
    run_id = str(uuid.uuid4())
    tmp_run = out_root / ".tmp" / run_id
    tmp_traj = tmp_run / f"traj_{traj_id}"
    tmp_run.mkdir(parents=True)
    tmp_traj.mkdir()
    _write_file_fsync(tmp_traj / "meta.json", dumps_json(meta))
    _cleanup_tree(tmp_run)
    final = out_root / "trajectories" / traj_id
    if final.exists():
        raise PilotWriteError("cleanup failed; final traj exists")
