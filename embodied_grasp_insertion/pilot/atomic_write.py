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


def _prepare_out_root(out_root: Path) -> None:
    """Create empty out_root scaffold if absent; refuse unexpected dirty roots."""
    reject_symlinks_along_path(out_root)
    if out_root.exists():
        if out_root.is_symlink():
            raise PilotPathError(f"out_root symlink forbidden: {out_root}")
        kids = list(out_root.iterdir())
        allowed_names = {".tmp", "trajectories", "manifests", "README.md", "PILOT_BANNER.json"}
        for k in kids:
            if k.name not in allowed_names:
                raise PilotWriteError(f"out_root has unexpected entry: {k}")
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
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _commit_into_root(
    *,
    out_root: Path,
    meta: dict[str, Any],
    labels: dict[str, Any],
    states: dict[str, np.ndarray],
    run_extra: dict[str, Any] | None = None,
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
    if tmp_run.exists():
        raise PilotWriteError(f"tmp run exists: {tmp_run}")
    tmp_run.mkdir(parents=True)
    tmp_traj.mkdir()

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

        (out_root / "trajectories").mkdir(exist_ok=True)
        _atomic_rename(tmp_traj, traj_final)

        man = {
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

        man_name = f"run_{man['created_at'].replace(':', '').replace('-', '')}_{run_id}.json"
        man_final = out_root / "manifests" / man_name
        (out_root / "manifests").mkdir(exist_ok=True)
        man_partial = out_root / ".tmp" / run_id / "manifest.json.partial"
        # tmp_run may have been emptied of traj; recreate for partial
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
    except Exception:
        _cleanup_tree(tmp_run)
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
) -> CommitResult:
    """Unit-test only: write under /tmp mock root. Never targets formal ALLOWED_OUT_ROOT."""
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
