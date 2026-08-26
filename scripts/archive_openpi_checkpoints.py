#!/usr/bin/env python3
"""Archive finalized OpenPI checkpoints before retention deletes older steps."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _parse_steps(value: str) -> set[int]:
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Training run checkpoint directory.")
    parser.add_argument("--archive", type=Path, required=True, help="Persistent policy-checkpoint archive.")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--expected-steps", type=_parse_steps, default=set())
    parser.add_argument("--training-pid", type=int, default=None)
    parser.add_argument("--once", action="store_true", help="Scan once instead of watching continuously.")
    return parser.parse_args()


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _complete_policy_tree(path: Path) -> bool:
    if not path.is_dir():
        return False
    required = (
        path / "_CHECKPOINT_METADATA",
        path / "params" / "_METADATA",
        path / "params" / "manifest.ocdbt",
        path / "assets",
    )
    if not all(item.exists() for item in required):
        return False
    return not any("tmp" in child.name.lower() for child in path.iterdir())


def _finalized_checkpoint(path: Path) -> bool:
    return path.name.isdigit() and _complete_policy_tree(path)


def _copy_tree(source: Path, destination: Path) -> str:
    attempts = (
        ("hardlink", ["cp", "-al", str(source), str(destination)]),
        ("reflink", ["cp", "-a", "--reflink=auto", str(source), str(destination)]),
    )
    for mode, command in attempts:
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return mode
        except (OSError, subprocess.CalledProcessError):
            shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination, symlinks=True)
    return "copy"


def _archive_checkpoint(source: Path, archive_root: Path) -> bool:
    destination = archive_root / source.name
    if _finalized_checkpoint(destination):
        return False

    temporary = archive_root / f".{source.name}.tmp-{os.getpid()}"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)

    modes: set[str] = set()
    for name in ("params", "assets"):
        modes.add(_copy_tree(source / name, temporary / name))
    shutil.copy2(source / "_CHECKPOINT_METADATA", temporary / "_CHECKPOINT_METADATA")

    manifest = {
        "step": int(source.name),
        "source": str(source.resolve()),
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "storage_modes": sorted(modes),
        "policy_only": True,
    }
    (temporary / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not _complete_policy_tree(temporary):
        shutil.rmtree(temporary, ignore_errors=True)
        raise RuntimeError(f"Archived checkpoint failed validation: {source}")

    if destination.exists():
        shutil.rmtree(destination)
    temporary.rename(destination)
    print(f"archived step {source.name} -> {destination} ({', '.join(sorted(modes))})", flush=True)
    return True


def _scan(source_root: Path, archive_root: Path) -> set[int]:
    archived: set[int] = set()
    if not source_root.is_dir():
        return archived
    for checkpoint in sorted(source_root.iterdir(), key=lambda path: int(path.name) if path.name.isdigit() else -1):
        if _finalized_checkpoint(checkpoint):
            _archive_checkpoint(checkpoint, archive_root)
            archived.add(int(checkpoint.name))
    return archived


def main() -> int:
    args = _parse_args()
    source = args.source.expanduser().resolve()
    archive = args.archive.expanduser().resolve()
    archive.mkdir(parents=True, exist_ok=True)
    print(f"watching {source}", flush=True)
    print(f"archiving to {archive}", flush=True)

    while True:
        _scan(source, archive)
        available = {int(path.name) for path in archive.iterdir() if _finalized_checkpoint(path)}
        if args.expected_steps and args.expected_steps.issubset(available):
            print(f"all expected steps archived: {sorted(args.expected_steps)}", flush=True)
            return 0
        if args.once:
            return 0
        if not _pid_alive(args.training_pid):
            print("training process exited; final scan complete", flush=True)
            return 0
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
