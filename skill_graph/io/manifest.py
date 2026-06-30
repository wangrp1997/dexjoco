"""Load sidecar manifest.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_graspable_episodes(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in manifest.get("episodes", []):
        timing = entry.get("timing") or {}
        if timing.get("left_grasp_frame") is None or timing.get("right_grasp_frame") is None:
            continue
        if not entry.get("has_tray") or not entry.get("has_peg"):
            continue
        out.append(entry)
    return out
