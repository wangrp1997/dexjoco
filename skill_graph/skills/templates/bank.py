"""Grasp template bank load/save."""

from __future__ import annotations

import json
from pathlib import Path

from skill_graph.constants import ObjectName
from skill_graph.paths import template_bank_dir
from skill_graph.skills.templates.schema import GraspTemplate


def bank_index_path(root: Path | None = None) -> Path:
    return (root or template_bank_dir()) / "index.json"


def save_template(template: GraspTemplate, root: Path | None = None) -> Path:
    root = root or template_bank_dir()
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{template.template_id}.npz"
    template.save_npz(out)
    _refresh_index(root)
    return out


def _refresh_index(root: Path) -> None:
    entries = []
    for p in sorted(root.glob("ep*_*.npz")):
        t = GraspTemplate.load_npz(p)
        entries.append(
            {
                "template_id": t.template_id,
                "path": p.name,
                "episode_index": t.episode_index,
                "object_name": t.object_name,
                "side": t.side,
                "export_contact_count": t.export_contact_count,
            }
        )
    bank_index_path(root).write_text(json.dumps({"templates": entries}, indent=2), encoding="utf-8")


def load_bank(root: Path | None = None, *, object_name: ObjectName | None = None) -> list[GraspTemplate]:
    root = root or template_bank_dir()
    if not root.is_dir():
        return []
    templates: list[GraspTemplate] = []
    for p in sorted(root.glob("ep*_*.npz")):
        t = GraspTemplate.load_npz(p)
        if object_name is None or t.object_name == object_name:
            templates.append(t)
    return templates
