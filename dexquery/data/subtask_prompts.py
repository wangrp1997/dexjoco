"""Subtask language prompts for DexQuery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SUBTASK_PROMPTS: dict[str, dict[str, str]] = {
    "bimanual_assembly": {
        "tray": "Grasp the tray with the left hand.",
        "peg": "Grasp the peg with the right hand.",
        "insert": "Insert the peg into the hole.",
    },
}


@dataclass(frozen=True)
class SubtaskPrompts:
    tray: str
    peg: str
    insert: str

    def as_list(self) -> list[str]:
        return [self.tray, self.peg, self.insert]

    @classmethod
    def from_mapping(cls, mapping: dict[str, str]) -> SubtaskPrompts:
        missing = {"tray", "peg", "insert"} - set(mapping)
        if missing:
            raise KeyError(f"subtask_prompts missing keys: {sorted(missing)}")
        return cls(tray=mapping["tray"], peg=mapping["peg"], insert=mapping["insert"])

    @classmethod
    def for_task(cls, task: str, *, config_path: Path | None = None) -> SubtaskPrompts:
        if config_path is not None:
            return cls.from_yaml(config_path)
        if task in DEFAULT_SUBTASK_PROMPTS:
            return cls.from_mapping(DEFAULT_SUBTASK_PROMPTS[task])
        raise KeyError(
            f"No subtask prompts for task {task!r}. "
            f"Add to dexquery/configs/{task}.yaml or DEFAULT_SUBTASK_PROMPTS."
        )

    @classmethod
    def from_yaml(cls, path: Path) -> SubtaskPrompts:
        path = path.expanduser()
        with open(path, "r", encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}
        mapping = cfg.get("subtask_prompts")
        if not mapping:
            raise KeyError(f"No subtask_prompts in {path}")
        return cls.from_mapping(mapping)


def infer_subtask_phase(
    tray_ok: bool,
    peg_ok: bool,
    *,
    tray_prob: float = 1.0,
    peg_prob: float = 1.0,
    insert_min_prob: float = 0.8,
) -> int:
    """Return active subtask index: 0=tray, 1=peg, 2=insert."""
    if not tray_ok:
        return 0
    if not peg_ok:
        return 1
    if tray_prob >= insert_min_prob and peg_prob >= insert_min_prob:
        return 2
    return 1
