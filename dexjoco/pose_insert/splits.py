"""Train/val episode splits for PoseInsert sim."""

from __future__ import annotations

# Hold-out by episode_index: every 10 + a few extras (never ep35 alone as val gate).
VAL_EPISODE_INDICES = frozenset({0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 96})


def is_val_episode(episode_index: int) -> bool:
    return int(episode_index) in VAL_EPISODE_INDICES


def filter_demo_dir(path, *, split: str) -> bool:
    ep = int(path.name)
    if split == "val":
        return is_val_episode(ep)
    if split == "train":
        return not is_val_episode(ep)
    return True
