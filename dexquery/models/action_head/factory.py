"""Construct DexQuery action heads from config."""

from __future__ import annotations

from typing import Any

from .act import ActActionHead
from .base import BaseActionHead
from .diffusion import DiffusionActionHead
from .dit import DitActionHead
from .flow_matching import FlowMatchingActionHead

ACTION_HEAD_TYPES: dict[str, type[BaseActionHead]] = {
    "act": ActActionHead,
    "diffusion": DiffusionActionHead,
    "flow_matching": FlowMatchingActionHead,
    "dit": DitActionHead,
}


def build_action_head(action_head_type: str, **kwargs: Any) -> BaseActionHead:
    """Instantiate a registered action-head backend."""
    key = action_head_type.lower().replace("-", "_")
    if key not in ACTION_HEAD_TYPES:
        supported = ", ".join(sorted(ACTION_HEAD_TYPES))
        raise ValueError(f"Unknown action_head_type={action_head_type!r}. Supported: {supported}")
    return ACTION_HEAD_TYPES[key](**kwargs)
