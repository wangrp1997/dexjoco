"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def detect_action_dim(ckpt_path: Path | str) -> int:
    """Infer PoseDP action_dim from checkpoint (9=pose9, 12=dual wrist)."""
    load_kw: dict = {"map_location": "cpu"}
    try:
        state = torch.load(Path(ckpt_path), **load_kw, weights_only=True)
    except TypeError:
        state = torch.load(Path(ckpt_path), **load_kw)
    for key, val in state.items():
        if "action_decoder.model.final_conv.1.weight" in key and hasattr(val, "shape"):
            return int(val.shape[0])
        if key.endswith("action_decoder.model.final_conv.1.bias") and hasattr(val, "shape"):
            return int(val.shape[0])
    # fallback: diffusion UNet output dim
    for key, val in state.items():
        if "action_decoder.model.final_conv.1.weight" in key:
            return int(val.shape[0])
    return 9
