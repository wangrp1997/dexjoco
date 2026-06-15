#!/usr/bin/env python3
"""Expand GR00T N1.5 action head action_dim (default 32 -> 44) for DexJoCo dual-arm IL.

Mirrors openpi/scripts/convert_to_action_dim_44_model.py:
  - keep pretrained weights on the first SOURCE_ACTION_DIM dimensions
  - randomly initialize new dimensions with CategorySpecificLinear defaults (std=0.02)

Only touches action encoder W1 (input) and action decoder layer2 (output). Backbone and
DiT weights are copied unchanged.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
import tyro
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

SOURCE_ACTION_DIM = 32
DEFAULT_TARGET_ACTION_DIM = 44
INIT_STD = 0.02  # lerobot CategorySpecificLinear default

W1_KEY = "action_head.action_encoder.W1.W"
DECODER_W_KEY = "action_head.action_decoder.layer2.W"
DECODER_B_KEY = "action_head.action_decoder.layer2.b"
ACTION_SHARD = "model-00003-of-00003.safetensors"


def _resolve_input_path(input_path: Path | str) -> Path:
    path = Path(input_path)
    if path.exists():
        return path.resolve()
    print(f"Downloading {input_path} from Hugging Face Hub...")
    return Path(snapshot_download(str(input_path), repo_type="model")).resolve()


def _expand_w1(weight: torch.Tensor, target_dim: int) -> torch.Tensor:
    # (num_embodiments, action_dim, hidden)
    num_cat, in_dim, hidden = weight.shape
    if in_dim != SOURCE_ACTION_DIM:
        raise ValueError(f"{W1_KEY}: expected in_dim={SOURCE_ACTION_DIM}, got {in_dim}")
    expanded = INIT_STD * torch.randn(num_cat, target_dim, hidden, dtype=weight.dtype)
    expanded[:, :SOURCE_ACTION_DIM, :] = weight
    return expanded


def _expand_decoder_w(weight: torch.Tensor, target_dim: int) -> torch.Tensor:
    # (num_embodiments, hidden, action_dim)
    num_cat, hidden, out_dim = weight.shape
    if out_dim != SOURCE_ACTION_DIM:
        raise ValueError(f"{DECODER_W_KEY}: expected out_dim={SOURCE_ACTION_DIM}, got {out_dim}")
    expanded = INIT_STD * torch.randn(num_cat, hidden, target_dim, dtype=weight.dtype)
    expanded[:, :, :SOURCE_ACTION_DIM] = weight
    return expanded


def _expand_decoder_b(bias: torch.Tensor, target_dim: int) -> torch.Tensor:
    # (num_embodiments, action_dim)
    num_cat, out_dim = bias.shape
    if out_dim != SOURCE_ACTION_DIM:
        raise ValueError(f"{DECODER_B_KEY}: expected out_dim={SOURCE_ACTION_DIM}, got {out_dim}")
    expanded = torch.zeros(num_cat, target_dim, dtype=bias.dtype)
    expanded[:, :SOURCE_ACTION_DIM] = bias
    return expanded


def _update_config(config_path: Path, target_dim: int) -> None:
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if config.get("action_dim") != SOURCE_ACTION_DIM:
        raise ValueError(
            f"Expected config action_dim={SOURCE_ACTION_DIM}, got {config.get('action_dim')}"
        )
    config["action_dim"] = target_dim
    head_cfg = config.get("action_head_cfg")
    if not isinstance(head_cfg, dict):
        raise ValueError("Missing action_head_cfg in config.json")
    if head_cfg.get("action_dim") != SOURCE_ACTION_DIM:
        raise ValueError(
            f"Expected action_head_cfg.action_dim={SOURCE_ACTION_DIM}, "
            f"got {head_cfg.get('action_dim')}"
        )
    head_cfg["action_dim"] = target_dim
    head_cfg["max_action_dim"] = target_dim
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def convert_action_head_shard(shard_path: Path, target_dim: int) -> None:
    tensors = load_file(shard_path)
    missing = [k for k in (W1_KEY, DECODER_W_KEY, DECODER_B_KEY) if k not in tensors]
    if missing:
        raise KeyError(f"Missing expected tensors in {shard_path}: {missing}")

    w1_before = tuple(tensors[W1_KEY].shape)
    dec_w_before = tuple(tensors[DECODER_W_KEY].shape)
    dec_b_before = tuple(tensors[DECODER_B_KEY].shape)

    tensors[W1_KEY] = _expand_w1(tensors[W1_KEY], target_dim)
    tensors[DECODER_W_KEY] = _expand_decoder_w(tensors[DECODER_W_KEY], target_dim)
    tensors[DECODER_B_KEY] = _expand_decoder_b(tensors[DECODER_B_KEY], target_dim)

    save_file(tensors, shard_path)

    print(f"Updated {shard_path.name}:")
    print(f"  {W1_KEY}: {w1_before} -> {tuple(tensors[W1_KEY].shape)}")
    print(f"  {DECODER_W_KEY}: {dec_w_before} -> {tuple(tensors[DECODER_W_KEY].shape)}")
    print(f"  {DECODER_B_KEY}: {dec_b_before} -> {tuple(tensors[DECODER_B_KEY].shape)}")


def main(
    output_path: Path,
    input_path: Path | str = "nvidia/GR00T-N1.5-3B",
    target_action_dim: int = DEFAULT_TARGET_ACTION_DIM,
    seed: int = 0,
) -> None:
    if target_action_dim <= SOURCE_ACTION_DIM:
        raise ValueError(
            f"target_action_dim must be > {SOURCE_ACTION_DIM}, got {target_action_dim}"
        )

    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_path}. "
            "Remove it or choose a different --output-path."
        )

    torch.manual_seed(seed)

    source_dir = _resolve_input_path(input_path)
    print(f"Copying {source_dir} -> {output_path}")
    shutil.copytree(source_dir, output_path)

    shard_path = output_path / ACTION_SHARD
    if not shard_path.exists():
        raise FileNotFoundError(f"Expected action-head shard not found: {shard_path}")

    convert_action_head_shard(shard_path, target_action_dim)
    _update_config(output_path / "config.json", target_action_dim)

    print(f"Saved converted GR00T checkpoint to: {output_path}")
    print(f"Use --policy.base_model_path={output_path} with --policy.max_action_dim={target_action_dim}")


if __name__ == "__main__":
    tyro.cli(main)
