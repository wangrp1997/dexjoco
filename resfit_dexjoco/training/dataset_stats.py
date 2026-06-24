"""Dataset statistics for ResFiT-style normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer


def load_dataset_stats(
    dataset_root: Path,
    *,
    state_dim: int,
    min_action_range: float = 0.1,
    min_state_std: float = 0.1,
    device: str = "cpu",
):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to compute dataset stats.") from exc

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for parquet_path in sorted((dataset_root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(parquet_path, columns=["action", "observation.state"])
        for action, state in zip(
            table.column("action").to_pylist(),
            table.column("observation.state").to_pylist(),
            strict=True,
        ):
            states.append(np.asarray(state, dtype=np.float32).reshape(-1)[:state_dim])
            actions.append(np.asarray(action, dtype=np.float32).reshape(-1))

    state_arr = np.stack(states, axis=0)
    action_arr = np.stack(actions, axis=0)

    state_mean = torch.as_tensor(state_arr.mean(axis=0), dtype=torch.float32)
    state_std = torch.as_tensor(state_arr.std(axis=0), dtype=torch.float32)
    action_min = torch.as_tensor(action_arr.min(axis=0), dtype=torch.float32)
    action_max = torch.as_tensor(action_arr.max(axis=0), dtype=torch.float32)

    state_standardizer = StateStandardizer(
        state_mean=state_mean,
        state_std=state_std,
        min_std=min_state_std,
        device=device,
    )
    action_scaler = ActionScaler(
        action_min=action_min,
        action_max=action_max,
        action_scale=0.0,
        min_range_per_dim=min_action_range,
        device=device,
    )
    return state_standardizer, action_scaler


def pack_norm_stats(state_standardizer: StateStandardizer, action_scaler: ActionScaler) -> dict:
    """Serialize normalization params for checkpoint / eval reload."""
    return {
        "state_mean": state_standardizer._mean.detach().cpu(),
        "state_std": state_standardizer._std.detach().cpu(),
        "min_state_std": state_standardizer.min_std,
        "action_limits_min": action_scaler.limits.min.detach().cpu(),
        "action_limits_max": action_scaler.limits.max.detach().cpu(),
    }


def load_norm_from_checkpoint(
    ckpt: dict,
    *,
    device: str = "cpu",
) -> tuple[StateStandardizer, ActionScaler]:
    """Restore normalization from a training checkpoint."""
    if "norm_stats" not in ckpt:
        raise KeyError("Checkpoint missing norm_stats; recompute via load_dataset_stats().")
    ns = ckpt["norm_stats"]
    state_standardizer = StateStandardizer(
        state_mean=ns["state_mean"],
        state_std=ns["state_std"],
        min_std=float(ns.get("min_state_std", 0.1)),
        device=device,
    )
    action_scaler = ActionScaler(
        action_min=ns["action_limits_min"],
        action_max=ns["action_limits_max"],
        action_scale=0.0,
        min_range_per_dim=0.0,
        device=device,
    )
    return state_standardizer, action_scaler
