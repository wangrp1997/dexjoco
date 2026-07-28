"""Compute q01/q99 norm stats for DexJoCo post-train (action chunk + tactile).

Uses absolute action chunks [16,44] with edge-padding (no invented delta-base).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.dexjoco_schema import (  # noqa: E402
    ACTION_CHUNK,
    ACTION_DIM,
    DEFAULT_FORCE_PARQUET,
    DEFAULT_LEROBOT_ROOT,
    F6_DIM,
    STATS_KEY,
)
from adapters.prep_vqvae_data import forces_to_tactile_f6, _list_to_arr  # noqa: E402
from utils.lerobot_common import calculate_stats  # noqa: E402


def load_actions(lerobot_root: str) -> tuple[np.ndarray, np.ndarray]:
    data_dir = os.path.join(lerobot_root, "data")
    acts, eps = [], []
    for dp, _, fns in os.walk(data_dir):
        for fn in sorted(fns):
            if not fn.endswith(".parquet"):
                continue
            t = pq.read_table(
                os.path.join(dp, fn), columns=["action", "episode_index"]
            )
            for a in t["action"].to_pylist():
                acts.append(np.asarray(a, dtype=np.float32).reshape(ACTION_DIM))
            eps.append(t["episode_index"].to_numpy())
    if not acts:
        raise FileNotFoundError(f"No action parquet under {data_dir}")
    return np.stack(acts, axis=0), np.concatenate(eps, axis=0)


def build_action_chunks(actions: np.ndarray, ep: np.ndarray) -> np.ndarray:
    """Per-frame absolute chunk [N, ACTION_CHUNK, ACTION_DIM], edge-padded."""
    n = actions.shape[0]
    out = np.zeros((n, ACTION_CHUNK, ACTION_DIM), dtype=np.float32)
    for ep_id in np.unique(ep):
        idx = np.where(ep == ep_id)[0]
        a = actions[idx]
        L = a.shape[0]
        for local_i, global_i in enumerate(idx):
            for k in range(ACTION_CHUNK):
                j = min(local_i + k, L - 1)
                out[global_i, k] = a[j]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lerobot_root", type=str, default=DEFAULT_LEROBOT_ROOT)
    p.add_argument("--force_parquet", type=str, default=DEFAULT_FORCE_PARQUET)
    p.add_argument(
        "--output",
        type=str,
        default="/mnt/ssd/datasets/trex_dexjoco/bimanual_assembly/trex_norm_stats.json",
    )
    args = p.parse_args()

    actions, ep = load_actions(args.lerobot_root)
    chunks = build_action_chunks(actions, ep)

    ft = pq.read_table(args.force_parquet)
    if ft.num_rows != actions.shape[0]:
        raise ValueError(
            f"force rows {ft.num_rows} != action rows {actions.shape[0]}"
        )
    left = _list_to_arr(ft["left_finger_force"].to_pylist(), ft.num_rows, 12)
    right = _list_to_arr(ft["right_finger_force"].to_pylist(), ft.num_rows, 12)
    f6 = forces_to_tactile_f6(left, right).reshape(-1, F6_DIM)

    act_flat = chunks.reshape(-1, ACTION_DIM)
    stats = {
        STATS_KEY: {
            "action": calculate_stats(act_flat, [True] * ACTION_DIM),
            "state": calculate_stats(actions, [True] * ACTION_DIM),
            "tactile_f6": calculate_stats(f6, [True] * F6_DIM),
            "num_trajectories": int(len(np.unique(ep))),
            "action_repr": "absolute_chunk_edge_pad",
            "action_dim": ACTION_DIM,
            "action_chunk": ACTION_CHUNK,
            "tactile_shape": [8, 3],
        }
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f)
    print(f"Wrote {args.output}")
    print(f"frames={actions.shape[0]} action_dim={ACTION_DIM} f6_dim={F6_DIM}")


if __name__ == "__main__":
    main()
