"""Build native DexJoCo tactile [N,8,3] windows for VQ-VAE retrain (T1).

Layout per frame (matches utils/lerobot_common):
  left 4 fingers × (fx,fy,fz)  then  right 4 fingers × (fx,fy,fz)

Writes a midtrain-compatible tree:
  OUT/dexjoco_assembly/pretrain_manifest.json
  OUT/dexjoco_assembly/episode_XXXXXX/pretrain.hdf5  # tactile_f6 [N,8,3]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import pyarrow.parquet as pq

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.dexjoco_schema import (  # noqa: E402
    DEFAULT_FORCE_PARQUET,
    F6_DIM,
    F6_PER_FINGER,
    N_FINGERS,
    N_FINGERS_PER_HAND,
)
from utils.lerobot_common import calculate_stats  # noqa: E402


def _list_to_arr(col, n: int, dim: int) -> np.ndarray:
    out = np.zeros((n, dim), dtype=np.float32)
    for i, v in enumerate(col):
        out[i] = np.asarray(v, dtype=np.float32).reshape(dim)
    return out


def forces_to_tactile_f6(
    left_12: np.ndarray, right_12: np.ndarray
) -> np.ndarray:
    """[N,12] + [N,12] → [N,8,3] with left fingers then right fingers."""
    n = left_12.shape[0]
    left = left_12.reshape(n, N_FINGERS_PER_HAND, F6_PER_FINGER)
    right = right_12.reshape(n, N_FINGERS_PER_HAND, F6_PER_FINGER)
    return np.concatenate([left, right], axis=1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force_parquet", type=str, default=DEFAULT_FORCE_PARQUET)
    p.add_argument(
        "--output_root",
        type=str,
        default="/mnt/ssd/datasets/trex_dexjoco/vqvae_f6_data",
    )
    p.add_argument("--task_name", type=str, default="dexjoco_assembly")
    args = p.parse_args()

    table = pq.read_table(args.force_parquet)
    ep = table["episode_index"].to_numpy()
    left = _list_to_arr(table["left_finger_force"].to_pylist(), len(ep), 12)
    right = _list_to_arr(table["right_finger_force"].to_pylist(), len(ep), 12)
    f6_all = forces_to_tactile_f6(left, right)  # [N,8,3]

    task_dir = os.path.join(args.output_root, args.task_name)
    os.makedirs(task_dir, exist_ok=True)

    episodes = []
    flat_for_stats = []
    for ep_id in sorted(set(ep.tolist())):
        mask = ep == ep_id
        f6 = f6_all[mask]
        ep_name = f"episode_{int(ep_id):06d}"
        ep_dir = os.path.join(task_dir, ep_name)
        os.makedirs(ep_dir, exist_ok=True)
        h5_path = os.path.join(ep_dir, "pretrain.hdf5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("tactile_f6", data=f6, compression="gzip")
        episodes.append(
            {
                "episode_dir": ep_dir,
                "num_frames": int(f6.shape[0]),
                "episode_index": int(ep_id),
            }
        )
        flat_for_stats.append(f6.reshape(f6.shape[0], F6_DIM))

    flat = np.concatenate(flat_for_stats, axis=0)
    tac_stats = calculate_stats(flat, mask=[True] * F6_DIM)

    manifest = {
        "task": args.task_name,
        "n_fingers": N_FINGERS,
        "fingers_per_hand": N_FINGERS_PER_HAND,
        "per_finger_dim": F6_PER_FINGER,
        "layout": "left_4x3 then right_4x3 (Allegro ff,mf,rf,th xyz)",
        "episodes": episodes,
        "statistics": {"tactile_f6": tac_stats},
    }
    man_path = os.path.join(task_dir, "pretrain_manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(episodes)} episodes → {task_dir}")
    print(f"Manifest: {man_path}")
    print(f"tactile_f6 shape per frame: [{N_FINGERS}, {F6_PER_FINGER}] "
          f"flat={F6_DIM}")


if __name__ == "__main__":
    main()
