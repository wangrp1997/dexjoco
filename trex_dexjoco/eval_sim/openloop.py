"""Open-loop action MSE on DexJoCo demos (multi-ckpt compare).

Uses the same inference path as sim eval (left→right wrists, denorm [16,44]).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_TREX_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _TREX_ROOT.parent
for p in (str(_TREX_ROOT), str(_TREX_ROOT / "scripts"), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapters.dexjoco_schema import (  # noqa: E402
    ACTION_CHUNK,
    DEFAULT_INSTRUCTION,
    SRC_EGO,
    SRC_WRIST_L,
    SRC_WRIST_R,
)
from eval_sim.policy import (  # noqa: E402
    TrexPolicy,
    _uint8_to_pil,
    resolve_trex_ckpt_label,
)


# action44: Right(xyz3+rotvec3+hand16) + Left(same)
_R_XYZ, _R_ROT, _R_HAND = slice(0, 3), slice(3, 6), slice(6, 22)
_L_XYZ, _L_ROT, _L_HAND = slice(22, 25), slice(25, 28), slice(28, 44)


class _Acc:
    is_main_process = True

    def print(self, *a, **k):
        print(*a, **k)


def _tensor_to_uint8(img_t: torch.Tensor) -> np.ndarray:
    arr = (img_t.detach().float().clamp(0, 1) * 255.0).to(torch.uint8)
    return arr.permute(1, 2, 0).cpu().numpy()


def _slice_mse(pred: np.ndarray, gt: np.ndarray, sl: slice) -> float:
    d = pred[..., sl] - gt[..., sl]
    return float(np.mean(d * d))


def _chunk_step_delta(actions: np.ndarray) -> dict[str, float]:
    """Mean L2 step-to-step delta inside a chunk [T,44]."""
    if actions.shape[0] < 2:
        return {"step_xyz": 0.0, "step_rot": 0.0, "step_all": 0.0}
    d = np.diff(actions, axis=0)
    xyz = np.concatenate([d[:, _R_XYZ], d[:, _L_XYZ]], axis=-1)
    rot = np.concatenate([d[:, _R_ROT], d[:, _L_ROT]], axis=-1)
    return {
        "step_xyz": float(np.linalg.norm(xyz, axis=-1).mean()),
        "step_rot": float(np.linalg.norm(rot, axis=-1).mean()),
        "step_all": float(np.linalg.norm(d.reshape(d.shape[0], -1), axis=-1).mean()),
    }


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    m = {
        "mse_all": float(np.mean((pred - gt) ** 2)),
        "mse_r_xyz": _slice_mse(pred, gt, _R_XYZ),
        "mse_r_rot": _slice_mse(pred, gt, _R_ROT),
        "mse_r_hand": _slice_mse(pred, gt, _R_HAND),
        "mse_l_xyz": _slice_mse(pred, gt, _L_XYZ),
        "mse_l_rot": _slice_mse(pred, gt, _L_ROT),
        "mse_l_hand": _slice_mse(pred, gt, _L_HAND),
    }
    pd = _chunk_step_delta(pred)
    gd = _chunk_step_delta(gt)
    m["pred_step_xyz"] = pd["step_xyz"]
    m["gt_step_xyz"] = gd["step_xyz"]
    m["pred_step_rot"] = pd["step_rot"]
    m["gt_step_rot"] = gd["step_rot"]
    return m


def _load_dataset(args: argparse.Namespace):
    from adapters.dexjoco_dataset import DexJoCoSftDataset
    from transformers import AutoProcessor

    cfg = argparse.Namespace(
        lerobot_root=args.lerobot_root,
        force_labels_path=args.force_labels_path,
        norm_stats_path=args.norm_stats_path,
        lerobot_repo_id=args.lerobot_repo_id or "",
        language_instruction=args.prompt,
        image_size=[args.image_w, args.image_h],
        use_flare=0,
        n_flare_steps=0,
        flare_frame_stride=4,
        use_tactile_vec=1,
        use_tactile_vqvae=1,
        vqvae_window=16,
        action_dim=44,
        video_backend=args.video_backend or "pyav",
    )
    processor = AutoProcessor.from_pretrained(
        args.base_model_path, trust_remote_code=True
    )
    ds = DexJoCoSftDataset(cfg, processor, _Acc())
    val = None
    if args.val_ratio > 0:
        val = ds.create_val_split(val_ratio=args.val_ratio, seed=args.seed)
    return ds, val


def _sample_indices(ds, n: int, seed: int) -> list[int]:
    rng = np.random.RandomState(seed)
    idxs = list(ds._indices)
    if n <= 0 or n >= len(idxs):
        return idxs
    pick = rng.choice(len(idxs), size=n, replace=False)
    return [idxs[i] for i in pick]


def _predict_one(policy: TrexPolicy, ds, global_i: int) -> np.ndarray:
    item = ds.ds[global_i]
    ego = _uint8_to_pil(_tensor_to_uint8(item[SRC_EGO]))
    wl = _uint8_to_pil(_tensor_to_uint8(item[SRC_WRIST_L]))
    wr = _uint8_to_pil(_tensor_to_uint8(item[SRC_WRIST_R]))
    hist = ds._f6_history(global_i)
    policy.reset()
    out = policy._predict_left_first(
        ego=ego,
        wrist_left=wl,
        wrist_right=wr,
        tactile_f6=hist,
        task_description=policy.prompt,
    )
    actions = np.asarray(out["actions"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    return actions


def _default_openloop_dir(env_name: str, seed: int, checkpoint: str) -> Path:
    """Same layout as sim eval, with ``_openloop`` suffix (no videos)."""
    label = resolve_trex_ckpt_label(checkpoint)
    return (
        _REPO_ROOT
        / "outputs"
        / "trex"
        / f"{env_name}_seed{seed}_{label}_openloop"
    )


def run(args: argparse.Namespace) -> None:
    train_ds, val_ds = _load_dataset(args)
    train_ids = _sample_indices(train_ds, args.n_train, args.seed)
    val_ids = _sample_indices(val_ds, args.n_val, args.seed + 1) if val_ds else []

    # Bind dataset lookup by global index (train/val share underlying arrays).
    root_ds = train_ds

    samples: list[tuple[str, int]] = (
        [("train", i) for i in train_ids] + [("val", i) for i in val_ids]
    )
    print(
        f"Open-loop: {len(train_ids)} train + {len(val_ids)} val frames, "
        f"{len(args.checkpoints)} ckpt(s)",
        flush=True,
    )

    table: list[dict[str, Any]] = []
    for ckpt in args.checkpoints:
        ckpt = os.path.abspath(ckpt)
        if args.dump_dir:
            out_dir = Path(args.dump_dir) / resolve_trex_ckpt_label(ckpt)
        else:
            out_dir = _default_openloop_dir(args.env_name, args.seed, ckpt)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {ckpt} ===", flush=True)
        print(f"Output: {out_dir}", flush=True)

        policy = TrexPolicy(
            ckpt,
            cuda=args.cuda,
            image_size=(args.image_w, args.image_h),
            base_model_path=args.base_model_path,
            prompt=args.prompt,
        )
        per_split: dict[str, list[dict[str, float]]] = {"train": [], "val": []}
        dump_traj = None
        for si, (split, gi) in enumerate(samples):
            gt = np.asarray(root_ds.chunks[gi], dtype=np.float32)
            pred = _predict_one(policy, root_ds, gi)
            T = min(pred.shape[0], gt.shape[0], ACTION_CHUNK)
            pred, gt = pred[:T], gt[:T]
            m = _metrics(pred, gt)
            per_split[split].append(m)
            if si == 0 or (si + 1) % max(1, args.log_every) == 0:
                print(
                    f"  [{split} i={gi}] mse={m['mse_all']:.5f} "
                    f"r_xyz={m['mse_r_xyz']:.5f} l_xyz={m['mse_l_xyz']:.5f} "
                    f"predΔxyz={m['pred_step_xyz']:.4f} gtΔxyz={m['gt_step_xyz']:.4f}",
                    flush=True,
                )
            if dump_traj is None and split == ("val" if val_ids else "train"):
                dump_traj = {
                    "checkpoint": ckpt,
                    "split": split,
                    "global_index": gi,
                    "gt": gt.tolist(),
                    "pred": pred.tolist(),
                    "metrics": m,
                }

        summary: dict[str, Any] = {"checkpoint": ckpt, "output_dir": str(out_dir)}
        lines = [f"checkpoint: {ckpt}", f"output: {out_dir}"]
        for split, rows in per_split.items():
            if not rows:
                continue
            keys = rows[0].keys()
            agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
            summary[split] = agg
            line = (
                f"[{split} n={len(rows)}] "
                f"mse_all={agg['mse_all']:.5f} "
                f"mse_xyz={(agg['mse_r_xyz'] + agg['mse_l_xyz']) / 2:.5f} "
                f"mse_rot={(agg['mse_r_rot'] + agg['mse_l_rot']) / 2:.5f} "
                f"predΔxyz={agg['pred_step_xyz']:.4f} "
                f"gtΔxyz={agg['gt_step_xyz']:.4f}"
            )
            print(f"  {line}", flush=True)
            lines.append(line)
        table.append(summary)

        with open(out_dir / "openloop_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(out_dir / "metrics.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        if dump_traj is not None:
            with open(out_dir / "sample_traj.json", "w") as f:
                json.dump(dump_traj, f)
        print(f"  wrote {out_dir}", flush=True)

    print("\n======== Summary ========", flush=True)
    for s in table:
        ck = Path(s["checkpoint"]).name
        for split in ("train", "val"):
            if split not in s:
                continue
            a = s[split]
            print(
                f"{ck:30s} {split:5s}  "
                f"mse={a['mse_all']:.5f}  "
                f"xyz={0.5 * (a['mse_r_xyz'] + a['mse_l_xyz']):.5f}  "
                f"rot={0.5 * (a['mse_r_rot'] + a['mse_l_rot']):.5f}  "
                f"predΔxyz={a['pred_step_xyz']:.4f}  "
                f"gtΔxyz={a['gt_step_xyz']:.4f}",
                flush=True,
            )


def main():
    p = argparse.ArgumentParser(description="DexJoCo open-loop action MSE")
    p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more checkpoint dirs",
    )
    p.add_argument(
        "--lerobot_root",
        default="/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly",
    )
    p.add_argument(
        "--force_labels_path",
        default="",
        help="Default: <lerobot_root>/force_labels/forces.parquet",
    )
    p.add_argument(
        "--norm_stats_path",
        default="/mnt/ssd/datasets/trex_dexjoco/bimanual_assembly/trex_norm_stats.json",
    )
    p.add_argument("--lerobot_repo_id", default="")
    p.add_argument(
        "--base_model_path",
        default="/mnt/hdd/checkpoints/trex/Qwen3-VL-2B-Instruct",
    )
    p.add_argument("--prompt", default=DEFAULT_INSTRUCTION)
    p.add_argument("--n_train", type=int, default=64)
    p.add_argument("--n_val", type=int, default=64)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--image_w", type=int, default=384)
    p.add_argument("--image_h", type=int, default=288)
    p.add_argument("--video_backend", default="pyav")
    p.add_argument("--log_every", type=int, default=16)
    p.add_argument("--env_name", default="bimanual_assembly")
    p.add_argument(
        "--dump_dir",
        default="",
        help="Optional parent dir; default = "
             "outputs/trex/{env}_seed{seed}_{ckpt_label}_openloop/",
    )
    args = p.parse_args()
    if not args.force_labels_path:
        args.force_labels_path = os.path.join(
            args.lerobot_root, "force_labels", "forces.parquet"
        )
    run(args)


if __name__ == "__main__":
    main()
