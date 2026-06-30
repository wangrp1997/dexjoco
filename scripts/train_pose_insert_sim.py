#!/usr/bin/env python3
"""Train PoseInsert PoseDP on DexJoCo sim-exported insert trajectories."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEXJOJO_ROOT = _REPO_ROOT / "dexjoco"
if str(_DEXJOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXJOJO_ROOT))

from pose_insert.dataset_sim import (
    BimanualAction44Dataset,
    BimanualPoseInsertDataset,
    SimPoseInsertDataset,
    collate_action44_batch,
    collate_bimanual_batch,
    collate_pose_batch,
    load_or_build_workspace,
)
from pose_insert.models.pose_dp import PoseDP
from pose_insert.paths import default_poseinsert_data_dir
from pose_insert.wrist_actions import DUAL_ACTION44_DIM, DUAL_WRIST_DIM, load_or_build_wrist_workspace
from interaction_retarget.constants import TASK_ID


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"default: {default_poseinsert_data_dir(TASK_ID)}",
    )
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=_REPO_ROOT / "outputs" / "poseinsert_sim" / "checkpoints",
    )
    p.add_argument("--num-action", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=240)
    p.add_argument("--num-epochs", type=int, default=500)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=233)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--smoke", action="store_true", help="2 epochs, small batch, for CI")
    p.add_argument(
        "--wrist12",
        action="store_true",
        help="Train dual wrist12 labels (deprecated: fingers locked at exec).",
    )
    p.add_argument(
        "--action44",
        action="store_true",
        help="Train full dual-arm44 (wrist rotvec + hands); recommended for bimanual insert.",
    )
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = _parse_args()
    if args.smoke:
        args.num_epochs = 2
        args.batch_size = min(args.batch_size, 8)
        args.num_workers = 0

    data_root = args.data_root if args.data_root is not None else default_poseinsert_data_dir(TASK_ID)
    data_root = data_root.expanduser()
    if not data_root.is_dir():
        print(f"data root not found: {data_root}", file=sys.stderr)
        return 1

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalize = not args.no_normalize
    use_wrist12 = bool(args.wrist12)
    use_action44 = bool(args.action44)
    if use_wrist12 and use_action44:
        print("choose only one of --wrist12 or --action44", file=sys.stderr)
        return 1
    action_dim = DUAL_ACTION44_DIM if use_action44 else (DUAL_WRIST_DIM if use_wrist12 else 9)
    if normalize:
        ws = load_or_build_workspace(data_root)
        print(f"pose workspace: {data_root / 'source_workspace.npy'} shape={ws.shape}", flush=True)
        if use_wrist12:
            wws = load_or_build_wrist_workspace(data_root)
            print(f"wrist workspace: {data_root / 'wrist_workspace.npy'} shape={wws.shape}", flush=True)
        if use_action44:
            from pose_insert.wrist_actions import load_or_build_action44_workspace

            aws = load_or_build_action44_workspace(data_root)
            print(f"action44 workspace: {data_root / 'action44_workspace.npy'} shape={aws.shape}", flush=True)

    if use_action44:
        dataset = BimanualAction44Dataset(
            data_root,
            split="train",
            num_obs=1,
            num_action=args.num_action,
            normalize=normalize,
        )
        collate_fn = collate_action44_batch
        action_key = "action44"
    elif use_wrist12:
        dataset = BimanualPoseInsertDataset(
            data_root,
            split="train",
            num_obs=1,
            num_action=args.num_action,
            normalize=normalize,
        )
        collate_fn = collate_bimanual_batch
        action_key = "action_dual_wrist"
    else:
        dataset = SimPoseInsertDataset(
            data_root,
            split="train",
            num_obs=1,
            num_action=args.num_action,
            normalize=normalize,
        )
        collate_fn = collate_pose_batch
        action_key = "action_source_pose"

    print(
        f"dataset ({'action44' if use_action44 else ('wrist12' if use_wrist12 else 'pose9')}): "
        f"{len(dataset)} samples from {len(dataset.demo_dirs)} demos",
        flush=True,
    )
    if len(dataset) == 0:
        err = "no training samples; run export_insert_poses.py --all"
        if use_wrist12:
            err += " (wrist12 needs dual_wrist_action.npy)"
        if use_action44:
            err += " (action44 needs action44.npy; run backfill_action44.py)"
        print(err, file=sys.stderr)
        return 1

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset),
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    policy = PoseDP(
        num_action=args.num_action,
        obs_feature_dim=128,
        action_dim=action_dim,
    ).to(device)
    if args.resume is not None:
        policy.load_state_dict(torch.load(args.resume, map_location=device), strict=False)
        print(f"resumed {args.resume}", flush=True)

    if use_action44:
        args.ckpt_dir = args.ckpt_dir / "action44"
    elif use_wrist12:
        args.ckpt_dir = args.ckpt_dir / "wrist12"
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_steps),
        num_training_steps=len(loader) * args.num_epochs,
    )

    policy.train()
    for epoch in range(args.num_epochs):
        losses: list[float] = []
        for batch in loader:
            obs = batch["obs_source_pose"].to(device)
            action = batch[action_key].to(device)
            if not use_wrist12:
                action = action.reshape(action.shape[0], action.shape[1], -1)
            loss = policy(obs, action, batch_size=action.shape[0])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            losses.append(float(loss.item()))

        avg = float(np.mean(losses)) if losses else float("nan")
        print(f"epoch {epoch + 1}/{args.num_epochs} loss={avg:.6f}", flush=True)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.num_epochs:
            ckpt = args.ckpt_dir / f"policy_epoch_{epoch + 1}_seed_{args.seed}.ckpt"
            torch.save(policy.state_dict(), ckpt)
            print(f"saved {ckpt}", flush=True)

    last = args.ckpt_dir / "policy_last.ckpt"
    torch.save(policy.state_dict(), last)
    print(f"done -> {last}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
