#!/usr/bin/env python3
"""Short ForceVLA train smoke on exported force-collect LeRobot data."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np


def _make_smoke_config(
    *,
    data_root: Path,
    assets_dir: Path,
    ckpt_dir: Path,
    batch_size: int,
    num_steps: int,
    exp_name: str,
):
    from openpi.training import config as _config
    from openpi.training.config import AssetsConfig

    base = _config.get_config("bimanual_assembly_forcevla_both")
    data = dataclasses.replace(
        base.data,
        root=data_root,
        assets=AssetsConfig(assets_dir=str(assets_dir)),
    )
    return dataclasses.replace(
        base,
        data=data,
        batch_size=batch_size,
        num_workers=2,
        num_train_steps=num_steps,
        save_interval=max(num_steps, 1),
        log_interval=1,
        wandb_enabled=False,
        exp_name=exp_name,
        checkpoint_base_dir=str(ckpt_dir),
        overwrite=True,
        resume=False,
    )


def _audit_join(data_root: Path) -> dict:
    from openpi.forcevla.data.force_dataset import ForceAugmentedDataset
    from openpi.forcevla.data.force_labels import ForceInputMode
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local_repo", root=data_root)
    aug = ForceAugmentedDataset(ds, dataset_root=data_root, mode=ForceInputMode.BOTH)
    n = len(aug)
    forces = []
    for i in (0, n // 2, n - 1):
        sample = aug[i]
        force = np.asarray(sample["force"], dtype=np.float32)
        forces.append(
            {
                "i": int(i),
                "index": int(np.asarray(sample["index"]).item()),
                "force_shape": list(force.shape),
                "finite": bool(np.isfinite(force).all()),
                "abs_mean": float(np.abs(force).mean()),
                "abs_max": float(np.abs(force).max()),
            }
        )
    return {"n_frames": n, "samples": forces}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/datasets/bimanual_assembly_force_smoke"),
    )
    p.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/outputs/forcevla_smoke_assets/forcevla_both"),
    )
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/outputs/forcevla_smoke_ckpt"),
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--exp-name", type=str, default="collect_force_smoke")
    p.add_argument("--skip-norm", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    openpi_root = Path(__file__).resolve().parents[1] / "openpi"
    if str(openpi_root) not in sys.path:
        sys.path.insert(0, str(openpi_root))
    # train/compute_norm_stats import config.yaml relative to cwd
    import os

    os.chdir(openpi_root)

    if not (args.data_root / "force_labels" / "forces.parquet").is_file():
        raise SystemExit(f"missing force labels under {args.data_root}")

    print("[smoke] audit force join...", flush=True)
    audit = _audit_join(args.data_root)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    if any(not s["finite"] for s in audit["samples"]):
        raise SystemExit("force join produced non-finite values")
    if any(s["force_shape"] != [36] for s in audit["samples"]):
        raise SystemExit("force dim != 36 for both mode")

    config = _make_smoke_config(
        data_root=args.data_root,
        assets_dir=args.assets_dir,
        ckpt_dir=args.ckpt_dir,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        exp_name=args.exp_name,
    )

    if not args.skip_norm:
        print("[smoke] compute_norm_stats...", flush=True)
        args.assets_dir.mkdir(parents=True, exist_ok=True)
        from scripts.compute_norm_stats import main as compute_norm_stats_main

        compute_norm_stats_main(config, max_frames=min(512, audit["n_frames"]))

    if not args.skip_train:
        print(f"[smoke] train {args.num_steps} steps batch={args.batch_size}...", flush=True)
        from scripts.train import main as train_main

        train_main(config)

    report = {
        "ok": True,
        "data_root": str(args.data_root),
        "assets_dir": str(args.assets_dir),
        "ckpt_dir": str(Path(args.ckpt_dir) / "bimanual_assembly" / args.exp_name),
        "audit": audit,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
    }
    out = args.ckpt_dir / "smoke_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[smoke] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
