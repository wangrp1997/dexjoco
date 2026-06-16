#!/usr/bin/env python3
"""DexQuery training entry point (standalone loop; does not call lerobot-train)."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_DEXQUERY_ROOT = Path(__file__).resolve().parents[1]
_DEXJOCo_ROOT = _DEXQUERY_ROOT.parent
if str(_DEXJOCo_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXJOCo_ROOT))

from dexquery.data import (  # noqa: E402
    DexQueryCollator,
    DexQueryDataset,
    DexQueryDatasetConfig,
    SubtaskPrompts,
    resolve_dataset_root,
)
from dexquery.models import DexQueryModel, DexQueryModelConfig  # noqa: E402

_BASELINE_BY_ROBOT = {
    "dual_arm": _DEXJOCo_ROOT / "configs/training/dual_arm_baseline.yaml",
    "single_arm": _DEXJOCo_ROOT / "configs/training/single_arm_baseline.yaml",
}
_POLICY_CONFIG = _DEXJOCo_ROOT / "configs/training/policies/dexquery.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _robot_type_for_task(task: str) -> str:
    eval_cfg_path = _DEXJOCo_ROOT / "configs/rand_obj" / f"{task}.yaml"
    if not eval_cfg_path.exists():
        raise FileNotFoundError(f"Unknown task config: {eval_cfg_path}")
    eval_cfg = _load_yaml(eval_cfg_path)
    return str(eval_cfg.get("robot_type", "single_arm"))


def load_training_config(task: str, *, task_config: Path | None = None) -> dict[str, Any]:
    robot_type = _robot_type_for_task(task)
    baseline_path = _BASELINE_BY_ROBOT.get(robot_type)
    if baseline_path is None:
        raise ValueError(f"Unsupported robot_type: {robot_type}")

    cfg = _deep_merge(_load_yaml(baseline_path), _load_yaml(_POLICY_CONFIG))
    task_cfg_path = task_config or (_DEXQUERY_ROOT / "configs" / f"{task}.yaml")
    if task_cfg_path.exists():
        cfg = _deep_merge(cfg, _load_yaml(task_cfg_path))
    cfg["task"] = task
    cfg["robot_type"] = robot_type
    return cfg


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model_config(cfg: dict[str, Any], prompts: SubtaskPrompts) -> DexQueryModelConfig:
    model_cfg = cfg.get("model", {})
    return DexQueryModelConfig(
        cross_attn_layers=int(model_cfg.get("cross_attn_layers", 2)),
        action_head_type=str(model_cfg.get("action_head_type", "act")),
        outcome_loss_weight=float(model_cfg.get("outcome_loss_weight", 0.1)),
        freeze_vision=bool(model_cfg.get("freeze_vision", False)),
        state_dim=int(cfg.get("state_dim", 46)),
        action_dim=int(cfg.get("action_dim", 44)),
        chunk_size=int(cfg.get("action_horizon", 30)),
        subtask_prompts=prompts.as_list(),
    )


def _build_optimizer(model: DexQueryModel, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = cfg.get("training", {})
    lr = float(train_cfg.get("lr", 1.0e-4))
    lr_backbone = float(train_cfg.get("lr_backbone", 1.0e-5))
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-4))

    backbone_params = list(model.backbone.parameters())
    text_params = list(model.subtask_encoder.text_model.parameters())
    backbone_ids = {id(p) for p in backbone_params + text_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids and p.requires_grad]

    param_groups = [
        {"params": [p for p in backbone_params if p.requires_grad], "lr": lr_backbone},
        {"params": [p for p in text_params if p.requires_grad], "lr": lr_backbone},
        {"params": other_params, "lr": lr},
    ]
    param_groups = [g for g in param_groups if g["params"]]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    return obj


def _save_checkpoint(
    path: Path,
    *,
    model: DexQueryModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "model_config": asdict(model.config),
    }
    torch.save(payload, path)


def _maybe_init_wandb(cfg: dict[str, Any], *, output_dir: Path, enabled: bool):
    wandb_cfg = cfg.get("wandb", {})
    if not enabled or not wandb_cfg.get("enable", False):
        return None
    import wandb

    run = wandb.init(
        project=wandb_cfg.get("project", "dexjoco"),
        entity=wandb_cfg.get("entity"),
        name=f"dexquery_{cfg['task']}",
        config=cfg,
        dir=str(output_dir),
    )
    return run


def train(cfg: dict[str, Any], *, device: str, output_dir: Path, no_wandb: bool) -> None:
    task = cfg["task"]
    train_cfg = cfg.get("training", {})
    dataset_cfg = cfg.get("dataset", {})

    seed = int(train_cfg.get("seed", 0))
    _set_seed(seed)

    dataset_root = resolve_dataset_root(dataset_cfg.get("root", ""), task)
    task_config_path = _DEXQUERY_ROOT / "configs" / f"{task}.yaml"
    prompts = SubtaskPrompts.for_task(task, config_path=task_config_path if task_config_path.exists() else None)

    print("Building dataset...", flush=True)
    ds = DexQueryDataset(
        DexQueryDatasetConfig(
            task=task,
            dataset_root=dataset_root,
            action_horizon=int(cfg.get("action_horizon", 30)),
            outcome_fields=tuple(dataset_cfg.get("outcome_fields", ("tray_ok", "peg_ok"))),
            video_backend=str(dataset_cfg.get("video_backend", "pyav")),
        ),
        subtask_prompts=prompts,
    )
    print(f"Dataset ready: {len(ds)} frames", flush=True)
    collator = DexQueryCollator.from_dataset_stats(ds.meta.stats)
    loader = DataLoader(
        ds,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=device.startswith("cuda"),
        drop_last=True,
        collate_fn=collator,
    )

    print("Building model (SigLIP may take a minute on first load)...", flush=True)
    model = DexQueryModel(_build_model_config(cfg, prompts)).to(device)
    optimizer = _build_optimizer(model, cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(output_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(ds.meta.stats), f, indent=2)

    wandb_run = _maybe_init_wandb(cfg, output_dir=output_dir, enabled=not no_wandb)

    total_steps = int(train_cfg.get("steps", 60000))
    log_freq = int(train_cfg.get("log_freq", 100))
    save_freq = int(train_cfg.get("save_freq", 10000))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    batch_size = int(train_cfg.get("batch_size", 32))

    print(
        f"Start training: steps={total_steps} batch_size={batch_size} "
        f"log_freq={log_freq} save_freq={save_freq}",
        flush=True,
    )

    model.train()
    data_iter = iter(loader)
    running = {"loss": 0.0, "loss_action": 0.0, "loss_outcome": 0.0}
    log_count = 0
    log_t0 = time.time()

    progbar = tqdm(total=total_steps, desc="Training", unit="step", dynamic_ncols=True)
    for step in range(1, total_steps + 1):
        step_t0 = time.time()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        data_s = time.time() - step_t0

        fwd_t0 = time.time()
        images = batch["images"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        tray_ok = batch["tray_ok"].to(device, non_blocking=True)
        peg_ok = batch["peg_ok"].to(device, non_blocking=True)
        subtask_phase = batch["subtask_phase"].to(device, non_blocking=True)

        outputs = model(
            images,
            state,
            actions=actions,
            tray_ok=tray_ok,
            peg_ok=peg_ok,
            subtask_phase=subtask_phase,
            subtask_prompts=batch["subtask_prompts"],
        )
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("DexQuery forward did not produce a training loss.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        updt_s = time.time() - fwd_t0
        step_s = data_s + updt_s

        running["loss"] += float(loss.item())
        running["loss_action"] += float(outputs.loss_action.item() if outputs.loss_action is not None else 0.0)
        running["loss_outcome"] += float(outputs.loss_outcome.item() if outputs.loss_outcome is not None else 0.0)
        log_count += 1

        progbar.update(1)
        progbar.set_postfix(
            loss=f"{loss.item():.3f}",
            action=f"{outputs.loss_action.item():.3f}" if outputs.loss_action is not None else "n/a",
            step_s=f"{step_s:.2f}",
            refresh=False,
        )

        if step == 1:
            print(
                f"[step 1/{total_steps}] first batch ok | "
                f"loss={loss.item():.4f} data_s={data_s:.2f} updt_s={updt_s:.2f}",
                flush=True,
            )

        if step % log_freq == 0:
            elapsed = time.time() - log_t0
            metrics = {k: v / log_count for k, v in running.items()}
            samples_per_s = (batch_size * log_count) / max(elapsed, 1e-6)
            mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0.0
            msg = (
                f"step:{step // 1000}K smpl:{step * batch_size // 1000}K "
                f"loss:{metrics['loss']:.3f} action:{metrics['loss_action']:.3f} "
                f"outcome:{metrics['loss_outcome']:.3f} "
                f"updt_s:{updt_s:.3f} data_s:{data_s:.3f} smp/s:{samples_per_s:.1f} mem_gb:{mem_gb:.2f}"
            )
            tqdm.write(msg)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        **{f"train/{k}": v for k, v in metrics.items()},
                        "train/updt_s": updt_s,
                        "train/data_s": data_s,
                        "train/samples_per_s": samples_per_s,
                        "train/mem_gb": mem_gb,
                    },
                    step=step,
                )
            running = {k: 0.0 for k in running}
            log_count = 0
            log_t0 = time.time()

        if step % save_freq == 0 or step == total_steps:
            ckpt_path = output_dir / f"checkpoint_step_{step:06d}.pt"
            _save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                step=step,
                cfg=cfg,
            )
            _save_checkpoint(
                output_dir / "checkpoint_last.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                cfg=cfg,
            )
            tqdm.write(f"Saved checkpoint: {ckpt_path}")

    progbar.close()

    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DexQuery on a DexJoCo LeRobot dataset.")
    parser.add_argument("--task", required=True, help="Task name, e.g. bimanual_assembly")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Task-specific DexQuery yaml (default: dexquery/configs/<task>.yaml)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override dataset.root from config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Checkpoint directory (default: checkpoints/dexquery_dexjoco_ckpt/<task>/)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    args = parser.parse_args()

    cfg = load_training_config(args.task, task_config=args.config)
    if args.dataset_root is not None:
        cfg.setdefault("dataset", {})["root"] = str(args.dataset_root.expanduser())

    if args.output_dir is None:
        output_root = Path(cfg.get("output", {}).get("root", "checkpoints"))
        if not output_root.is_absolute():
            output_root = _DEXJOCo_ROOT / output_root
        output_dir = output_root / "dexquery_dexjoco_ckpt" / args.task
    else:
        output_dir = args.output_dir.expanduser()

    print(f"Task: {args.task} ({cfg.get('robot_type')})")
    print(f"Dataset root: {resolve_dataset_root(cfg['dataset']['root'], args.task)}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")

    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA requested but not available.")

    train(cfg, device=args.device, output_dir=output_dir, no_wandb=args.no_wandb)


if __name__ == "__main__":
    main()
