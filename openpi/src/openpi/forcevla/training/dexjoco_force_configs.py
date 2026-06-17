# Source: openpi/training/dexjoco_configs.py + refs/ForceVLA/src/openpi/training/config.py (forcevla_lora)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

import openpi.training.optimizer as _optimizer
from openpi.forcevla.data.force_labels import ForceInputMode
from openpi.forcevla.pi05_force_dexjoco import Pi05ForceDexJoCoConfig
from openpi.forcevla.training.dual_arm_force_data_config import (
    DualArmForceDataConfig,
    force_suffix,
    make_force_model_config,
)
from openpi.training.config import AssetsConfig, DataConfig, TrainConfig
from openpi.training import weight_loaders

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

PRETRAINED_MODEL_ACTION_DIM_44_PATH = _cfg["pretrained_model_action_dim_44_path"]
DATASET_ROOT = Path(_cfg["dataset_root"])
CKPTS_ROOT = Path(_cfg["ckpts_root"])
FORCEVLA_CKPTS_ROOT = Path(_cfg["forcevla_ckpts_root"])
BATCH_SIZE = _cfg["batch_size"]
DUAL_ARM_STEPS = _cfg["dual_arm_steps"]
WANDB_ENABLED = _cfg["wandb_enabled"]
ASSETS_BASE_DIR = Path("./assets")


@dataclass
class ForceDexJoCoTask:
    name: str
    data_root: Path


FORCE_TASKS = [
    ForceDexJoCoTask(
        name="bimanual_assembly",
        data_root=Path(f"{DATASET_ROOT}/bimanual_assembly"),
    ),
]


def _make_force_train_config(task: ForceDexJoCoTask, *, force_mode: ForceInputMode) -> TrainConfig:
    suffix = force_suffix(force_mode)
    config_name = f"{task.name}_forcevla_{suffix}"
    exp_name = f"forcevla_{suffix}"
    model = make_force_model_config(task.name, force_mode=force_mode)
    return TrainConfig(
        name=config_name,
        exp_name=exp_name,
        checkpoint_subdir=task.name,
        model=model,
        data=DualArmForceDataConfig(
            root=task.data_root,
            repo_id="local_repo",
            force_mode=force_mode,
            assets=AssetsConfig(
                assets_dir=str(ASSETS_BASE_DIR / task.name / f"forcevla_{suffix}"),
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=BATCH_SIZE,
        num_workers=4,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=Pi05ForceDexJoCoConfig(
            action_dim=44,
            action_horizon=30,
            max_token_len=250,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            proprio_dim=44,
            force_dim=model.force_dim,
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.Pi0GuidanceWeightLoader(PRETRAINED_MODEL_ACTION_DIM_44_PATH),
        num_train_steps=DUAL_ARM_STEPS,
        save_interval=10000,
        wandb_enabled=WANDB_ENABLED,
        checkpoint_base_dir=f"{FORCEVLA_CKPTS_ROOT}",
    )


def get_forcevla_dexjoco_configs() -> list[TrainConfig]:
    configs: list[TrainConfig] = []
    for task in FORCE_TASKS:
        for mode in (ForceInputMode.WRIST, ForceInputMode.FINGER, ForceInputMode.BOTH):
            configs.append(_make_force_train_config(task, force_mode=mode))
    return configs
