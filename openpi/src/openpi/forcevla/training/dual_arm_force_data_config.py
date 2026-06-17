# Source: openpi/training/config.py DualArmDataConfig + ForceVLA data wiring
from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Sequence
from typing import Literal

import tyro
from typing_extensions import override

import openpi.transforms as _transforms
from openpi.forcevla.data.force_labels import ForceInputMode, force_dim_for_mode
from openpi.forcevla.policies import dexjoco_force_policy
from openpi.training.config import DataConfig, DataConfigFactory


@dataclasses.dataclass(frozen=True)
class DualArmForceDataConfig(DataConfigFactory):
    """DexJoCo dual-arm dataset with privileged ``force_labels`` sidecar."""

    root: pathlib.Path = tyro.MISSING
    action_sequence_keys: Sequence[str] = ("action",)
    base_img_name: str | None = None
    wrist_left_img_name: str | None = None
    wrist_right_img_name: str | None = None
    force_mode: ForceInputMode = ForceInputMode.WRIST
    proprio_dim: int = 44

    @override
    def create(self, assets_dirs: pathlib.Path, model_config) -> DataConfig:
        base_img_name = self.base_img_name or "observation.images.ego"
        wrist_left_img_name = self.wrist_left_img_name or "observation.images.wrist_left"
        wrist_right_img_name = self.wrist_right_img_name or "observation.images.wrist_right"

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "base": base_img_name,
                    "wrist_left": wrist_left_img_name,
                    "wrist_right": wrist_right_img_name,
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                    "index": "index",
                    "force": "force",
                })
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                dexjoco_force_policy.DexJoCoForceInputs(
                    model_type=model_config.model_type,
                    proprio_dim=self.proprio_dim,
                    force_mode=self.force_mode,
                )
            ],
            outputs=[dexjoco_force_policy.DexJoCoForceOutputs()],
        )
        from openpi.forcevla.training.model_transform_factory import ForcePi05ModelTransformFactory

        model_transforms = ForcePi05ModelTransformFactory(proprio_dim=self.proprio_dim)(model_config)

        base = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(
            base,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            root=self.root if self.root is not tyro.MISSING else None,
            force_mode=self.force_mode.value,
            proprio_dim=self.proprio_dim,
        )


def force_suffix(mode: ForceInputMode) -> str:
    return {
        ForceInputMode.WRIST: "wrist",
        ForceInputMode.FINGER: "finger",
        ForceInputMode.BOTH: "both",
    }[mode]


def make_force_model_config(
    task_name: str,
    *,
    force_mode: ForceInputMode,
    action_horizon: int = 30,
):
    from openpi.forcevla.pi05_force_dexjoco import Pi05ForceDexJoCoConfig

    return Pi05ForceDexJoCoConfig(
        action_dim=44,
        action_horizon=action_horizon,
        max_token_len=250,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        proprio_dim=44,
        force_dim=force_dim_for_mode(force_mode),
    )
