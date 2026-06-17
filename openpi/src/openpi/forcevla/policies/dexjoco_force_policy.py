# Source: refs/ForceVLA/src/openpi/policies/forcevla_policy.py + openpi/policies/dual_arm_policy.py
from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.forcevla.data.force_labels import ForceInputMode, force_dim_for_mode
from openpi.forcevla.data.state_layout import state46_to_proprio44
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DexJoCoForceInputs(transforms.DataTransformFn):
    """Fuse DexJoCo dual-arm proprio with privileged force sidecar."""

    model_type: _model.ModelType
    proprio_dim: int = 44
    force_mode: ForceInputMode = ForceInputMode.WRIST

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] == 46:
            proprio = state46_to_proprio44(state)
        elif state.shape[-1] == self.proprio_dim:
            proprio = state
        else:
            proprio = transforms.pad_to_dim(state, self.proprio_dim)
        if proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"Expected proprio dim {self.proprio_dim}, got {proprio.shape}")
        force = np.asarray(data.get("force", np.zeros(force_dim_for_mode(self.force_mode))), dtype=np.float32)
        if force.shape[-1] != force_dim_for_mode(self.force_mode):
            raise ValueError(
                f"Expected force dim {force_dim_for_mode(self.force_mode)}, got shape {force.shape}"
            )
        front_image = _parse_image(data["base"])
        wrist_left_image = _parse_image(data["wrist_left"])
        wrist_right_image = _parse_image(data["wrist_right"])
        inputs = {
            "state": proprio,
            "force": force,
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": wrist_left_image,
                "right_wrist_0_rgb": wrist_right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class DexJoCoForceOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :44])}
