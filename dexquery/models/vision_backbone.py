"""Multi-view SigLIP vision encoder returning patch tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel

# Image normalize [0,1]->[-1,1]: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0/modeling_pi0.py
# SigLIP vision tower usage: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py


def preprocess_images(images: torch.Tensor, *, image_size: int) -> torch.Tensor:
    """Resize to SigLIP input size and map LeRobot ``[0, 1]`` floats to ``[-1, 1]``."""
    if images.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W), got {tuple(images.shape)}")
    images = images.float()
    if images.shape[-2] != image_size or images.shape[-1] != image_size:
        images = F.interpolate(
            images,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    return images * 2.0 - 1.0


class SiglipPatchBackbone(nn.Module):
    """Extract patch tokens from one or more RGB views with a shared SigLIP vision encoder."""

    def __init__(
        self,
        *,
        model_name: str = "google/siglip-base-patch16-224",
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.vision = SiglipVisionModel.from_pretrained(model_name)
        self.embed_dim = int(self.vision.config.hidden_size)
        self.image_size = int(getattr(self.vision.config, "image_size", 224))
        if freeze:
            for param in self.vision.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return patch tokens ``(B, num_views * num_patches, embed_dim)``.

        Args:
            images: ``(B, num_views, C, H, W)`` in ``[0, 1]``.
        """
        if images.ndim != 5:
            raise ValueError(f"Expected (B, num_views, C, H, W), got {tuple(images.shape)}")

        batch_size, num_views, channels, height, width = images.shape
        flat = preprocess_images(
            images.reshape(batch_size * num_views, channels, height, width),
            image_size=self.image_size,
        )
        outputs = self.vision(pixel_values=flat, return_dict=True)
        patches = outputs.last_hidden_state
        num_patches = patches.shape[1]
        return patches.reshape(batch_size, num_views * num_patches, self.embed_dim)
