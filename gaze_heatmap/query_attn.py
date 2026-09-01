# Spatial query attention head adapted from RF-DETR SpatialKeypointHead (einsum heatmaps).
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GazeQueryNet(nn.Module):
    """Two learnable queries (hole, tip) attend over ViT patch features -> 2ch heatmaps."""

    def __init__(
        self,
        *,
        num_keypoints: int = 2,
        image_size: int = 224,
        kpt_embed_dim: int = 64,
        backbone: str = "vit_small_patch16_224",
        freeze_blocks: int = 6,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.image_size = image_size
        self.kpt_embed_dim = kpt_embed_dim

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.hidden_dim = int(self.backbone.embed_dim)
        patch = getattr(self.backbone.patch_embed, "patch_size", (16, 16))
        self.patch_size = int(patch[0])

        for i, blk in enumerate(self.backbone.blocks):
            if i < freeze_blocks:
                for p in blk.parameters():
                    p.requires_grad = False

        self.queries = nn.Parameter(torch.randn(num_keypoints, self.hidden_dim) * 0.02)
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.spatial_proj = nn.Conv2d(self.hidden_dim, kpt_embed_dim, kernel_size=1)
        self.query_proj = nn.Linear(self.hidden_dim, kpt_embed_dim)

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    def _spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(x)
        patch_tokens = tokens[:, 1:, :]
        h = w = self.grid_size
        return patch_tokens.transpose(1, 2).reshape(x.shape[0], self.hidden_dim, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sp = self.spatial_proj(self._spatial_features(x))
        q = self.query_norm(self.queries).unsqueeze(0).expand(x.shape[0], -1, -1)
        qf = self.query_proj(q)
        heatmaps = torch.einsum("bchw,bkc->bkhw", sp, qf)
        heatmaps = F.interpolate(
            heatmaps,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return heatmaps

    def attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        b, k, h, w = logits.shape
        flat = logits.reshape(b, k, -1)
        return F.softmax(flat, dim=-1).reshape(b, k, h, w)
