"""DexQuery model assembly and training losses."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_head import BaseActionHead, build_action_head
from .outcome_head import OutcomeHead
from .subtask_query import SubtaskQueryEncoder
from .vision_backbone import SiglipPatchBackbone


@dataclass
class DexQueryModelConfig:
    vision_model: str = "google/siglip-base-patch16-224"
    embed_dim: int = 768
    num_heads: int = 8
    cross_attn_layers: int = 2
    action_decoder_layers: int = 2
    action_head_type: str = "act"  # act | diffusion | flow_matching | dit
    num_cameras: int = 3
    state_dim: int = 46
    action_dim: int = 44
    chunk_size: int = 30
    outcome_loss_weight: float = 0.1
    freeze_vision: bool = False
    subtask_prompts: list[str] = field(
        default_factory=lambda: [
            "Grasp the tray with the left hand.",
            "Grasp the peg with the right hand.",
            "Insert the peg into the hole.",
        ]
    )


@dataclass
class DexQueryOutputs:
    pred_actions: torch.Tensor
    tray_logit: torch.Tensor
    peg_logit: torch.Tensor
    z_subtasks: torch.Tensor
    loss: torch.Tensor | None = None
    loss_action: torch.Tensor | None = None
    loss_outcome: torch.Tensor | None = None


@dataclass
class DexQueryPredictOutput:
    pred_actions: torch.Tensor
    tray_logit: torch.Tensor
    peg_logit: torch.Tensor
    tray_prob: torch.Tensor
    peg_prob: torch.Tensor
    subtask_phase: torch.Tensor
    z_subtasks: torch.Tensor


class DexQueryModel(nn.Module):
    """Subtask-query policy with outcome prediction and action-chunk regression."""

    def __init__(self, config: DexQueryModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or DexQueryModelConfig()
        self.backbone = SiglipPatchBackbone(
            model_name=self.config.vision_model,
            freeze=self.config.freeze_vision,
        )
        if self.config.embed_dim != self.backbone.embed_dim:
            self.config.embed_dim = self.backbone.embed_dim
        self.subtask_encoder = SubtaskQueryEncoder(
            model_name=self.config.vision_model,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.cross_attn_layers,
        )
        self.outcome_head = OutcomeHead(self.config.embed_dim)
        self.action_head: BaseActionHead = build_action_head(
            self.config.action_head_type,
            embed_dim=self.config.embed_dim,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            chunk_size=self.config.chunk_size,
            num_heads=self.config.num_heads,
            num_layers=self.config.action_decoder_layers,
        )

    @property
    def subtask_prompts(self) -> list[str]:
        return list(self.config.subtask_prompts)

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        *,
        actions: torch.Tensor | None = None,
        tray_ok: torch.Tensor | None = None,
        peg_ok: torch.Tensor | None = None,
        subtask_phase: torch.Tensor | None = None,
        subtask_prompts: list[str] | None = None,
    ) -> DexQueryOutputs:
        """Run DexQuery forward pass.

        Args:
            images: ``(B, num_cameras, C, H, W)`` in ``[0, 1]``.
            state: ``(B, state_dim)`` proprioception.
            actions: optional ``(B, chunk_size, action_dim)`` training targets.
            tray_ok / peg_ok: optional ``(B,)`` binary outcome targets.
            subtask_phase: optional ``(B,)`` ints in ``{0, 1, 2}`` selecting z for action BC.
            subtask_prompts: optional override for the 3 fixed subtask sentences.
        """
        prompts = subtask_prompts or self.subtask_prompts
        if len(prompts) != 3:
            raise ValueError(f"DexQuery expects 3 subtask prompts, got {len(prompts)}")

        patch_tokens = self.backbone(images)
        z_subtasks = self.subtask_encoder(patch_tokens, prompts)
        z_tray = z_subtasks[:, 0, :]
        z_peg = z_subtasks[:, 1, :]
        z_insert = z_subtasks[:, 2, :]

        tray_logit, peg_logit = self.outcome_head(z_tray, z_peg)
        if subtask_phase is not None:
            phase = subtask_phase.long().reshape(-1)
            batch_indices = torch.arange(z_subtasks.shape[0], device=z_subtasks.device)
            z_action = z_subtasks[batch_indices, phase, :]
        else:
            z_action = z_insert
        action_out = self.action_head(z_action, state, actions=actions)
        pred_actions = action_out.pred_actions

        loss = loss_action = loss_outcome = None
        if action_out.loss is not None:
            loss_action = action_out.loss
            loss = loss_action
        if tray_ok is not None and peg_ok is not None:
            tray_target = tray_ok.float()
            peg_target = peg_ok.float()
            loss_outcome = (
                F.binary_cross_entropy_with_logits(tray_logit, tray_target)
                + F.binary_cross_entropy_with_logits(peg_logit, peg_target)
            ) * 0.5
            loss = loss_outcome if loss is None else loss + self.config.outcome_loss_weight * loss_outcome

        return DexQueryOutputs(
            pred_actions=pred_actions,
            tray_logit=tray_logit,
            peg_logit=peg_logit,
            z_subtasks=z_subtasks,
            loss=loss,
            loss_action=loss_action,
            loss_outcome=loss_outcome,
        )

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        *,
        subtask_phase: torch.Tensor | int | None = None,
        subtask_prompts: list[str] | None = None,
    ) -> DexQueryPredictOutput:
        """Inference forward pass with optional explicit subtask phase selection."""
        prompts = subtask_prompts or self.subtask_prompts
        patch_tokens = self.backbone(images)
        z_subtasks = self.subtask_encoder(patch_tokens, prompts)
        tray_logit, peg_logit = self.outcome_head(z_subtasks[:, 0, :], z_subtasks[:, 1, :])
        tray_prob = torch.sigmoid(tray_logit)
        peg_prob = torch.sigmoid(peg_logit)

        if subtask_phase is None:
            z_action = z_subtasks[:, 2, :]
            phase_tensor = torch.full(
                (z_subtasks.shape[0],),
                2,
                dtype=torch.long,
                device=z_subtasks.device,
            )
        else:
            if isinstance(subtask_phase, int):
                phase_tensor = torch.full(
                    (z_subtasks.shape[0],),
                    subtask_phase,
                    dtype=torch.long,
                    device=z_subtasks.device,
                )
            else:
                phase_tensor = subtask_phase.long().reshape(-1)
            batch_indices = torch.arange(z_subtasks.shape[0], device=z_subtasks.device)
            z_action = z_subtasks[batch_indices, phase_tensor, :]

        pred_actions = self.action_head(z_action, state).pred_actions
        return DexQueryPredictOutput(
            pred_actions=pred_actions,
            tray_logit=tray_logit,
            peg_logit=peg_logit,
            tray_prob=tray_prob,
            peg_prob=peg_prob,
            subtask_phase=phase_tensor,
            z_subtasks=z_subtasks,
        )
