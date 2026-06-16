"""ACT action head conditioned on DexQuery subtask embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTDecoder, ACTEncoder

from .base import ActionHeadOutput, BaseActionHead

# ACT policy / transformer blocks: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py


def _build_act_config(
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    cond_dim: int,
    dim_model: int = 512,
) -> ACTConfig:
    """Minimal ACT config: DexQuery ``z_subtask`` maps to env-state token, no images."""
    return ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            "observation.environment_state": PolicyFeature(
                type=FeatureType.ENV,
                shape=(cond_dim,),
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        dim_model=dim_model,
        use_vae=False,
    )


class ActActionHead(BaseActionHead):
    """LeRobot ACT decoder stack with DexQuery (z_subtask, state) conditioning."""

    action_type = "act"

    def __init__(
        self,
        *,
        embed_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        dim_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 1,
        **_kwargs,
    ) -> None:
        super().__init__()
        self._chunk_size = chunk_size
        self._action_dim = action_dim
        self.config = _build_act_config(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            cond_dim=embed_dim,
            dim_model=dim_model,
        )
        self.config.n_heads = num_heads
        self.config.n_encoder_layers = num_encoder_layers
        self.config.n_decoder_layers = num_decoder_layers

        self.subtask_proj = nn.Linear(embed_dim, dim_model)
        self.encoder = ACTEncoder(self.config)
        self.decoder = ACTDecoder(self.config)
        self.encoder_latent_input_proj = nn.Linear(self.config.latent_dim, dim_model)
        self.encoder_robot_state_input_proj = nn.Linear(state_dim, dim_model)
        self.encoder_env_state_input_proj = nn.Linear(embed_dim, dim_model)
        self.encoder_1d_feature_pos_embed = nn.Embedding(3, dim_model)
        self.decoder_pos_embed = nn.Embedding(chunk_size, dim_model)
        self.action_head = nn.Linear(dim_model, action_dim)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def forward(
        self,
        z_subtask: torch.Tensor,
        state: torch.Tensor,
        *,
        actions: torch.Tensor | None = None,
    ) -> ActionHeadOutput:
        batch_size = state.shape[0]
        device = state.device
        latent = torch.zeros(
            batch_size,
            self.config.latent_dim,
            dtype=state.dtype,
            device=device,
        )
        encoder_tokens = [
            self.encoder_latent_input_proj(latent).unsqueeze(0),
            self.encoder_robot_state_input_proj(state).unsqueeze(0),
            self.encoder_env_state_input_proj(z_subtask).unsqueeze(0),
        ]
        encoder_in = torch.cat(encoder_tokens, dim=0)
        pos_embed = self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
        encoder_out = self.encoder(encoder_in, pos_embed=pos_embed)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in.dtype,
            device=device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        pred_actions = self.action_head(decoder_out.transpose(0, 1))

        loss = None
        if actions is not None:
            loss = F.l1_loss(pred_actions, actions)
        return ActionHeadOutput(pred_actions=pred_actions, loss=loss)
