"""Language subtask queries with cross-attention over visual patch tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import SiglipProcessor, SiglipTextModel

# Cross-attn query-over-visual-latents pattern: https://github.com/peract/peract/blob/master/arm/llm_peract/layers/perceiver_lang_io.py


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        hidden = int(embed_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        q = self.norm_q(query)
        kv = self.norm_kv(context)
        attn_out, attn_weights = self.attn(
            q,
            kv,
            kv,
            need_weights=return_attn_weights,
            average_attn_weights=True,
        )
        query = query + attn_out
        query = query + self.ff(query)
        if return_attn_weights:
            return query, attn_weights
        return query


class SubtaskQueryEncoder(nn.Module):
    """Encode fixed subtask prompts and cross-attend over multi-view patch tokens."""

    def __init__(
        self,
        *,
        model_name: str = "google/siglip-base-patch16-224",
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.text_model = SiglipTextModel.from_pretrained(model_name)
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, embed_dim)
        self.layers = nn.ModuleList(
            CrossAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_prompts(self, prompts: list[str], batch_size: int) -> torch.Tensor:
        """Return text query tokens ``(B, num_prompts, embed_dim)``."""
        inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        text_hidden = self.text_model(**inputs).last_hidden_state
        pooled = text_hidden.mean(dim=1)
        query = self.text_proj(pooled)
        return query.unsqueeze(0).expand(batch_size, len(prompts), -1)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        prompts: list[str],
        *,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Cross-attend each prompt query over shared patch tokens.

        Returns:
            Tensor of shape ``(B, num_prompts, embed_dim)``, or with attention weights
            ``(B, num_prompts, num_patches)`` when ``return_attn_weights=True``.
        """
        batch_size = patch_tokens.shape[0]
        queries = self.encode_prompts(prompts, batch_size)
        outputs: list[torch.Tensor] = []
        attn_outputs: list[torch.Tensor] = []
        for prompt_idx in range(queries.shape[1]):
            q = queries[:, prompt_idx : prompt_idx + 1, :]
            last_attn: torch.Tensor | None = None
            for layer in self.layers:
                if return_attn_weights:
                    q, last_attn = layer(q, patch_tokens, return_attn_weights=True)
                else:
                    q = layer(q, patch_tokens)
            outputs.append(self.output_norm(q))
            if return_attn_weights:
                assert last_attn is not None
                attn_outputs.append(last_attn.squeeze(1))
        encoded = torch.cat(outputs, dim=1)
        if return_attn_weights:
            return encoded, torch.stack(attn_outputs, dim=1)
        return encoded
