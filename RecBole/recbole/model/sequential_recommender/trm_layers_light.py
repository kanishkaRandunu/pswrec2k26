# -*- coding: utf-8 -*-
"""
TRM building blocks for TRM4RecLight: LightSANs-style low-rank attention.
Item-to-interest aggregation (masked), TRMAttentionLowRank, TRMBlockLight.
Reuses rms_norm, SwiGLU, apply_rotary_pos_emb from trm_layers (no change to trm_layers).
"""

from typing import Optional, Tuple

import torch
from torch import nn

from recbole.model.sequential_recommender.trm_layers import (
    _find_multiple,
    apply_rotary_pos_emb,
    rms_norm,
    trunc_normal_init_,
)
from recbole.model.sequential_recommender.trm_layers import SwiGLU


class ItemToInterestAggregationMasked(nn.Module):
    """Maps [B, L, d] to [B, k, d]; padding positions do not contribute when mask is provided."""

    def __init__(self, feature_dim: int, k_interests: int):
        super().__init__()
        self.k_interests = k_interests
        self.theta = nn.Parameter(torch.randn(feature_dim, k_interests))
        trunc_normal_init_(self.theta, std=1.0)

    def forward(
        self,
        input_tensor: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # input_tensor: [B, L, d], mask: [B, L] True = valid, False = pad
        B, L, d = input_tensor.shape
        D_matrix = torch.matmul(input_tensor, self.theta)  # [B, L, k]
        if mask is not None:
            # mask: [B, L], True = valid. Set invalid to large negative before softmax.
            D_matrix = D_matrix.masked_fill(~mask.unsqueeze(-1), -1e9)
        D_matrix = nn.functional.softmax(D_matrix, dim=-2)  # over sequence dim
        if mask is not None:
            D_matrix = D_matrix * mask.unsqueeze(-1).to(D_matrix.dtype)
            # renormalize so that for each batch and interest, weights sum to 1 over valid positions
            denom = D_matrix.sum(dim=1, keepdim=True).clamp(min=1e-9)
            D_matrix = D_matrix / denom
        result = torch.einsum("bld,blk->bkd", input_tensor, D_matrix)
        return result  # [B, k, d]


def _padding_mask_from_attn_mask(attn_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Derive [B, L] boolean (True = valid position) from additive attn_mask [B, 1, L, L]."""
    if attn_mask is None:
        return None
    # attn_mask: 0.0 = valid, -10000 = invalid. Diagonal (j,j) is valid iff position j is valid.
    if attn_mask.dim() == 4:
        diag = attn_mask[:, 0, :, :].diagonal(dim1=1, dim2=2)  # [B, L]
    else:
        diag = attn_mask.diagonal(dim1=1, dim2=2)  # [B, L]
    return (diag == 0)  # True where valid


class TRMAttentionLowRank(nn.Module):
    """LightSANs-style low-rank attention: Q (L) x K_agg (k), same interface as TRMAttention."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        seq_len: int,
        k_interests: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.k_interests = k_interests

        self.q_proj = nn.Linear(hidden_size, self.output_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.output_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.output_size, bias=False)
        self.agg_key = ItemToInterestAggregationMasked(self.output_size, k_interests)
        self.agg_value = ItemToInterestAggregationMasked(self.output_size, k_interests)
        self.o_proj = nn.Linear(self.output_size, hidden_size, bias=False)

        trunc_normal_init_(self.q_proj.weight, std=1.0 / (hidden_size ** 0.5))
        trunc_normal_init_(self.k_proj.weight, std=1.0 / (hidden_size ** 0.5))
        trunc_normal_init_(self.v_proj.weight, std=1.0 / (hidden_size ** 0.5))
        trunc_normal_init_(self.o_proj.weight, std=1.0 / (self.output_size ** 0.5))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len_in, _ = hidden_states.shape

        # Q: [B, L, H, D], apply RoPE
        query = self.q_proj(hidden_states)
        query = query.view(
            batch_size, seq_len_in, self.num_heads, self.head_dim
        )
        if cos_sin is not None:
            cos, sin = cos_sin
            # apply_rotary_pos_emb expects q,k [bs, seq_len, num_heads, head_dim]
            query, _ = apply_rotary_pos_emb(query, query, cos, sin)

        # K, V: project then aggregate to [B, k, H]
        key_full = self.k_proj(hidden_states)   # [B, L, H]
        value_full = self.v_proj(hidden_states)
        padding_mask = _padding_mask_from_attn_mask(attn_mask)
        key_agg = self.agg_key(key_full, padding_mask)    # [B, k, H]
        value_agg = self.agg_value(value_full, padding_mask)

        # Reshape for attention: key_agg [B, k, num_heads, head_dim], value_agg same
        key_agg = key_agg.view(
            batch_size, self.k_interests, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)   # [B, H, k, D]
        value_agg = value_agg.view(
            batch_size, self.k_interests, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)   # [B, H, k, D]
        query = query.permute(0, 2, 1, 3)   # [B, H, L, D]

        # scores [B, H, L, k]
        scale = self.head_dim ** -0.5
        scores = torch.matmul(query, key_agg.transpose(-2, -1)) * scale

        # Mask padding positions (output at padding is not used; readout is last valid)
        if attn_mask is not None:
            # attn_mask [B, 1, L, L]: row j is -10000 for invalid. We want scores[b,h,j,:] masked if j is pad.
            if attn_mask.dim() == 4:
                pad_row = (attn_mask[:, 0, :, 0] == -10000.0)  # [B, L]
            else:
                pad_row = (attn_mask[:, :, 0] == -10000.0)
            scores = scores.masked_fill(pad_row.unsqueeze(1).unsqueeze(-1), -1e9)

        attn_weights = nn.functional.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value_agg)  # [B, H, L, D]
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len_in, self.output_size
        )
        return self.o_proj(attn_output)


class TRMBlockLight(nn.Module):
    """One TRM block with low-rank attention; same interface as TRMBlock."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        expansion: float,
        rms_norm_eps: float,
        seq_len: int,
        k_interests: int,
    ):
        super().__init__()
        self.rms_norm_eps = rms_norm_eps
        self.self_attn = TRMAttentionLowRank(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            seq_len=seq_len,
            k_interests=k_interests,
        )
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states + self.self_attn(hidden_states, cos_sin, attn_mask),
            self.rms_norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            self.rms_norm_eps,
        )
        return hidden_states
