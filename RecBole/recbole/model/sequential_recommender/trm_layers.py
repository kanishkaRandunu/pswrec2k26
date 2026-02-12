# -*- coding: utf-8 -*-
"""
TRM building blocks for TRM4Rec (aligned with TinyRecursiveModels).
RMSNorm, RoPE, SwiGLU, Attention with causal+padding mask support, TRMBlock.
No dependency on TinyRecursiveModels or einops.
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    """Truncated normal init (JAX/Flax style)."""
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2
            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(
                1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
            )
            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor


def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """RoPE cache; forward() returns (cos, sin) for [seq_len, head_dim]."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached, self.sin_cached


class SwiGLU(nn.Module):
    """SwiGLU with expansion; same structure as TinyRecursiveModels."""

    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = nn.Linear(hidden_size, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)
        trunc_normal_init_(self.gate_up_proj.weight, std=1.0 / (hidden_size ** 0.5))
        trunc_normal_init_(self.down_proj.weight, std=1.0 / (inter ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TRMAttention(nn.Module):
    """TRM-style attention with RoPE and optional additive attn_mask (causal+padding)."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
    ):
        super().__init__()
        num_key_value_heads = num_key_value_heads or num_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads

        self.qkv_proj = nn.Linear(
            hidden_size,
            (num_heads + 2 * num_key_value_heads) * head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(self.output_size, hidden_size, bias=False)
        trunc_normal_init_(self.qkv_proj.weight, std=1.0 / (hidden_size ** 0.5))
        trunc_normal_init_(self.o_proj.weight, std=1.0 / (self.output_size ** 0.5))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(
            batch_size,
            seq_len,
            self.num_heads + 2 * self.num_key_value_heads,
            self.head_dim,
        )
        query = qkv[:, :, : self.num_heads]
        key = qkv[
            :,
            :,
            self.num_heads : self.num_heads + self.num_key_value_heads,
        ]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # B S H D -> B H S D for SDPA
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        if attn_mask is not None:
            # attn_mask: [B, 1, L, L] additive; broadcast to [B, H, L, L] if needed
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_output = scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, is_causal=False
            )
        else:
            attn_output = scaled_dot_product_attention(
                query, key, value, is_causal=True
            )

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.output_size
        )
        return self.o_proj(attn_output)


class TRMBlock(nn.Module):
    """One TRM block: post-norm attention then post-norm SwiGLU (same order as TinyRecursiveModels)."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        expansion: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.rms_norm_eps = rms_norm_eps
        self.self_attn = TRMAttention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
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
