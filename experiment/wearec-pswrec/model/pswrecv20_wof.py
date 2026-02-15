#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PSWRecV20: Polyphonic Subspace Attention (PSA).

Aligns independent wavelet filterbands strictly to attention-head subspaces:
- Head 0 (kernel 3): first head_dim dimensions → hyper-local phase/magnitude.
- Head 3 (kernel 31): last head_dim dimensions → global phase/magnitude.
Magnitude is bounded with tanh so softmax gradients never explode.
"""

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from model._abstract_model import SequentialRecModel


# ---------------------------------------------------------------------------
# 1. Polyphonic Subspace Filterbank
# ---------------------------------------------------------------------------
class PolyphonicWaveletFilter(nn.Module):
    r"""Extracts independent Phase and Magnitude for each attention-head subspace."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        kernel_sizes: List[int],
        dilations: List[int],
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.n_bands = len(kernel_sizes)
        if self.n_bands != self.n_heads:
            raise ValueError(
                f"V20 requires one band per head: n_bands={self.n_bands} != n_heads={self.n_heads}"
            )

        self.real_convs = nn.ModuleList()
        self.imag_convs = nn.ModuleList()
        self.pads: List[int] = []

        for k, d in zip(kernel_sizes, dilations):
            pad = (k - 1) * d
            self.pads.append(pad)
            self.real_convs.append(
                nn.Conv1d(
                    self.head_dim,
                    self.head_dim,
                    kernel_size=k,
                    dilation=d,
                    padding=0,
                    groups=self.head_dim,
                    bias=False,
                )
            )
            self.imag_convs.append(
                nn.Conv1d(
                    self.head_dim,
                    self.head_dim,
                    kernel_size=k,
                    dilation=d,
                    padding=0,
                    groups=self.head_dim,
                    bias=False,
                )
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.size()
        x_sub = x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 3, 1)

        us: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []
        for i in range(self.n_heads):
            x_i = x_sub[:, i, :, :]
            x_padded = F.pad(x_i, (self.pads[i], 0))
            u_i = self.real_convs[i](x_padded).mean(dim=1)
            v_i = self.imag_convs[i](x_padded).mean(dim=1)
            us.append(u_i)
            vs.append(v_i)

        U = torch.stack(us, dim=1)
        V = torch.stack(vs, dim=1)
        mag = torch.sqrt(U * U + V * V + 1e-8)
        phi = torch.atan2(V, U + 1e-8)
        return phi, mag


# ---------------------------------------------------------------------------
# 2. Phase rotation helper
# ---------------------------------------------------------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_polyphonic_rope(
    q: torch.Tensor, k: torch.Tensor, phase: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    phase = torch.cat([phase, phase], dim=-1)
    sin_phase = torch.sin(phase)
    cos_phase = torch.cos(phase)
    q_rotated = (q * cos_phase) + (rotate_half(q) * sin_phase)
    k_rotated = (k * cos_phase) + (rotate_half(k) * sin_phase)
    return q_rotated, k_rotated


# ---------------------------------------------------------------------------
# 3. Polyphonic Subspace Attention
# ---------------------------------------------------------------------------
class PolyphonicSubspaceAttentionV20(nn.Module):
    """AM-B-RoPE per head with subspace-aligned phi/mag and tanh-bounded magnitude."""

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by n_heads {n_heads}"
            )
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim {self.head_dim} must be even for RoPE.")

        self.gamma = nn.Parameter(torch.ones(1, self.n_heads, 1, 1) * 0.1)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        return x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        phi: torch.Tensor,
        mag: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = hidden_states.size()
        residual = hidden_states

        q = self._shape(self.query(hidden_states))
        k = self._shape(self.key(hidden_states))
        v = self._shape(self.value(hidden_states))

        phase_head = phi.unsqueeze(-1)
        phase_rope = phase_head.expand(-1, -1, -1, self.head_dim // 2)

        bounded_mag = torch.tanh(mag.unsqueeze(-1))

        q_scaled = q * (1.0 + self.gamma * bounded_mag)
        k_scaled = k * (1.0 + self.gamma * bounded_mag)

        q_rot, k_rot = apply_polyphonic_rope(q_scaled, k_scaled, phase_rope)

        attn_scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        eye = torch.eye(L, dtype=torch.bool, device=hidden_states.device).view(1, 1, L, L)
        row_max = attn_scores.max(dim=-1, keepdim=True)[0]
        attn_scores = torch.where(
            (row_max <= -1e8) & eye,
            torch.zeros_like(attn_scores),
            attn_scores,
        )

        attn_probs = self.attn_dropout(F.softmax(attn_scores, dim=-1))
        context = (
            torch.matmul(attn_probs, v)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, L, self.hidden_size)
        )
        return self.layer_norm(self.out_dropout(self.out_proj(context)) + residual)


# ---------------------------------------------------------------------------
# 4. Feed-forward (reuse V13)
# ---------------------------------------------------------------------------
class FeedForwardV13(nn.Module):
    """Position-wise FFN with residual + LayerNorm."""

    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        act_map = {
            "gelu": lambda x: x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))),
            "relu": F.relu,
            "swish": lambda x: x * torch.sigmoid(x),
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        if hidden_act not in act_map:
            raise ValueError(f"Unsupported hidden_act '{hidden_act}' in pswrecv20_wof.")
        self.act_fn = act_map[hidden_act]
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dense_1(x)
        x = self.act_fn(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)


class PSWBlockV20(nn.Module):
    """Single layer: Polyphonic Subspace Attention + FFN."""

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.attn = PolyphonicSubspaceAttentionV20(
            n_heads=n_heads,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.ffn = FeedForwardV13(
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        phi: torch.Tensor,
        mag: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.attn(hidden_states, attention_mask, phi, mag)
        return self.ffn(hidden_states)


class PSWEncoderV20(nn.Module):
    """Stack of PSWBlockV20 layers."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            PSWBlockV20(
                n_heads=n_heads,
                hidden_size=hidden_size,
                inner_size=inner_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attn_dropout_prob=attn_dropout_prob,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        phi: torch.Tensor,
        mag: torch.Tensor,
        output_all_encoded_layers: bool = True,
    ) -> List[torch.Tensor]:
        all_layers: List[torch.Tensor] = []
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, phi, mag)
            if output_all_encoded_layers:
                all_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_layers.append(hidden_states)
        return all_layers


# ---------------------------------------------------------------------------
# 5. WOF model shell
# ---------------------------------------------------------------------------
class PSWRecV20WOFModel(SequentialRecModel):
    r"""PSWRecV20: Polyphonic Subspace Attention — one wavelet band per attention head."""

    def __init__(self, args):
        super().__init__(args)
        self.n_layers = args.num_hidden_layers
        self.n_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.inner_size = getattr(args, "inner_size", 4 * args.hidden_size)
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.attn_dropout_prob = args.attention_probs_dropout_prob
        self.hidden_act = args.hidden_act
        self.layer_norm_eps = 1e-12
        self.initializer_range = args.initializer_range
        band_kernel_sizes = getattr(args, "band_kernel_sizes", [3, 7, 15, 31])
        band_dilations = getattr(args, "band_dilations", [1, 2, 4, 8])
        self.n_bands = getattr(args, "n_bands", len(band_kernel_sizes))
        self.phase_aux = getattr(args, "phase_aux", False)
        self.phase_aux_weight = getattr(args, "phase_aux_weight", 0.0)
        self._last_phase_reg: Optional[torch.Tensor] = None

        if self.n_heads != self.n_bands:
            raise ValueError(
                f"PSWRecV20 requires num_attention_heads == n_bands. "
                f"Got num_attention_heads={self.n_heads}, n_bands={self.n_bands}."
            )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.phase_filter = PolyphonicWaveletFilter(
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            kernel_sizes=band_kernel_sizes,
            dilations=band_dilations,
        )
        self.encoder = PSWEncoderV20(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        sequence_emb = self.add_position_embedding(input_ids)
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        phi, mag = self.phase_filter(sequence_emb)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        if self.phase_aux:
            cos_diff = cos_phi[:, :, 1:] - cos_phi[:, :, :-1]
            sin_diff = sin_phi[:, :, 1:] - sin_phi[:, :, :-1]
            self._last_phase_reg = (cos_diff.pow(2) + sin_diff.pow(2)).mean()
        else:
            self._last_phase_reg = None

        extended_attention_mask = self.get_attention_mask(input_ids)
        encoder_outputs = self.encoder(
            sequence_emb,
            extended_attention_mask,
            phi,
            mag,
            output_all_encoded_layers=True,
        )
        if all_sequence_output:
            return encoder_outputs
        return encoder_outputs[-1]

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)
        if self.phase_aux and self._last_phase_reg is not None:
            loss = loss + self.phase_aux_weight * self._last_phase_reg
        return loss

    def predict(self, input_ids, user_ids=None, all_sequence_output=False):
        return self.forward(input_ids, user_ids, all_sequence_output)
