#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantum-Inspired Interference Attention (QIA / VQIA) adapted for the WEARec Official Framework.

Original QIA:
1) Quantum State Preparation (complex filterbank).
2) CVNN (ComplexLinear) for Q, K, V.
3) Hermitian attention: softmax(|Q*K^H|^2).
4) Wave collapse to amplitude only.

Improved QIA (QIARecImprovedWOFModel):
- Attention: real part of Q·K^H / sqrt(d) as logits (standard scaling) + learnable temperature.
- Readout: phase-aware = concat(amplitude, cos(phase), sin(phase)) -> Linear -> logits (no phase discard).
- Stabilization: proper scaling and temperature so optimization matches standard attention.
"""

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from model._abstract_model import SequentialRecModel


# ---------------------------------------------------------------------------
# 1. Complex-Valued Neural Network (CVNN) Base Layers
# ---------------------------------------------------------------------------
class ComplexLinear(nn.Module):
    """
    Simulates native complex linear projections using real-valued tensors
    to guarantee strict compatibility across all PyTorch versions.
    (Q_r + i Q_i) = (X_r W_r - X_i W_i) + i(X_r W_i + X_i W_r)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(
        self, x_r: torch.Tensor, x_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_r = self.fc_r(x_r) - self.fc_i(x_i)
        out_i = self.fc_r(x_i) + self.fc_i(x_r)
        return out_r, out_i


class ComplexLayerNorm(nn.Module):
    """Applies LayerNorm independently to the real and imaginary components."""

    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.ln_r = nn.LayerNorm(hidden_size, eps=eps)
        self.ln_i = nn.LayerNorm(hidden_size, eps=eps)

    def forward(
        self, x_r: torch.Tensor, x_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ln_r(x_r), self.ln_i(x_i)


# ---------------------------------------------------------------------------
# 2. Quantum State Preparator (The Complex Filterbank)
# ---------------------------------------------------------------------------
class QuantumStatePreparator(nn.Module):
    """
    Replaces the V13 filterbank. Instead of squashing U and V into scalar
    magnitude and phase, this explicitly preserves the multi-dimensional
    complex wave state of the user sequence.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_sizes: List[int],
        dilations: Optional[List[int]] = None,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1 for _ in kernel_sizes]

        real_convs, imag_convs, pads = [], [], []
        for k, d in zip(kernel_sizes, dilations):
            pad = (k - 1) * d
            pads.append(pad)
            real_convs.append(
                nn.Conv1d(
                    hidden_size,
                    hidden_size,
                    kernel_size=k,
                    dilation=d,
                    padding=0,
                    groups=hidden_size,
                    bias=False,
                )
            )
            imag_convs.append(
                nn.Conv1d(
                    hidden_size,
                    hidden_size,
                    kernel_size=k,
                    dilation=d,
                    padding=0,
                    groups=hidden_size,
                    bias=False,
                )
            )

        self.real_convs = nn.ModuleList(real_convs)
        self.imag_convs = nn.ModuleList(imag_convs)
        self.band_pads = pads

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_t = x.transpose(1, 2)  # (B, D, L)

        u_sum = torch.zeros_like(x_t)
        v_sum = torch.zeros_like(x_t)

        for pad, conv_r, conv_i in zip(self.band_pads, self.real_convs, self.imag_convs):
            x_padded = F.pad(x_t, (pad, 0))
            u_sum = u_sum + conv_r(x_padded)
            v_sum = v_sum + conv_i(x_padded)

        return u_sum.transpose(1, 2), v_sum.transpose(1, 2)


# ---------------------------------------------------------------------------
# 3. Quantum-Inspired Interference Attention (QIA)
# ---------------------------------------------------------------------------
class InterferenceAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_heads

        self.q_proj = ComplexLinear(hidden_size, hidden_size)
        self.k_proj = ComplexLinear(hidden_size, hidden_size)
        self.v_proj = ComplexLinear(hidden_size, hidden_size)

        self.out_proj = ComplexLinear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        return x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(
        self,
        x_r: torch.Tensor,
        x_i: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x_r.size()

        qr_raw, qi_raw = self.q_proj(x_r, x_i)
        kr_raw, ki_raw = self.k_proj(x_r, x_i)
        vr_raw, vi_raw = self.v_proj(x_r, x_i)

        q_r, q_i = self._shape(qr_raw), self._shape(qi_raw)
        k_r, k_i = self._shape(kr_raw), self._shape(ki_raw)
        v_r, v_i = self._shape(vr_raw), self._shape(vi_raw)

        # S = Q * K^H: S_real = Q_r*K_r + Q_i*K_i, S_imag = Q_i*K_r - Q_r*K_i
        s_real = torch.matmul(q_r, k_r.transpose(-1, -2)) + torch.matmul(
            q_i, k_i.transpose(-1, -2)
        )
        s_imag = torch.matmul(q_i, k_r.transpose(-1, -2)) - torch.matmul(
            q_r, k_i.transpose(-1, -2)
        )

        interference = (s_real.pow(2) + s_imag.pow(2)) / self.head_dim

        if attention_mask is not None:
            interference = interference + attention_mask

        attn_probs = self.attn_dropout(F.softmax(interference, dim=-1))

        c_r = (
            torch.matmul(attn_probs, v_r)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, L, D)
        )
        c_i = (
            torch.matmul(attn_probs, v_i)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, L, D)
        )

        out_r, out_i = self.out_proj(c_r, c_i)
        return self.out_dropout(out_r), self.out_dropout(out_i)


# ---------------------------------------------------------------------------
# 3b. Improved Interference Attention (real-part logits + temperature)
# ---------------------------------------------------------------------------
class ImprovedInterferenceAttention(nn.Module):
    """
    Same complex Q,K,V and Hermitian product, but:
    - Logits = real part of Q·K^H / sqrt(head_dim) (standard scaling, better gradients).
    - Learnable temperature for softmax (default softplus so > 0).
    """

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        init_temperature: float = 1.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = ComplexLinear(hidden_size, hidden_size)
        self.k_proj = ComplexLinear(hidden_size, hidden_size)
        self.v_proj = ComplexLinear(hidden_size, hidden_size)
        self.out_proj = ComplexLinear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        # Temperature > 0: logits / temp; larger temp = softer attention
        self.log_temperature = nn.Parameter(torch.tensor(math.log(max(init_temperature, 0.01))))

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        return x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(
        self,
        x_r: torch.Tensor,
        x_i: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x_r.size()

        qr_raw, qi_raw = self.q_proj(x_r, x_i)
        kr_raw, ki_raw = self.k_proj(x_r, x_i)
        vr_raw, vi_raw = self.v_proj(x_r, x_i)

        q_r, q_i = self._shape(qr_raw), self._shape(qi_raw)
        k_r, k_i = self._shape(kr_raw), self._shape(ki_raw)
        v_r, v_i = self._shape(vr_raw), self._shape(vi_raw)

        # S = Q * K^H: S_real = Q_r*K_r + Q_i*K_i, S_imag = Q_i*K_r - Q_r*K_i
        s_real = torch.matmul(q_r, k_r.transpose(-1, -2)) + torch.matmul(
            q_i, k_i.transpose(-1, -2)
        )
        # Use real part only as logits (standard attention-like), scaled by 1/sqrt(d)
        logits = s_real * self.scale
        temperature = F.softplus(self.log_temperature) + 0.01
        logits = logits / temperature

        if attention_mask is not None:
            logits = logits + attention_mask

        attn_probs = self.attn_dropout(F.softmax(logits, dim=-1))

        c_r = (
            torch.matmul(attn_probs, v_r)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, L, D)
        )
        c_i = (
            torch.matmul(attn_probs, v_i)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, L, D)
        )

        out_r, out_i = self.out_proj(c_r, c_i)
        return self.out_dropout(out_r), self.out_dropout(out_i)


# ---------------------------------------------------------------------------
# 4. Complex Feed Forward & Encoder Blocks
# ---------------------------------------------------------------------------
class ComplexFFN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
    ):
        super().__init__()
        self.fc1 = ComplexLinear(hidden_size, inner_size)
        self.fc2 = ComplexLinear(inner_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self, x_r: torch.Tensor, x_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_r, h_i = self.fc1(x_r, x_i)
        h_r, h_i = F.gelu(h_r), F.gelu(h_i)
        out_r, out_i = self.fc2(h_r, h_i)
        return self.dropout(out_r), self.dropout(out_i)


class QIABlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        layer_norm_eps = getattr(args, "layer_norm_eps", 1e-12)
        self.attn = InterferenceAttention(
            args.num_attention_heads,
            args.hidden_size,
            args.hidden_dropout_prob,
            args.attention_probs_dropout_prob,
        )
        self.ffn = ComplexFFN(
            args.hidden_size,
            getattr(args, "inner_size", 4 * args.hidden_size),
            args.hidden_dropout_prob,
        )
        self.ln1 = ComplexLayerNorm(args.hidden_size, eps=layer_norm_eps)
        self.ln2 = ComplexLayerNorm(args.hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        x_r: torch.Tensor,
        x_i: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a_r, a_i = self.attn(x_r, x_i, attention_mask)
        x_r, x_i = self.ln1(x_r + a_r, x_i + a_i)

        f_r, f_i = self.ffn(x_r, x_i)
        x_r, x_i = self.ln2(x_r + f_r, x_i + f_i)

        return x_r, x_i


class QIABlockImproved(nn.Module):
    """QIABlock using ImprovedInterferenceAttention (real-part logits + temperature)."""

    def __init__(self, args):
        super().__init__()
        layer_norm_eps = getattr(args, "layer_norm_eps", 1e-12)
        init_temp = getattr(args, "qia_init_temperature", 1.0)
        self.attn = ImprovedInterferenceAttention(
            args.num_attention_heads,
            args.hidden_size,
            args.hidden_dropout_prob,
            args.attention_probs_dropout_prob,
            init_temperature=init_temp,
        )
        self.ffn = ComplexFFN(
            args.hidden_size,
            getattr(args, "inner_size", 4 * args.hidden_size),
            args.hidden_dropout_prob,
        )
        self.ln1 = ComplexLayerNorm(args.hidden_size, eps=layer_norm_eps)
        self.ln2 = ComplexLayerNorm(args.hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        x_r: torch.Tensor,
        x_i: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a_r, a_i = self.attn(x_r, x_i, attention_mask)
        x_r, x_i = self.ln1(x_r + a_r, x_i + a_i)
        f_r, f_i = self.ffn(x_r, x_i)
        x_r, x_i = self.ln2(x_r + f_r, x_i + f_i)
        return x_r, x_i


# ---------------------------------------------------------------------------
# 5. WOF Model Shell
# ---------------------------------------------------------------------------
class QIARecWOFModel(SequentialRecModel):
    """
    Quantum-Inspired Interference Attention Model (QIA-Rec / VQIA).
    Lifts the WOF framework to the complex plane; captures transient bursts
    and long-term periodicity via constructive and destructive wave interference.
    """

    def __init__(self, args):
        super().__init__(args)
        band_kernel_sizes = getattr(args, "band_kernel_sizes", [3, 7, 15, 31])
        band_dilations = getattr(args, "band_dilations", [1, 2, 4, 8])

        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.state_preparator = QuantumStatePreparator(
            args.hidden_size, band_kernel_sizes, band_dilations
        )

        self.layers = nn.ModuleList(
            [QIABlock(args) for _ in range(args.num_hidden_layers)]
        )

        self.collapse_proj = nn.Linear(args.hidden_size, args.hidden_size)

        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        sequence_emb = self.dropout(
            self.LayerNorm(self.add_position_embedding(input_ids))
        )

        x_r, x_i = self.state_preparator(sequence_emb)

        extended_attention_mask = self.get_attention_mask(input_ids)
        all_layers = []

        for layer in self.layers:
            x_r, x_i = layer(x_r, x_i, extended_attention_mask)

            amplitude = torch.sqrt(x_r.pow(2) + x_i.pow(2) + 1e-8)
            all_layers.append(self.collapse_proj(amplitude))

        return all_layers if all_sequence_output else all_layers[-1]

    def calculate_loss(
        self, input_ids, answers, neg_answers, same_target, user_ids
    ):
        seq_output = self.forward(input_ids)[:, -1, :]
        logits = torch.matmul(
            seq_output, self.item_embeddings.weight.transpose(0, 1)
        )
        return nn.CrossEntropyLoss()(logits, answers)

    def predict(self, input_ids, user_ids=None, all_sequence_output=False):
        return self.forward(input_ids, user_ids, all_sequence_output)


# ---------------------------------------------------------------------------
# 6. Improved QIA Model (real-part attention + phase-aware readout)
# ---------------------------------------------------------------------------
class QIARecImprovedWOFModel(SequentialRecModel):
    """
    Improved QIA: (1) Real-part attention logits + learnable temperature.
    (2) Phase-aware readout: concat(amplitude, cos(phase), sin(phase)) -> projection
    so phase is used in the final prediction instead of discarding it.
    """

    def __init__(self, args):
        super().__init__(args)
        band_kernel_sizes = getattr(args, "band_kernel_sizes", [3, 7, 15, 31])
        band_dilations = getattr(args, "band_dilations", [1, 2, 4, 8])

        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.state_preparator = QuantumStatePreparator(
            args.hidden_size, band_kernel_sizes, band_dilations
        )

        self.layers = nn.ModuleList(
            [QIABlockImproved(args) for _ in range(args.num_hidden_layers)]
        )

        # Phase-aware readout: amplitude + cos(phase) + sin(phase) -> hidden_size
        self.readout_proj = nn.Linear(3 * args.hidden_size, args.hidden_size)

        self.apply(self.init_weights)

    def _complex_to_readout(self, x_r: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        """(B, L, D) complex -> (B, L, 3*D) real [amplitude, cos(phase), sin(phase)]."""
        amplitude = torch.sqrt(x_r.pow(2) + x_i.pow(2) + 1e-8)
        phase = torch.atan2(x_i, x_r + 1e-8)
        return torch.cat([amplitude, torch.cos(phase), torch.sin(phase)], dim=-1)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        sequence_emb = self.dropout(
            self.LayerNorm(self.add_position_embedding(input_ids))
        )
        x_r, x_i = self.state_preparator(sequence_emb)
        extended_attention_mask = self.get_attention_mask(input_ids)
        all_layers = []

        for layer in self.layers:
            x_r, x_i = layer(x_r, x_i, extended_attention_mask)
            readout_feat = self._complex_to_readout(x_r, x_i)
            all_layers.append(self.readout_proj(readout_feat))

        return all_layers if all_sequence_output else all_layers[-1]

    def calculate_loss(
        self, input_ids, answers, neg_answers, same_target, user_ids
    ):
        seq_output = self.forward(input_ids)[:, -1, :]
        logits = torch.matmul(
            seq_output, self.item_embeddings.weight.transpose(0, 1)
        )
        return nn.CrossEntropyLoss()(logits, answers)

    def predict(self, input_ids, user_ids=None, all_sequence_output=False):
        return self.forward(input_ids, user_ids, all_sequence_output)
