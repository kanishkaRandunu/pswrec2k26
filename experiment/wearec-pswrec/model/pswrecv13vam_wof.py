#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PSWRecV13 VAM (Value Amplitude Modulation) — WEARec Official Framework (WOF).

Same as PSWRecV13withoutphaserot except: amplitude scale is applied to Q, K, and V.
- Router (Q, K): attention probabilities (SOTA precision/NDCG).
- Payload (V): scale = 1.0 + 0.5 * mag_mix so high-magnitude periodic items
  inject more energy into the output (bridges Hit Ratio gap vs WEARec).
"""

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from model._abstract_model import SequentialRecModel


# ---------------------------------------------------------------------------
# 1. Filter bank and FFN (same as V13withoutphaserot; duplicated for standalone load)
# ---------------------------------------------------------------------------
class LocalPhaseFilterBankV13NoRot(nn.Module):
    """Extracts ungated phase features and magnitude for residual phase bias and amplitude modulation."""

    def __init__(
        self,
        hidden_size: int,
        kernel_sizes: List[int],
        dilations: Optional[List[int]] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes
        if dilations is None:
            dilations = [1 for _ in kernel_sizes]
        if len(dilations) != len(kernel_sizes):
            raise ValueError(
                f"band_dilations length {len(dilations)} must match band_kernel_sizes length {len(kernel_sizes)}"
            )
        self.dilations = dilations
        self.n_bands = len(kernel_sizes)

        real_convs: List[nn.Conv1d] = []
        imag_convs: List[nn.Conv1d] = []
        pads: List[int] = []

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
        self.mag_eps = 1e-8

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.size()
        x_t = x.transpose(1, 2)

        us: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []

        for pad, conv_r, conv_i in zip(self.band_pads, self.real_convs, self.imag_convs):
            x_padded = F.pad(x_t, (pad, 0))
            u = conv_r(x_padded).mean(dim=1)
            v = conv_i(x_padded).mean(dim=1)
            us.append(u)
            vs.append(v)

        U = torch.stack(us, dim=1)
        V = torch.stack(vs, dim=1)
        mag = torch.sqrt(U * U + V * V + self.mag_eps)
        phi = torch.atan2(V, U + 1e-8)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        return cos_phi, sin_phi, mag


class FeedForwardV13NoRot(nn.Module):
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
            raise ValueError(f"Unsupported hidden_act '{hidden_act}' in pswrecv13vam_wof.")
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


# ---------------------------------------------------------------------------
# 2. Attention: Q,K amplitude modulation + Value Amplitude Modulation (VAM)
# ---------------------------------------------------------------------------
class AMResidualPhaseBiasAttentionV13NoRotVAM(nn.Module):
    """
    Same as AMResidualPhaseBiasAttentionV13NoRot plus Value Amplitude Modulation:
    v = v * scale so the payload (not just the router) is scaled by wave magnitude.
    """

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        n_bands: int,
        phase_bias_init: float,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by n_heads {n_heads}")

        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_heads
        self.n_bands = n_bands

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.band_logits = nn.Parameter(torch.zeros(n_heads, n_bands))
        self.phase_bias = nn.Parameter(torch.full((n_heads,), float(phase_bias_init)))
        self.gamma = nn.Parameter(torch.full((n_heads, n_bands), -3.0))

        with torch.no_grad():
            for h in range(n_heads):
                center = (h + 0.5) * (n_bands / max(n_heads, 1.0))
                for s in range(n_bands):
                    dist = (s + 0.5) - center
                    self.band_logits.data[h, s] = -dist * dist

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        x = x.view(B, L, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos_phi: torch.Tensor,
        sin_phi: torch.Tensor,
        mag: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = hidden_states.size()
        residual = hidden_states

        q = self._shape(self.query(hidden_states))
        k = self._shape(self.key(hidden_states))
        v = self._shape(self.value(hidden_states))

        cos_pl = cos_phi.permute(0, 2, 1)
        sin_pl = sin_phi.permute(0, 2, 1)
        mag_pl = mag.permute(0, 2, 1)

        mag_pl = mag_pl / (mag_pl.mean(dim=-1, keepdim=True) + 1e-8)

        gamma_pos = F.softplus(self.gamma)
        mag_mix = (mag_pl.unsqueeze(1) * gamma_pos.view(1, self.n_heads, 1, self.n_bands)).sum(dim=-1)
        mag_mix = torch.tanh(mag_mix)
        scale = 1.0 + 0.5 * mag_mix.unsqueeze(-1)
        q = q * scale
        k = k * scale
        v = v * scale  # Value Amplitude Modulation: payload energy scales with magnitude

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        band_weights = torch.softmax(self.band_logits, dim=-1)
        band_weights_sqrt = torch.sqrt(band_weights + 1e-8)
        cos_feat = cos_pl.unsqueeze(1) * band_weights_sqrt.view(1, self.n_heads, 1, self.n_bands)
        sin_feat = sin_pl.unsqueeze(1) * band_weights_sqrt.view(1, self.n_heads, 1, self.n_bands)
        phase_align = torch.matmul(cos_feat, cos_feat.transpose(-1, -2)) + torch.matmul(
            sin_feat, sin_feat.transpose(-1, -2)
        )
        phase_align = phase_align / math.sqrt(self.n_bands)
        phase_scale = F.softplus(self.phase_bias).view(1, self.n_heads, 1, 1)
        attn_scores = attn_scores + phase_scale * phase_align

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, L, self.hidden_size)
        hidden_states = self.out_proj(context)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + residual)
        return hidden_states


class PSWBlockV13NoRotVAM(nn.Module):
    """Single layer: VAM attention + FFN."""

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        n_bands: int,
        phase_bias_init: float,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.attn = AMResidualPhaseBiasAttentionV13NoRotVAM(
            n_heads=n_heads,
            hidden_size=hidden_size,
            n_bands=n_bands,
            phase_bias_init=phase_bias_init,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.ffn = FeedForwardV13NoRot(
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
        cos_phi: torch.Tensor,
        sin_phi: torch.Tensor,
        mag: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.attn(hidden_states, attention_mask, cos_phi, sin_phi, mag)
        return self.ffn(hidden_states)


class PSWEncoderV13NoRotVAM(nn.Module):
    """Stack of PSWBlockV13NoRotVAM layers."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        hidden_size: int,
        n_bands: int,
        phase_bias_init: float,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            PSWBlockV13NoRotVAM(
                n_heads=n_heads,
                hidden_size=hidden_size,
                n_bands=n_bands,
                phase_bias_init=phase_bias_init,
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
        cos_phi: torch.Tensor,
        sin_phi: torch.Tensor,
        mag: torch.Tensor,
        output_all_encoded_layers: bool = True,
    ) -> List[torch.Tensor]:
        all_layers: List[torch.Tensor] = []
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, cos_phi, sin_phi, mag)
            if output_all_encoded_layers:
                all_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_layers.append(hidden_states)
        return all_layers


# ---------------------------------------------------------------------------
# WOF model shell
# ---------------------------------------------------------------------------
class PSWRecV13VAMWOFModel(SequentialRecModel):
    """
    PSWRecV13 with Value Amplitude Modulation (Q,K,V scaled by magnitude).
    Same hyperparameters as V13withoutphaserot; use same v13 params (n_heads=4, n_bands=4, etc.).
    """

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
        self.phase_bias_init = getattr(args, "phase_bias_init", -5.0)
        self.phase_aux = getattr(args, "phase_aux", False)
        self.phase_aux_weight = getattr(args, "phase_aux_weight", 0.0)
        self._last_phase_reg: Optional[torch.Tensor] = None

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.phase_filter = LocalPhaseFilterBankV13NoRot(
            hidden_size=self.hidden_size,
            kernel_sizes=band_kernel_sizes,
            dilations=band_dilations,
        )
        self.encoder = PSWEncoderV13NoRotVAM(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            n_bands=self.n_bands,
            phase_bias_init=self.phase_bias_init,
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
        cos_phi, sin_phi, mag = self.phase_filter(sequence_emb)
        cos_phi = cos_phi[:, : self.n_bands, :]
        sin_phi = sin_phi[:, : self.n_bands, :]
        mag = mag[:, : self.n_bands, :]
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
            cos_phi,
            sin_phi,
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
