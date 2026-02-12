#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pswrecv4h: V4 + Harmonic-mean phase gating.

Same as pswrecv4 (geometric-mean gated PSWRec) but the magnitude gating uses
the harmonic mean instead of the geometric mean:

    Bias_ij = beta * (2 * A_i * A_j / (A_i + A_j)) * cos(Phi_i - Phi_j)

The harmonic mean is more aggressive at suppressing outliers: when one of
A_i or A_j is small, the gate is even smaller than sqrt(A_i * A_j).
"""

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class LocalPhaseFilterBankV4H(nn.Module):
    r"""LocalPhaseFilterBankV4H

    A lightweight, learnable, multi-band filterbank over the sequence axis.
    Same as V4 (no Gabor); quadrature pair (U_s, V_s) and phase/mag output.
    """

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
                f"band_dilations length {len(dilations)} "
                f"must match band_kernel_sizes length {len(kernel_sizes)}"
            )
        self.dilations = dilations

        self.n_bands = len(kernel_sizes)

        real_convs = []
        imag_convs = []
        pads = []
        for k, d in zip(kernel_sizes, dilations):
            pad = (k - 1) * d
            pads.append(pad)
            real_convs.append(
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=k,
                    dilation=d,
                    padding=0,
                    groups=hidden_size,
                    bias=False,
                )
            )
            imag_convs.append(
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
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
        self.mag_tau = 1e-3

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.size()
        x_t = x.transpose(1, 2)  # [B, D, L]

        us = []
        vs = []

        for pad, conv_r, conv_i in zip(
            self.band_pads, self.real_convs, self.imag_convs
        ):
            x_padded = F.pad(x_t, (pad, 0))
            u = conv_r(x_padded)
            v = conv_i(x_padded)
            u_band = u.mean(dim=1)
            v_band = v.mean(dim=1)
            us.append(u_band)
            vs.append(v_band)

        U = torch.stack(us, dim=1)
        V = torch.stack(vs, dim=1)

        mag = torch.sqrt(U * U + V * V + self.mag_eps)
        phi = torch.atan2(V, U)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        gate = (mag > self.mag_tau).float()
        cos_phi = cos_phi * gate
        sin_phi = sin_phi * gate

        return cos_phi, sin_phi, mag


class PhaseSyncAttentionV4H(nn.Module):
    r"""Multi-head self-attention with harmonic-mean phase gating.

    Same as V4 but the magnitude gate is the harmonic mean:
        gate_ij = 2 * A_i * A_j / (A_i + A_j + eps)
    """

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        n_bands: int,
        phase_bias_scale: float,
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
        self.n_bands = n_bands

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.band_logits = nn.Parameter(torch.zeros(n_heads, n_bands))
        self.phase_bias_scale = nn.Parameter(
            torch.full((n_heads,), float(phase_bias_scale))
        )

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

        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        cos_pl = cos_phi.permute(0, 2, 1)
        sin_pl = sin_phi.permute(0, 2, 1)
        mag_pl = mag.permute(0, 2, 1)

        band_weights = torch.softmax(self.band_logits, dim=-1)
        band_weights_sqrt = torch.sqrt(band_weights + 1e-8)

        cos_feat = cos_pl.unsqueeze(1) * band_weights_sqrt.view(
            1, self.n_heads, 1, self.n_bands
        )
        sin_feat = sin_pl.unsqueeze(1) * band_weights_sqrt.view(
            1, self.n_heads, 1, self.n_bands
        )

        phase_scores = torch.matmul(
            cos_feat, cos_feat.transpose(-1, -2)
        ) + torch.matmul(sin_feat, sin_feat.transpose(-1, -2))

        # ── Harmonic-mean magnitude gating (V4H) ──
        mag_weighted = mag_pl.unsqueeze(1) * band_weights.view(
            1, self.n_heads, 1, self.n_bands
        )
        mag_head = mag_weighted.sum(dim=-1)  # [B, H, L]

        mag_i = mag_head.unsqueeze(-1)   # [B, H, L, 1]
        mag_j = mag_head.unsqueeze(-2)   # [B, H, 1, L]
        harm_mag = 2.0 * mag_i * mag_j / (mag_i + mag_j + 1e-8)  # [B, H, L, L]

        phase_scores = phase_scores * harm_mag

        phase_scale = torch.exp(self.phase_bias_scale).view(
            1, self.n_heads, 1, 1
        )
        phase_scores = phase_scores * phase_scale

        attn_scores = attn_scores + phase_scores
        attn_scores = attn_scores + attention_mask

        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(*new_context_shape)

        hidden_states = self.out_proj(context)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + residual)
        return hidden_states


class FeedForwardV4H(nn.Module):
    """Standard position-wise feed-forward layer with residual + LayerNorm."""

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
            "gelu": self.gelu,
            "relu": nn.functional.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        if hidden_act not in act_map:
            raise ValueError(f"Unsupported hidden_act '{hidden_act}' in pswrecv4h.")
        self.act_fn = act_map[hidden_act]

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    @staticmethod
    def gelu(x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    @staticmethod
    def swish(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dense_1(x)
        x = self.act_fn(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


class PSWBlockV4H(nn.Module):
    """Single transformer-style layer: PhaseSyncAttentionV4H + FFN."""

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        n_bands: int,
        phase_bias_scale: float,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.attn = PhaseSyncAttentionV4H(
            n_heads=n_heads,
            hidden_size=hidden_size,
            n_bands=n_bands,
            phase_bias_scale=phase_bias_scale,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.ffn = FeedForwardV4H(
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
        hidden_states = self.attn(
            hidden_states, attention_mask, cos_phi, sin_phi, mag
        )
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class PSWEncoderV4H(nn.Module):
    """Stack of PSWBlockV4H layers."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        hidden_size: int,
        n_bands: int,
        phase_bias_scale: float,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [
                PSWBlockV4H(
                    n_heads=n_heads,
                    hidden_size=hidden_size,
                    n_bands=n_bands,
                    phase_bias_scale=phase_bias_scale,
                    inner_size=inner_size,
                    hidden_dropout_prob=hidden_dropout_prob,
                    attn_dropout_prob=attn_dropout_prob,
                    hidden_act=hidden_act,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos_phi: torch.Tensor,
        sin_phi: torch.Tensor,
        mag: torch.Tensor,
        output_all_encoded_layers: bool = True,
    ) -> List[torch.Tensor]:
        all_layers = []
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask, cos_phi, sin_phi, mag
            )
            if output_all_encoded_layers:
                all_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_layers.append(hidden_states)
        return all_layers


class pswrecv4h(SequentialRecommender):
    r"""pswrecv4h: V4 + harmonic-mean phase gating.

    Same as pswrecv4 but gate = 2*A_i*A_j/(A_i+A_j) for stronger outlier suppression.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        if "band_kernel_sizes" in config:
            band_kernel_sizes = list(config["band_kernel_sizes"])
        else:
            band_kernel_sizes = [3, 7, 15, 31]
        if "band_dilations" in config:
            band_dilations = list(config["band_dilations"])
        else:
            band_dilations = [1 for _ in band_kernel_sizes]
        self.n_bands = len(band_kernel_sizes)

        self.phase_bias_scale = (
            float(config["phase_bias_scale"])
            if "phase_bias_scale" in config
            else 0.1
        )
        self.phase_aux = bool(config["phase_aux"]) if "phase_aux" in config else False
        self.phase_aux_weight = (
            float(config["phase_aux_weight"])
            if "phase_aux_weight" in config
            else 0.0
        )
        self._last_phase_reg: Optional[torch.Tensor] = None

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )

        self.phase_filter = LocalPhaseFilterBankV4H(
            hidden_size=self.hidden_size,
            kernel_sizes=band_kernel_sizes,
            dilations=band_dilations,
        )
        self.encoder = PSWEncoderV4H(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            n_bands=self.n_bands,
            phase_bias_scale=self.phase_bias_scale,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq: torch.Tensor) -> torch.Tensor:
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.layer_norm(input_emb)
        input_emb = self.dropout(input_emb)

        cos_phi, sin_phi, mag = self.phase_filter(input_emb)

        if self.phase_aux:
            cos_diff = cos_phi[:, :, 1:] - cos_phi[:, :, :-1]
            sin_diff = sin_phi[:, :, 1:] - sin_phi[:, :, :-1]
            phase_reg = (cos_diff.pow(2) + sin_diff.pow(2)).mean()
            self._last_phase_reg = phase_reg
        else:
            self._last_phase_reg = None

        extended_attention_mask = self.get_attention_mask(item_seq)

        encoder_outputs = self.encoder(
            input_emb,
            extended_attention_mask,
            cos_phi,
            sin_phi,
            mag,
            output_all_encoded_layers=True,
        )
        output = encoder_outputs[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        if self.phase_aux and (self._last_phase_reg is not None):
            loss = loss + self.phase_aux_weight * self._last_phase_reg

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        if scores.size(1) > 0:
            scores[:, 0] = -1e9
        return scores
