#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Residual Dynamic PSWRec (RD-PSWRec / V1111) adapted for the WEARec Official Framework.

Innovations:
1) Residual Hyperpersonalization: Nudges the stable V13 global priors (gamma and
   band_logits) from a context MLP; zero-init on the MLP output so training starts
   identical to V13 (avoids catastrophic forgetting).
2) Transient Value Modulation (V-AM): Positive temporal derivative of magnitude
   (bursts) scales Value vectors only, amplifying payload at intent shifts without
   corrupting Q/K attention probabilities.
Static filterbank (LocalPhaseFilterBankV13NoRot) unchanged; no phase_aux.
"""

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from model._abstract_model import SequentialRecModel


# ---------------------------------------------------------------------------
# 1. Filterbank: ungated cos, sin, magnitude (identical to V13)
# ---------------------------------------------------------------------------
class LocalPhaseFilterBankV13NoRot(nn.Module):
    """Static phase filterbank: extracts cos, sin, mag per band (no modulator)."""

    def __init__(self, hidden_size: int, kernel_sizes: List[int], dilations: Optional[List[int]] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes
        if dilations is None:
            dilations = [1 for _ in kernel_sizes]
        if len(dilations) != len(kernel_sizes):
            raise ValueError("dilations length must match kernel_sizes length")
        self.dilations = dilations
        self.n_bands = len(kernel_sizes)

        real_convs: List[nn.Conv1d] = []
        imag_convs: List[nn.Conv1d] = []
        pads: List[int] = []
        for k, d in zip(kernel_sizes, dilations):
            pad = (k - 1) * d
            pads.append(pad)
            real_convs.append(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=k, dilation=d, padding=0, groups=hidden_size, bias=False)
            )
            imag_convs.append(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=k, dilation=d, padding=0, groups=hidden_size, bias=False)
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
            us.append(conv_r(x_padded).mean(dim=1))
            vs.append(conv_i(x_padded).mean(dim=1))

        U = torch.stack(us, dim=1)
        V = torch.stack(vs, dim=1)
        mag = torch.sqrt(U * U + V * V + self.mag_eps)
        phi = torch.atan2(V, U + 1e-8)
        return torch.cos(phi), torch.sin(phi), mag


# ---------------------------------------------------------------------------
# 2. Residual Dynamic Attention Block
# ---------------------------------------------------------------------------
class ResidualDynamicAMPhaseAttention(nn.Module):
    """
    Residual hyperpersonalization (base V13 priors + context nudges) and
    Transient Value Modulation (V-AM): burst scales V only, not Q/K.
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

        # Base V13 priors
        self.base_band_logits = nn.Parameter(torch.zeros(n_heads, n_bands))
        self.base_phase_bias = nn.Parameter(torch.full((n_heads,), float(phase_bias_init)))
        self.base_gamma = nn.Parameter(torch.full((n_heads, n_bands), -3.0))

        with torch.no_grad():
            for h in range(n_heads):
                center = (h + 0.5) * (n_bands / max(n_heads, 1.0))
                for s in range(n_bands):
                    dist = (s + 0.5) - center
                    self.base_band_logits.data[h, s] = -dist * dist

        # Hyperpersonalization nudge MLP (output zero-initialized so we start as V13)
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, n_heads * n_bands * 2),
        )
        nn.init.zeros_(self.context_mlp[-1].weight)
        nn.init.zeros_(self.context_mlp[-1].bias)

        self.burst_weight = nn.Parameter(torch.full((n_heads,), -5.0))

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        return x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

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

        # 1. Personalization nudges (zero-init so Epoch 0 = V13)
        context = hidden_states.mean(dim=1)  # (B, D)
        deltas = self.context_mlp(context)  # (B, 2 * heads * bands)
        delta_gamma, delta_logits = deltas.chunk(2, dim=-1)
        delta_gamma = delta_gamma.view(B, self.n_heads, self.n_bands)
        delta_logits = delta_logits.view(B, self.n_heads, self.n_bands)

        # 2. Residual integration (base + nudge)
        user_gamma = self.base_gamma.unsqueeze(0) + delta_gamma
        user_band_logits = self.base_band_logits.unsqueeze(0) + delta_logits

        # 3. Q/K amplitude modulation (no transient; keeps attention probabilities stable)
        gamma_pos = F.softplus(user_gamma).unsqueeze(2)  # (B, n_heads, 1, n_bands)
        mag_pl = mag.permute(0, 2, 1)  # (B, n_bands, L)
        mag_pl = mag_pl / (mag_pl.mean(dim=-1, keepdim=True) + 1e-8)
        mag_mix = (mag_pl.unsqueeze(1) * gamma_pos).sum(dim=-1)  # (B, n_heads, L)
        scale = 1.0 + 0.5 * torch.tanh(mag_mix).unsqueeze(-1)
        q = q * scale
        k = k * scale

        # 4. Transient Value Modulation (V-AM): burst scales V only
        # mag_pl is (B, L, n_bands); keep that layout so pad/diff are along L (sequence), not n_bands
        # #region agent log
        try:
            import json
            _log = open("/home/572/kd6504/TRACT/.cursor/debug.log", "a")
            _log.write(json.dumps({"hypothesisId": "H1", "location": "pswrecv1111_wof.py:V-AM", "message": "mag and mag_pl shapes", "data": {"mag_shape": list(mag.shape), "mag_pl_shape": list(mag_pl.shape)}, "timestamp": __import__("time").time()}) + "\n")
            _log.close()
        except Exception: pass
        # #endregion
        mag_pl_norm = mag_pl  # (B, L, n_bands); do not permute to (B, n_bands, L) or burst_mix becomes (B,1,n_bands)
        mag_norm = mag_pl_norm / (mag_pl_norm.mean(dim=1, keepdim=True) + 1e-8)
        mag_padded = F.pad(mag_norm, (0, 0, 1, 0))[:, :-1, :]  # (B, L, n_bands), shift right in L
        mag_burst = F.relu(mag_norm - mag_padded)
        burst_mix = mag_burst.sum(dim=-1).unsqueeze(1)  # (B, 1, L)
        # #region agent log
        try:
            _log = open("/home/572/kd6504/TRACT/.cursor/debug.log", "a")
            _log.write(json.dumps({"hypothesisId": "H2", "location": "pswrecv1111_wof.py:V-AM", "message": "mag_pl_norm burst_mix v v_scale shapes", "data": {"mag_pl_norm_shape": list(mag_pl_norm.shape), "mag_burst_shape": list(mag_burst.shape), "burst_mix_shape": list(burst_mix.shape), "v_shape": list(v.shape)}, "timestamp": __import__("time").time()}) + "\n")
            _log.close()
        except Exception: pass
        # #endregion
        burst_pos = F.softplus(self.burst_weight).view(1, self.n_heads, 1)  # (1, H, 1)
        v_scale = (1.0 + burst_pos * burst_mix).unsqueeze(-1)  # (B, H, L, 1) to match v (B, H, L, head_dim)
        # #region agent log
        try:
            _log = open("/home/572/kd6504/TRACT/.cursor/debug.log", "a")
            _log.write(json.dumps({"hypothesisId": "H3", "location": "pswrecv1111_wof.py:V-AM", "message": "v_scale shape before multiply", "data": {"v_scale_shape": list(v_scale.shape)}, "timestamp": __import__("time").time()}) + "\n")
            _log.close()
        except Exception: pass
        # #endregion
        v = v * v_scale

        # 5. Dynamic phase alignment
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        band_weights = torch.softmax(user_band_logits, dim=-1).unsqueeze(2)  # (B, n_heads, 1, n_bands)
        band_weights_sqrt = torch.sqrt(band_weights + 1e-8)
        cos_pl = cos_phi.permute(0, 2, 1)  # (B, L, n_bands)
        sin_pl = sin_phi.permute(0, 2, 1)
        cos_feat = cos_pl.unsqueeze(1) * band_weights_sqrt
        sin_feat = sin_pl.unsqueeze(1) * band_weights_sqrt
        phase_align = torch.matmul(cos_feat, cos_feat.transpose(-1, -2)) + torch.matmul(
            sin_feat, sin_feat.transpose(-1, -2)
        )
        phase_align = phase_align / math.sqrt(self.n_bands)
        phase_scale = F.softplus(self.base_phase_bias).view(1, self.n_heads, 1, 1)
        attn_scores = attn_scores + phase_scale * phase_align

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = self.attn_dropout(F.softmax(attn_scores, dim=-1))
        context_out = torch.matmul(attn_probs, v).permute(0, 2, 1, 3).contiguous().view(B, L, self.hidden_size)
        return self.layer_norm(self.out_dropout(self.out_proj(context_out)) + residual)


# ---------------------------------------------------------------------------
# 3. Feed forward and encoder blocks
# ---------------------------------------------------------------------------
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
            raise ValueError(f"Unsupported hidden_act '{hidden_act}' in pswrecv1111_wof.")
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


class ResidualDynamicPSWBlock(nn.Module):
    """Single layer: Residual dynamic AM phase attention + FFN."""

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
        self.attn = ResidualDynamicAMPhaseAttention(
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


class ResidualDynamicPSWEncoder(nn.Module):
    """Stack of ResidualDynamicPSWBlock layers."""

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
            ResidualDynamicPSWBlock(
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
# 4. WOF model shell (V1111: Residual Dynamic, no phase_aux)
# ---------------------------------------------------------------------------
class ResidualDynamicPSWRecWOFModel(SequentialRecModel):
    """
    RD-PSWRec: residual hyperpersonalization (base V13 + context nudges) and
    V-AM (transient burst scales Values only). Starts training as V13.
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
        band_kernel_sizes = getattr(args, "band_kernel_sizes", [3, 7, 15, 31])
        band_dilations = getattr(args, "band_dilations", [1, 2, 4, 8])
        self.n_bands = getattr(args, "n_bands", len(band_kernel_sizes))
        self.phase_bias_init = getattr(args, "phase_bias_init", -5.0)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.phase_filter = LocalPhaseFilterBankV13NoRot(
            hidden_size=self.hidden_size,
            kernel_sizes=band_kernel_sizes,
            dilations=band_dilations,
        )
        self.encoder = ResidualDynamicPSWEncoder(
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
        return loss

    def predict(self, input_ids, user_ids=None, all_sequence_output=False):
        return self.forward(input_ids, user_ids, all_sequence_output)


# Alias for main.py registration (model_type pswrecv1111)
HyperpersonalizedPSWRecWOFModel = ResidualDynamicPSWRecWOFModel
