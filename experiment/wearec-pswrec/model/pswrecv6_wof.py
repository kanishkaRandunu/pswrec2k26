#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PSWRecV6 adapted for the WEARec Official Framework (WOF).

V6 = V5 + **Dynamic Phase Evolution (DPE)**.

Key change: the ``LocalPhaseFilterBank`` is called inside the encoder loop
at every layer (with *shared* filterbank weights), so each layer's attention
is guided by phase computed from its own input -- not from the stale initial
embeddings.  This ensures phase-attention alignment remains accurate as
hidden states evolve through the transformer stack.

Everything else (architecture components, WOF shell) is identical to V5.
"""

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# WEARec base class -- imported at runtime via sys.path set in main.py
from model._abstract_model import SequentialRecModel


# ---------------------------------------------------------------------------
# Architecture components (same as V5 -- LocalPhaseFilterBank, Attention, FFN)
# ---------------------------------------------------------------------------

class LocalPhaseFilterBankV6(nn.Module):
    r"""Multi-band quadrature filterbank over the sequence axis.  (Same as V5.)"""

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

        real_convs: List[nn.Conv1d] = []
        imag_convs: List[nn.Conv1d] = []
        pads: List[int] = []
        for k, d in zip(kernel_sizes, dilations):
            pad = (k - 1) * d
            pads.append(pad)
            real_convs.append(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=k,
                          dilation=d, padding=0, groups=hidden_size, bias=False)
            )
            imag_convs.append(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=k,
                          dilation=d, padding=0, groups=hidden_size, bias=False)
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

        us: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []

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


class PhaseSyncAttentionV6(nn.Module):
    r"""Multi-head self-attention with harmonic-mean phase bias + post-softmax phase gating.
    (Same as V5.)"""

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        n_bands: int,
        phase_bias_scale: float,
        phase_gate_scale: float,
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
        self.phase_gate_scale = nn.Parameter(
            torch.tensor(float(phase_gate_scale), dtype=torch.float32)
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

        mag_weighted = mag_pl.unsqueeze(1) * band_weights.view(
            1, self.n_heads, 1, self.n_bands
        )
        mag_head = mag_weighted.sum(dim=-1)

        mag_i = mag_head.unsqueeze(-1)
        mag_j = mag_head.unsqueeze(-2)
        harm_mag = 2.0 * mag_i * mag_j / (mag_i + mag_j + 1e-8)

        phase_scores = phase_scores * harm_mag

        phase_scale = torch.exp(self.phase_bias_scale).view(
            1, self.n_heads, 1, 1
        )
        phase_scores = phase_scores * phase_scale

        attn_scores = attn_scores + phase_scores
        attn_scores = attn_scores + attention_mask

        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        attn_probs = self.attn_dropout(attn_probs)

        gate = torch.sigmoid(self.phase_gate_scale * phase_scores)
        attn_probs_gated = attn_probs * gate
        attn_probs_gated = attn_probs_gated / (
            attn_probs_gated.sum(dim=-1, keepdim=True) + 1e-9
        )

        context = torch.matmul(attn_probs_gated, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(*new_context_shape)

        hidden_states = self.out_proj(context)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + residual)
        return hidden_states


class FeedForwardV6(nn.Module):
    """Position-wise feed-forward with residual + LayerNorm.  (Same as V5.)"""

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
            raise ValueError(f"Unsupported hidden_act '{hidden_act}' in pswrecv6_wof.")
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


class PSWBlockV6(nn.Module):
    """Single transformer-style layer: PhaseSyncAttentionV6 + FFN.  (Same as V5.)"""

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        n_bands: int,
        phase_bias_scale: float,
        phase_gate_scale: float,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.attn = PhaseSyncAttentionV6(
            n_heads=n_heads,
            hidden_size=hidden_size,
            n_bands=n_bands,
            phase_bias_scale=phase_bias_scale,
            phase_gate_scale=phase_gate_scale,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.ffn = FeedForwardV6(
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


# ---------------------------------------------------------------------------
# V6 CHANGE: PSWEncoderV6 -- phase_filter is called INSIDE the loop
# ---------------------------------------------------------------------------

class PSWEncoderV6(nn.Module):
    """Stack of PSWBlockV6 layers with Dynamic Phase Evolution.

    Unlike V5 where phase is computed once from input embeddings, V6
    recomputes phase at every layer from the current hidden states using
    a *shared* filterbank.  The filterbank is passed in (not owned) so
    the outer model can also use it for auxiliary regularisation.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        hidden_size: int,
        n_bands: int,
        phase_bias_scale: float,
        phase_gate_scale: float,
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
                PSWBlockV6(
                    n_heads=n_heads,
                    hidden_size=hidden_size,
                    n_bands=n_bands,
                    phase_bias_scale=phase_bias_scale,
                    phase_gate_scale=phase_gate_scale,
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
        phase_filter: "LocalPhaseFilterBankV6",
        output_all_encoded_layers: bool = True,
    ) -> Tuple[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass with per-layer phase recomputation (DPE).

        Args:
            hidden_states: [B, L, D]
            attention_mask: [B, 1, L, L]
            phase_filter: shared filterbank module
            output_all_encoded_layers: whether to collect every layer's output

        Returns:
            (all_layers, (last_cos_phi, last_sin_phi, last_mag))
            The last phase outputs are returned for auxiliary regularisation.
        """
        all_layers: List[torch.Tensor] = []
        last_cos_phi = last_sin_phi = last_mag = None

        for layer in self.layers:
            # --- DPE: recompute phase from current hidden states ---
            cos_phi, sin_phi, mag = phase_filter(hidden_states)
            last_cos_phi, last_sin_phi, last_mag = cos_phi, sin_phi, mag

            hidden_states = layer(
                hidden_states, attention_mask, cos_phi, sin_phi, mag
            )
            if output_all_encoded_layers:
                all_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_layers.append(hidden_states)

        return all_layers, (last_cos_phi, last_sin_phi, last_mag)


# ---------------------------------------------------------------------------
# WOF model shell
# ---------------------------------------------------------------------------

class PSWRecV6WOFModel(SequentialRecModel):
    r"""PSWRecV6 on WEARec Official Framework.

    V6 = V5 + Dynamic Phase Evolution (DPE).
    Phase is recomputed from each layer's hidden states using a shared filterbank.
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
        self.phase_bias_scale = getattr(args, "phase_bias_scale", 0.1)
        self.phase_gate_scale = getattr(args, "phase_gate_scale", 1.0)
        self.phase_aux = getattr(args, "phase_aux", False)
        self.phase_aux_weight = getattr(args, "phase_aux_weight", 0.0)
        self._last_phase_reg: Optional[torch.Tensor] = None

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # Shared filterbank -- used inside encoder loop (DPE)
        self.phase_filter = LocalPhaseFilterBankV6(
            hidden_size=self.hidden_size,
            kernel_sizes=band_kernel_sizes,
            dilations=band_dilations,
        )

        self.encoder = PSWEncoderV6(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            n_bands=self.n_bands,
            phase_bias_scale=self.phase_bias_scale,
            phase_gate_scale=self.phase_gate_scale,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        sequence_emb = self.add_position_embedding(input_ids)

        extended_attention_mask = self.get_attention_mask(input_ids)

        # V6: phase_filter is passed to encoder; recomputed at each layer
        encoder_outputs, (last_cos, last_sin, last_mag) = self.encoder(
            sequence_emb,
            extended_attention_mask,
            phase_filter=self.phase_filter,
            output_all_encoded_layers=True,
        )

        # Phase-smoothness auxiliary regularisation (on last layer's phase)
        if self.phase_aux and last_cos is not None:
            cos_diff = last_cos[:, :, 1:] - last_cos[:, :, :-1]
            sin_diff = last_sin[:, :, 1:] - last_sin[:, :, :-1]
            phase_reg = (cos_diff.pow(2) + sin_diff.pow(2)).mean()
            self._last_phase_reg = phase_reg
        else:
            self._last_phase_reg = None

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
