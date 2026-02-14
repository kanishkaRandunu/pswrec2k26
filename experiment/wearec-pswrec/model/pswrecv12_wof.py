#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PSWRecV12 adapted for the WEARec Official Framework (WOF).

Integrates B-RoPE from experiment/B-RoPE-Code:
  - Identity mapping for phase: no nn.Linear on wavelet_phases; (B, L, 4) maps to num_heads=4.
  - Adaptive Magnitude Gating: Dynamically shuts off B-RoPE for sparse noise (e.g., Beauty)
    while engaging it for dense rhythmic data (e.g., Last.FM).
  - Only Q and K are rotated by apply_behavioral_rope; V stays pure (semantic integrity).
  - Sync-gate: cos(Δφ) < sync_threshold -> -1e9 before softmax (hard filter on behavioral noise).

For Beauty (short/sparse) use sync_threshold=-0.7 or -0.9; for LastFM/MovieLens use 0.0.
"""

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# WEARec base class -- imported at runtime via sys.path set in main.py
from model._abstract_model import SequentialRecModel


# ---------------------------------------------------------------------------
# Architecture components (Filterbank with Adaptive Magnitude Gating)
# ---------------------------------------------------------------------------


class LocalPhaseFilterBankV5(nn.Module):
    r"""Multi-band quadrature filterbank with Adaptive Magnitude Gating."""

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
        self.base_mag_tau = 1e-3

        # Adaptive Magnitude Gate: [Mean, Variance] per band -> dynamic threshold per band
        self.tau_proj = nn.Sequential(
            nn.Linear(self.n_bands * 2, self.n_bands * 2),
            nn.ReLU(),
            nn.Linear(self.n_bands * 2, self.n_bands),
            nn.Softplus(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        # Compute phi safely (U + 1e-8 avoids atan2(0,0) and division-by-zero in backward)
        phi = torch.atan2(V, U + 1e-8)

        # Adaptive Magnitude Gating: sequence-level stats -> dynamic tau per user/band
        mag_mean = mag.mean(dim=-1)  # (B, n_bands)
        mag_var = mag.var(dim=-1, unbiased=False)  # (B, n_bands)
        mag_stats = torch.cat([mag_mean, mag_var], dim=-1)  # (B, n_bands * 2)
        dynamic_tau = self.tau_proj(mag_stats).unsqueeze(-1)  # (B, n_bands, 1)
        tau = dynamic_tau + self.base_mag_tau
        gate = (mag > tau).float()
        phi = phi * gate
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        return phi, cos_phi, sin_phi, mag


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last dimension by 90 degrees in complex plane (for RoPE)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_behavioral_rope(q: torch.Tensor, k: torch.Tensor, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the Behavioral Rotary Position Embedding.
    Only Q and K are rotated by wavelet phase; V is never passed in (semantic integrity).
    """
    # Duplicate the phase for the two halves of the embedding dimension
    phase = torch.cat([phase, phase], dim=-1)
    sin_phase = torch.sin(phase)
    cos_phase = torch.cos(phase)
    q_rotated = (q * cos_phase) + (rotate_half(q) * sin_phase)
    k_rotated = (k * cos_phase) + (rotate_half(k) * sin_phase)
    return q_rotated, k_rotated


class BehavioralRotaryAttentionV12(nn.Module):
    """B-RoPE with identity phase mapping and sync-gate. Only Q and K rotated; V untouched."""

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        n_bands: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
        sync_threshold: float = -0.7,
    ):
        super().__init__()
        assert n_heads == n_bands, (
            f"V12 requires n_heads == n_bands for identity phase mapping (got n_heads={n_heads}, n_bands={n_bands})"
        )
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by n_heads {n_heads}"
            )

        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim {self.head_dim} must be even for rotary splitting."
            )
        self.n_bands = n_bands
        self.sync_threshold = sync_threshold

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # No phase_proj: identity mapping from (B, L, n_bands) to heads

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        x = x.view(B, L, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [B, H, L, d]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = hidden_states.size()
        residual = hidden_states

        # phi: [B, L, n_bands] -> wavelet_phases (identity)
        q = self._shape(self.query(hidden_states))   # [B, H, L, d]
        k = self._shape(self.key(hidden_states))
        v = self._shape(self.value(hidden_states))

        # Identity mapping: (B, L, n_bands) -> (B, H, L, 1) then expand to head_dim//2 for RoPE
        phase_head = phi.transpose(1, 2).unsqueeze(-1)   # [B, n_bands, L, 1] -> [B, H, L, 1]
        phase_rope = phase_head.expand(-1, -1, -1, self.head_dim // 2)   # [B, H, L, d/2]

        # Apply Behavioral RoPE (only Q and K)
        q_rot, k_rot = apply_behavioral_rope(q, k, phase_rope)

        # Geometric attention scores
        attn_scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 5. The "Sync-Gate" Resonance Mask
        phi_i = phase_head
        phi_j = phase_head.transpose(-2, -1)   # [B, H, 1, L]
        delta_phi = phi_i - phi_j
        cos_delta_phi = torch.cos(delta_phi)
        sync_mask = cos_delta_phi < self.sync_threshold

        # FIX 1 (Option A): Protect the diagonal. An item is always in phase with itself (Δφ=0, cos=1).
        eye = torch.eye(L, dtype=torch.bool, device=hidden_states.device).view(1, 1, L, L)
        sync_mask = sync_mask.masked_fill(eye, False)
        attn_scores = attn_scores.masked_fill(sync_mask, -1e9)

        # 6. Causal + padding mask from base
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # FIX 2 (Option B): NaN safety net. If a row is entirely masked, force diagonal to 0.0 so softmax never 0/0.
        row_max = attn_scores.max(dim=-1, keepdim=True)[0]
        all_masked = row_max <= -1e8
        attn_scores = torch.where(all_masked & eye, torch.zeros_like(attn_scores), attn_scores)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(B, L, self.hidden_size)

        hidden_states = self.out_proj(context)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + residual)
        return hidden_states


class FeedForwardV11(nn.Module):
    """Position-wise feed-forward with residual + LayerNorm (same as V5/V11)."""

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
            raise ValueError(f"Unsupported hidden_act '{hidden_act}' in pswrecv12_wof.")
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


class PSWBlockV12(nn.Module):
    """Single transformer-style layer: BehavioralRotaryAttentionV12 + FFN."""

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        n_bands: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
        sync_threshold: float,
    ):
        super().__init__()
        self.attn = BehavioralRotaryAttentionV12(
            n_heads=n_heads,
            hidden_size=hidden_size,
            n_bands=n_bands,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            sync_threshold=sync_threshold,
        )
        self.ffn = FeedForwardV11(
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
    ) -> torch.Tensor:
        hidden_states = self.attn(hidden_states, attention_mask, phi)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class PSWEncoderV12(nn.Module):
    """Stack of PSWBlockV12 layers."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        hidden_size: int,
        n_bands: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
        sync_threshold: float,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [
                PSWBlockV12(
                    n_heads=n_heads,
                    hidden_size=hidden_size,
                    n_bands=n_bands,
                    inner_size=inner_size,
                    hidden_dropout_prob=hidden_dropout_prob,
                    attn_dropout_prob=attn_dropout_prob,
                    hidden_act=hidden_act,
                    layer_norm_eps=layer_norm_eps,
                    sync_threshold=sync_threshold,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        phi: torch.Tensor,
        output_all_encoded_layers: bool = True,
    ) -> List[torch.Tensor]:
        all_layers: List[torch.Tensor] = []
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, phi)
            if output_all_encoded_layers:
                all_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_layers.append(hidden_states)
        return all_layers


# ---------------------------------------------------------------------------
# WOF model shell -- adapts PSWRecV12 to WEARec's SequentialRecModel interface
# ---------------------------------------------------------------------------


class PSWRecV12WOFModel(SequentialRecModel):
    r"""PSWRecV12 on WEARec Official Framework.

    B-RoPE with identity phase mapping, Adaptive Magnitude Gating, and sync-gate.
    For Beauty use sync_threshold=-0.7 or -0.9; for LastFM/MovieLens use 0.0.
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
        self.phase_aux = getattr(args, "phase_aux", False)
        self.phase_aux_weight = getattr(args, "phase_aux_weight", 0.0)
        self._last_phase_reg: Optional[torch.Tensor] = None

        # Sync-gate: Beauty (sparse) -> -0.7/-0.9; LastFM/MovieLens (dense) -> 0.0
        self.sync_threshold = getattr(args, "sync_threshold", -0.9)

        if self.n_heads != self.n_bands:
            raise ValueError(
                f"PSWRecV12 requires num_attention_heads == n_bands (identity mapping). "
                f"Got num_attention_heads={self.n_heads}, n_bands={self.n_bands}."
            )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.phase_filter = LocalPhaseFilterBankV5(
            hidden_size=self.hidden_size,
            kernel_sizes=band_kernel_sizes,
            dilations=band_dilations,
        )

        self.encoder = PSWEncoderV12(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            n_bands=self.n_bands,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            sync_threshold=self.sync_threshold,
        )

        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        sequence_emb = self.add_position_embedding(input_ids)
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        phi, cos_phi, sin_phi, mag = self.phase_filter(sequence_emb)

        if self.phase_aux:
            cos_diff = cos_phi[:, :, 1:] - cos_phi[:, :, :-1]
            sin_diff = sin_phi[:, :, 1:] - sin_phi[:, :, :-1]
            phase_reg = (cos_diff.pow(2) + sin_diff.pow(2)).mean()
            self._last_phase_reg = phase_reg
        else:
            self._last_phase_reg = None

        phi_pl = phi.permute(0, 2, 1).contiguous()

        extended_attention_mask = self.get_attention_mask(input_ids)

        encoder_outputs = self.encoder(
            sequence_emb,
            extended_attention_mask,
            phi_pl,
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
