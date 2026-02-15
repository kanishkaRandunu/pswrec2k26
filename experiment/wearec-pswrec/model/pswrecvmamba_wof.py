#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VMamba: Hyperpersonalized Phase-Modulated Structured State Space Duality (PM-SSD)
Adapted for the WEARec Official Framework (WOF).

Developed upon PSWRecV13withoutphaserot, integrating Mamba-2 / Selective SSM concepts.
Innovations:
1) Local Hyperpersonalized Phase FilterBank: 1D Conv modulator for per-step phase/mag.
2) Phase-Modulated Mamba (SSM) Core: Replaces O(N^2) self-attention with O(L) selective scan.
   - Inject 1: Delta Mag scales the Mamba step size (dt) for short-term transient bursts.
   - Inject 2: Phase features (cos/sin) mapped into SSM output matrix C for temporal alignment.
Pure PyTorch sequential scan (no Triton/CUDA kernels); suitable for L <= 200 (e.g. LastFM).
"""

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from model._abstract_model import SequentialRecModel


# ---------------------------------------------------------------------------
# 1. Hyperpersonalized Time-Aware Phase Filterbank
# ---------------------------------------------------------------------------
class LocalHyperpersonalizedPhaseFilterBank(nn.Module):
    """
    Time-Aware Hyperpersonalization: Uses a 1D Conv modulator to generate
    local, time-specific Delta Mag and Delta Phi adjustments for every step.
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
            raise ValueError("dilations length must match kernel_sizes length")

        self.dilations = dilations
        self.n_bands = len(kernel_sizes)

        real_convs: List[nn.Conv1d] = []
        imag_convs: List[nn.Conv1d] = []
        self.band_pads: List[int] = []

        for k, d in zip(kernel_sizes, dilations):
            pad = (k - 1) * d
            self.band_pads.append(pad)
            real_convs.append(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=k, dilation=d, groups=hidden_size, bias=False)
            )
            imag_convs.append(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=k, dilation=d, groups=hidden_size, bias=False)
            )

        self.real_convs = nn.ModuleList(real_convs)
        self.imag_convs = nn.ModuleList(imag_convs)
        self.mag_eps = 1e-8

        # Local Time-Aware Modulator
        self.modulator = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_size // 2, 2 * self.n_bands, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.size()
        x_t = x.transpose(1, 2)

        # Generate hyperpersonalized modulations per time-step
        mod_params = self.modulator(x_t)  # (B, 2*n_bands, L)
        delta_mag, delta_phi = mod_params.chunk(2, dim=1)

        # Bounded modulation
        delta_mag = torch.tanh(delta_mag)
        delta_phi = torch.tanh(delta_phi) * (math.pi / 4.0)

        us, vs = [], []
        for pad, conv_r, conv_i in zip(self.band_pads, self.real_convs, self.imag_convs):
            x_padded = F.pad(x_t, (pad, 0))
            us.append(conv_r(x_padded).mean(dim=1))
            vs.append(conv_i(x_padded).mean(dim=1))

        U = torch.stack(us, dim=1)
        V = torch.stack(vs, dim=1)

        mag_base = torch.sqrt(U * U + V * V + self.mag_eps)
        phi_base = torch.atan2(V, U + 1e-8)

        # Apply Modulations
        mag_personalized = mag_base * (1.0 + delta_mag)
        phi_personalized = phi_base + delta_phi

        cos_phi = torch.cos(phi_personalized)
        sin_phi = torch.sin(phi_personalized)

        return cos_phi, sin_phi, mag_personalized, delta_mag


# ---------------------------------------------------------------------------
# 2. Phase-Modulated Selective State Space (Mamba) Block
# ---------------------------------------------------------------------------
class PhaseModulatedMambaBlock(nn.Module):
    """
    Pure PyTorch implementation of a Mamba block with Phase and Magnitude injections.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        n_bands: int = 4,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * 2
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Phase Injection Projection
        self.phase_proj = nn.Linear(n_bands * 2, self.d_state, bias=False)

    def forward(self, x: torch.Tensor, cos_phi: torch.Tensor, sin_phi: torch.Tensor, delta_mag: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x

        xz = self.in_proj(x)
        x_mamba, z = xz.chunk(2, dim=-1)

        # Local Convolution
        x_mamba = x_mamba.transpose(1, 2)
        x_mamba = self.conv1d(x_mamba)[:, :, :L]
        x_mamba = x_mamba.transpose(1, 2)
        x_mamba = F.silu(x_mamba)

        # State Space Projections
        x_proj = self.x_proj(x_mamba)
        dt, B_state, C_state = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt))

        # --- HYPERPERSONALIZATION INJECTION 1: Transient Step Modulation ---
        # Averages the burst signal across bands and scales the SSM time-step (dt)
        delta_mag_mean = delta_mag.mean(dim=1).unsqueeze(-1)  # (B, L, 1)
        dt = dt * (1.0 + F.softplus(delta_mag_mean))

        # --- HYPERPERSONALIZATION INJECTION 2: Phase Alignment ---
        # Embeds periodic phase alignment natively into the output projection matrix C
        phase_feats = torch.cat([cos_phi, sin_phi], dim=1).transpose(1, 2)  # (B, L, 2*n_bands)
        C_state = C_state + self.phase_proj(phase_feats)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Selective Scan Operation
        y = self._selective_scan(x_mamba, dt, A, B_state, C_state)

        y = y + x_mamba * self.D
        y = y * F.silu(z)

        out = self.out_proj(y)
        return self.layer_norm(out + residual)

    def _selective_scan(self, u: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch Sequential Scan (efficient for RecSys L <= 200)."""
        B_sz, L, D = u.shape
        N = A.shape[1]

        # Discretize
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A))  # (B, L, D, N)
        dB = torch.einsum('bld,bln->bldn', dt, B)            # (B, L, D, N)

        h = torch.zeros(B_sz, D, N, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
            y = torch.einsum('bdn,bn->bd', h, C[:, t])
            ys.append(y)

        return torch.stack(ys, dim=1)


# ---------------------------------------------------------------------------
# 3. PM-SSD Encoder Stack
# ---------------------------------------------------------------------------
class PM_SSDEncoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_bands: int, layer_norm_eps: float):
        super().__init__()
        self.layers = nn.ModuleList([
            PhaseModulatedMambaBlock(
                d_model=d_model,
                d_state=64,
                n_bands=n_bands,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, cos_phi: torch.Tensor, sin_phi: torch.Tensor, delta_mag: torch.Tensor, output_all_encoded_layers: bool = True) -> List[torch.Tensor]:
        all_layers: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x, cos_phi, sin_phi, delta_mag)
            if output_all_encoded_layers:
                all_layers.append(x)
        if not output_all_encoded_layers:
            all_layers.append(x)
        return all_layers


# ---------------------------------------------------------------------------
# 4. WOF Model Shell (VMamba)
# ---------------------------------------------------------------------------
class HyperpersonalizedPMSSDModel(SequentialRecModel):
    """
    VMamba: Hyperpersonalized Phase-Modulated State Space model for WEARec Framework.
    Replaces Transformer blocks with PM-SSD (Mamba) blocks; causal by construction.
    """

    def __init__(self, args):
        super().__init__(args)
        self.n_layers = args.num_hidden_layers
        self.hidden_size = args.hidden_size
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.layer_norm_eps = 1e-12

        band_kernel_sizes = getattr(args, "band_kernel_sizes", [3, 7, 15, 31])
        band_dilations = getattr(args, "band_dilations", [1, 2, 4, 8])
        self.n_bands = getattr(args, "n_bands", len(band_kernel_sizes))

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # Gamma parameter for base input amplitude scaling
        self.gamma = nn.Parameter(torch.full((self.n_bands,), -3.0))

        self.phase_filter = LocalHyperpersonalizedPhaseFilterBank(
            hidden_size=self.hidden_size,
            kernel_sizes=band_kernel_sizes,
            dilations=band_dilations,
        )

        self.encoder = PM_SSDEncoder(
            n_layers=self.n_layers,
            d_model=self.hidden_size,
            n_bands=self.n_bands,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        sequence_emb = self.add_position_embedding(input_ids)
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        # Extract features and hyperpersonalized adjustments
        cos_phi, sin_phi, mag, delta_mag = self.phase_filter(sequence_emb)

        # Input amplitude modulation (base scaling)
        gamma_pos = F.softplus(self.gamma)
        mag_mix = (mag.transpose(1, 2) * gamma_pos.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        mag_scale = 1.0 + 0.5 * torch.tanh(mag_mix).unsqueeze(-1)
        sequence_emb = sequence_emb * mag_scale

        encoder_outputs = self.encoder(
            sequence_emb,
            cos_phi,
            sin_phi,
            delta_mag,
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
