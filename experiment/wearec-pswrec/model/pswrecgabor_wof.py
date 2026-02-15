#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GaborRec: Learnable Gabor filterbank in place of standard conv filterbank.

Builds on the best V13withoutphaserot model (pswrecv13withoutphaserot_wof_best.py):
only the phase/filter module is replaced with LearnableGaborFilterBank. The rest
(AMResidualPhaseBiasAttentionV13NoRot, FeedForwardV13NoRot, PSWEncoderV13NoRot)
is reused from the best model—no code in the best file is overwritten.

Learnable Gabor filters: per-band per-channel center frequency (omega) and Gaussian
bandwidth (sigma) so each embedding dimension can specialize. Kernels are generated
on the fly (real: exp(-x²/(2σ²))*cos(ωx), imag: sin). Reduces spectral leakage vs
unconstrained conv while matching the per-channel capacity of the original filterbank.
"""

import math
import os
import importlib.util
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from model._abstract_model import SequentialRecModel


# ---------------------------------------------------------------------------
# 1. Learnable Gabor filterbank (drop-in for LocalPhaseFilterBankV13NoRot)
# ---------------------------------------------------------------------------
class LearnableGaborFilterBank(nn.Module):
    """
    Replaces standard 1D convs with Gabor filters. Learns center frequency (omega)
    and bandwidth (sigma) per band per channel so each embedding dimension can
    specialize. Kernels generated on the forward pass. Same interface as
    LocalPhaseFilterBankV13NoRot: forward(x) -> cos_phi, sin_phi, mag (B, n_bands, L).
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
        self.dilations = dilations
        self.n_bands = len(kernel_sizes)
        self.mag_eps = 1e-8
        self.band_pads = [(k - 1) * d for k, d in zip(kernel_sizes, dilations)]

        # Per-band per-channel: (n_bands, hidden_size) so each channel has its own Gabor per band
        init_freqs = torch.linspace(0.1, math.pi / 2, self.n_bands)
        self.omega = nn.Parameter(init_freqs.unsqueeze(1).expand(self.n_bands, hidden_size).clone())
        init_sigmas = torch.tensor([k / 4.0 for k in kernel_sizes], dtype=torch.float32)
        self.sigma = nn.Parameter(init_sigmas.unsqueeze(1).expand(self.n_bands, hidden_size).clone())

    def _generate_gabor_kernels(
        self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        real_kernels = []
        imag_kernels = []
        for i, k in enumerate(self.kernel_sizes):
            limit = (k - 1) / 2
            x = torch.linspace(
                -limit, limit, k, device=self.omega.device, dtype=self.omega.dtype
            )
            # (hidden_size,) per band
            w = torch.clamp(self.omega[i], min=1e-3, max=math.pi)
            s = F.softplus(self.sigma[i]) + 1e-3
            # x: (k,); w, s: (hidden_size,) -> broadcast to (hidden_size, k)
            x_b = x.unsqueeze(0)
            gaussian = torch.exp(-(x_b ** 2) / (2 * s.unsqueeze(1) ** 2))
            real_wave = torch.cos(w.unsqueeze(1) * x_b)
            imag_wave = torch.sin(w.unsqueeze(1) * x_b)
            gabor_real = gaussian * real_wave
            gabor_imag = gaussian * imag_wave
            # Normalize per channel (each row)
            gabor_real = gabor_real / (torch.sum(torch.abs(gabor_real), dim=1, keepdim=True) + 1e-8)
            gabor_imag = gabor_imag / (torch.sum(torch.abs(gabor_imag), dim=1, keepdim=True) + 1e-8)
            real_kernels.append(gabor_real.unsqueeze(1))
            imag_kernels.append(gabor_imag.unsqueeze(1))
        return real_kernels, imag_kernels

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.size()
        x_t = x.transpose(1, 2)
        real_kernels, imag_kernels = self._generate_gabor_kernels()
        us: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []
        for i, (pad, d) in enumerate(zip(self.band_pads, self.dilations)):
            x_padded = F.pad(x_t, (pad, 0))
            u = F.conv1d(x_padded, real_kernels[i], dilation=d, groups=D)
            v = F.conv1d(x_padded, imag_kernels[i], dilation=d, groups=D)
            us.append(u.mean(dim=1))
            vs.append(v.mean(dim=1))
        U = torch.stack(us, dim=1)
        V = torch.stack(vs, dim=1)
        mag = torch.sqrt(U * U + V * V + self.mag_eps)
        phi = torch.atan2(V, U + 1e-8)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        return cos_phi, sin_phi, mag


# ---------------------------------------------------------------------------
# 2. Reuse encoder from best model (no overwrite of best file)
# ---------------------------------------------------------------------------
def _load_best_encoder():
    _dir = os.path.dirname(os.path.abspath(__file__))
    _path = os.path.join(_dir, "pswrecv13withoutphaserot_wof_best.py")
    _spec = importlib.util.spec_from_file_location(
        "pswrecv13withoutphaserot_wof_best", _path
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.PSWEncoderV13NoRot


PSWEncoderV13NoRot = _load_best_encoder()


# ---------------------------------------------------------------------------
# 3. WOF model shell (same as best, but phase_filter = LearnableGaborFilterBank)
# ---------------------------------------------------------------------------
class GaborRecWOFModel(SequentialRecModel):
    """
    GaborRec: V13withoutphaserot attention stack with LearnableGaborFilterBank
    instead of LocalPhaseFilterBankV13NoRot. All other components (encoder,
    attention, FFN) are identical to the best model.
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
        self.phase_aux = getattr(args, "phase_aux", False)
        self.phase_aux_weight = getattr(args, "phase_aux_weight", 0.0)
        self._last_phase_reg: Optional[torch.Tensor] = None

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.phase_filter = LearnableGaborFilterBank(
            hidden_size=self.hidden_size,
            kernel_sizes=band_kernel_sizes,
            dilations=band_dilations,
        )
        self.encoder = PSWEncoderV13NoRot(
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
