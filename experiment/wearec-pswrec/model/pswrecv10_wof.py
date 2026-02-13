#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PSWRecV10 adapted for the WEARec Official Framework (WOF).

This version implements Wave Superposition Attention (WSA), which moves phase
rotation from attention logits to the Value matrix, enabling destructive interference
where out-of-phase signals can subtract noise rather than just being ignored.

Architecture differences from V5:
- PhaseSyncAttentionWSA replaces PhaseSyncAttentionV5
- Phase rotation applied directly to Values using O(N) trigonometric computation
- Pure semantic attention (no phase bias in Q·K), enabling true destructive interference
"""

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# WEARec base class -- imported at runtime via sys.path set in main.py
from model._abstract_model import SequentialRecModel


# ---------------------------------------------------------------------------
# Architecture components
# ---------------------------------------------------------------------------

class LocalPhaseFilterBankV5(nn.Module):
    r"""Multi-band quadrature filterbank over the sequence axis.
    
    Same as V5 - unchanged.
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


class PhaseSyncAttentionWSA(nn.Module):
    r"""Wave Superposition Attention (WSA).
    
    Moves phase rotation from attention logits to Value matrix, enabling
    destructive interference where out-of-phase signals subtract noise.
    
    Key innovation: Uses O(N) trigonometric identity to compute phase rotation
    on Values without O(N^2) pairwise phase difference matrix.
    """

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        n_bands: int,
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

        # Projection to map filterbank bands to attention heads
        # This aligns phase/magnitude dimensions [B, L, n_bands] -> [B, H, L, 1]
        self.phase_proj = nn.Linear(n_bands, n_heads)

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

        # 1. Standard Linear Projections
        q = self._shape(self.query(hidden_states))  # [B, H, L, d]
        k = self._shape(self.key(hidden_states))    # [B, H, L, d]
        v = self._shape(self.value(hidden_states))  # [B, H, L, d]

        # 2. Align Phase/Magnitude dimensions to Attention Heads
        # Phase filter returns [B, n_bands, L], need to permute to [B, L, n_bands] for phase_proj
        # Then phase_proj outputs [B, L, n_heads], permute to [B, n_heads, L], unsqueeze to [B, H, L, 1]
        cos_phi_pl = cos_phi.permute(0, 2, 1)  # [B, n_bands, L] -> [B, L, n_bands]
        sin_phi_pl = sin_phi.permute(0, 2, 1)  # [B, n_bands, L] -> [B, L, n_bands]
        mag_pl = mag.permute(0, 2, 1)          # [B, n_bands, L] -> [B, L, n_bands]
        
        p_cos = self.phase_proj(cos_phi_pl).permute(0, 2, 1).unsqueeze(-1)  # [B, H, L, 1]
        p_sin = self.phase_proj(sin_phi_pl).permute(0, 2, 1).unsqueeze(-1)  # [B, H, L, 1]
        p_mag = self.phase_proj(mag_pl).permute(0, 2, 1).unsqueeze(-1)      # [B, H, L, 1]

        # --- THE WAVE SUPERPOSITION NOVELTY STARTS HERE ---
        
        # Step A: Encode the Values into Complex Wave Space
        # Instead of gating the logits, we physically scale and rotate the Values.
        V_real = v * p_mag * p_cos  # [B, H, L, d]
        V_imag = v * p_mag * p_sin  # [B, H, L, d]

        # Step B: Pure Semantic Attention (No phase bias here!)
        # Let the Transformer learn pure semantic item-to-item similarity
        attention_scores = torch.matmul(q, k.transpose(-1, -2))  # [B, H, L, L]
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        # Step C: Attend to both the Real and Imaginary waves independently
        context_real = torch.matmul(attention_probs, V_real)  # [B, H, L, d]
        context_imag = torch.matmul(attention_probs, V_imag)  # [B, H, L, d]

        # Step D: Decode and Superimpose (The Destructive Interference)
        # O_i = cos(phi_i) * sum(Real) + sin(phi_i) * sum(Imag)
        # If an attended item was out-of-phase, this math naturally subtracts its semantic payload.
        context_layer = (context_real * p_cos) + (context_imag * p_sin)  # [B, H, L, d]

        # --- THE WAVE SUPERPOSITION NOVELTY ENDS HERE ---

        # 4. Standard Output formatting
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [B, L, H, d]
        new_context_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_shape)  # [B, L, D]

        hidden_states = self.out_proj(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + residual)
        return hidden_states


class FeedForwardV5(nn.Module):
    """Position-wise feed-forward with residual + LayerNorm.
    
    Same as V5 - unchanged.
    """

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
            raise ValueError(f"Unsupported hidden_act '{hidden_act}' in pswrecv10_wof.")
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


class PSWBlockV10(nn.Module):
    """Single transformer-style layer: PhaseSyncAttentionWSA + FFN."""

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
    ):
        super().__init__()
        self.attn = PhaseSyncAttentionWSA(
            n_heads=n_heads,
            hidden_size=hidden_size,
            n_bands=n_bands,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.ffn = FeedForwardV5(
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


class PSWEncoderV10(nn.Module):
    """Stack of PSWBlockV10 layers."""

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
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [
                PSWBlockV10(
                    n_heads=n_heads,
                    hidden_size=hidden_size,
                    n_bands=n_bands,
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
        all_layers: List[torch.Tensor] = []
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask, cos_phi, sin_phi, mag
            )
            if output_all_encoded_layers:
                all_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_layers.append(hidden_states)
        return all_layers


# ---------------------------------------------------------------------------
# WOF model shell -- adapts PSWRecV10 to WEARec's SequentialRecModel interface
# ---------------------------------------------------------------------------

class PSWRecV10WOFModel(SequentialRecModel):
    r"""PSWRecV10 on WEARec Official Framework.

    Implements Wave Superposition Attention (WSA) that moves phase rotation
    from attention logits to Value matrix, enabling destructive interference.
    """

    def __init__(self, args):
        super().__init__(args)

        # ---- map WEARec args to PSWRecV10 params ----
        self.n_layers = args.num_hidden_layers
        self.n_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.inner_size = getattr(args, "inner_size", 4 * args.hidden_size)
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.attn_dropout_prob = args.attention_probs_dropout_prob
        self.hidden_act = args.hidden_act
        self.layer_norm_eps = 1e-12
        self.initializer_range = args.initializer_range

        # PSWRecV10-specific hyperparameters (passed via extra CLI args)
        # Note: phase_bias_scale and phase_gate_scale are not used in V10 (WSA)
        # but kept for CLI compatibility
        band_kernel_sizes = getattr(args, "band_kernel_sizes", [3, 7, 15, 31])
        band_dilations = getattr(args, "band_dilations", [1, 2, 4, 8])
        self.n_bands = getattr(args, "n_bands", len(band_kernel_sizes))
        self.phase_aux = getattr(args, "phase_aux", False)
        self.phase_aux_weight = getattr(args, "phase_aux_weight", 0.0)
        self._last_phase_reg: Optional[torch.Tensor] = None

        # ---- layers ----
        # item_embeddings & position_embeddings are created by the base class.
        # We add our own LayerNorm and dropout (matching RecBole pswrecv5).
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.phase_filter = LocalPhaseFilterBankV5(
            hidden_size=self.hidden_size,
            kernel_sizes=band_kernel_sizes,
            dilations=band_dilations,
        )

        self.encoder = PSWEncoderV10(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            n_bands=self.n_bands,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.apply(self.init_weights)

    # ------------------------------------------------------------------
    # Forward -- returns FULL sequence [B, L, D] (WEARec convention)
    # ------------------------------------------------------------------

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        # Embed items + positions (reuses base-class embeddings)
        sequence_emb = self.add_position_embedding(input_ids)

        # Phase filterbank
        cos_phi, sin_phi, mag = self.phase_filter(sequence_emb)

        # Phase-smoothness auxiliary regularisation
        if self.phase_aux:
            cos_diff = cos_phi[:, :, 1:] - cos_phi[:, :, :-1]
            sin_diff = sin_phi[:, :, 1:] - sin_phi[:, :, :-1]
            phase_reg = (cos_diff.pow(2) + sin_diff.pow(2)).mean()
            self._last_phase_reg = phase_reg
        else:
            self._last_phase_reg = None

        # Causal attention mask
        extended_attention_mask = self.get_attention_mask(input_ids)

        # Encoder
        encoder_outputs = self.encoder(
            sequence_emb,
            extended_attention_mask,
            cos_phi,
            sin_phi,
            mag,
            output_all_encoded_layers=True,
        )

        if all_sequence_output:
            return encoder_outputs          # list of [B, L, D]
        return encoder_outputs[-1]          # [B, L, D]

    # ------------------------------------------------------------------
    # Loss -- WEARec trainer calls model.calculate_loss(input_ids,
    #          answers, neg_answers, same_target, user_ids)
    # ------------------------------------------------------------------

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]                   # last position
        item_emb = self.item_embeddings.weight               # [V, D]
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        if self.phase_aux and self._last_phase_reg is not None:
            loss = loss + self.phase_aux_weight * self._last_phase_reg

        return loss

    # ------------------------------------------------------------------
    # Predict -- WEARec trainer calls model.predict(input_ids, user_ids)
    # and then takes [:, -1, :] on the result.
    # ------------------------------------------------------------------

    def predict(self, input_ids, user_ids=None, all_sequence_output=False):
        return self.forward(input_ids, user_ids, all_sequence_output)
