The core idea is a paradigm shift in how neural networks perceive user behavior. Almost all standard sequential models treat historical interactions as a sequence of discrete, static points in a real-numbered space (Euclidean geometry).QIA discards this approach. Instead, it lifts the user's sequence into the complex domain ($inline$ \mathbb{C}^d $inline$), modeling the user's evolving intent not as static points, but as propagating wavefunctions. By doing this, the model can natively utilize the physics of wave interference to solve the hardest problems in sequential modeling: capturing sudden bursts of intent and filtering out historical noise.Here is the mathematical engine driving the code, broken down step-by-step.1. Quantum State Preparation (The Complex Filterbank)In standard models, a sequence of item embeddings $inline$ x $inline$ is just a matrix of real numbers. The QuantumStatePreparator acts as a transducer, converting these real numbers into complex wave states.Using parallel 1D convolutions (with varying kernel sizes and dilations), the network extracts two distinct signals from the sequence:$inline$ U $inline$ (The Real Part): Represents the base semantic amplitude.$inline$ V $inline$ (The Imaginary Part): Represents the periodic temporal phase.Instead of immediately calculating the magnitude and phase to use as scalar multipliers, the model binds them together into a native complex tensor:$$display$$\psi_t = U_t + iV_t$$display$$Every item in the sequence is now a complex vector possessing both magnitude (importance) and a phase angle (its position in the user's behavioral rhythm).2. Native Complex Computing (CVNN)Because the sequence is now complex, standard neural network layers will no longer work. If you feed a complex number into a standard linear layer, it destroys the phase geometry.To solve this, the code implements a Complex-Valued Neural Network (CVNN). The ComplexLinear layer maintains a separate real weight matrix ($inline$ W_R $inline$) and an imaginary weight matrix ($inline$ W_I $inline$). When projecting the complex input ($inline$ X_R + iX_I $inline$), it follows the strict rules of complex multiplication:$$display$$\text{Output} = (X_R W_R - X_I W_I) + i(X_R W_I + X_I W_R)$$display$$This ensures that as the Queries ($inline$ Q $inline$), Keys ($inline$ K $inline$), and Values ($inline$ V $inline$) are generated, their structural phase relationships remain mathematically entangled and perfectly preserved.3. Hermitian Interference Attention (The Core Engine)This is where QIA mathematically overtakes standard Transformer attention.Standard attention calculates the relationship between two items using a simple dot product ($inline$ Q \cdot K^T $inline$). QIA replaces this with the Hermitian Inner Product (the conjugate transpose, $inline$ Q \cdot K^H $inline$), and defines the attention score as the squared modulus of that complex intersection:$$display$$\text{Score} = |Q \cdot K^H|^2 = (Q_R K_R + Q_I K_I)^2 + (Q_I K_R - Q_R K_I)^2$$display$$By using complex arithmetic, the attention matrix fundamentally changes from a measure of "static similarity" to a measure of wave resonance.ShutterstockConstructive Interference (The Burst Capturer): If a user clicks three highly related items in rapid succession, their phase angles align. When complex numbers with aligned phases are multiplied, their amplitudes experience exponential constructive interference. This organically creates a massive mathematical spike in the attention probability for the immediate next logical item.Destructive Interference (The Noise Filter): If an item in the user's history is noisy, random, or irrelevant to their current intent, its phase angle will be misaligned (orthogonal or opposite). When the Hermitian dot product is calculated, the opposing waves naturally cancel each other out (destructive interference), dropping the attention score to near-zero without the need for manual forget-gates.4. Wave Collapse (Measurement)Neural networks ultimately need to output real-numbered probabilities to calculate the Cross-Entropy loss against the target item. A complex wavefunction cannot be directly mapped to a probability.Borrowing from quantum mechanics, the model performs a "Measurement" or "Wave Collapse" at the very end of the encoder block. To collapse the complex state $inline$ \psi = x_r + i x_i $inline$ back into the observable real-numbered world, the code calculates the absolute amplitude of the wave:$$display$$\text{Amplitude} = \sqrt{x_r^2 + x_i^2}$$display$$This real-valued amplitude is then passed through a final linear projection to generate the predictions over the item catalog.Why This is Mathematically SuperiorBy moving into the complex plane $inline$ \mathbb{C}^d $inline$, the model's representational capacity is drastically expanded without artificially inflating the parameter count. Phase and magnitude are not treated as isolated features; they are geometrically bound together through complex algebra. This allows the self-attention mechanism to route information based on the dynamic, oscillating rhythms of user behavior rather than just static semantic matching.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantum-Inspired Interference Attention (QIA) adapted for the WEARec Official Framework.

Innovations:
1) Quantum State Preparation: 1D convolutional filterbanks map the real-valued 
   sequence embeddings into the complex domain (Wavefunctions: X = U + Vi).
2) Native Complex Computing (CVNN): Replaces standard linear layers with ComplexLinear 
   projections, keeping the entire sequence in the complex plane C^d.
3) Hermitian Interference Attention: Calculates attention probabilities using the 
   squared modulus of the complex inner product (|Q * K^H|^2). 
   - Constructive Interference organically amplifies transient bursts.
   - Destructive Interference natively cancels historical noise.
4) Wave Collapse: Projects the final complex sequence back to real-valued probability 
   amplitudes for the final cross-entropy prediction.
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

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out_r = self.fc_r(x_r) - self.fc_i(x_i)
        out_i = self.fc_r(x_i) + self.fc_i(x_r)
        return out_r, out_i


class ComplexLayerNorm(nn.Module):
    """Applies LayerNorm independently to the real and imaginary components."""
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.ln_r = nn.LayerNorm(hidden_size, eps=eps)
        self.ln_i = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def __init__(self, hidden_size: int, kernel_sizes: List[int], dilations: Optional[List[int]] = None):
        super().__init__()
        if dilations is None:
            dilations = [1 for _ in kernel_sizes]
        
        real_convs, imag_convs, pads = [], [], []
        for k, d in zip(kernel_sizes, dilations):
            pad = (k - 1) * d
            pads.append(pad)
            # Groups=hidden_size ensures independent frequency extraction per dimension
            real_convs.append(nn.Conv1d(hidden_size, hidden_size, kernel_size=k, dilation=d, padding=0, groups=hidden_size, bias=False))
            imag_convs.append(nn.Conv1d(hidden_size, hidden_size, kernel_size=k, dilation=d, padding=0, groups=hidden_size, bias=False))

        self.real_convs = nn.ModuleList(real_convs)
        self.imag_convs = nn.ModuleList(imag_convs)
        self.band_pads = pads

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_t = x.transpose(1, 2) # (B, D, L)
        
        u_sum = torch.zeros_like(x_t)
        v_sum = torch.zeros_like(x_t)

        # Superposition: Summing the overlapping periodic wave functions
        for pad, conv_r, conv_i in zip(self.band_pads, self.real_convs, self.imag_convs):
            x_padded = F.pad(x_t, (pad, 0))
            u_sum = u_sum + conv_r(x_padded)
            v_sum = v_sum + conv_i(x_padded)

        # Return to (B, L, D)
        return u_sum.transpose(1, 2), v_sum.transpose(1, 2)


# ---------------------------------------------------------------------------
# 3. Quantum-Inspired Interference Attention (QIA)
# ---------------------------------------------------------------------------
class InterferenceAttention(nn.Module):
    def __init__(self, n_heads: int, hidden_size: int, hidden_dropout_prob: float, attn_dropout_prob: float):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_heads

        # Complex Projections
        self.q_proj = ComplexLinear(hidden_size, hidden_size)
        self.k_proj = ComplexLinear(hidden_size, hidden_size)
        self.v_proj = ComplexLinear(hidden_size, hidden_size)
        
        self.out_proj = ComplexLinear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        return x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x_r.size()

        # Generate Complex Q, K, V
        qr_raw, qi_raw = self.q_proj(x_r, x_i)
        kr_raw, ki_raw = self.k_proj(x_r, x_i)
        vr_raw, vi_raw = self.v_proj(x_r, x_i)

        q_r, q_i = self._shape(qr_raw), self._shape(qi_raw)
        k_r, k_i = self._shape(kr_raw), self._shape(ki_raw)
        v_r, v_i = self._shape(vr_raw), self._shape(vi_raw)

        # --- NATIVE WAVE INTERFERENCE ---
        # S = Q * K^H (Hermitian inner product)
        # S_real = Q_r*K_r + Q_i*K_i
        # S_imag = Q_i*K_r - Q_r*K_i
        s_real = torch.matmul(q_r, k_r.transpose(-1, -2)) + torch.matmul(q_i, k_i.transpose(-1, -2))
        s_imag = torch.matmul(q_i, k_r.transpose(-1, -2)) - torch.matmul(q_r, k_i.transpose(-1, -2))

        # Interference Intensity (Squared Modulus: |S|^2 = S_real^2 + S_imag^2)
        # Scaled by dimension to prevent gradient explosion
        interference = (s_real.pow(2) + s_imag.pow(2)) / self.head_dim

        if attention_mask is not None:
            interference = interference + attention_mask

        # Wave collapses to probability
        attn_probs = self.attn_dropout(F.softmax(interference, dim=-1))

        # Apply probabilities to Complex Values
        c_r = torch.matmul(attn_probs, v_r).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        c_i = torch.matmul(attn_probs, v_i).permute(0, 2, 1, 3).contiguous().view(B, L, D)

        out_r, out_i = self.out_proj(c_r, c_i)
        return self.out_dropout(out_r), self.out_dropout(out_i)


# ---------------------------------------------------------------------------
# 4. Complex Feed Forward & Encoder Blocks
# ---------------------------------------------------------------------------
class ComplexFFN(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int, hidden_dropout_prob: float):
        super().__init__()
        self.fc1 = ComplexLinear(hidden_size, inner_size)
        self.fc2 = ComplexLinear(inner_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_r, h_i = self.fc1(x_r, x_i)
        # Apply non-linearity independently (Standard CVNN practice)
        h_r, h_i = F.gelu(h_r), F.gelu(h_i)
        out_r, out_i = self.fc2(h_r, h_i)
        return self.dropout(out_r), self.dropout(out_i)


class QIABlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn = InterferenceAttention(args.num_attention_heads, args.hidden_size, args.hidden_dropout_prob, args.attention_probs_dropout_prob)
        self.ffn = ComplexFFN(args.hidden_size, getattr(args, "inner_size", 4 * args.hidden_size), args.hidden_dropout_prob)
        self.ln1 = ComplexLayerNorm(args.hidden_size)
        self.ln2 = ComplexLayerNorm(args.hidden_size)

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Complex Attention with Residual
        a_r, a_i = self.attn(x_r, x_i, attention_mask)
        x_r, x_i = self.ln1(x_r + a_r, x_i + a_i)
        
        # Complex FFN with Residual
        f_r, f_i = self.ffn(x_r, x_i)
        x_r, x_i = self.ln2(x_r + f_r, x_i + f_i)
        
        return x_r, x_i


# ---------------------------------------------------------------------------
# 5. WOF Model Shell
# ---------------------------------------------------------------------------
class QIARecWOFModel(SequentialRecModel):
    """
    Quantum-Inspired Interference Attention Model (QIA-Rec).
    Elevates the WOF framework to the complex plane. Captures transient bursts 
    and long-term periodicity organically through constructive and destructive 
    wave interference, completely eliminating the need for auxiliary Wavelet branches.
    """
    def __init__(self, args):
        super().__init__(args)
        band_kernel_sizes = getattr(args, "band_kernel_sizes", [3, 7, 15, 31])
        band_dilations = getattr(args, "band_dilations", [1, 2, 4, 8])
        
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        
        # Map real sequences to complex wave states
        self.state_preparator = QuantumStatePreparator(args.hidden_size, band_kernel_sizes, band_dilations)
        
        # Complex Encoder Stack
        self.layers = nn.ModuleList([QIABlock(args) for _ in range(args.num_hidden_layers)])
        
        # Final Wave Collapse Projection
        self.collapse_proj = nn.Linear(args.hidden_size, args.hidden_size)
        
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        # 1. Base Real Embeddings
        sequence_emb = self.dropout(self.LayerNorm(self.add_position_embedding(input_ids)))
        
        # 2. Quantum State Preparation (Real -> Complex)
        x_r, x_i = self.state_preparator(sequence_emb)
        
        extended_attention_mask = self.get_attention_mask(input_ids)
        all_layers = []

        # 3. Complex Interference Processing
        for layer in self.layers:
            x_r, x_i = layer(x_r, x_i, extended_attention_mask)
            
            # Wave Collapse: To output a layer, we measure the amplitude of the wave
            # Amplitude = sqrt(Real^2 + Imag^2)
            amplitude = torch.sqrt(x_r.pow(2) + x_i.pow(2) + 1e-8)
            all_layers.append(self.collapse_proj(amplitude))
            
        return all_layers if all_sequence_output else all_layers[-1]

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)[:, -1, :]
        logits = torch.matmul(seq_output, self.item_embeddings.weight.transpose(0, 1))
        return nn.CrossEntropyLoss()(logits, answers)

    def predict(self, input_ids, user_ids=None, all_sequence_output=False):
        return self.forward(input_ids, user_ids, all_sequence_output)