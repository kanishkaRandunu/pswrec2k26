# -*- coding: utf-8 -*-
# RecBole implementation of the WEARec model architecture. Evaluation follows the unified RecBole protocol, which differs from the original WEARec repository’s custom pipeline.
# Reference: Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation (AAAI 2026).

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender


class LayerNorm(nn.Module):
    """Custom LayerNorm (TF style, eps inside sqrt) as in official WEARec."""

    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class FeedForward(nn.Module):
    """Feed-forward block: Linear -> act -> Linear -> dropout -> residual + LayerNorm."""

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self._get_hidden_act(hidden_act)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def _get_hidden_act(self, act):
        if act == "gelu":
            return self._gelu
        if act == "relu":
            return F.relu
        if act == "tanh":
            return torch.tanh
        if act == "sigmoid":
            return torch.sigmoid
        if act == "swish":
            return lambda x: x * torch.sigmoid(x)
        raise ValueError(f"unknown act: {act}")

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class WEARecLayer(nn.Module):
    """Single WEARec layer: FFT branch + wavelet branch, combined by alpha gate."""

    def __init__(
        self,
        max_seq_length,
        hidden_size,
        num_heads,
        alpha,
        hidden_dropout_prob,
        layer_norm_eps,
    ):
        super(WEARecLayer, self).__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.seq_len = max_seq_length
        self.freq_bins = max_seq_length // 2 + 1
        self.alpha = alpha
        self.combine_mode = "gate"

        self.complex_weight = nn.Parameter(
            torch.randn(1, num_heads, max_seq_length // 2, self.head_dim, dtype=torch.float32) * 0.02
        )
        self.base_filter = nn.Parameter(torch.ones(num_heads, self.freq_bins, 1))
        self.base_bias = nn.Parameter(torch.full((num_heads, self.freq_bins, 1), -0.1))
        self.adaptive_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_heads * self.freq_bins * 2),
        )
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def wavelet_transform(self, x_heads):
        B, H, N, D = x_heads.shape
        N_even = N if (N % 2) == 0 else (N - 1)
        x_heads = x_heads[:, :, :N_even, :]
        x_even = x_heads[:, :, 0::2, :]
        x_odd = x_heads[:, :, 1::2, :]
        approx = 0.5 * (x_even + x_odd)
        detail = 0.5 * (x_even - x_odd)
        detail = detail * self.complex_weight
        x_even_recon = approx + detail
        x_odd_recon = approx - detail
        out = torch.zeros_like(x_heads)
        out[:, :, 0::2, :] = x_even_recon
        out[:, :, 1::2, :] = x_odd_recon
        if N_even < N:
            pad = torch.zeros((B, H, 1, D), device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=2)
        return out

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x_heads = input_tensor.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # FFT branch
        F_fft = torch.fft.rfft(x_heads, dim=2, norm="ortho")
        context = input_tensor.mean(dim=1)
        adapt_params = self.adaptive_mlp(context)
        adapt_params = adapt_params.view(batch, self.num_heads, self.freq_bins, 2)
        adaptive_scale = adapt_params[..., 0:1]
        adaptive_bias = adapt_params[..., 1:2]
        effective_filter = self.base_filter * (1 + adaptive_scale)
        effective_bias = self.base_bias + adaptive_bias
        F_fft_mod = F_fft * effective_filter + effective_bias
        x_fft = torch.fft.irfft(F_fft_mod, dim=2, n=self.seq_len, norm="ortho")

        # Wavelet branch
        x_wavelet = self.wavelet_transform(x_heads)

        # Combine
        x_combined = (1.0 - self.alpha) * x_wavelet + self.alpha * x_fft
        x_out = x_combined.permute(0, 2, 1, 3).reshape(batch, seq_len, hidden)
        hidden_states = self.out_dropout(x_out)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class WEARecBlock(nn.Module):
    """WEARecLayer + FeedForward."""

    def __init__(self, max_seq_length, hidden_size, num_heads, alpha, inner_size,
                 hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(WEARecBlock, self).__init__()
        self.layer = WEARecLayer(
            max_seq_length, hidden_size, num_heads, alpha, hidden_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
        )

    def forward(self, hidden_states):
        hidden_states = self.layer(hidden_states)
        return self.feed_forward(hidden_states)


class WEARecEncoder(nn.Module):
    """Stack of n_layers WEARecBlocks."""

    def __init__(self, n_layers, max_seq_length, hidden_size, num_heads, alpha, inner_size,
                 hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(WEARecEncoder, self).__init__()
        block = WEARecBlock(
            max_seq_length, hidden_size, num_heads, alpha, inner_size,
            hidden_dropout_prob, hidden_act, layer_norm_eps
        )
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class WEARec(SequentialRecommender):
    """WEARec: Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation."""

    def __init__(self, config, dataset):
        super(WEARec, self).__init__(config, dataset)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"] if "inner_size" in config else (4 * self.hidden_size)
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.alpha = config["alpha"]

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.item_encoder = WEARecEncoder(
            n_layers=self.n_layers,
            max_seq_length=self.max_seq_length,
            hidden_size=self.hidden_size,
            num_heads=self.n_heads,
            alpha=self.alpha,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        if self.loss_type != "CE":
            raise NotImplementedError("WEARec only supports loss_type 'CE'")
        self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        encoded_layers = self.item_encoder(input_emb, output_all_encoded_layers=True)
        output = encoded_layers[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
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
        return scores
