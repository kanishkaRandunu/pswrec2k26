# -*- coding: utf-8 -*-
"""
TRM4Rec: Tiny Recursive Model for Sequential Recommendation

Vanilla TRM alignment: same building blocks as TinyRecursiveModels (RMSNorm, SwiGLU,
RoPE, their attention), gradient-saving trick (T-1 cycles no_grad, last cycle with grad).
Minimal RecSys changes: item sequence input, last-position readout, causal+padding mask,
item scoring head and CE loss.

Reference: TRM4Rec_Experiment.md, TinyRecursiveModels, TRM paper (Less is More: Recursive Reasoning with Tiny Networks)
"""

import copy
import math
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.sequential_recommender.trm_layers import (
    RotaryEmbedding,
    TRMBlock,
)


class TRM4Rec(SequentialRecommender):
    r"""
    TRM4Rec: Tiny Recursive Model for Sequential Recommendation (vanilla TRM alignment).

    Uses TRM blocks (RMSNorm, SwiGLU, RoPE, TRM attention), no learned position embedding.
    Two latent states (z_reason, z_answer) refined for trm_steps; gradient-saving trick:
    T-1 cycles under no_grad, last cycle with gradients (training only).
    Readout at last valid position (item_seq_len - 1); item scoring head and CE loss.
    """

    def __init__(self, config, dataset):
        super(TRM4Rec, self).__init__(config, dataset)

        self.trm_steps = config["trm_steps"]
        self.trm_blocks_per_step = config["trm_blocks_per_step"]
        self.check_item_id_range = (
            config["check_item_id_range"]
            if "check_item_id_range" in config
            else False
        )

        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.expansion = config["expansion"]
        self.rope_theta = (
            config["rope_theta"] if "rope_theta" in config else 10000.0
        )
        self.rms_norm_eps = (
            config["rms_norm_eps"] if "rms_norm_eps" in config else 1e-5
        )
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        self.embed_scale = math.sqrt(self.hidden_size)
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )

        head_dim = self.hidden_size // self.n_heads
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=self.max_seq_length,
            base=self.rope_theta,
        )

        trm_block = TRMBlock(
            hidden_size=self.hidden_size,
            head_dim=head_dim,
            num_heads=self.n_heads,
            expansion=self.expansion,
            rms_norm_eps=self.rms_norm_eps,
        )
        self.trm_blocks = nn.ModuleList(
            [copy.deepcopy(trm_block) for _ in range(self.trm_blocks_per_step)]
        )

        self.z_answer_init = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        self.z_reason_init = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        self._id_range_logged = False

        if self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(
                "TRM4Rec Vanilla uses CE only. Make sure 'loss_type' is 'CE'."
            )

        self.apply(self._init_weights)
        self._init_latent_states()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _init_latent_states(self):
        nn.init.normal_(self.z_answer_init, mean=0.0, std=0.02)
        nn.init.normal_(self.z_reason_init, mean=0.0, std=0.02)

    def _one_cycle(self, z_reason, z_answer, input_emb, cos_sin, attn_mask):
        for block in self.trm_blocks:
            z_reason = block(z_reason + z_answer + input_emb, cos_sin, attn_mask)
            z_answer = block(z_answer + z_reason, cos_sin, attn_mask)
        return z_reason, z_answer

    def forward(self, item_seq, item_seq_len):
        B, L = item_seq.shape
        device = item_seq.device
        if self.check_item_id_range:
            assert item_seq.max().item() < self.n_items, (
                "item_seq ids must be in [0, n_items-1]"
            )

        input_emb = self.embed_scale * self.item_embedding(item_seq)

        extended_attention_mask = self.get_attention_mask(item_seq)
        cos_sin = self.rotary_emb()

        z_answer = self.z_answer_init.expand(B, L, -1)
        z_reason = self.z_reason_init.expand(B, L, -1)

        if self.training and self.trm_steps > 1:
            with torch.no_grad():
                for _ in range(self.trm_steps - 1):
                    z_reason, z_answer = self._one_cycle(
                        z_reason, z_answer, input_emb, cos_sin, extended_attention_mask
                    )
            z_reason = z_reason.detach()
            z_answer = z_answer.detach()
            z_reason, z_answer = self._one_cycle(
                z_reason, z_answer, input_emb, cos_sin, extended_attention_mask
            )
        else:
            for _ in range(self.trm_steps):
                z_reason, z_answer = self._one_cycle(
                    z_reason, z_answer, input_emb, cos_sin, extended_attention_mask
                )

        last_pos = torch.clamp(item_seq_len - 1, min=0, max=L - 1)
        seq_output = self.gather_indexes(z_answer, last_pos)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.check_item_id_range:
            assert pos_items.max().item() < self.n_items, (
                "pos_items must be in [0, n_items-1]"
            )
            if not self._id_range_logged:
                self._id_range_logged = True
                print(
                    "[TRM4Rec] id range (once): "
                    f"n_items={self.n_items}, item_seq.max()={item_seq.max().item()}, "
                    f"pos_items.max()={pos_items.max().item()}"
                )

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
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )
        scores[:, 0] = -1e9
        return scores
