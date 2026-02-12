# -*- coding: utf-8 -*-
"""
RLI-TRM: Recursive Latent Interest TRM.

Projects sequence L into K interest slots, runs recursive z_reason/z_answer refinement
on K with bidirectional attention, then a gated readout over (z_answer, last_emb).
CE loss, next-item prediction. Full backprop in all steps.
"""

import copy
import math
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import ItemToInterestAggregation
from recbole.model.sequential_recommender.trm_layers import TRMBlock


class RLITRM(SequentialRecommender):
    r"""
    RLI-TRM: Recursive Latent Interest TRM.

    L-to-K interest bottleneck, recursive bidirectional reasoning on K slots,
    hybrid readout (mean(z_answer) + last_emb). Full backprop, CE loss.
    """

    def __init__(self, config, dataset):
        super(RLITRM, self).__init__(config, dataset)

        self.k_interests = config["k_interests"]
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
        self.rms_norm_eps = (
            config["rms_norm_eps"] if "rms_norm_eps" in config else 1e-5
        )
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.use_history_skip = (
            config["use_history_skip"] if "use_history_skip" in config else False
        )
        self.step_drop_rate = (
            config["step_drop_rate"] if "step_drop_rate" in config else 0.0
        )

        self.embed_scale = math.sqrt(self.hidden_size)
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )

        self.interest_proj = ItemToInterestAggregation(
            seq_len=self.max_seq_length,
            hidden_size=self.hidden_size,
            k_interests=self.k_interests,
        )

        # Gated readout: combines pooled reasoning state and last-item embedding.
        self.readout_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        head_dim = self.hidden_size // self.n_heads
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

        # Learned gate on static interests_0 to allow up/down-weighting during recursion.
        self.interest_gate = nn.Parameter(torch.tensor(1.0))
        # Optional compressed skip connection from original history into the bottleneck.
        self.history_skip_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.z_answer_init = nn.Parameter(
            torch.empty(1, self.k_interests, self.hidden_size)
        )
        self.z_reason_init = nn.Parameter(
            torch.empty(1, self.k_interests, self.hidden_size)
        )
        self._id_range_logged = False

        if self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(
                "RLITRM uses CE only. Make sure 'loss_type' is 'CE'."
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

    def _one_cycle(
        self,
        z_reason,
        z_answer,
        interests_0,
        attn_mask,
        history_skip=None,
    ):
        cos_sin = None  # No RoPE for unordered interest slots
        for block in self.trm_blocks:
            # Residual thinking step over current reasoning/answer and static interests.
            residual = z_reason + z_answer + self.interest_gate * interests_0
            if history_skip is not None:
                residual = residual + history_skip
            z_reason = block(residual, cos_sin, attn_mask)
            z_answer = block(z_answer + z_reason, cos_sin, attn_mask)
        return z_reason, z_answer

    def forward(self, item_seq, item_seq_len):
        B, L = item_seq.shape
        if self.check_item_id_range:
            assert item_seq.max().item() < self.n_items, (
                "item_seq ids must be in [0, n_items-1]"
            )

        position_ids = torch.arange(
            item_seq.size(1), device=item_seq.device, dtype=torch.long
        ).unsqueeze(0).expand_as(item_seq)
        position_emb = self.position_embedding(position_ids)
        item_emb = self.embed_scale * self.item_embedding(item_seq)
        input_emb = item_emb + position_emb

        interests_0 = self.interest_proj(input_emb)

        last_pos = torch.clamp(item_seq_len - 1, min=0, max=L - 1)
        last_emb = self.gather_indexes(input_emb, last_pos)

        z_reason = self.z_reason_init.expand(B, self.k_interests, -1)
        z_answer = self.z_answer_init.expand(B, self.k_interests, -1)

        K = self.k_interests
        attn_mask = torch.zeros(
            B, 1, K, K, device=input_emb.device, dtype=input_emb.dtype
        )

        history_skip = None
        if self.use_history_skip:
            # Compressed view of original sequence (mean-pooled then projected).
            history_summary = input_emb.mean(dim=1, keepdim=True)
            history_skip = self.history_skip_proj(history_summary)

        if (not self.training) or self.step_drop_rate <= 0.0:
            # No stochastic depth: always run all steps.
            for _ in range(self.trm_steps):
                z_reason, z_answer = self._one_cycle(
                    z_reason, z_answer, interests_0, attn_mask, history_skip
                )
        else:
            # Stochastic depth over recursion steps: randomly skip intermediate steps
            # during training, but never skip the final step so at least one update
            # is always applied.
            for step in range(self.trm_steps):
                if step < self.trm_steps - 1:
                    rand_val = torch.rand((), device=input_emb.device)
                    if rand_val < self.step_drop_rate:
                        # Skip this step: identity mapping on z_reason / z_answer.
                        continue
                z_reason, z_answer = self._one_cycle(
                    z_reason, z_answer, interests_0, attn_mask, history_skip
                )

        pooled = z_answer.mean(dim=1)
        combined = torch.cat([pooled, last_emb], dim=-1)
        seq_output = self.readout_mlp(combined)
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
                    "[RLITRM] id range (once): "
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
