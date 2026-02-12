# Benchmark Results (Amazon Beauty)

Same evaluation setup as MovieLens-1M: `user_inter_num_interval: [5,inf)`, `item_inter_num_interval: [5,inf)`, `MAX_ITEM_LIST_LENGTH: 50`, seed 2026, full ranking eval, NDCG@10 for early stopping.

**Filtered dataset (amazon-beauty):** 22,364 users, 12,102 items, 198,502 interactions, sparsity 99.93% (from SASRec run 160183476).

## Test metrics (Amazon Beauty)

*Best per metric: **bold**. Second best: asterisk (*).*

| Model | Params | Hit@1 | Hit@5 | Hit@10 | NDCG@1 | NDCG@5 | NDCG@10 | MRR@1 | MRR@5 | MRR@10 | Best valid epoch | Job ID | Checkpoint |
|-------|--------|-------|-------|--------|--------|--------|---------|-------|-------|--------|------------------|--------|------------|
| SASRec | 877,824 | 0.0126 | 0.0497 | 0.0736 | 0.0126 | 0.0315 | 0.0391 | 0.0126 | 0.0255 | 0.0286 | 14 | 160183476 | `outputs/sasrec_beauty/SASRec-Feb-09-2026_07-43-11.pth` |
| BERT4Rec | 894,278 | 0.0072 | 0.0242 | 0.0404 | 0.0072 | 0.0158 | 0.0211 | 0.0072 | 0.0131 | 0.0152 | 33 | 160184018 | `outputs/bert4rec_beauty/BERT4Rec-Feb-09-2026_08-27-00.pth` |
| LightSANs | 896,000 | **0.0158** | 0.0502* | 0.0753 | **0.0158** | 0.0335* | 0.0416 | **0.0158** | **0.0280** | **0.0313** | 24 | 160184019 | `outputs/lightsans_beauty/LightSANs-Feb-09-2026_08-27-01.pth` |
| FEARec | 877,824 | — | — | **0.0854** | — | — | **0.0433** | — | — | 0.0304 | 38 | 160234598 | `outputs/fearec_beauty/FEARec-Feb-09-2026_15-56-14.pth` |
| TRM4Rec-Vanilla S4 | 840,192 | 0.0136* | **0.0523** | 0.0769 | 0.0136* | **0.0336** | 0.0415 | 0.0136* | 0.0274* | 0.0306* | 11 | 160184063 | `outputs/trm4rec_beauty/TRM4Rec-Feb-09-2026_09-02-35.pth` |
| TRM4Rec-Vanilla S4 B2 | 905,728 | 0.0118 | 0.0495 | 0.0780 | 0.0118 | 0.0311 | 0.0403 | 0.0118 | 0.0251 | 0.0288 | 5 | 160185361 | `outputs/trm4rec_beauty_b2/TRM4Rec-Feb-09-2026_09-15-18.pth` |
| TRM4RecFullBP S4 | 840,192 | — | — | 0.0781 | — | — | 0.0407 | — | — | 0.0294 | 4 | 160466127 | `outputs/trm4rec_fullbp_beauty/TRM4RecFullBP-Feb-10-2026_06-43-17.pth` |
| TRM4RecFullBP S4 B2 | 905,728 | — | — | 0.0797 | — | — | 0.0411 | — | — | 0.0293 | 5 | 160466275 | `outputs/trm4rec_fullbp_beauty_b2/TRM4RecFullBP-Feb-10-2026_07-10-41.pth` |
| TRM4RecPos pos-only vanilla B1 | 843,392 | — | — | 0.0779 | — | — | 0.0408 | — | — | 0.0295 | 10 | 160471515 | `outputs/trm4rec_pos_posonly_vanilla_beauty/TRM4RecPos-Feb-10-2026_09-00-25.pth` |
| TRM4RecPos pos-only vanilla B2 | 908,928 | — | — | 0.0789 | — | — | 0.0415 | — | — | 0.0301 | 8 | 160471556 | `outputs/trm4rec_pos_posonly_vanilla_beauty_b2/TRM4RecPos-Feb-10-2026_09-07-09.pth` |
| TRM4RecPos pos+RoPE vanilla B1 | 843,392 | — | — | 0.0771 | — | — | 0.0409 | — | — | 0.0299 | 10 | 160473289 | `outputs/trm4rec_pos_posrope_vanilla_beauty/TRM4RecPos-Feb-10-2026_09-24-34.pth` |
| TRM4RecPos pos+RoPE vanilla B2 | 908,928 | — | — | 0.0791 | — | — | 0.0407 | — | — | 0.0289 | 8 | 160473503 | `outputs/trm4rec_pos_posrope_vanilla_beauty_b2/TRM4RecPos-Feb-10-2026_09-33-44.pth` |
| TRM4RecPos pos-only fullbp B1 | 843,392 | — | — | 0.0808* | — | — | 0.0419* | — | — | 0.0301 | 4 | 160478663 | `outputs/trm4rec_pos_posonly_fullbp_beauty/TRM4RecPos-Feb-10-2026_10-09-13.pth` |
| TRM4RecPos pos-only fullbp B2 | 908,928 | — | — | 0.0785 | — | — | 0.0405 | — | — | 0.0289 | 5 | 160478675 | `outputs/trm4rec_pos_posonly_fullbp_beauty_b2/TRM4RecPos-Feb-10-2026_10-09-36.pth` |
| TRM4RecPos pos+RoPE fullbp B1 | 843,392 | — | — | 0.079 | — | — | 0.041 | — | — | 0.0294 | 4 | 160484265 | `outputs/trm4rec_pos_posrope_fullbp_beauty/TRM4RecPos-Feb-10-2026_10-56-00.pth` |
| TRM4RecPos pos+RoPE fullbp B2 | 908,928 | — | — | 0.0787 | — | — | 0.0407 | — | — | 0.0291 | 4 | 160485777 | `outputs/trm4rec_pos_posrope_fullbp_beauty_b2/TRM4RecPos-Feb-10-2026_11-09-13.pth` |

**Source for SASRec (amazon-beauty):** Job 160183476, A100, config `configs/beauty/beauty_SASRec.yaml`. Early stopping at epoch 14. Walltime used ~17 min. Output: `outputs/sasrec_beauty/sasrec_beauty_a100_ce.o160183476`.

**Source for BERT4Rec (amazon-beauty):** Job 160184018, A100, config `configs/beauty/beauty_BERT4Rec.yaml`. Best valid epoch 33. Walltime used ~35 min. Output: `bert4rec_beauty_a100_ce.o160184018`.

**Source for LightSANs (amazon-beauty):** Job 160184019, A100, config `configs/beauty/beauty_LightSANs.yaml`. Best valid epoch 24. Walltime used ~43 min. Output: `lightsans_beauty_a100_ce.o160184019`.

**Source for TRM4Rec S4 (amazon-beauty):** Job 160184063, A100, config `configs/beauty/beauty_TRM4Rec.yaml`. Best valid epoch 11 (NDCG@10 0.0526). Walltime used ~15 min. Output: `trm4rec_beauty_a100_ce.o160184063`.

**Source for TRM4Rec S4 B2 (amazon-beauty):** Job 160185361, A100, config `configs/beauty/beauty_TRM4Rec_B2.yaml`. Best valid epoch 5 (NDCG@10 0.0537). Walltime used ~21 min. Output: `trm4rec_beauty_b2_ce.o160185361`.

**Source for FEARec (amazon-beauty):** Job 160234598, A100, config `configs/beauty/beauty_FEARec.yaml`. 877,824 params. Best valid epoch 38 (NDCG@10 0.0538). Test: hit@10 0.0854, ndcg@10 0.0433, mrr@10 0.0304. Live log: `fearec_beauty_live_160234598.gadi-pbs.log`.

**Source for TRM4RecFullBP S4 (amazon-beauty):** Job 160466127, A100, config `configs/beauty/beauty_TRM4Rec_FullBP.yaml`. 840,192 params, full backprop in all recursive steps. Best valid epoch 4 (NDCG@10 0.0528). Test: hit@10 0.0781, ndcg@10 0.0407, mrr@10 0.0294. Walltime used ~35 min. Output: `trm4rec_fullbp_beauty_a100_ce.o160466127`. Live log: `trm4rec_fullbp_beauty_live_160466127.gadi-pbs.log`.

**Source for TRM4RecFullBP S4 B2 (amazon-beauty):** Job 160466275, A100, config `configs/beauty/beauty_TRM4Rec_FullBP_B2.yaml`. 905,728 params, trm_blocks_per_step=2, full backprop in all recursive steps. Best valid epoch 5 (NDCG@10 0.0534). Test: hit@10 0.0797, ndcg@10 0.0411, mrr@10 0.0293. Walltime used ~71 min. Output: `trm4rec_fullbp_beauty_b2_ce.o160466275`. Live log: `trm4rec_fullbp_beauty_b2_live_160466275.gadi-pbs.log`.

**Source for TRM4RecPos pos-only vanilla B1 (amazon-beauty):** Job 160471515, A100, config `configs/beauty-posattention/beauty_pos_posonly_vanilla_s4_b1.yaml`. 843,392 params, use_rope=false, trm_blocks_per_step=1. Best valid epoch 10 (NDCG@10 0.0519). Test: hit@10 0.0779, ndcg@10 0.0408, mrr@10 0.0295. Walltime ~13 min. Output: `trm4rec_pos_posonly_van_b1_ce.o160471515`.

**Source for TRM4RecPos pos-only vanilla B2 (amazon-beauty):** Job 160471556, A100, config `configs/beauty-posattention/beauty_pos_posonly_vanilla_s4_b2.yaml`. 908,928 params, use_rope=false, trm_blocks_per_step=2. Best valid epoch 8 (NDCG@10 0.0532). Test: hit@10 0.0789, ndcg@10 0.0415, mrr@10 0.0301. Walltime ~20 min. Output: `trm4rec_pos_posonly_van_b2_ce.o160471556`.

**Source for TRM4RecPos pos+RoPE vanilla B1 (amazon-beauty):** Job 160473289, A100, config `configs/beauty-posattention/beauty_pos_posrope_vanilla_s4_b1.yaml`. 843,392 params, use_rope=true, trm_blocks_per_step=1. Best valid epoch 10 (NDCG@10 0.0528). Test: hit@10 0.0771, ndcg@10 0.0409, mrr@10 0.0299. Walltime ~15 min. Output: `trm4rec_pos_posrope_van_b1_ce.o160473289`.

**Source for TRM4RecPos pos+RoPE vanilla B2 (amazon-beauty):** Job 160473503, A100, config `configs/beauty-posattention/beauty_pos_posrope_vanilla_s4_b2.yaml`. 908,928 params, use_rope=true, trm_blocks_per_step=2. Best valid epoch 8 (NDCG@10 0.0527). Test: hit@10 0.0791, ndcg@10 0.0407, mrr@10 0.0289. Walltime ~25 min. Output: `trm4rec_pos_posrope_van_b2_ce.o160473503`.

**Source for TRM4RecPos pos-only fullbp B1 (amazon-beauty):** Job 160478663, A100, config `configs/beauty-posattention/beauty_pos_posonly_fullbp_s4_b1.yaml`. 843,392 params, use_rope=false, use_full_backprop=true, trm_blocks_per_step=1. Best valid epoch 4 (NDCG@10 0.0544). Test: hit@10 0.0808, ndcg@10 0.0419, mrr@10 0.0301. Walltime ~27 min. Output: `trm4rec_pos_posonly_fbp_b1_ce.o160478663`.

**Source for TRM4RecPos pos-only fullbp B2 (amazon-beauty):** Job 160478675, A100, config `configs/beauty-posattention/beauty_pos_posonly_fullbp_s4_b2.yaml`. 908,928 params, use_rope=false, use_full_backprop=true, trm_blocks_per_step=2. Best valid epoch 5 (NDCG@10 0.0542). Test: hit@10 0.0785, ndcg@10 0.0405, mrr@10 0.0289. Walltime ~57 min. Output: `trm4rec_pos_posonly_fbp_b2_ce.o160478675`.

**Source for TRM4RecPos pos+RoPE fullbp B1 (amazon-beauty):** Job 160484265, A100, config `configs/beauty-posattention/beauty_pos_posrope_fullbp_s4_b1.yaml`. 843,392 params, use_rope=true, use_full_backprop=true, trm_blocks_per_step=1. Best valid epoch 4 (NDCG@10 0.0535). Test: hit@10 0.079, ndcg@10 0.041, mrr@10 0.0294. Walltime ~36 min. Output: `trm4rec_pos_posrope_fbp_b1_ce.o160484265`.

**Source for TRM4RecPos pos+RoPE fullbp B2 (amazon-beauty):** Job 160485777, A100, config `configs/beauty-posattention/beauty_pos_posrope_fullbp_s4_b2.yaml`. 908,928 params, use_rope=true, use_full_backprop=true, trm_blocks_per_step=2. Best valid epoch 4 (NDCG@10 0.0527). Test: hit@10 0.0787, ndcg@10 0.0407, mrr@10 0.0291. Walltime ~68 min. Output: `trm4rec_pos_posrope_fbp_b2_ce.o160485777`.

*Hit@1, Hit@5, NDCG@1, NDCG@5, MRR@1, MRR@5:* From [beauty_metrics_at_1_5_10.md](beauty_metrics_at_1_5_10.md). To recompute: `python3 scripts/run_benchmark_metrics_at_1_5_10.py --benchmark beauty --out results/beauty_metrics_at_1_5_10.md` or on Gadi `qsub pbs/run_benchmark_metrics_beauty.pbs`.
