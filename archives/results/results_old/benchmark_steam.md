# Benchmark Results (Steam, duplicate removal)

Same evaluation setup as MovieLens-1M and Amazon Beauty: `user_inter_num_interval: [5,inf)`, `item_inter_num_interval: [5,inf)`, `MAX_ITEM_LIST_LENGTH: 50`, seed 2026, full ranking eval, NDCG@10 for early stopping.

**Filtered dataset (steam-duprem):** 25,390 users, 4,090 items, 328,278 interactions, sparsity 99.68% (from SASRec run 160193586).

## Test metrics (Steam)

| Model | Params | Hit@1 | Hit@5 | Hit@10 | NDCG@1 | NDCG@5 | NDCG@10 | MRR@1 | MRR@5 | MRR@10 | Best valid epoch | Job ID | Checkpoint |
|-------|--------|-------|-------|--------|--------|--------|---------|-------|-------|--------|------------------|--------|------------|
| SASRec | 365,056 | 0.0252 | 0.0803 | 0.1383 | 0.0252 | 0.0528 | **0.0714** | 0.0252 | 0.0439 | **0.0514** | 2 | 160193586 | `outputs/sasrec_steam/SASRec-Feb-09-2026_10-45-16.pth` |
| BERT4Rec | 373,498 | 0.0228 | 0.0784 | 0.132 | 0.0228 | 0.0505 | 0.0677 | 0.0228 | 0.0414 | 0.0484 | 21 | 160196053 | `outputs/bert4rec_steam/BERT4Rec-Feb-09-2026_11-03-42.pth` |
| LightSANs | 383,232 | 0.0230 | 0.0806 | **0.1405** | 0.0230 | 0.0516 | 0.0709 | 0.0230 | 0.0421 | 0.05 | 6 | 160197089 | `outputs/lightsans_steam/LightSANs-Feb-09-2026_11-08-39.pth` |
| TRM4Rec S4 | 327,424 | 0.0234 | **0.0844** | 0.1376 | 0.0234 | **0.0538** | 0.0709 | 0.0234 | **0.0438** | 0.0508 | 8 | 160205252 | `outputs/trm4rec_steam/TRM4Rec-Feb-09-2026_14-09-44.pth` |
| TRM4Rec S4 B2 | 392,960 | **0.0254** | 0.0798 | 0.1349 | **0.0254** | 0.0525 | 0.0701 | **0.0254** | 0.0436 | 0.0507 | 10 | 160205259 | `outputs/trm4rec_steam_b2/TRM4Rec-Feb-09-2026_14-09-44.pth` |

**Source for SASRec (steam-duprem):** Job 160193586, A100, config `configs/steam/steam_SASRec.yaml`. Early stopping at epoch 2 (best valid NDCG@10 0.0771). Walltime used ~19 min. Output: `sasrec_steam_a100_ce.o160193586`.

**Source for BERT4Rec (steam-duprem):** Job 160196053, A100, config `configs/steam/steam_BERT4Rec.yaml`. Best valid epoch 21 (NDCG@10 0.0748). Walltime used ~48 min. Output: `bert4rec_steam_a100_ce.o160196053`.

**Source for LightSANs (steam-duprem):** Job 160197089, A100, config `configs/steam/steam_LightSANs.yaml`. Early stopping at epoch 6 (best valid NDCG@10 0.0762). Walltime used ~42 min. Output: `lightsans_steam_a100_ce.o160197089`.

**Source for TRM4Rec S4 (steam-duprem):** Job 160205252, A100, config `configs/steam/steam_TRM4Rec.yaml`. trm_steps=4, trm_blocks_per_step=1, 327,424 params. Best valid epoch 8 (NDCG@10 0.0769). Test: hit@10 0.1376, ndcg@10 0.0709, mrr@10 0.0508. Walltime used ~27 min. Output: `trm4rec_steam_a100_ce.o160205252`.

**Source for TRM4Rec S4 B2 (steam-duprem):** Job 160205259, A100, config `configs/steam/steam_TRM4Rec_B2.yaml`. trm_steps=4, trm_blocks_per_step=2, 392,960 params. Best valid epoch 10 (NDCG@10 0.077). Test: hit@10 0.1349, ndcg@10 0.0701, mrr@10 0.0507. Walltime used ~54 min. Output: `trm4rec_steam_b2_ce.o160205259`.

*Hit@1, Hit@5, NDCG@1, NDCG@5, MRR@1, MRR@5:* From [steam_metrics_at_1_5_10.md](steam_metrics_at_1_5_10.md). To regenerate: `python3 scripts/run_benchmark_metrics_at_1_5_10.py --benchmark steam --out results/steam_metrics_at_1_5_10.md`.
