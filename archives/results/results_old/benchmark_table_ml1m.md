# Main Benchmark Results (MovieLens-1M)

SASRec row is from our run **sasrec_ml1m_v100_ce** (job 160090820, config `configs/ml-1m_SASRec.yaml`).  
Test metrics from best valid epoch (43): NDCG@10 0.1662, Hit@10 0.2952, MRR@10 0.127.

Parameter counts and (where completed) test metrics from run logs. TRM4Rec variants (S4/S8/S16/Adaptive) share the same architecture; only inference depth (steps) differs.

**Table 1. Main Benchmark Results (MovieLens-1M)**

| Model | Params | Compute (relative) | Hit@1 | Hit@5 | Hit@10 | NDCG@1 | NDCG@5 | NDCG@10 | MRR@1 | MRR@5 | MRR@10 |
|-------|--------|--------------------|-------|-------|--------|--------|--------|---------|-------|-------|--------|
| SASRec | 321,984 | 1.0× | 0.0690 | 0.2015 | 0.2952 | 0.0690 | 0.1361 | 0.1662 | 0.0690 | 0.1146 | 0.127 |
| BERT4Rec | 329,753 | 1.04× | 0.0629 | 0.1952 | 0.2858 | 0.0629 | 0.1292 | 0.1583 | 0.0629 | 0.1076 | 0.1196 |
| LightSANs | 340,160 | 1.16× | 0.0732 | 0.2073 | 0.299 | 0.0732 | 0.1418 | 0.1713 | 0.0732 | 0.1203 | 0.1324 |
| TRM4Rec-Vanilla S4 | 284,352 | 5.26× | 0.0717 | 0.2050 | 0.2909 | 0.0717 | 0.1395 | 0.1672 | 0.0717 | 0.1180 | 0.1294 |
| TRM4Rec-Vanilla S8 | 284,352 | 10.5× | 0.0725 | 0.2008 | 0.2922 | 0.0725 | 0.1378 | 0.1672 | 0.0725 | 0.1171 | 0.1291 |
| TRM4Rec-Vanilla S4 B2 | 349,888 | 10.5× | **0.0760** | **0.2205** | **0.3076** | **0.0760** | **0.1503** | **0.1784** | **0.0760** | **0.1272** | **0.1387** |
| TRM4Rec-Vanilla S16 | 284,352 | — | — | — | — | — | — | — | — | — | — |
| TRM4Rec-Adaptive | 284,352 | — | — | — | — | — | — | — | — | — | — |

*Hit@1, Hit@5, NDCG@1, NDCG@5, MRR@1, MRR@5:* From same checkpoints as Table 1; see `results/ml1m_metrics_at_1_5_10.md`. To recompute: `python3 scripts/run_benchmark_metrics_at_1_5_10.py --out results/ml1m_metrics_at_1_5_10.md` or on Gadi `qsub pbs/run_benchmark_metrics_ml1m.pbs`.

*Compute (relative):* FLOPs per forward from RecBole `get_flops` (run logs); baseline SASRec = 1.0×.

**Table 2. Compute vs accuracy (MovieLens-1M test)**

| Model | Steps / depth | Relative FLOPs | NDCG@10 | Hit@10 |
|-------|----------------|----------------|---------|--------|
| SASRec | 2 layers | 1.0× | 0.1662 | 0.2952 |
| BERT4Rec | 2 layers | 1.04× | 0.1583 | 0.2858 |
| LightSANs | 2 layers | 1.16× | 0.1713 | 0.299 |
| TRM4Rec-Vanilla S4 | 4 steps × 1 block | 5.26× | 0.1672 | 0.2909 |
| TRM4Rec-Vanilla S8 | 8 steps × 1 block | 10.5× | 0.1672 | 0.2922 |
| TRM4Rec-Vanilla S4 B2 | 4 steps × 2 blocks | 10.5× | **0.1784** | **0.3076** |
| TRM4Rec-Adaptive | (avg TBD) | — | — | — |

**TRM4Rec-Vanilla S4 B2 vs previous best (LightSANs, MovieLens-1M test):**

| Metric   | LightSANs | TRM4Rec S4 B2 | Absolute Δ | Relative Δ |
|----------|-----------|---------------|------------|------------|
| NDCG@1   | 0.0732    | 0.0760       | +0.0028    | +3.8%      |
| NDCG@5   | 0.1418    | 0.1503       | +0.0085    | +6.0%      |
| NDCG@10  | 0.1713    | 0.1784       | +0.0071    | +4.1%      |
| Hit@1    | 0.0732    | 0.0760       | +0.0028    | +3.8%      |
| Hit@5    | 0.2073    | 0.2205       | +0.0132    | +6.4%      |
| Hit@10   | 0.299     | 0.3076       | +0.0086    | +2.9%      |
| MRR@1    | 0.0732    | 0.0760       | +0.0028    | +3.8%      |
| MRR@5    | 0.1203    | 0.1272       | +0.0069    | +5.7%      |
| MRR@10   | 0.1324    | 0.1387       | +0.0063    | +4.8%      |

**Source for SASRec (ml-1m):** `sasrec_ml1m_v100_ce.o160090820` (PBS job 160090820, V100, best valid epoch 43, checkpoint `outputs/sasrec_ml1m/SASRec-Feb-07-2026_07-32-42.pth`).  
**Source for TRM4Rec-Vanilla S4 (ml-1m):** job 160117254, A100, config `configs/ml-1m_TRM4Rec.yaml` (trm_steps=4), checkpoint `outputs/trm4rec_ml1m/TRM4Rec-Feb-07-2026_15-58-33.pth`.  
**Source for BERT4Rec (ml-1m):** job 160124170, A100, best valid epoch 45, test from `bert4rec_ml1m_a100_ce.o160124170`.  
**Source for LightSANs (ml-1m):** run 160127593, A100, best valid epoch 32, checkpoint `outputs/lightsans_ml1m/LightSANs-Feb-08-2026_07-18-22.pth`, test from `lightsans_ml1m_live_160127593.gadi-pbs.log`.  
**Source for TRM4Rec-Vanilla S8 (ml-1m):** job 160124196, A100, config `configs/ml-1m_TRM4Rec_S8.yaml` (trm_steps=8), best valid epoch 32, checkpoint `outputs/trm4rec_ml1m_s8/TRM4Rec-Feb-08-2026_05-04-32.pth`, test from `trm4rec_ml1m_s8_a100_ce.o160124196`.  
**Source for TRM4Rec-Vanilla S4 B2 (ml-1m):** job 160141304, A100, config `configs/ml-1m_TRM4Rec_B2.yaml` (trm_blocks_per_step=2), 349,888 params, best valid epoch 31, checkpoint `outputs/trm4rec_ml1m_b2/TRM4Rec-Feb-08-2026_14-02-03.pth`, test from `trm4rec_ml1m_b2_ce.o160141304`.

---

**Interpretation (compute vs parameters):** Although TRM4Rec uses fewer parameters than SASRec, its inference cost is higher in fixed-depth settings because the same parameters are reused across multiple recursive refinement steps. Each step applies a full attention–feedforward block, increasing FLOPs without increasing parameter count. This decoupling of parameters from inference compute enables TRM4Rec to trade compute for accuracy at inference time, motivating our adaptive halting mechanism.”