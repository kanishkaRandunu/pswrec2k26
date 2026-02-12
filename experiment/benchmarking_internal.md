# Benchmarking internal checklist and plan

Internal operational checklist. Do not share externally.

---

## Pre-run checklist

Before starting any benchmark run, verify:

- [ ] **Per-model YAML is self-contained** (beauty configs in `configs/beauty/` include all benchmark settings; no separate base file needed)
- [ ] **Full-ranking candidate universe** is identical across all runs (same vocab, same item id mapping)
- [ ] **Seeds fixed** and enforced by the aggregator (identical seed list per dataset and model)
- [ ] **Early stopping** recorded: `best_valid_epoch`, `best_valid_score`, `test_score_at_best_valid`; never report best test across epochs
- [ ] **Dataset lock file** written (see below)
- [ ] **WEARec mapping verified** (see sanity checks below)
- [ ] **Determinism**: `torch.backends.cudnn.deterministic` set and recorded in metadata if RecBole does not already

---

## 1. Base benchmark config

- Each per-model YAML in `configs/beauty/` is **self-contained** — all benchmark-defining settings are inlined.
- A reference file `configs/benchmark_base.yaml` still exists for documentation and future multi-dataset benchmarks, but beauty runs pass only the single per-model YAML via `--config_files`.
- Model configs include both benchmark-defining settings and model-specific knobs in one file.

---

## 2. Candidate set and item mapping

- Confirm test candidates = all items in the dataset vocab.
- Vocab construction = same filtered interactions for every model.
- Ensure no model uses a different item universe (e.g. WEARec must consume RecBole’s item id mapping, not its own).
- **Pitfall**: A repo baseline (WEARec) may build its own item mapping or filter differently — force it to use RecBole’s pipeline.

---

## 3. Train objective parity and metadata

For each model, record in `results/metadata.json` (or equivalent):

- `loss_type` (full softmax CE vs sampled)
- `train_neg_sample_args` (negative sampling settings)
- Any deviation from CE (e.g. BPR, sampled softmax)

If a model cannot run full softmax CE at our scale: **do not silently switch it**. Either document it in the paper table footnote or drop the model.

---

## 4. Seed policy and variance

- **Decision**: 1 seed for quick iteration; 3 seeds for final paper tables.
- Aggregator must enforce **identical seed list** per dataset and model.
- For 3-seed runs: store and report **mean ± std**.
- Lock determinism: set `torch.backends.cudnn.deterministic` and record in metadata.

---

## 5. Early stopping

- `valid_metric: NDCG@10`, `stopping_step: 10`, `eval_step: 1`.
- Store for every run:
  - `best_valid_epoch`
  - `best_valid_score`
  - `test_score_at_best_valid`
- **Do not** report best test across epochs.

---

## 6. Compute and throughput logging

Log compute evidence (not vibes). Add to every run:

- Wall clock training time
- Inference time per user batch at test
- Peak GPU memory
- Params count (trainable)
- Optional: tokens or interactions processed per second

---

## 7. Dataset lock file

Maintain `datasets.lock` (or equivalent) with:

- Dataset source and version
- Download hash (if available)
- Filtering thresholds used
- Resulting counts: users, items, interactions after filtering

If preprocessing changes, earlier runs become invalid. Freeze and version.

---

## 8. BERT4Rec masking

- Keep BERT4Rec mask ratio and training settings at RecBole defaults.
- Record them in metadata.
- Do not change sequence construction for BERT4Rec relative to other models.

---

## 9. WEARec integration sanity checks

**High risk.** Run before benchmarking:

1. **Identity test**: Run SASRec (pure RecBole). Then run a wrapper path that mimics WEARec ingestion but still uses SASRec. Verify **identical metrics**. If metrics differ, the pipeline diverged.
2. **Mapping test**: Confirm item ids seen by WEARec match RecBole internal token ids **one to one**.

If either fails, runs are **not comparable**.

---

## 10. Result schema and naming

- **Run ID format**: `{dataset}_{model}_{seed}_{timestamp}`
- **Per run**: one JSON with config snapshot and metrics at K ∈ {1, 5, 10, 20}
- **Per dataset**: one aggregated CSV with mean ± std and best-epoch metadata
- Zero manual table editing in the pipeline

---

## Minimal pre-run checklist (summary)

1. Base benchmark YAML exists and is included in every config  
2. Full ranking candidate universe identical across all runs  
3. Seeds fixed and enforced by aggregator  
4. Early stopping recorded; test reported at best valid only  
5. Dataset lock file written  
6. WEARec mapping verified (identity + mapping tests)

If all above pass, you can start benchmarking without contaminating results later.
