## Benchmarking experiments (RecBole)

### Datasets

We will run the benchmark on:

- **Beauty** (Amazon Beauty)
- **MovieLens-1M**
- **Steam**

### Configuration policy

- **All RecBole baselines**: align to **RecBole defaults** (for that model), while using the same dataset pipeline/evaluation protocol for fair comparison.
- **WEARec**: follow **their repo defaults** for model/training hyperparameters, but use **RecBole’s data pipeline + data splitting** so evaluation remains comparable across all models.

### Canonical benchmark config

Each per-model YAML in `configs/beauty/` is **self-contained** — it includes all benchmark-defining settings (seed, filtering, evaluation protocol, training budget, etc.) alongside model-specific hyperparameters. No separate base file is needed at run time.

Common parameters are found in the **`configs/benchmark_base.yaml`** file. all these paramters should be present in each separate configuration files. we do not pass this as an argument to .pbs files. 

### Directory layout

| Path | Purpose |
|------|---------|
| `configs/` | All config files (YAML, per dataset and model); each `configs/beauty/beauty_*.yaml` is self-contained |
| `log/` | Run logs and job markers |
| `log_tensorboard/` | TensorBoard event files |
| `outputs/` | Checkpoints, best models, and job output files |
| `results/` | Aggregated benchmark results and tables |

### Candidate set semantics (full ranking)

Full evaluation is only comparable if the candidate universe is identical across all models:

- Test candidates must be **all items in the dataset vocabulary**.
- Vocab construction must use the **same filtered interactions** for every model.
- No model may use a different item universe due to feature files or different preprocessing.
- External baselines (e.g. WEARec) must consume the same item id mapping produced by RecBole’s dataset build.

### Fixed across all model runs (benchmark definition)

#### Data and sequence definition

Keep fixed across all model runs:

- **`data_path`, dataset version and preprocessing**
- **Interaction filtering thresholds** (user/item minimum interactions)
- **`MAX_ITEM_LIST_LENGTH` (max sequence length)**: 50
- **Sequence construction logic**: chronological ordering (`TO`) + truncation policy (RecBole sequence truncation)
- **Train/valid/test split scheme**: RecBole `LS` with `valid_and_test`

#### Evaluation protocol

Keep identical because it defines the benchmark:

- **`eval_args`**:
  - `split: {"LS": "valid_and_test"}`
  - `order: TO`
  - `group_by: user`
  - `mode: {"valid":"full","test":"full"}`
- **Metrics set and \(K\)**:
  - For paper tables, keep `topk` fixed (e.g., `[10, 20]` if reporting both).
- **Validation metric / early stopping target**:
  - `valid_metric: NDCG@10`
- **`eval_batch_size`**: 4096 (or as large as fits)

#### Repro and logging

- **Seed policy**: use 1 seed for quick iteration; use 3 seeds for final paper tables. Be consistent per phase. If using 3 seeds, store and report **mean ± std**.
- **`reproducibility: true`**
- **Same hardware class** if possible (A100).
- **Same mixed precision policy** (`enable_amp`) across models unless a model breaks.

#### Hardware specifications

All benchmark experiments are run on **A100** GPUs:

- **GPU**: NVIDIA A100 (1 per job)
- **Queue**: `dgxa100` (Gadi DGX A100 nodes)
- **Compute**: 16 CPUs, 16 GB RAM per job
- **CUDA**: 12.9.0 (module `cuda/12.9.0`)
- **Environment**: Python venv `hopper-cu124` (PyTorch 2.6.0+cu124)

#### Training budget and stopping rule

Keep the stopping policy identical:

- `epochs: 200` (upper bound)
- `train_batch_size: 128` (fixed for comparability)
- `eval_step: 1`
- `stopping_step: 10`

**Early stopping comparability**: Report **test score at the best-validation epoch** only. Do not report best test across epochs. For each run, store: `best_valid_epoch`, `best_valid_score`, `test_score_at_best_valid`.

### Train objective parity

- Under full-softmax CE, keep `train_neg_sample_args: null`.
- If a model cannot run full softmax CE at our scale, do not silently switch it: either **document the exception** in the paper table footnote, or **exclude it**.
- For each model, record in results metadata: `loss_type`, negative sampling settings, and any deviation from CE.

### Dataset preprocessing

Beauty, ML-1M, and Steam differ in sparsity and timestamp quality. **Preprocessing must be frozen and versioned**. If filtering or thresholds change later, earlier runs become invalid. Document dataset source, version, filtering thresholds, and resulting user/item/interaction counts.

### BERT4Rec

BERT4Rec trains bidirectionally; evaluation is next-item prediction. Use the standard RecBole BERT4Rec setup. Do not change mask ratio or sequence construction per dataset. Record BERT4Rec-specific settings in metadata.

### Result schema and naming

- **Run ID format**: `{dataset}_{model}_{seed}_{timestamp}`
- **Per run**: one JSON with config snapshot and metrics at K ∈ {1, 5, 10, 20}
- **Per dataset**: one aggregated CSV/table with mean ± std and best-epoch metadata

### Not fixed across models (allowed to vary)

Do **not** force these to be fixed across all models. These are optimization knobs that interact with the model
architecture; forcing them equal can be unfair and may undertrain some baselines.

#### Batch size

- Benchmark uses **`train_batch_size: 128`** for all models (fixed for comparability).

#### Learning rate and weight decay

- Do **not** keep `learning_rate = 0.001` fixed across all models unless you intentionally accept that some baselines
  may be undertrained.
- Weight decay should be tuned per model (or at least per model family).

#### Dropout and hidden sizes

- Each baseline has its own recommended hyperparameters.
- Do **not** force PSWRec-specific architecture settings on baselines.

#### Negative sampling configuration

- Under full-softmax **CE**, keep `train_neg_sample_args: null` consistently.
- If a baseline is designed for sampled loss and cannot reasonably run full-softmax CE, either:
  - run it under its standard loss and **document the exception**, or
  - exclude it from the benchmark.
- In our current benchmark list, **GRU4Rec, SASRec, BERT4Rec, FEARec, WEARec** should be fine with CE.

#### Model-specific defaults

Some models require different regularization or training schedules to reach their reported performance. Allow that
within a defined tuning budget, and document any deviations.

---

## Beauty benchmark: final model list

We will benchmark the following models on the Beauty dataset under the aligned RecBole protocol:

- `GRU4Rec`
- `SASRec`
- `BERT4Rec`
- `FEARec`
- `wearec` (WEARec)
- `PSWRec` (our model)

## Metrics to store

For each model, we will store the following metrics at \(K \in \{1, 5, 10, 20\}\):

- **HR@K** (Hit Rate / Hit@K)
- **NDCG@K**
- **MRR@K**

## Training policy

- **Max epochs**: 200
- **Early stopping patience**: 10 (RecBole `stopping_step: 10`, evaluated every epoch)

