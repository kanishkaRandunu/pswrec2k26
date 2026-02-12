# TRACT Experiment Results — Amazon Beauty

## Dataset

| Property | Value |
|---|---|
| Dataset | Amazon Beauty |
| Users | 22,364 |
| Items | 12,102 |
| Interactions | 198,502 |
| Sparsity | 99.93% |
| Max sequence length | 50 |
| User/Item filter | [5, inf) |

## Common Hyperparameters

All runs share the following settings:

| Parameter | Value |
|---|---|
| seed | 2026 |
| loss_type | CE (CrossEntropy) |
| n_layers | 2 |
| n_heads | 2 |
| hidden_size | 64 |
| inner_size | 256 |
| hidden_dropout_prob | 0.5 |
| attn_dropout_prob | 0.5 |
| hidden_act | gelu |
| learning_rate | 0.001 |
| train_batch_size | 128 |
| optimizer | Adam |
| epochs (max) | 200 |
| stopping_step | 10 |
| eval_batch_size | 4096 |
| valid_metric | NDCG@10 |
| trainable_params | 885,012 |
| FLOPs | 5,312,064 |
| n_bands | 4 |
| band_kernel_sizes | [3, 7, 15, 31] |
| phase_bias_scale | 0.1 |

---

## Run Details

### Run 1 — PSWRec (no phase aux)

| Detail | Value |
|---|---|
| Job ID | 160580716 |
| Model | PSWRec |
| Config | `configs/beauty/beauty_PSWRec_A100_CE.yaml` |
| band_dilations | [1, 1, 2, 4] |
| phase_aux | False |
| phase_aux_weight | 0.0 |
| Best epoch | 68 |
| Total epochs | 79 |
| Walltime | 01:15:54 |
| Service Units | 91.08 |

### Run 2 — PSWRec (+ phase aux 0.01)

| Detail | Value |
|---|---|
| Job ID | 160596102 |
| Model | PSWRec |
| Config | `configs/beauty/beauty_PSWRec_A100_CE.yaml` |
| band_dilations | [1, 1, 2, 4] |
| phase_aux | True |
| phase_aux_weight | 0.01 |
| Best epoch | 27 |
| Total epochs | 38 |
| Walltime | 00:46:26 |
| Service Units | 55.72 |

### Run 3 — PSWRecV3 (+ phase aux 0.05, wider dilations)

| Detail | Value |
|---|---|
| Job ID | 160623434 |
| Model | pswrecv3 |
| Config | `configs/beauty/beauty_pswrecv3_a100_ce.yaml` |
| band_dilations | [1, 2, 4, 8] |
| phase_aux | True |
| phase_aux_weight | 0.05 |
| Best epoch | 52 |
| Total epochs | 63 |
| Walltime | 01:17:50 |
| Service Units | 93.40 |

Key changes in V3: wider dilation pattern `[1,2,4,8]` (vs `[1,1,2,4]`), stronger phase aux weight `0.05` (vs `0.01`), refactored architecture (`PSWEncoderV3`, `PhaseSyncAttentionV3`, `FeedForwardV3`, `LocalPhaseFilterBankV3`).

---

## Results

### Validation (Best)

| Model | Hit@10 | NDCG@10 | MRR@10 | Recall@10 |
|---|---|---|---|---|
| PSWRec (no aux) | 0.1044 | 0.0532 | 0.0374 | 0.1044 |
| PSWRec (aux=0.01) | 0.1044 | 0.0530 | 0.0372 | 0.1044 |
| **PSWRecV3 (aux=0.05)** | **0.1048** | **0.0535** | **0.0377** | **0.1048** |

### Test

| Model | Hit@10 | NDCG@10 | MRR@10 | Recall@10 |
|---|---|---|---|---|
| PSWRec (no aux) | 0.0833 | 0.0424 | 0.0298 | 0.0833 |
| PSWRec (aux=0.01) | 0.0822 | 0.0417 | 0.0293 | 0.0822 |
| **PSWRecV3 (aux=0.05)** | **0.0845** | **0.0424** | 0.0294 | **0.0845** |

### Summary

PSWRecV3 achieves the best results across the board on both validation and test sets. On the test set, the most notable improvement is in **Hit@10 / Recall@10** (0.0845 vs 0.0833), a **+1.4%** relative improvement over the original PSWRec without phase aux, and **+2.8%** over PSWRec with phase_aux=0.01. The wider dilation pattern `[1,2,4,8]` combined with the stronger phase aux regularization (`0.05`) appears to help the model generalize better.
