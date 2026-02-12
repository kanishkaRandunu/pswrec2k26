# Baseline model default parameters

For benchmarking, we use **RecBole defaults** for GRU4Rec, SASRec, BERT4Rec, and FEARec. For **WEARec**, we use the defaults from their official repo. For **PSWRec** (our model), we use the pswrecv4h parameters — our best model so far.

---

## PSWRec (pswrecv4h — our model)

Source: `archives/configs/beauty/beauty_pswrecv4h_a100_ce.yaml` (best model so far)

| Parameter | Value |
|-----------|-------|
| `model` | pswrecv4h |
| `loss_type` | CE |
| `n_layers` | 2 |
| `n_heads` | 2 |
| `hidden_size` | 64 |
| `inner_size` | 256 |
| `hidden_dropout_prob` | 0.5 |
| `attn_dropout_prob` | 0.5 |
| `hidden_act` | gelu |
| `layer_norm_eps` | 1e-12 |
| `initializer_range` | 0.02 |
| `n_bands` | 4 |
| `band_kernel_sizes` | [3, 7, 15, 31] |
| `band_dilations` | [1, 2, 4, 8] |
| `phase_bias_scale` | 0.1 |
| `phase_aux` | True |
| `phase_aux_weight` | 0.05 |
| `train_batch_size` | 128 |
| `learning_rate` | 0.001 |
| `weight_decay` | 0.0 |

---

## GRU4Rec (RecBole)

Source: `RecBole/recbole/properties/model/GRU4Rec.yaml`; training from `overall.yaml`.

| Parameter | Default |
|-----------|---------|
| `embedding_size` | 64 |
| `hidden_size` | 128 |
| `num_layers` | 1 |
| `dropout_prob` | 0.3 |
| `loss_type` | CE |
| `train_batch_size` | 128 (benchmark fixed) |

---

## SASRec (RecBole)

Source: `RecBole/recbole/properties/model/SASRec.yaml`

| Parameter | Default |
|-----------|---------|
| `n_layers` | 2 |
| `n_heads` | 2 |
| `hidden_size` | 64 |
| `inner_size` | 256 |
| `hidden_dropout_prob` | 0.5 |
| `attn_dropout_prob` | 0.5 |
| `hidden_act` | gelu |
| `layer_norm_eps` | 1e-12 |
| `initializer_range` | 0.02 |
| `loss_type` | CE |
| `train_batch_size` | 128 (benchmark fixed) |

---

## BERT4Rec (RecBole)

Source: `RecBole/recbole/properties/model/BERT4Rec.yaml`

| Parameter | Default |
|-----------|---------|
| `n_layers` | 2 |
| `n_heads` | 2 |
| `hidden_size` | 64 |
| `inner_size` | 256 |
| `hidden_dropout_prob` | 0.2 |
| `attn_dropout_prob` | 0.2 |
| `hidden_act` | gelu |
| `layer_norm_eps` | 1e-12 |
| `initializer_range` | 0.02 |
| `mask_ratio` | 0.2 |
| `ft_ratio` | 0.5 |
| `loss_type` | CE |
| `transform` | mask_itemseq |
| `train_batch_size` | 128 (benchmark fixed) |

---

## FEARec (RecBole)

Source: `RecBole/recbole/properties/model/FEARec.yaml`

| Parameter | Default |
|-----------|---------|
| `n_layers` | 2 |
| `n_heads` | 2 |
| `hidden_size` | 64 |
| `inner_size` | 256 |
| `hidden_dropout_prob` | 0.5 |
| `attn_dropout_prob` | 0.5 |
| `hidden_act` | gelu |
| `layer_norm_eps` | 1e-12 |
| `initializer_range` | 0.02 |
| `loss_type` | CE |
| `lmd` | 0.1 |
| `lmd_sem` | 0.1 |
| `global_ratio` | 1 |
| `dual_domain` | False |
| `std` | False |
| `spatial_ratio` | 0 |
| `fredom` | False |
| `fredom_type` | null |
| `topk_factor` | 1 |
| `train_batch_size` | 128 (benchmark fixed) |

---

## WEARec (official repo)

Source: `WEARec/src/utils.py` `parse_args()` — WEARec-specific and shared model args.

| Parameter | Default |
|-----------|---------|
| `max_seq_length` | 50 |
| `hidden_size` | 64 |
| `num_hidden_layers` | 2 |
| `hidden_act` | gelu |
| `num_heads` | 2 |
| `alpha` | 0.3 |
| `attention_probs_dropout_prob` | 0.5 |
| `hidden_dropout_prob` | 0.5 |
| `initializer_range` | 0.02 |
| **Training (WEARec repo)** | |
| `batch_size` | 256 (repo); 128 (benchmark) |
| `lr` | 0.001 |
| `epochs` | 200 |
| `weight_decay` | 0.0 |
| `patience` | 10 |
| `seed` | 42 |

**Note:** Our RecBole port of WEARec uses our benchmark data pipeline and eval protocol. We keep WEARec’s repo defaults including batch size 256 .
