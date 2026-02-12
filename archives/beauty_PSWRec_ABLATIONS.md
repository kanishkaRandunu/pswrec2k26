# PSWRec (Beauty) Ablations & Starting Point

## Starting configuration (Beauty)
- **Base config**: `configs/beauty/beauty_PSWRec.yaml`
- **Suggested initial values**:
  - `MAX_ITEM_LIST_LENGTH: 50`
  - `hidden_size: 64`, `n_heads: 2`, `n_layers: 2`
  - `band_kernel_sizes: [3, 7, 15, 31]`
  - `band_dilations: [1, 1, 2, 4]` (more “wavelet-like” multi-resolution)
  - `phase_bias_scale: 0.1`
  - `hidden_dropout_prob: 0.5`, `attn_dropout_prob: 0.5`

## Key ablations (what to run)
### 1) Remove phase term (PSWRec → SASRec-like backbone)
- **Change**: set `phase_bias_scale: 0.0`
- **Expectation**: drops gains on users/items with periodicity; similar to standard causal attention.

### 2) Single-band vs multi-band
- **Single band**:
  - `band_kernel_sizes: [15]`
  - `band_dilations: [1]`
- **Multi-band (default)**:
  - `band_kernel_sizes: [3, 7, 15, 31]`
  - `band_dilations: [1, 1, 2, 4]`
- **Expectation**: multi-band helps mixed behaviors (short bursts + longer habits).

### 3) “Local TF” vs “more global-ish”
- **Local emphasis**: keep smaller kernels and modest dilations.
- **More global-ish**: use larger kernels/dilations (e.g. `[7, 15, 31, 63]` with `[1, 2, 4, 8]`).
- **Expectation**: too-global can blur phase; may hurt when behavior changes mid-history.

### 4) Phase smoothness regularization (usually off)
- **Enable**:
  - `phase_aux: True`
  - `phase_aux_weight: 0.01` (start small)
- **Expectation**: can help on very noisy sequences; can hurt if user intent shifts rapidly.

## Failure modes (what to watch for)
- **Highly non-periodic users**: phase bias may add noise; mitigate with smaller `phase_bias_scale` or higher dropout.
- **Very short sequences**: phase estimates are unreliable; multi-band may not help. Consider fewer bands for such datasets.
- **Very long max length**: attention cost dominates; PSWRec adds only a small conv overhead, but \(O(L^2)\) still rules.

