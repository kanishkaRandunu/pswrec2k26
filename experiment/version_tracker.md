# Version tracker — best performing model and tests

## Best performing version: **PSWRecV13withoutphaserot (v13 params)**

- **Model:** `experiment/wearec-pswrec/model/pswrecv13withoutphaserot_wof.py`  
  - No RoPE; amplitude modulation on Q/K; additive residual phase bias (no phase rotation).  
  - Stable init: gamma -3, phase_bias softplus(-5) ≈ 0.007; bounded scale (tanh), magnitude normalized.
- **Config:** `experiment/wearec-pswrec/configs/pswrecv13withoutphaserot_wof_lastfm_v13params.yaml`  
  - Same training setup as V13 best (v13lastfm_s1_lr0p001) but with `model_type: pswrecv13withoutphaserot`.

### V13 params (hyperparameters)

| Param | Value |
|-------|--------|
| num_attention_heads | 4 |
| n_bands | 4 |
| band_kernel_sizes | [3, 7, 15, 31] |
| band_dilations | [1, 2, 4, 8] |
| hidden_dropout_prob / attention_probs_dropout_prob | 0.5 |
| phase_aux | true |
| phase_aux_weight | 0.05 |
| phase_bias_init | -5.0 |
| lr | 0.001 |
| hidden_size | 64 |
| num_hidden_layers | 2 |
| inner_size | 256 |
| max_seq_length | 50 |
| batch_size | 256 |
| epochs | 200 |
| patience | 10 |
| seed | 42 |

### Best results (test set, best-validation checkpoint)

**LastFM**

| Metric | Value |
|--------|--------|
| HR@5 | 0.0532 |
| NDCG@5 | 0.0368 |
| HR@10 | 0.0743 |
| **NDCG@10** | **0.0434** |
| HR@20 | 0.1037 |
| NDCG@20 | 0.0507 |

- Job: 160932802. Logs: `v13nophase_v13params.o160932802`, `v13withoutphaserot_lastfm_v13params_live_160932802.gadi-pbs.log`.
- Gap vs WEARec Official: NDCG@10 −0.0006 (−1.4%); WEARec still leads on HR@10 (0.0817 vs 0.0743) and HR@20.

**Beauty**

- Same v13 params; config/PBS: `configs/pswrecv13withoutphaserot_wof_beauty_v13params.yaml`, `pbs/pswrecv13withoutphaserot_wof_beauty_v13params.pbs` (e.g. job v13nophase_beauty).

---

## V1111 — Hyperpersonalized Phase Filter Bank

- **Model:** `experiment/wearec-pswrec/model/pswrecv1111_wof.py`  
  - **HyperpersonalizedPhaseFilterBank:** context-aware modulator; sequence mean → Δmag, Δφ per band; `mag_personalized = mag_base * (1 + softplus(Δmag))`; `φ_personalized = φ_base + Δφ`.  
  - **HyperpersonalizedPSWRecWOFModel:** same encoder stack as V13 (AM residual phase bias, no RoPE); **no phase_aux** so the modulator can create strong transient magnitude spikes for @10 metrics.
- **Config:** `experiment/wearec-pswrec/configs/pswrecv1111_wof_lastfm.yaml`
- **PBS:** `experiment/wearec-pswrec/pbs/pswrecv1111_wof_lastfm.pbs`
- **Rationale:** User-specific adaptivity (bursty vs slow users); hyper-sensitivity to last 1–3 items for @10.

### V1111 params (LastFM; same as V13 where applicable, no phase_aux)

| Param | Value |
|-------|--------|
| num_attention_heads | 4 |
| n_bands | 4 |
| band_kernel_sizes | [3, 7, 15, 31] |
| band_dilations | [1, 2, 4, 8] |
| phase_aux | false (not used) |
| phase_bias_init | -5.0 |
| lr | 0.001 |
| hidden_size | 64 |
| num_hidden_layers | 2 |
| inner_size | 256 |
| max_seq_length | 50 |
| batch_size | 256 |
| epochs | 200 |
| patience | 10 |
| seed | 42 |

### LastFM (V1111)

- **Results:** To be run (train_name `v1111_lastfm`; output `tuning_v1111_lastfm/output`, log `log/pswrecv1111`).

---

## VMamba — Phase-Modulated Mamba (PM-SSD)

- **Model:** `experiment/wearec-pswrec/model/pswrecvmamba_wof.py`  
  - **LocalHyperpersonalizedPhaseFilterBank:** time-aware 1D Conv modulator; per-step Δmag, Δφ; returns cos_phi, sin_phi, mag_personalized, delta_mag.  
  - **PhaseModulatedMambaBlock:** pure PyTorch Mamba-2-style selective SSM; dt scaled by (1 + softplus(delta_mag_mean)); phase (cos/sin) injected into SSM C matrix.  
  - **HyperpersonalizedPMSSDModel:** replaces self-attention with O(L) sequential scan; causal by construction; no phase_aux.
- **Config:** `experiment/wearec-pswrec/configs/pswrecvmamba_wof_lastfm.yaml`
- **PBS:** `experiment/wearec-pswrec/pbs/pswrecvmamba_wof_lastfm.pbs`
- **Rationale:** Integrates Mamba concepts; transient step modulation + phase alignment in SSM for LastFM @10.

### VMamba params (LastFM)

| Param | Value |
|-------|--------|
| hidden_size | 64 |
| num_hidden_layers | 2 |
| n_bands | 4 |
| band_kernel_sizes | [3, 7, 15, 31] |
| band_dilations | [1, 2, 4, 8] |
| hidden_dropout_prob | 0.5 |
| max_seq_length | 50 |
| lr | 0.001 |
| batch_size | 256 |
| epochs | 200 |
| patience | 10 |
| seed | 42 |

### LastFM (VMamba)

- **Results:** To be run (train_name `vmamba_lastfm`; output `tuning_vmamba_lastfm/output`, log `log/pswrecvmamba`).

---

## What we tested in this version

Experiments and variants built on PSWRecV13withoutphaserot (v13 params):

1. **Phase bias strength (Option A)**  
   - Configs: `phase_bias_init` = -4.5 (pb4p5), -4.0 (pb4p0).  
   - Goal: close the 0.0006 NDCG@10 gap to WEARec.  
   - Configs: `pswrecv13withoutphaserot_wof_lastfm_phasebias_4p5.yaml` / `_4p0.yaml` and corresponding PBS.

2. **Value Amplitude Modulation (VAM)**  
   - Model: `pswrecv13vam_wof.py` — same as v13withoutphaserot but scales V by the same magnitude as Q/K (`v = v * scale`).  
   - Goal: improve HR by amplifying payload (WEARec-style).  
   - Config/PBS: `pswrecv13vam_wof_lastfm_v13params.yaml` / `.pbs`, train_name `v13vam_lastfm_v13params`.

3. **Soft magnitude gate (softmag)**  
   - Branch: `v13norot_softmag_gate`. In `LocalPhaseFilterBankV13NoRot`: soft gating of phase by magnitude (`gate = sigmoid((mag - tau_eff)/temp_eff)`), phase only (mag unchanged).  
   - Grid: mag_tau ∈ {0.2, 0.5, 1.0}, mag_temp ∈ {0.25, 0.5, 1.0}, mag_gate_mode = mean.  
   - LastFM: 9 runs; best test NDCG@10 0.0426 (tau=0.5, temp=0.5); all below v13 params (0.0434) because gating attenuates phase.  
   - Experiment files: `experiment/pswrecv13withoutphaserot/` (configs, pbs, results.md).

4. **Beauty with v13 params**  
   - Same architecture and v13 params on Beauty dataset.  
   - YAML/PBS: `pswrecv13withoutphaserot_wof_beauty_v13params.yaml` / `.pbs`.

5. **Checkpoint verification (Step A)**  
   - Before final test, log checkpoint path, file mtime, and size; load with `map_location=trainer.device`.  
   - In `experiment/wearec-pswrec/main.py` (post–early stopping).

6. **V1111 — Hyperpersonalized Phase Filter Bank**  
   - Model: `pswrecv1111_wof.py` (modulator: sequence mean → Δmag, Δφ per band; no phase_aux).  
   - Config/PBS: `pswrecv1111_wof_lastfm.yaml`, `pswrecv1111_wof_lastfm.pbs`; train_name `v1111_lastfm`.  
   - LastFM: to be run.

7. **VMamba — Phase-Modulated Mamba (PM-SSD)**  
   - Model: `pswrecvmamba_wof.py` (local hyperpersonalized phase filter + Mamba-2 selective SSM; phase/mag inject into dt and C).  
   - Config/PBS: `pswrecvmamba_wof_lastfm.yaml`, `pswrecvmamba_wof_lastfm.pbs`; train_name `vmamba_lastfm`.  
   - LastFM: to be run.

---

## Summary

- **Best model:** PSWRecV13withoutphaserot with v13 params; best reported test NDCG@10 on LastFM is **0.0434** (job 160932802).  
- **Variants tested:** phase bias -4.5/-4.0, VAM (Q/K/V magnitude scaling), softmag gate grid on LastFM, Beauty v13 params, checkpoint logging for correct best-checkpoint evaluation, V1111 (Hyperpersonalized Phase Filter Bank), and VMamba (PM-SSD / Mamba-2 selective SSM, LastFM to be run).
