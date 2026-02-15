# PSWRecV13withoutphaserot soft magnitude gate tuning

Soft, scale-aware phase gating in `LocalPhaseFilterBankV13NoRot`: phase (cos_phi, sin_phi) is gated by a sigmoid on magnitude relative to batch mean; magnitude is left intact for AM/mixing.

## Grid

- **mag_tau:** 0.2, 0.5, 1.0  
- **mag_temp:** 0.25, 0.5, 1.0  
- **mag_gate_mode:** mean (only mode implemented)  
- **9 runs per dataset** (Beauty, LastFM). Same v13 params: n_heads=4, n_bands=4, dropout 0.5, phase_aux 0.05, phase_bias_init -5.0, lr=0.001.

## Expected outcomes

- **Beauty:** Gate mean should be noticeably lower than 1 but not collapse to 0 (reduces hurt from random phase injection while keeping expressivity when mag is real).
- **LastFM:** Gate mean higher, closer to 1. If LastFM drops, gate is too aggressive: reduce mag_tau or increase mag_temp.

## Run

From repo root:

```bash
# Beauty
qsub experiment/pswrecv13withoutphaserot/pbs/v13nophase_softmag_beauty_tau0p2_temp0p25.pbs
# ... (9 Beauty PBS)

# LastFM
qsub experiment/pswrecv13withoutphaserot/pbs/v13nophase_softmag_lastfm_tau0p2_temp0p25.pbs
# ... (9 LastFM PBS)
```

Checkpoints: `experiment/pswrecv13withoutphaserot/output/`. Logs: `log/pswrecv13withoutphaserot/` and live logs in run directory.
