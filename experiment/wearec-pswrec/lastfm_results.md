# LastFM Dataset — Run Results

Results on the **LastFM** dataset (WEARec repo data & evaluation protocol). Test metrics are from the best-validation checkpoint evaluated on the test set (last-item holdout).

---

## Summary Table

| Model | Job / Log | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|-------|-----------|------|--------|-------|---------|-------|---------|
| **WEARec Official** | 160907220 | 0.0541 | 0.0353 | 0.0817 | **0.0440** | 0.1202 | **0.0537** |
| **PSWRecV5 (WOF, tuned)** | 160925507 | 0.0477 | 0.0345 | 0.0679 | 0.0410 | 0.0945 | 0.0477 |
| **PSWRecV12 (WOF)** | 160925462 | 0.0523 | 0.0352 | 0.0697 | 0.0407 | 0.1046 | 0.0496 |
| **PSWRecV13 (AM-B-RoPE)** | 160925607 | 0.0514 | 0.0347 | 0.0743 | 0.0419 | 0.1037 | 0.0492 |
| **PSWRecV14 (Gamma Wake-Up, silver)** | 160930100 | 0.0468 | 0.0318 | 0.0697 | 0.0392 | 0.1028 | 0.0475 |
| **PSWRecV13withoutphaserot (v13 params)** | 160932802 | 0.0532 | 0.0368 | 0.0743 | 0.0434 | 0.1037 | 0.0507 |

---

## Gap vs WEARec Official

| Model | NDCG@10 | Δ vs WEARec | % |
|-------|---------|-------------|---|
| WEARec Official | 0.0440 | — | — |
| PSWRecV5 (WOF, tuned) | 0.0410 | -0.0030 | -6.8% |
| PSWRecV12 (WOF) | 0.0407 | -0.0033 | -7.5% |
| **PSWRecV13 (AM-B-RoPE)** | **0.0419** | **-0.0021** | **-4.8%** |
| PSWRecV14 (Gamma Wake-Up, silver) | 0.0392 | -0.0048 | -10.9% |
| **PSWRecV13withoutphaserot (v13 params)** | **0.0434** | **-0.0006** | **-1.4%** |

V13withoutphaserot (v13 params) is the closest to WEARec Official (NDCG@10 gap -1.4%). V13 (AM-B-RoPE) is next at -4.8%. WEARec still leads on HR@10 (0.0817 vs 0.0743) and HR@20 (0.1202 vs 0.1037). V14 (gamma init 0.1) silver run: NDCG@10 0.0392, HR@10 0.0697 (best-validation checkpoint from early stopping).

---

## Source Logs

| Model | Log file |
|-------|----------|
| WEARec Official (repo defaults) | `wearec_official_lastfm_repo_live_160907220.gadi-pbs.log` |
| PSWRecV5 (WOF, tuned) | `experiment/V12_and_V13_logs/pswrecv5_wof_lastfm.o160925507` |
| PSWRecV12 (WOF) | `pswrecv12_wof_lastfm.o160925462` |
| PSWRecV13 (AM-B-RoPE) | `pswrecv13_wof_lastfm.o160925607` |
| PSWRecV14 (Gamma Wake-Up, silver) | `v14lastfm_silver.o160930100` / `v14lastfm_silver_live_160930100.gadi-pbs.log` |
| PSWRecV13withoutphaserot (v13 params) | `v13nophase_v13params.o160932802` / `v13withoutphaserot_lastfm_v13params_live_160932802.gadi-pbs.log` |

---

## Notes

- **WEARec Official**: Baseline (WEARec repo defaults: lr=0.001, α=0.3, num_heads=2). Best validation at epoch 30; test run reported as Epoch 0 after loading best checkpoint.
- **PSWRecV5**: Tuned config (lr=0.0003, dropout=0.5, n_heads=2, hidden=64, inner=256, layers=2). Best checkpoint loaded before final test.
- **PSWRecV12**: B-RoPE + sync-gate (LastFM: sync_threshold=0.0, lr=0.001). Early stopping at epoch 39; test on best checkpoint.
- **PSWRecV13**: AM-B-RoPE (amplitude modulation + phase rotation, no sync-gate; lr=0.001). Test on best checkpoint.
- **PSWRecV14**: Same as V13 with gamma initialized at 0.1 (gamma wake-up). Silver run: s3_in256_paw0p05 config; test on best checkpoint (job 160930100).
- **PSWRecV13withoutphaserot (v13 params)**: No RoPE; amplitude modulation + residual phase bias; stable init (gamma -3, phase_bias softplus -5). V13 best params: n_heads=4, n_bands=4, dropout 0.5, phase_aux 0.05, lr=0.001. Test on best checkpoint (job 160932802).

---

## Full Tuning & Micro-Sweep Results (sorted by NDCG@10)

Complete table from `collect_results.sh` (Stage 1–3 + micro-sweep), as of 2026-02-15.

| Run | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|-----|------|--------|-------|---------|-------|---------|
| **WEARec_Official_LastFM** | 0.0541 | 0.0353 | 0.0817 | **0.0440** | 0.1202 | 0.0537 |
| v13withoutphaserot_lastfm_v13params | 0.0532 | 0.0368 | 0.0743 | 0.0434 | 0.1037 | 0.0507 |
| v13lastfm_s3_in256_paw0p05 | 0.0514 | 0.0347 | 0.0743 | 0.0419 | 0.1037 | 0.0492 |
| v13lastfm_s2_d0p5_h4 | 0.0514 | 0.0347 | 0.0743 | 0.0419 | 0.1037 | 0.0492 |
| v13lastfm_s1_lr0p001 | 0.0514 | 0.0347 | 0.0743 | 0.0419 | 0.1037 | 0.0492 |
| v13lastfm_micro_d0p1_paw0p2 | 0.0523 | 0.0361 | 0.0706 | 0.0419 | 0.0982 | 0.0487 |
| v13lastfm_s2_d0p5_h2 | 0.0523 | 0.0358 | 0.0706 | 0.0418 | 0.1083 | 0.0511 |
| v13lastfm_s2_d0p4_h2 | 0.0523 | 0.0338 | 0.0725 | 0.0404 | 0.1064 | 0.0489 |
| v14lastfm_silver | 0.0468 | 0.0318 | 0.0697 | 0.0392 | 0.1028 | 0.0475 |
| v13lastfm_micro_d0p2_paw0p2 | 0.0477 | 0.0345 | 0.0661 | 0.0403 | 0.0972 | 0.0481 |
| v13lastfm_micro_d0p2_paw0p1 | 0.0468 | 0.0338 | 0.0670 | 0.0403 | 0.0963 | 0.0476 |
| v13lastfm_s3_in512_paw0p0 | 0.0486 | 0.0318 | 0.0725 | 0.0396 | 0.1028 | 0.0471 |
| v13lastfm_s1_lr0p0002 | 0.0413 | 0.0308 | 0.0670 | 0.0393 | 0.0954 | 0.0463 |
| v13lastfm_s3_in256_paw0p0 | 0.0459 | 0.0321 | 0.0661 | 0.0387 | 0.1018 | 0.0477 |
| v13lastfm_s1_lr0p0005 | 0.0422 | 0.0313 | 0.0651 | 0.0385 | 0.0945 | 0.0460 |
| v13lastfm_s2_d0p4_h4 | 0.0431 | 0.0320 | 0.0587 | 0.0369 | 0.1000 | 0.0471 |
| v13lastfm_s3_in512_paw0p05 | 0.0440 | 0.0296 | 0.0661 | 0.0366 | 0.1018 | 0.0457 |
| v13lastfm_micro_d0p1_paw0p1 | 0.0486 | 0.0325 | 0.0615 | 0.0366 | 0.0917 | 0.0442 |
| v13lastfm_s1_lr0p0003 | 0.0339 | 0.0240 | 0.0486 | 0.0287 | 0.0725 | 0.0347 |

WEARec Official LastFM NDCG@10 = 0.0440 (target to beat). Best V13 runs tie at NDCG@10 = 0.0419; micro-sweep d0p1_paw0p2 matches that and improves NDCG@5 (0.0361). V14 silver (gamma wake-up): Test Score NDCG@10 = 0.0392, HR@10 = 0.0697 (from `v14lastfm_silver_live_160930100.gadi-pbs.log` / `.o160930100`).

---

*Last updated: 2026-02-15*
