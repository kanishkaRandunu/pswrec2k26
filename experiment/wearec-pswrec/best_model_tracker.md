# PSWRec vs WEARec — Best Model Tracker (Beauty Dataset)

> **Target:** Beat WEARec Official by 3%+ on NDCG@10
> **Baseline (WEARec Official):** NDCG@10 = **0.0599**

---

## Current Best PSWRec Model

| | Value |
|---|---|
| **Model** | PSWRecV5 (WOF, tuned) |
| **Config** | lr=0.0003, dropout=0.5, n_heads=2, hidden=64, inner=256, layers=2 |
| **NDCG@10** | **0.0588** |
| **Gap to WEARec** | -0.0011 (-1.8%) |
| **Target (3%+ over WEARec)** | >= 0.0617 |
| **Remaining to target** | +0.0029 needed |

---

## Full Results Table

| # | Model | Changes over V5 | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 | Job ID |
|---|-------|-----------------|------|--------|-------|---------|-------|---------|--------|
| - | **WEARec Official** | — | 0.0721 | 0.0505 | 0.1016 | **0.0599** | 0.1370 | 0.0688 | 160794226 |
| 1 | PSWRecV5 (WOF, original) | baseline (lr=0.001) | 0.0695 | 0.0496 | 0.0948 | 0.0578 | 0.1299 | 0.0666 | 160796466 |
| 2 | PSWRecV5 (WOF, tuned) | lr=0.0003 | — | — | — | 0.0588 | — | — | 160799619 (tune_s1_lr0p0003) |
| 3 | PSWRecV6 (DPE) | +phase inside encoder loop | 0.0701 | 0.0501 | 0.0961 | 0.0585 | 0.1290 | 0.0668 | 160833188 |
| 4 | PSWRecV7 (DPE+CFPC) | +cross-band coupling matrix | 0.0689 | 0.0495 | 0.0967 | 0.0584 | 0.1310 | 0.0670 | 160833192 |
| 5 | PSWRecV8 (DPE+CFPC+PAV) | +phase-aware value modulation | — | — | — | *running* | — | — | 160835604 |

## Gap to WEARec (NDCG@10)

| Model | NDCG@10 | Δ vs WEARec | % |
|-------|---------|-------------|---|
| WEARec Official | 0.0599 | — | — |
| PSWRecV5 (original) | 0.0578 | -0.0021 | -3.5% |
| PSWRecV5 (tuned) | 0.0588 | -0.0011 | **-1.8%** |
| PSWRecV6 (DPE) | 0.0585 | -0.0014 | -2.3% |
| PSWRecV7 (DPE+CFPC) | 0.0584 | -0.0015 | -2.5% |
| PSWRecV8 (DPE+CFPC+PAV) | *pending* | — | — |

## Key Observations

1. **Tuning helped:** LR sweep (0.001 -> 0.0003) closed the gap from -3.5% to -1.8%.
2. **DPE hurt:** Recomputing phase inside the encoder loop degraded performance. The filterbank extracts cleaner phase from initial embeddings than from already-transformed hidden states.
3. **CFPC made no difference on top of DPE:** V7 ~ V6, confirming DPE is the root problem, not CFPC.
4. **Next steps:** If V8 also underperforms, revisit architecture — try CFPC and PAV on top of V5 (static phase) instead of V6 (dynamic phase).

---

*Last updated: 2026-02-13*
