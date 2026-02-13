# WEARec Official vs PSWRecV5 Comparison — Beauty Dataset

Same data pipeline (WEARec's leave-last-two-out), evaluation protocol, and metrics.
Data: `WEARec/src/data/Beauty.txt`

## Results

| Model | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|-------|:----:|:------:|:-----:|:-------:|:-----:|:-------:|
| **WEARec Official** | 0.0721 | 0.0505 | 0.1016 | 0.0599 | 0.1370 | 0.0688 |
| **PSWRecV5 (WOF, original)** | 0.0695 | 0.0496 | 0.0948 | 0.0578 | 0.1299 | 0.0666 |
| **PSWRecV5 (WOF, tuned)** | 0.0694 | 0.0501 | 0.0963 | 0.0588 | 0.1312 | 0.0676 |

### Difference vs ours (PSWRecV5 as baseline)

WEARec outperforms ours on all metrics:

| Metric | Δ (WEARec − PSWRecV5 tuned) | % vs PSWRecV5 tuned |
|--------|:---------------------------:|:--------------------:|
| HR@5   | +0.0027                     | +3.9%                |
| NDCG@5 | +0.0004                     | +0.8%                |
| HR@10  | +0.0053                     | +5.5%                |
| NDCG@10| +0.0011                     | +1.9%                |
| HR@20  | +0.0058                     | +4.4%                |
| NDCG@20| +0.0012                     | +1.8%                |

## Run details

| Model | Job ID | Log | Checkpoint |
|-------|--------|-----|------------|
| WEARec Official | 160794226 | `wearec_official_beauty.o160794226` | `output/wearec_official_beauty.pt` |
| PSWRecV5 (WOF, original) | 160796466 | `pswrecv5_wof_beauty.o160796466` | `output/pswrecv5_wof_beauty.pt` |
| PSWRecV5 (WOF, tuned) | 160799619 | `tune_s1_lr0p0003.o160799619` | `output/tune_s1_lr0p0003.pt` |

Both: final **test** scores (after early stopping).
