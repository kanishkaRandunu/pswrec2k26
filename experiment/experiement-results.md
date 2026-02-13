# WEARec Official vs PSWRecV5 Comparison — Beauty Dataset

Same data pipeline (WEARec's leave-last-two-out), evaluation protocol, and metrics.
Data: `WEARec/src/data/Beauty.txt`

## Results

| Model | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|-------|:----:|:------:|:-----:|:-------:|:-----:|:-------:|
| **WEARec Official** | 0.0721 | 0.0505 | 0.1016 | 0.0599 | 0.1370 | 0.0688 |
| **PSWRecV5 (WOF)** | 0.0695 | 0.0496 | 0.0948 | 0.0578 | 0.1299 | 0.0666 |

### Difference vs ours (PSWRecV5 as baseline)

WEARec outperforms ours on all metrics:

| Metric | Δ (WEARec − Ours) | % vs Ours |
|--------|:-----------------:|:---------:|
| HR@5   | +0.0026           | +3.7%     |
| NDCG@5 | +0.0009           | +1.8%     |
| HR@10  | +0.0068           | +7.2%     |
| NDCG@10| +0.0021           | +3.6%     |
| HR@20  | +0.0071           | +5.5%     |
| NDCG@20| +0.0022           | +3.3%     |

## Run details

| Model | Job ID | Log | Checkpoint |
|-------|--------|-----|------------|
| WEARec Official | 160794226 | `wearec_official_beauty.o160794226` | `output/wearec_official_beauty.pt` |
| PSWRecV5 (WOF) | 160796466 | `pswrecv5_wof_beauty.o160796466` | `output/pswrecv5_wof_beauty.pt` |

Both: final **test** scores (after early stopping).
