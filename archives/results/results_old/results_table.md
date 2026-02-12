# Results table

Metrics: Hit@1, Hit@5, Hit@10, NDCG@1, NDCG@5, NDCG@10, MRR@1, MRR@5, MRR@10. Fill in values as runs complete. MovieLens-1M @1 and @5 from [ml1m_metrics_at_1_5_10.md](ml1m_metrics_at_1_5_10.md). Amazon Beauty @1 and @5 from [beauty_metrics_at_1_5_10.md](beauty_metrics_at_1_5_10.md). Steam @1 and @5 from [steam_metrics_at_1_5_10.md](steam_metrics_at_1_5_10.md) (run script to generate).

**MovieLens-1M:** Best model is TRM4Rec S4 B2. Improvement over previous best (LightSANs): NDCG@10 +4.1%, Hit@10 +2.9%, MRR@10 +4.8% (see [benchmark_table_ml1m.md](benchmark_table_ml1m.md) for full comparison).

<div style="overflow-x: auto;">

*Best per metric: **bold**. Second best: asterisk (*).*

<table>
<thead>
<tr><th>Dataset</th><th>Metric</th><th>SASRec</th><th>BERT4Rec</th><th>LightSANs</th><th>TRM4Rec S4</th><th>TRM4Rec S8</th><th>TRM4Rec S4 B2</th><th>TRM4RecFullBP S4</th><th>TRM4RecFullBP S4 B2</th><th>FEARec</th><th>TRM4RecPos FullBP B1</th></tr>
</thead>
<tbody>
<tr><td>MovieLens-1M</td><td>hit@1</td><td>0.0690</td><td>0.0629</td><td>0.0732*</td><td>0.0717</td><td>0.0725</td><td><strong>0.0760</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>MovieLens-1M</td><td>hit@5</td><td>0.2015</td><td>0.1952</td><td>0.2073*</td><td>0.2050</td><td>0.2008</td><td><strong>0.2205</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>MovieLens-1M</td><td>hit@10</td><td>0.2952</td><td>0.2858</td><td>0.299*</td><td>0.2909</td><td>0.2922</td><td><strong>0.3076</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>MovieLens-1M</td><td>ndcg@1</td><td>0.0690</td><td>0.0629</td><td>0.0732*</td><td>0.0717</td><td>0.0725</td><td><strong>0.0760</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>MovieLens-1M</td><td>ndcg@5</td><td>0.1361</td><td>0.1292</td><td>0.1418*</td><td>0.1395</td><td>0.1378</td><td><strong>0.1503</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>MovieLens-1M</td><td>ndcg@10</td><td>0.1662</td><td>0.1583</td><td>0.1713*</td><td>0.1672</td><td>0.1672</td><td><strong>0.1784</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>MovieLens-1M</td><td>mrr@1</td><td>0.0690</td><td>0.0629</td><td>0.0732*</td><td>0.0717</td><td>0.0725</td><td><strong>0.0760</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>MovieLens-1M</td><td>mrr@5</td><td>0.1146</td><td>0.1076</td><td>0.1203*</td><td>0.1180</td><td>0.1171</td><td><strong>0.1272</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>MovieLens-1M</td><td>mrr@10</td><td>0.127</td><td>0.1196</td><td>0.1324*</td><td>0.1294</td><td>0.1291</td><td><strong>0.1387</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>Amazon Beauty</td><td>hit@1</td><td>0.0126</td><td>0.0072</td><td><strong>0.0158</strong></td><td>0.0136*</td><td>—</td><td>0.0118</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>Amazon Beauty</td><td>hit@5</td><td>0.0497</td><td>0.0242</td><td>0.0502*</td><td><strong>0.0523</strong></td><td>—</td><td>0.0495</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>Amazon Beauty</td><td>hit@10</td><td>0.0736</td><td>0.0404</td><td>0.0753</td><td>0.0769</td><td>—</td><td>0.0780</td><td>0.0781</td><td>0.0797</td><td><strong>0.0854</strong></td><td>0.0808*</td></tr>
<tr><td>Amazon Beauty</td><td>ndcg@1</td><td>0.0126</td><td>0.0072</td><td><strong>0.0158</strong></td><td>0.0136*</td><td>—</td><td>0.0118</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>Amazon Beauty</td><td>ndcg@5</td><td>0.0315</td><td>0.0158</td><td>0.0335*</td><td><strong>0.0336</strong></td><td>—</td><td>0.0311</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>Amazon Beauty</td><td>ndcg@10</td><td>0.0391</td><td>0.0211</td><td>0.0416</td><td>0.0415</td><td>—</td><td>0.0403</td><td>0.0407</td><td>0.0411</td><td><strong>0.0433</strong></td><td>0.0419*</td></tr>
<tr><td>Amazon Beauty</td><td>mrr@1</td><td>0.0126</td><td>0.0072</td><td><strong>0.0158</strong></td><td>0.0136*</td><td>—</td><td>0.0118</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>Amazon Beauty</td><td>mrr@5</td><td>0.0255</td><td>0.0131</td><td><strong>0.0280</strong></td><td>0.0274*</td><td>—</td><td>0.0251</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>Amazon Beauty</td><td>mrr@10</td><td>0.0286</td><td>0.0152</td><td><strong>0.0313</strong></td><td>0.0306*</td><td>—</td><td>0.0288</td><td>0.0294</td><td>0.0293</td><td>0.0304</td><td>0.0301</td></tr>
<tr><td>Steam</td><td>hit@1</td><td>0.0252*</td><td>0.0228</td><td>0.0230</td><td>0.0234</td><td>—</td><td><strong>0.0254</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>Steam</td><td>hit@5</td><td>0.0803</td><td>0.0784</td><td>0.0806*</td><td><strong>0.0844</strong></td><td>—</td><td>0.0798</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>Steam</td><td>hit@10</td><td>0.1383*</td><td>0.132</td><td><strong>0.1405</strong></td><td>0.1376</td><td>—</td><td>0.1349</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>Steam</td><td>ndcg@1</td><td>0.0252*</td><td>0.0228</td><td>0.0230</td><td>0.0234</td><td>—</td><td><strong>0.0254</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>Steam</td><td>ndcg@5</td><td>0.0528*</td><td>0.0505</td><td>0.0516</td><td><strong>0.0538</strong></td><td>—</td><td>0.0525</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>Steam</td><td>ndcg@10</td><td><strong>0.0714</strong></td><td>0.0677</td><td>0.0709*</td><td>0.0709*</td><td>—</td><td>0.0701</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>Steam</td><td>mrr@1</td><td>0.0252*</td><td>0.0228</td><td>0.0230</td><td>0.0234</td><td>—</td><td><strong>0.0254</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>Steam</td><td>mrr@5</td><td><strong>0.0439</strong></td><td>0.0414</td><td>0.0421</td><td>0.0438*</td><td>—</td><td>0.0436</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>Steam</td><td>mrr@10</td><td><strong>0.0514</strong></td><td>0.0484</td><td>0.05</td><td>0.0508*</td><td>—</td><td>0.0507</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
</tbody>
</table>

</div>

## Example (fill from logs)

From a RecBole log line:
```text
hit@10 : 0.2851    ndcg@10 : 0.1537    mrr@10 : 0.1138
```
put each value in the row for that **dataset** and **metric**, under the correct **model** column.

For Hit@1, Hit@5, NDCG@1, NDCG@5, MRR@1, MRR@5: run the benchmark script, then copy from the generated file.

- **MovieLens-1M:** `python3 scripts/run_benchmark_metrics_at_1_5_10.py --benchmark ml1m --out results/ml1m_metrics_at_1_5_10.md` (or `qsub pbs/run_benchmark_metrics_ml1m.pbs` on Gadi).
- **Amazon Beauty:** `python3 scripts/run_benchmark_metrics_at_1_5_10.py --benchmark beauty --out results/beauty_metrics_at_1_5_10.md` (or `qsub pbs/run_benchmark_metrics_beauty.pbs` on Gadi).
- **Steam:** `python3 scripts/run_benchmark_metrics_at_1_5_10.py --benchmark steam --out results/steam_metrics_at_1_5_10.md`.
