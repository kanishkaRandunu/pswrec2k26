# PSWRecV13 on Beauty (WEARec data)

Single run of PSWRecV13 (AM-B-RoPE) on the WEARec Beauty dataset using the best LastFM Stage 2 config: **v13lastfm_s2_d0p5_h4**.

## Parameters (from v13lastfm_s2_d0p5_h4)

- **lr** 0.001, **dropout** 0.5, **n_heads** 4, **n_bands** 4  
- **inner_size** 256, **phase_aux_weight** 0.05  
- **max_seq_length** 50, **hidden_size** 64, **num_hidden_layers** 2  
- **batch_size** 256, **epochs** 200, **patience** 10  

See `../configs/pswrecv13_wof_beauty.yaml` for the full reference config.

## How to run

From the **repo root**:

```bash
# Submit PBS job
bash experiment/wearec-pswrec/v13_beauty/run_v13_beauty.sh
```

Or submit the PBS script directly:

```bash
qsub experiment/wearec-pswrec/pbs/pswrecv13_wof_beauty.pbs
```

## Outputs

- **Checkpoint:** `experiment/wearec-pswrec/v13_beauty/output/pswrecv13_wof_beauty.pt`  
- **Training log:** `experiment/wearec-pswrec/v13_beauty/output/pswrecv13_wof_beauty.log`  
- **PBS stdout:** `pswrecv13_wof_beauty.o<jobid>` (in repo root)  
- **Live log:** `pswrecv13_wof_beauty_live_<jobid>.log` (in repo root)
