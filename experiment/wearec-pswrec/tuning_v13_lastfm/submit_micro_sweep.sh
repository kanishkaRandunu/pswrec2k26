#!/bin/bash
# ============================================================================
# PSWRecV13 LastFM — Micro-sweep: break WEARec ceiling (HR@10 0.0817 / NDCG@10 0.0440)
# ============================================================================
# Locked: lr=0.001, inner_size=256, num_attention_heads=4.
# Sweep: dropout ∈ {0.1, 0.2}, phase_aux_weight ∈ {0.1, 0.2}  →  4 jobs.
#
# Submit from repo root:
#   bash experiment/wearec-pswrec/tuning_v13_lastfm/submit_micro_sweep.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_TEMPLATE="${SCRIPT_DIR}/pbs/pswrecv13_micro_lastfm.pbs"

echo "===== Micro-sweep: dropout × phase_aux_weight (lr=0.001, inner=256, n_heads=4) ====="
for dropout in 0.1 0.2; do
    for paw in 0.1 0.2; do
        dtag=$(echo "${dropout}" | tr '.' 'p')
        pawtag=$(echo "${paw}" | tr '.' 'p')
        run_name="v13lastfm_micro_d${dtag}_paw${pawtag}"
        echo "Submitting: ${run_name}  (dropout=${dropout}, phase_aux_weight=${paw})"
        qsub \
            -N "${run_name}" \
            -v "TUNE_DROPOUT=${dropout},TUNE_PHASE_AUX_W=${paw},TUNE_RUN_NAME=${run_name}" \
            "${PBS_TEMPLATE}"
    done
done
echo ""
echo "Micro-sweep submitted (4 jobs). Collect with: bash experiment/wearec-pswrec/tuning_v13_lastfm/collect_results.sh"
