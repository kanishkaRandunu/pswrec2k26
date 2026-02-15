#!/bin/bash
# ============================================================================
# PSWRecV13 (AM-B-RoPE) Hyperparameter Tuning — Job Launcher (LastFM)
# ============================================================================
# Submit from repo root:
#   bash experiment/wearec-pswrec/tuning_v13_lastfm/submit_tuning.sh [stage1|stage2|stage3|all]
#
# Stage 1 — LR sweep (4 jobs):
#   lr ∈ {0.0002, 0.0003, 0.0005, 0.001}
#   dropout = 0.5, n_heads = 4
#
# Stage 2 — dropout × n_heads grid (4 jobs; set BEST_LR after stage 1):
#   dropout ∈ {0.4, 0.5}, n_heads ∈ {2, 4}
#   lr = BEST_LR
#
# Stage 3 (optional) — inner_size × phase_aux_weight (4 jobs; set best from stage 2):
#   inner_size ∈ {256, 512}, phase_aux_weight ∈ {0.0, 0.05}
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_TEMPLATE="${SCRIPT_DIR}/pbs/pswrecv13_tune_lastfm.pbs"

# ---- Edit after Stage 1 completes ----
BEST_LR="0.001"
# ---- Edit after Stage 2 completes (for stage 3) ----
BEST_DROPOUT="0.5"
BEST_NHEADS="4"
# --------------------------------------

STAGE="${1:-all}"

submit_job() {
    local lr="$1"
    local dropout="$2"
    local nheads="$3"
    local run_name="$4"
    local inner_size="${5:-256}"
    local phase_aux_w="${6:-0.05}"

    echo "Submitting: ${run_name}  (lr=${lr}, dropout=${dropout}, nheads=${nheads}, inner=${inner_size}, paw=${phase_aux_w})"
    qsub \
        -N "${run_name}" \
        -v "TUNE_LR=${lr},TUNE_DROPOUT=${dropout},TUNE_NHEADS=${nheads},TUNE_RUN_NAME=${run_name},TUNE_INNER_SIZE=${inner_size},TUNE_PHASE_AUX_W=${phase_aux_w}" \
        "${PBS_TEMPLATE}"
}

# ============================================================================
# Stage 1: Learning-rate sweep
# ============================================================================
submit_stage1() {
    echo "===== Stage 1: LR sweep (dropout=0.5, nheads=4) ====="
    for lr in 0.0002 0.0003 0.0005 0.001; do
        tag=$(echo "${lr}" | tr '.' 'p')
        submit_job "${lr}" 0.5 4 "v13lastfm_s1_lr${tag}"
    done
    echo ""
    echo "Stage 1 submitted (4 jobs)."
    echo "After completion, run collect_results.sh and set BEST_LR in this script."
}

# ============================================================================
# Stage 2: Dropout × n_heads grid (with best LR from stage 1)
# ============================================================================
submit_stage2() {
    echo "===== Stage 2: dropout × n_heads grid (lr=${BEST_LR}) ====="
    for dropout in 0.4 0.5; do
        for nheads in 2 4; do
            dtag=$(echo "${dropout}" | tr '.' 'p')
            submit_job "${BEST_LR}" "${dropout}" "${nheads}" "v13lastfm_s2_d${dtag}_h${nheads}"
        done
    done
    echo ""
    echo "Stage 2 submitted (4 jobs)."
    echo "After completion, set BEST_DROPOUT and BEST_NHEADS in this script for stage 3."
}

# ============================================================================
# Stage 3 (optional): inner_size × phase_aux_weight
# ============================================================================
submit_stage3() {
    echo "===== Stage 3: inner_size × phase_aux_weight (lr=${BEST_LR}, dropout=${BEST_DROPOUT}, nheads=${BEST_NHEADS}) ====="
    for inner in 256 512; do
        for paw in 0.0 0.05; do
            pawtag=$(echo "${paw}" | tr '.' 'p')
            submit_job "${BEST_LR}" "${BEST_DROPOUT}" "${BEST_NHEADS}" "v13lastfm_s3_in${inner}_paw${pawtag}" "${inner}" "${paw}"
        done
    done
    echo ""
    echo "Stage 3 submitted (4 jobs)."
}

# ============================================================================
# Dispatch
# ============================================================================
case "${STAGE}" in
    stage1|s1|1)
        submit_stage1
        ;;
    stage2|s2|2)
        submit_stage2
        ;;
    stage3|s3|3)
        submit_stage3
        ;;
    all)
        submit_stage1
        echo ""
        echo "WARNING: Stage 2 is also being submitted with BEST_LR=${BEST_LR}."
        echo "         If you haven't updated BEST_LR yet, cancel stage 2 jobs."
        echo ""
        submit_stage2
        echo ""
        echo "WARNING: Stage 3 is also being submitted with BEST_DROPOUT=${BEST_DROPOUT}, BEST_NHEADS=${BEST_NHEADS}."
        echo ""
        submit_stage3
        ;;
    *)
        echo "Usage: $0 [stage1|stage2|stage3|all]"
        exit 1
        ;;
esac
