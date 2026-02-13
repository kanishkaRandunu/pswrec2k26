#!/bin/bash
# ============================================================================
# PSWRecV5 (WOF) Hyperparameter Tuning — Job Launcher
# ============================================================================
# Submit from repo root:
#   bash experiment/wearec-pswrec/tuning/submit_tuning.sh [stage1|stage2|all]
#
# Stage 1 — LR sweep (5 jobs):
#   LR ∈ {0.0001, 0.0003, 0.0005, 0.001, 0.003}
#   dropout = 0.5, n_heads = 2  (baseline architecture)
#
# Stage 2 — dropout × n_heads grid (9 jobs, pick best LR from stage 1):
#   dropout ∈ {0.2, 0.3, 0.5}
#   n_heads ∈ {2, 4, 8}
#   LR = <best from stage 1>  (edit BEST_LR below after stage 1)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_TEMPLATE="${SCRIPT_DIR}/pswrecv5_tune.pbs"

# ---- Edit this after Stage 1 completes ----
BEST_LR="0.003"
# -------------------------------------------

STAGE="${1:-all}"

submit_job() {
    local lr="$1"
    local dropout="$2"
    local nheads="$3"
    local run_name="$4"

    echo "Submitting: ${run_name}  (lr=${lr}, dropout=${dropout}, nheads=${nheads})"
    qsub \
        -N "${run_name}" \
        -v "TUNE_LR=${lr},TUNE_DROPOUT=${dropout},TUNE_NHEADS=${nheads},TUNE_RUN_NAME=${run_name}" \
        "${PBS_TEMPLATE}"
}

# ============================================================================
# Stage 1: Learning-rate sweep
# ============================================================================
submit_stage1() {
    echo "===== Stage 1: LR sweep (dropout=0.5, nheads=2) ====="
    for lr in 0.0001 0.0003 0.0005 0.001 0.003; do
        tag=$(echo "${lr}" | tr '.' 'p')
        submit_job "${lr}" 0.5 2 "tune_s1_lr${tag}"
    done
    echo ""
    echo "Stage 1 submitted (5 jobs)."
    echo "After completion, check results and set BEST_LR in this script."
}

# ============================================================================
# Stage 2: Dropout × n_heads grid (with best LR from stage 1)
# ============================================================================
submit_stage2() {
    echo "===== Stage 2: dropout × n_heads grid (lr=${BEST_LR}) ====="
    for dropout in 0.2 0.3 0.5; do
        for nheads in 2 4 8; do
            dtag=$(echo "${dropout}" | tr '.' 'p')
            submit_job "${BEST_LR}" "${dropout}" "${nheads}" "tune_s2_d${dtag}_h${nheads}"
        done
    done
    echo ""
    echo "Stage 2 submitted (9 jobs)."
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
    all)
        submit_stage1
        echo ""
        echo "WARNING: Stage 2 is also being submitted with BEST_LR=${BEST_LR}."
        echo "         If you haven't updated BEST_LR yet, cancel stage 2 jobs."
        echo ""
        submit_stage2
        ;;
    *)
        echo "Usage: $0 [stage1|stage2|all]"
        exit 1
        ;;
esac
