#!/bin/bash
# ============================================================================
# V15 Receptive-Field Tuning — Submit 4 jobs (tight + ultra-tight × paw 0.05, 0.1)
# ============================================================================
# Gadi may not allow "qsub -v"; we inline job parameters into a generated PBS script.
# From repo root:  bash experiment/wearec-pswrec/tuning_v13_lastfm/submit_v15_receptive.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PBS_TEMPLATE="${SCRIPT_DIR}/pbs/pswrecv13_v15_receptive_lastfm.pbs"
cd "${REPO_ROOT}"

echo "===== V15 Receptive-Field (V13, lr=0.001, dropout 0.1/0.2, phase_aux) ====="

submit_one() {
    local run_name="$1"
    local kernels="$2"
    local dilations="$3"
    local nheads="$4"
    local paw="$5"
    local hidden_size="${6:-64}"
    local tmp_pbs
    tmp_pbs=$(mktemp "${TMPDIR:-/tmp}/v15_${run_name}.pbs.XXXXXX")
    {
        sed -n '1,22p' "${PBS_TEMPLATE}"
        echo "set -e"
        echo 'cd "$PBS_O_WORKDIR"'
        echo "TUNE_RUN_NAME=${run_name}"
        echo "TUNE_KERNELS=${kernels}"
        echo "TUNE_DILATIONS=${dilations}"
        echo "TUNE_NHEADS=${nheads}"
        echo "TUNE_PAW=${paw}"
        echo "TUNE_HIDDEN_SIZE=${hidden_size}"
        echo "export TUNE_RUN_NAME TUNE_KERNELS TUNE_DILATIONS TUNE_NHEADS TUNE_PAW TUNE_HIDDEN_SIZE"
        echo ""
        sed -n '25,$p' "${PBS_TEMPLATE}"
    } > "${tmp_pbs}"
    qsub -N "${run_name}" "${tmp_pbs}"
    rm -f "${tmp_pbs}"
}

# Tight: kernels 3 5 7, dilations 1 2 4, n_heads=3; hidden_size=66 (64 not divisible by 3)
echo "Submitting: v15lastfm_tight_paw0p05  (kernels=3 5 7, n_heads=3, hidden=66, paw=0.05)"
submit_one "v15lastfm_tight_paw0p05" "3,5,7" "1,2,4" "3" "0.05" "66"

echo "Submitting: v15lastfm_tight_paw0p1  (kernels=3 5 7, n_heads=3, hidden=66, paw=0.1)"
submit_one "v15lastfm_tight_paw0p1" "3,5,7" "1,2,4" "3" "0.1" "66"

# Ultra-tight: kernels 3 7, dilations 1 2, n_heads=2; hidden_size=64
echo "Submitting: v15lastfm_ultratight_paw0p05  (kernels=3 7, n_heads=2, paw=0.05)"
submit_one "v15lastfm_ultratight_paw0p05" "3,7" "1,2" "2" "0.05"

echo "Submitting: v15lastfm_ultratight_paw0p1  (kernels=3 7, n_heads=2, paw=0.1)"
submit_one "v15lastfm_ultratight_paw0p1" "3,7" "1,2" "2" "0.1"

echo ""
echo "V15 receptive-field submitted (4 jobs). Collect with: bash experiment/wearec-pswrec/tuning_v13_lastfm/collect_results.sh"
