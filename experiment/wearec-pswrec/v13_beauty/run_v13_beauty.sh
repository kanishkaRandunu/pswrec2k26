#!/bin/bash
# ============================================================================
# PSWRecV13 on Beauty — submit single job (v13lastfm_s2_d0p5_h4 params)
# ============================================================================
# Run from repo root:
#   bash experiment/wearec-pswrec/v13_beauty/run_v13_beauty.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PBS_SCRIPT="experiment/wearec-pswrec/pbs/pswrecv13_wof_beauty.pbs"

echo "Submitting PSWRecV13 Beauty (params from v13lastfm_s2_d0p5_h4)..."
qsub "${PBS_SCRIPT}"
echo "Done. Check queue: qstat -u \$USER"
