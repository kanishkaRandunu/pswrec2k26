#!/bin/bash
# ============================================================================
# Collect tuning results from PBS output / live-log files.
#
# Scans for final test scores in all tune_s*_*.o* and tune_s*_*_live_*.log
# files and prints a sorted table.
#
# Usage:
#   bash experiment/wearec-pswrec/tuning/collect_results.sh
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# WEARec baseline for reference
WEAREC_HR5="0.0721"
WEAREC_NDCG10="0.0599"

printf "\n%-30s  %8s  %8s  %8s  %8s  %8s  %8s\n" \
    "Run" "HR@5" "NDCG@5" "HR@10" "NDCG@10" "HR@20" "NDCG@20"
printf "%-30s  %8s  %8s  %8s  %8s  %8s  %8s\n" \
    "------------------------------" "--------" "--------" "--------" "--------" "--------" "--------"

# Reference: WEARec official
printf "%-30s  %8s  %8s  %8s  %8s  %8s  %8s  (baseline)\n" \
    "WEARec Official" "${WEAREC_HR5}" "0.0505" "0.1016" "${WEAREC_NDCG10}" "0.1370" "0.0688"

# Find all tuning output files  (PBS .o files and live logs)
for f in tune_s*.o* tune_s*_live_*.log output/tune_s*.log; do
    [ -f "$f" ] || continue

    # Extract the run name from the filename
    run_name=""
    if [[ "$f" == *.o* ]]; then
        # PBS output: tune_s1_lr0p001.o123456
        run_name="${f%%.*}"
    elif [[ "$f" == *_live_* ]]; then
        # Live log:  tune_s1_lr0p001_live_123456.log
        run_name="${f%%_live_*}"
    elif [[ "$f" == output/*.log ]]; then
        # WEARec trainer log
        run_name="$(basename "${f}" .log)"
    fi

    # Grab the LAST test score line (after "Test Score" banner)
    # Format: {'Epoch': 0, 'HR@5': '0.0695', ...}
    scores=$(grep -A2 "Test Score" "$f" 2>/dev/null \
        | grep "'HR@5'" \
        | tail -1)

    if [ -z "$scores" ]; then
        # Fallback: look for the very last line with HR@5 at epoch 0
        scores=$(grep "'Epoch': 0" "$f" 2>/dev/null | grep "'HR@5'" | tail -1)
    fi

    if [ -z "$scores" ]; then
        printf "%-30s  (no results yet)\n" "${run_name}"
        continue
    fi

    # Parse scores using python for reliability
    read -r hr5 ndcg5 hr10 ndcg10 hr20 ndcg20 <<< $(python3 -c "
import re, sys
s = '''${scores}'''
d = dict(re.findall(r\"'(\\w+@?\\d*)': '([\\d.]+)'\", s))
print(d.get('HR@5','?'), d.get('NDCG@5','?'), d.get('HR@10','?'),
      d.get('NDCG@10','?'), d.get('HR@20','?'), d.get('NDCG@20','?'))
")

    printf "%-30s  %8s  %8s  %8s  %8s  %8s  %8s\n" \
        "${run_name}" "${hr5}" "${ndcg5}" "${hr10}" "${ndcg10}" "${hr20}" "${ndcg20}"

done | sort

echo ""
echo "WEARec baseline NDCG@10 = ${WEAREC_NDCG10}  |  HR@5 = ${WEAREC_HR5}"
echo "Our original  NDCG@10 = 0.0578  |  HR@5 = 0.0695"
echo ""
