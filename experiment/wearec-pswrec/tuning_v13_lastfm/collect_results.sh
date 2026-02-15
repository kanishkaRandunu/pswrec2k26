#!/bin/bash
# ============================================================================
# Collect V13 LastFM tuning results from PBS output / live-log files.
#
# Scans for final test scores in v13lastfm_*.o*, v13lastfm_*_live_*.log,
# and optionally tuning_v13_lastfm/output/*.log. Prints table sorted by
# NDCG@10 descending. Writes results/summary_<date>.txt.
#
# Usage (from repo root):
#   bash experiment/wearec-pswrec/tuning_v13_lastfm/collect_results.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# WEARec Official LastFM baseline (to beat)
WEAREC_NDCG10="0.0440"
WEAREC_HR5="0.0541"
WEAREC_HR10="0.0817"
WEAREC_HR20="0.1202"
WEAREC_NDCG5="0.0353"
WEAREC_NDCG20="0.0537"

RESULTS_DIR="${SCRIPT_DIR}/results"
OUTPUT_DIR="${SCRIPT_DIR}/output"
mkdir -p "${RESULTS_DIR}"

TMPFILE=$(mktemp)
trap "rm -f ${TMPFILE}" EXIT

# Header
printf "\n%-32s  %8s  %8s  %8s  %8s  %8s  %8s\n" \
    "Run" "HR@5" "NDCG@5" "HR@10" "NDCG@10" "HR@20" "NDCG@20"
printf "%-32s  %8s  %8s  %8s  %8s  %8s  %8s\n" \
    "--------------------------------" "--------" "--------" "--------" "--------" "--------" "--------"

# Reference: WEARec Official LastFM
printf "%-32s  %8s  %8s  %8s  %8s  %8s  %8s  (WEARec baseline)\n" \
    "WEARec_Official_LastFM" "${WEAREC_HR5}" "${WEAREC_NDCG5}" "${WEAREC_HR10}" "${WEAREC_NDCG10}" "${WEAREC_HR20}" "${WEAREC_NDCG20}"

# Find all tuning output files (PBS .o in repo root, live logs; include V15 receptive-field)
for f in v13lastfm_*.o* v13lastfm_*_live_*.log v13nophaserot_*.o* v13nophaserot_*_live_*.log v13nophase_*.o* v13nophase_*_live_*.log v13withoutphaserot_*.o* v13withoutphaserot_*_live_*.log v15lastfm_*.o* v15lastfm_*_live_*.log v17lastfm_*.o* v17lastfm_*_live_*.log v20lastfm_*.o* v20lastfm_*_live_*.log v25lastfm_*.o* v25lastfm_*_live_*.log; do
    [ -f "$f" ] || continue
    run_name=""
    if [[ "$f" == *.o* ]]; then
        run_name="${f%%.*}"
    elif [[ "$f" == *_live_*.log ]]; then
        run_name="${f%%_live_*}"
    fi
    [ -n "$run_name" ] || continue

    scores=$(grep -A2 "Test Score" "$f" 2>/dev/null | grep "'HR@5'" | tail -1)
    if [ -z "$scores" ]; then
        scores=$(grep "'Epoch': 0" "$f" 2>/dev/null | grep "'HR@5'" | tail -1)
    fi
    if [ -z "$scores" ]; then
        continue
    fi

    read -r hr5 ndcg5 hr10 ndcg10 hr20 ndcg20 <<< $(python3 -c "
import re
s = '''${scores}'''
d = dict(re.findall(r\"'(\\w+@?\\d*)': '([\\d.]+)'\", s))
print(d.get('HR@5','?'), d.get('NDCG@5','?'), d.get('HR@10','?'),
      d.get('NDCG@10','?'), d.get('HR@20','?'), d.get('NDCG@20','?'))
" 2>/dev/null) || continue

    # Output: run_name ndcg10 hr5 ndcg5 hr10 ndcg10 hr20 ndcg20 (for sort by ndcg10 desc, then dedupe by run_name)
    echo "${run_name} ${ndcg10} ${hr5} ${ndcg5} ${hr10} ${ndcg10} ${hr20} ${ndcg20}"
done >> "${TMPFILE}"

# Optional: scan tuning output/*.log (train_name.log)
for f in "${OUTPUT_DIR}"/v13lastfm_*.log "${OUTPUT_DIR}"/v13withoutphaserot_*.log "${OUTPUT_DIR}"/v15lastfm_*.log "${OUTPUT_DIR}"/v17lastfm_*.log "${OUTPUT_DIR}"/v20lastfm_*.log "${OUTPUT_DIR}"/v25lastfm_*.log; do
    [ -f "$f" ] || continue
    run_name="$(basename "$f" .log)"
    scores=$(grep -A2 "Test Score" "$f" 2>/dev/null | grep "'HR@5'" | tail -1)
    if [ -z "$scores" ]; then
        scores=$(grep "'Epoch': 0" "$f" 2>/dev/null | grep "'HR@5'" | tail -1)
    fi
    if [ -z "$scores" ]; then
        continue
    fi
    read -r hr5 ndcg5 hr10 ndcg10 hr20 ndcg20 <<< $(python3 -c "
import re
s = '''${scores}'''
d = dict(re.findall(r\"'(\\w+@?\\d*)': '([\\d.]+)'\", s))
print(d.get('HR@5','?'), d.get('NDCG@5','?'), d.get('HR@10','?'),
      d.get('NDCG@10','?'), d.get('HR@20','?'), d.get('NDCG@20','?'))
" 2>/dev/null) || continue
    echo "${run_name} ${ndcg10} ${hr5} ${ndcg5} ${hr10} ${ndcg10} ${hr20} ${ndcg20}"
done >> "${TMPFILE}"

# Sort by NDCG@10 descending (column 2), then keep first occurrence per run_name (column 1) so we don't duplicate
sort -t' ' -k2 -rn "${TMPFILE}" 2>/dev/null | awk '!seen[$1]++' | while read -r run_name _ndcg10 hr5 ndcg5 hr10 ndcg10 hr20 ndcg20; do
    printf "%-32s  %8s  %8s  %8s  %8s  %8s  %8s\n" \
        "${run_name}" "${hr5}" "${ndcg5}" "${hr10}" "${ndcg10}" "${hr20}" "${ndcg20}"
done

echo ""
echo "WEARec Official LastFM NDCG@10 = ${WEAREC_NDCG10}  (target to beat)"
echo ""

# Write summary file
SUMMARY_FILE="${RESULTS_DIR}/summary_$(date +%Y%m%d_%H%M).txt"
{
    echo "V13 LastFM tuning results — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Target: NDCG@10 > ${WEAREC_NDCG10} (WEARec Official)"
    echo ""
    printf "%-32s  %8s  %8s  %8s  %8s  %8s  %8s\n" "Run" "HR@5" "NDCG@5" "HR@10" "NDCG@10" "HR@20" "NDCG@20"
    printf "%-32s  %8s  %8s  %8s  %8s  %8s  %8s\n" "--------------------------------" "--------" "--------" "--------" "--------" "--------" "--------"
    printf "%-32s  %8s  %8s  %8s  %8s  %8s  %8s\n" "WEARec_Official_LastFM" "${WEAREC_HR5}" "${WEAREC_NDCG5}" "${WEAREC_HR10}" "${WEAREC_NDCG10}" "${WEAREC_HR20}" "${WEAREC_NDCG20}"
    sort -t' ' -k2 -rn "${TMPFILE}" 2>/dev/null | awk '!seen[$1]++' | while read -r run_name _ndcg10 hr5 ndcg5 hr10 ndcg10 hr20 ndcg20; do
        printf "%-32s  %8s  %8s  %8s  %8s  %8s  %8s\n" "${run_name}" "${hr5}" "${ndcg5}" "${hr10}" "${ndcg10}" "${hr20}" "${ndcg20}"
    done
    echo ""
    best_run=$(sort -t' ' -k2 -rn "${TMPFILE}" 2>/dev/null | awk '!seen[$1]++' | head -1)
    if [ -n "$best_run" ]; then
        read -r best_name best_ndcg10 _ <<< "$best_run"
        echo "Best run: ${best_name}  NDCG@10 = ${best_ndcg10}"
    fi
} > "${SUMMARY_FILE}"
echo "Summary written to ${SUMMARY_FILE}"
