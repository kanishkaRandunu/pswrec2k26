#!/usr/bin/env python3
"""
Aggregate per-run results into per-dataset CSV with mean ± std and best-epoch metadata.

Supports two input formats:
  1. JSON files   – structured per-run results (original format)
  2. PBS .o logs  – RecBole training logs from PBS job output

Per experiment/benchmarking_internal.md Section 10.

Usage (from repo root):
  python scripts/aggregate_results.py results/runs/
  python scripts/aggregate_results.py results/runs/ -o results/beauty_aggregated.csv
"""
import argparse
import ast
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_ordered_dict(line: str) -> Optional[Dict[str, float]]:
    """Parse an OrderedDict([('key', value), ...]) string into a dict."""
    m = re.search(r"OrderedDict\(\[(.+)\]\)", line)
    if not m:
        return None
    try:
        pairs = ast.literal_eval("[" + m.group(1) + "]")
        return {k: float(v) for k, v in pairs}
    except (ValueError, SyntaxError):
        return None


def parse_pbs_log(filepath: str) -> Optional[dict]:
    """Extract model, dataset, best epoch, best valid, and test result from a PBS .o log."""
    model = None
    dataset = None
    best_epoch = None
    best_valid = None
    test_result = None

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            # Model and dataset from command line args
            # e.g.  command line args [-m SASRec -d amazon-beauty ...]
            # or    INFO  ['RecBole/run_recbole.py', '-m', 'FEARec', '-d', 'amazon-beauty', ...]
            if model is None:
                m_flag = re.search(r"-m['\s,\]]+\s*['\"]?(\w+)", line)
                if m_flag:
                    model = m_flag.group(1)
                d_flag = re.search(r"-d['\s,\]]+\s*['\"]?([^\s'\",\]]+)", line)
                if d_flag:
                    dataset = d_flag.group(1)

            # Best epoch
            # e.g.  Finished training, best eval result in epoch 32
            ep = re.search(r"best eval result in epoch\s+(\d+)", line)
            if ep:
                best_epoch = int(ep.group(1))

            # Best valid
            if "best valid" in line and "OrderedDict" in line:
                best_valid = parse_ordered_dict(line)

            # Test result
            if "test result" in line and "OrderedDict" in line:
                test_result = parse_ordered_dict(line)

    if test_result is None and best_valid is None:
        return None  # not a valid run log

    return {
        "model": model or "unknown",
        "dataset": dataset or "unknown",
        "best_valid_epoch": best_epoch,
        "best_valid_result": best_valid or {},
        "test_result": test_result or {},
        "_file": filepath,
    }


def load_runs(runs_dir: str) -> List[dict]:
    """Load all run results from directory (JSON files and PBS .o logs)."""
    runs = []
    runs_path = Path(runs_dir)

    # JSON files (original format)
    for p in runs_path.rglob("*.json"):
        if "metadata" in p.name:
            continue
        try:
            with open(p) as f:
                data = json.load(f)
            data["_file"] = str(p)
            runs.append(data)
        except (json.JSONDecodeError, IOError):
            pass

    # PBS .o log files (e.g. sasrec_beauty_a100_ce.o160691774)
    for p in runs_path.rglob("*.o*"):
        if not re.search(r"\.o\d+$", p.name):
            continue
        parsed = parse_pbs_log(str(p))
        if parsed:
            runs.append(parsed)

    return runs


def aggregate_by_dataset_model(runs: List[dict]) -> dict:
    """Group runs by dataset and model."""
    groups = defaultdict(list)
    for r in runs:
        dataset = r.get("dataset", "unknown")
        model = r.get("model", "unknown")
        groups[(dataset, model)].append(r)
    return groups


def compute_stats(values: List[float]) -> Tuple[str, str]:
    """Return (mean, std) as formatted strings."""
    if not values:
        return "", ""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return f"{mean:.4f}", ""
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    return f"{mean:.4f}", f"±{std:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", help="Directory containing per-run JSON files")
    parser.add_argument("-o", "--output", help="Output CSV path")
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    groups = aggregate_by_dataset_model(runs)

    rows = []
    for (dataset, model), group_runs in sorted(groups.items()):
        # Best valid epoch
        epochs = [r["best_valid_epoch"] for r in group_runs if r.get("best_valid_epoch") is not None]
        best_epoch_mean, best_epoch_std = compute_stats(epochs)

        # NDCG@10 test
        ndcg10 = [
            r["test_result"].get("ndcg@10")
            for r in group_runs
            if r.get("test_result") and "ndcg@10" in r["test_result"]
        ]
        ndcg10 = [x for x in ndcg10 if x is not None]
        ndcg_mean, ndcg_std = compute_stats(ndcg10)

        # Hit@10 test
        hit10 = [
            r["test_result"].get("hit@10")
            for r in group_runs
            if r.get("test_result") and "hit@10" in r["test_result"]
        ]
        hit10 = [x for x in hit10 if x is not None]
        hit_mean, hit_std = compute_stats(hit10)

        rows.append({
            "dataset": dataset,
            "model": model,
            "n_runs": len(group_runs),
            "best_valid_epoch_mean": best_epoch_mean,
            "best_valid_epoch_std": best_epoch_std,
            "ndcg@10_mean": ndcg_mean,
            "ndcg@10_std": ndcg_std,
            "hit@10_mean": hit_mean,
            "hit@10_std": hit_std,
        })

    out_path = args.output or "results/aggregated.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
