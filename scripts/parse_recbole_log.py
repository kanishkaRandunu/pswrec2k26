#!/usr/bin/env python3
"""
Parse RecBole log files to extract best_valid_epoch, best_valid_score, test_result.

Per experiment/benchmarking_internal.md: store best_valid_epoch, best_valid_score,
test_score_at_best_valid for every run.

Usage (from repo root):
  python scripts/parse_recbole_log.py path/to/logfile.log
  python scripts/parse_recbole_log.py path/to/logfile.log -o results/runs/beauty_sasrec_2026_123.json
  python scripts/parse_recbole_log.py path/to/logfile.log -o results/runs/run.json --dataset amazon-beauty --model SASRec
"""
import argparse
import ast
import json
import re
import sys


def parse_log(log_path: str) -> dict:
    """Parse RecBole log and return dict with best_valid_epoch, best_valid_score, test_result."""
    with open(log_path, "r") as f:
        content = f.read()

    result = {
        "best_valid_epoch": None,
        "best_valid_score": None,
        "best_valid_result": None,
        "test_result": None,
    }

    # "Finished training, best eval result in epoch 52"
    m = re.search(r"Finished training, best eval result in epoch\s+(\d+)", content)
    if m:
        result["best_valid_epoch"] = int(m.group(1))

    # "best valid : OrderedDict([('hit@10', 0.1051), ...])"
    m = re.search(r"best valid\s*:\s*OrderedDict\(\[([^\]]+)\]\)", content)
    if m:
        try:
            pairs = ast.literal_eval("[" + m.group(1) + "]")
            result["best_valid_result"] = dict(pairs)
            # best_valid_score = valid_metric value (NDCG@10 typically)
            if result["best_valid_result"]:
                # Use ndcg@10 if present, else first metric
                result["best_valid_score"] = result["best_valid_result"].get(
                    "ndcg@10",
                    next(iter(result["best_valid_result"].values())),
                )
        except (ValueError, SyntaxError):
            pass

    # "test result: OrderedDict([('hit@10', 0.0821), ...])"
    m = re.search(r"test result:\s*OrderedDict\(\[([^\]]+)\]\)", content)
    if m:
        try:
            pairs = ast.literal_eval("[" + m.group(1) + "]")
            result["test_result"] = dict(pairs)
        except (ValueError, SyntaxError):
            pass

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse RecBole log for benchmark metrics")
    parser.add_argument("log_path", help="Path to RecBole log file (.log or PBS .o file)")
    parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
    parser.add_argument("--dataset", help="Dataset name (for run JSON)")
    parser.add_argument("--model", help="Model name (for run JSON)")
    parser.add_argument("--run-id", help="Run ID (for run JSON)")
    args = parser.parse_args()

    result = parse_log(args.log_path)
    if args.dataset:
        result["dataset"] = args.dataset
    if args.model:
        result["model"] = args.model
    if args.run_id:
        result["run_id"] = args.run_id

    out = json.dumps(result, indent=2)

    if args.output:
        import os
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(out)
        print(f"Wrote {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
