#!/usr/bin/env python3
"""
Export TRACT datasets to WEARec one-line-per-user format using RecBole's
Dataset and the same LS (leave-one-out) split so WEARec official code
uses identical data/split as TRACT baselines.

Run from repo root. Requires RecBole and the same config files as baselines.
Output: WEARec/src/data/{ml100k_TRACT,lastfm_TRACT,beauty_TRACT}.txt

Usage:
  python scripts/export_recbole_to_wearec.py --datasets ml-100k lastfm amazon-beauty
  python scripts/export_recbole_to_wearec.py --dataset ml-100k

When data or benchmark config changes, re-run this so WEARec has TRACT-aligned data.
"""
from __future__ import annotations

import argparse
import os
import sys

# RecBole is under repo root (TRACT/RecBole/)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# After inserting repo root, RecBole package is at RecBole/recbole
_RECBOLE_ROOT = os.path.join(_REPO_ROOT, "RecBole")
if _RECBOLE_ROOT not in sys.path:
    sys.path.insert(0, _RECBOLE_ROOT)

import numpy as np

from recbole.config import Config
from recbole.data import create_dataset


# Dataset key -> (RecBole dataset name, config file, WEARec output name)
DATASET_MAP = {
    "ml-100k": ("ml-100k", "configs/ml-100k/ml100k_SASRec.yaml", "ml100k_TRACT"),
    "lastfm": ("lastfm", "configs/lastfm/lastfm_SASRec.yaml", "lastfm_TRACT"),
    "amazon-beauty": ("amazon-beauty", "configs/beauty/beauty_SASRec.yaml", "beauty_TRACT"),
}


def export_one(dataset_key: str, out_dir: str) -> None:
    """Export one dataset to WEARec .txt format."""
    if dataset_key not in DATASET_MAP:
        raise ValueError(f"Unknown dataset key: {dataset_key}. Use one of {list(DATASET_MAP)}")
    recbole_name, config_path, wearec_name = DATASET_MAP[dataset_key]
    config_path = os.path.join(_REPO_ROOT, config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = Config(
        model="SASRec",
        dataset=recbole_name,
        config_file_list=[os.path.join(_REPO_ROOT, "configs/benchmark_base.yaml"), config_path],
    )
    dataset = create_dataset(config)

    # Same order as RecBole build(): sort by time (ascending). Use sort_values (pandas API);
    # dataset.sort() uses deprecated DataFrame.sort() which was removed in pandas 2.x.
    inter_sorted = dataset.inter_feat.sort_values(by=dataset.time_field, ascending=True)
    uid_col = inter_sorted[dataset.uid_field].to_numpy()
    iid_col = inter_sorted[dataset.iid_field].to_numpy()

    # Group by user (same logic as leave_one_out); order = first occurrence in sorted table
    grouped_index = list(dataset._grouped_index(uid_col))

    lines = []
    for user_idx, indices in enumerate(grouped_index):
        indices = np.asarray(indices)
        item_ids = iid_col[indices]
        # WEARec format: first token = user (we use line index 0,1,2,...); rest = item ids space-separated
        line = f"{user_idx} " + " ".join(str(int(x)) for x in item_ids)
        lines.append(line)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{wearec_name}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path} ({len(lines)} users)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export TRACT datasets to WEARec format (same data/split as RecBole LS)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset key: ml-100k, lastfm, amazon-beauty",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Multiple dataset keys",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: WEARec/src/data)",
    )
    args = parser.parse_args()

    if args.dataset and args.datasets:
        parser.error("Use either --dataset or --datasets, not both")
    if not args.dataset and not args.datasets:
        args.datasets = ["ml-100k", "lastfm", "amazon-beauty"]

    keys = [args.dataset] if args.dataset else args.datasets
    out_dir = args.out_dir or os.path.join(_REPO_ROOT, "WEARec", "src", "data")

    for k in keys:
        export_one(k, out_dir)


if __name__ == "__main__":
    main()
