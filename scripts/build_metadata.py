#!/usr/bin/env python3
"""
Build results/metadata.json from config files.

Extracts loss_type, train_neg_sample_args per model per dataset.
Per experiment/benchmarking_internal.md Section 3.

Usage (from repo root):
  python scripts/build_metadata.py
"""
import json
import os

import yaml

CONFIGS = {
    "amazon-beauty": [
        ("GRU4Rec", "configs/beauty/beauty_GRU4Rec.yaml"),
        ("SASRec", "configs/beauty/beauty_SASRec.yaml"),
        ("BERT4Rec", "configs/beauty/beauty_BERT4Rec.yaml"),
        ("FEARec", "configs/beauty/beauty_FEARec.yaml"),
        ("pswrecv4h", "configs/beauty/beauty_pswrecv4h.yaml"),
    ]
}


def main():
    metadata = {}
    for dataset, models in CONFIGS.items():
        metadata[dataset] = {}
        for model_name, config_path in models:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                metadata[dataset][model_name] = {
                    "loss_type": cfg.get("loss_type"),
                    "train_neg_sample_args": cfg.get("train_neg_sample_args"),
                    "config": config_path,
                }
            else:
                metadata[dataset][model_name] = {"config": config_path}

    out_path = "results/metadata.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
