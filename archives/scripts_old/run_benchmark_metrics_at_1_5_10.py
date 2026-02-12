#!/usr/bin/env python3
"""
Run test evaluation with topk [1, 5, 10] for benchmark checkpoints and print
a table of Hit@1, Hit@5, NDCG@1, NDCG@5, MRR@1, MRR@5 (and @10).

Run from repo root:
  python scripts/run_benchmark_metrics_at_1_5_10.py
  python scripts/run_benchmark_metrics_at_1_5_10.py --benchmark ml1m --out results/ml1m_metrics_at_1_5_10.md
  python scripts/run_benchmark_metrics_at_1_5_10.py --benchmark beauty --out results/beauty_metrics_at_1_5_10.md
  python scripts/run_benchmark_metrics_at_1_5_10.py --benchmark steam --out results/steam_metrics_at_1_5_10.md
  python scripts/run_benchmark_metrics_at_1_5_10.py --topk 1,5,10 --out results/ml1m_metrics_at_1_5_10.md
  python scripts/run_benchmark_metrics_at_1_5_10.py --topk 1,3,5,10,20 --out results/ml1m_metrics_wide.md
"""

import argparse
import sys
from pathlib import Path

# Use this repo's RecBole (contains TRM4Rec); otherwise venv's recbole is used and TRM4Rec is missing
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "RecBole"))
sys.path.insert(0, str(_repo_root))

import numpy as np
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]

import torch
_orig_torch_load = torch.load
def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    if not torch.cuda.is_available() and "map_location" not in kwargs:
        kwargs["map_location"] = torch.device("cpu")
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load

from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_logger, init_seed, get_model

# Checkpoints from results/benchmark_table_ml1m.md (outputs/)
BENCHMARK_CHECKPOINTS_ML1M = [
    ("SASRec", "outputs/sasrec_ml1m/SASRec-Feb-07-2026_07-32-42.pth"),
    ("BERT4Rec", "outputs/bert4rec_ml1m/BERT4Rec-Feb-07-2026_23-49-04.pth"),
    ("LightSANs", "outputs/lightsans_ml1m/LightSANs-Feb-08-2026_07-18-22.pth"),
    ("TRM4Rec-Vanilla S4", "outputs/trm4rec_ml1m/TRM4Rec-Feb-07-2026_15-58-33.pth"),
    ("TRM4Rec-Vanilla S8", "outputs/trm4rec_ml1m_s8/TRM4Rec-Feb-08-2026_05-04-32.pth"),
    ("TRM4Rec-Vanilla S4 B2", "outputs/trm4rec_ml1m_b2/TRM4Rec-Feb-08-2026_14-02-03.pth"),
]

# Checkpoints from results/benchmark_beauty.md (outputs/)
BENCHMARK_CHECKPOINTS_BEAUTY = [
    ("SASRec", "outputs/sasrec_beauty/SASRec-Feb-09-2026_07-43-11.pth"),
    ("BERT4Rec", "outputs/bert4rec_beauty/BERT4Rec-Feb-09-2026_08-27-00.pth"),
    ("LightSANs", "outputs/lightsans_beauty/LightSANs-Feb-09-2026_08-27-01.pth"),
    ("TRM4Rec-Vanilla S4", "outputs/trm4rec_beauty/TRM4Rec-Feb-09-2026_09-02-35.pth"),
    ("TRM4Rec-Vanilla S4 B2", "outputs/trm4rec_beauty_b2/TRM4Rec-Feb-09-2026_09-15-18.pth"),
]

# Checkpoints from results/benchmark_steam.md (outputs/)
BENCHMARK_CHECKPOINTS_STEAM = [
    ("SASRec", "outputs/sasrec_steam/SASRec-Feb-09-2026_10-45-16.pth"),
    ("BERT4Rec", "outputs/bert4rec_steam/BERT4Rec-Feb-09-2026_11-03-42.pth"),
    ("LightSANs", "outputs/lightsans_steam/LightSANs-Feb-09-2026_11-08-39.pth"),
    ("TRM4Rec S4", "outputs/trm4rec_steam/TRM4Rec-Feb-09-2026_14-09-44.pth"),
    ("TRM4Rec S4 B2", "outputs/trm4rec_steam_b2/TRM4Rec-Feb-09-2026_14-09-44.pth"),
]

BENCHMARK_TITLES = {
    "ml1m": "MovieLens-1M test metrics @1, @5, @10",
    "beauty": "Amazon Beauty test metrics @1, @5, @10",
    "steam": "Steam (duplicate removal) test metrics @1, @5, @10",
}

BENCHMARK_DEFAULT_OUT = {
    "ml1m": "results/ml1m_metrics_at_1_5_10.md",
    "beauty": "results/beauty_metrics_at_1_5_10.md",
    "steam": "results/steam_metrics_at_1_5_10.md",
}

DEFAULT_TOPK = [1, 5, 10]


def evaluate_one(checkpoint_path: str, topk: list, show_progress: bool = False, verbose: bool = False):
    """Load checkpoint, override topk, run test evaluation. Returns dict of metric -> value."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(str(path), weights_only=False, map_location="cpu")
    config = checkpoint["config"]
    if not torch.cuda.is_available():
        config["device"] = "cpu"
    config["topk"] = sorted(topk)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    if not verbose:
        import logging
        logging.getLogger("recbole").setLevel(logging.WARNING)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    result = trainer.evaluate(test_data, load_best_model=False, show_progress=show_progress)
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmark checkpoints with Hit/NDCG/MRR @1, @5, @10.")
    parser.add_argument("--benchmark", type=str, choices=("ml1m", "beauty", "steam"), default="ml1m",
                        help="Which benchmark: ml1m, beauty, or steam (default: ml1m).")
    parser.add_argument("--out", type=str, default=None, help="Write markdown table to this file.")
    parser.add_argument("--show_progress", action="store_true", help="Show progress bar per model.")
    parser.add_argument("--verbose", action="store_true", help="Verbose RecBole logging.")
    parser.add_argument(
        "--topk",
        type=str,
        default="1,5,10",
        help="Comma-separated top-k values (default: 1,5,10). E.g. --topk 1,5,10,20",
    )
    args = parser.parse_args()

    topk_list = [int(k.strip()) for k in args.topk.split(",") if k.strip()]
    if not topk_list:
        topk_list = DEFAULT_TOPK
    else:
        topk_list = sorted(set(topk_list))

    benchmark = args.benchmark
    if benchmark == "ml1m":
        checkpoints = BENCHMARK_CHECKPOINTS_ML1M
    elif benchmark == "beauty":
        checkpoints = BENCHMARK_CHECKPOINTS_BEAUTY
    else:
        checkpoints = BENCHMARK_CHECKPOINTS_STEAM

    out_path_str = args.out
    if out_path_str is None:
        out_path_str = BENCHMARK_DEFAULT_OUT.get(benchmark)
    title = BENCHMARK_TITLES.get(benchmark, f"{benchmark} test metrics @1, @5, @10")

    repo_root = _repo_root
    results = []
    for name, ckpt in checkpoints:
        # Try repo root (script location) then cwd so both local and PBS runs find checkpoints
        path = repo_root / ckpt
        if not path.is_file():
            path = Path.cwd() / ckpt
        if not path.is_file():
            print(f"Skip {name}: checkpoint not found (tried {repo_root / ckpt} and {Path.cwd() / ckpt})", file=sys.stderr)
            results.append((name, None))
            continue
        print(f"Evaluating {name} ...", flush=True)
        try:
            metrics = evaluate_one(str(path), topk_list, show_progress=args.show_progress, verbose=args.verbose)
            results.append((name, metrics))
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            results.append((name, None))

    # Build markdown table (columns driven by --topk)
    headers = ["Model"]
    for metric in ("Hit", "NDCG", "MRR"):
        headers.extend([f"{metric}@{k}" for k in topk_list])
    rows = []
    for name, m in results:
        if m is None:
            rows.append([name] + ["—"] * (len(headers) - 1))
            continue
        # RecBole returns lowercase keys: hit@1, ndcg@5, mrr@10
        row = [name]
        for metric_key in ("hit", "ndcg", "mrr"):
            for k in topk_list:
                row.append(f"{m.get(f'{metric_key}@{k}', 0):.4f}")
        rows.append(row)

    sep = "| " + " | ".join(headers) + " |"
    border = "|" + "|".join(["---"] * len(headers)) + "|"
    lines = [sep, border] + ["| " + " | ".join(r) + " |" for r in rows]
    table = "\n".join(lines)
    print("\n" + table)

    if out_path_str:
        out_path = Path(out_path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(f"# {title}\n\n" + table + "\n", encoding="utf-8")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
