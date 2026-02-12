#!/usr/bin/env python3
"""
Evaluate a saved RecBole checkpoint on the test set only (no training).

Use when a job hit walltime before the final test evaluation.

Examples (from TRACT repo root):

  # Using eval YAML (for PBS / same config, only checkpoint changes per model):
  python scripts/evaluate_checkpoint.py --config_files configs/ml-1m_eval_SASRec.yaml

  # Or pass checkpoint path directly:
  python scripts/evaluate_checkpoint.py outputs_saved_sasrec_ml1m/SASRec-Feb-05-2026_08-42-46.pth

  # With Hit/NDCG/MRR @1, @5, @10 (overrides config topk):
  python scripts/evaluate_checkpoint.py outputs/sasrec_ml1m/SASRec-Feb-07-2026_07-32-42.pth --topk 1,5,10
"""

import argparse
import sys
import yaml
from pathlib import Path

# Run from repo root so RecBole and configs are on the path
sys.path.insert(0, ".")

# NumPy 2.0+ removed np.float; RecBole's evaluator uses it (metrics.py)
import numpy as np
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]

import torch

# PyTorch 2.6+ defaults weights_only=True; RecBole checkpoints include config etc., so need weights_only=False
_orig_torch_load = torch.load
def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    if not torch.cuda.is_available() and "map_location" not in kwargs:
        kwargs["map_location"] = torch.device("cpu")
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.quick_start import load_data_and_model
from recbole.utils import get_trainer, init_logger, init_seed, set_color, get_model
from logging import getLogger


def _load_checkpoint_path_from_config(config_file_list):
    """Merge YAML config files and return checkpoint_file. First file wins for checkpoint_file."""
    if not config_file_list:
        return None
    merged = {}
    for p in config_file_list:
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(path) as f:
            merged.update(yaml.safe_load(f) or {})
    return merged.get("checkpoint_file")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved RecBole model on the test set.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Path to the .pth checkpoint (or use --config_files).",
    )
    parser.add_argument(
        "--config_files",
        type=str,
        default=None,
        help="Eval config YAML(s) containing checkpoint_file (e.g. configs/ml-1m_eval_SASRec.yaml).",
    )
    parser.add_argument("--show_progress", action="store_true", help="Show evaluation progress bar.")
    parser.add_argument(
        "--topk",
        type=str,
        default=None,
        help='Comma-separated top-k values for metrics (e.g. "1,5,10"). Overrides config; use to get Hit@1, Hit@5, NDCG@1, etc.',
    )
    args = parser.parse_args()

    if args.config_files:
        config_list = [s.strip() for s in args.config_files.split() if s.strip()]
        checkpoint_path = _load_checkpoint_path_from_config(config_list)
        if not checkpoint_path:
            raise ValueError("config_files must contain 'checkpoint_file'.")
    else:
        checkpoint_path = args.checkpoint or "outputs_saved_sasrec_ml1m/SASRec-Feb-05-2026_08-42-46.pth"

    logger = getLogger()
    logger.info(set_color("Loading checkpoint and data", "green") + f": {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    config = checkpoint["config"]
    if not torch.cuda.is_available():
        config["device"] = "cpu"
    if args.topk:
        topk_list = [int(k.strip()) for k in args.topk.split(",") if k.strip()]
        if topk_list:
            config["topk"] = sorted(topk_list)
            logger.info(set_color("Overriding topk", "green") + f": {config['topk']}")
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    logger.info(set_color("Evaluating on test set", "green"))
    test_result = trainer.evaluate(
        test_data, load_best_model=False, show_progress=args.show_progress
    )

    logger.info(set_color("test result", "yellow") + f": {test_result}")
    print("\nTest metrics:", test_result)


if __name__ == "__main__":
    main()
