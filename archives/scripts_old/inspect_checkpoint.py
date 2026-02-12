#!/usr/bin/env python3
"""
Load a RecBole checkpoint (.pth) and print its contents and/or the model structure.

Usage (from TRACT root):
  python scripts/inspect_checkpoint.py outputs/saved/SASRec-Feb-02-2026_14-11-38.pth
  python scripts/inspect_checkpoint.py outputs/saved/SASRec-Feb-02-2026_14-11-38.pth --print-model
"""

import argparse
import sys
from pathlib import Path

# Allow running from TRACT root; RecBole may be in subdir
TRACT_ROOT = Path(__file__).resolve().parent.parent
if str(TRACT_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACT_ROOT))

import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect RecBole checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to .pth file")
    parser.add_argument(
        "--print-model",
        action="store_true",
        help="Rebuild model from config and print structure (needs RecBole + config/dataset)",
    )
    args = parser.parse_args()
    path = Path(args.checkpoint)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    print("=" * 60)
    print("Loading checkpoint:", path)
    print("=" * 60)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Top-level keys
    print("\nTop-level keys:", list(checkpoint.keys()))

    # Config (summary)
    if "config" in checkpoint:
        config = checkpoint["config"]
        print("\n--- Config (model/dataset/size) ---")
        for k in ["model", "dataset", "hidden_size", "embedding_size", "n_heads", "inner_size", "MAX_ITEM_LIST_LENGTH"]:
            if k in config:
                print(f"  {k}: {config[k]}")
        print("  ... (full config has many more keys)")

    # Training info
    for k in ["epoch", "cur_step", "best_valid_score"]:
        if k in checkpoint:
            print(f"\n{k}: {checkpoint[k]}")

    # State dict: parameter names and shapes
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        print("\n--- state_dict (parameter shapes) ---")
        for name, tensor in state.items():
            print(f"  {name}: {tuple(tensor.shape)}")

    # Optionally rebuild full model and print it (requires RecBole + data_path/dataset)
    if args.print_model and "config" in checkpoint:
        try:
            from recbole.config import Config
            from recbole.data import create_dataset, data_preparation
            from recbole.utils import init_logger, get_model, init_seed, get_logger
            config_dict = checkpoint["config"]
            # Ensure data_path and device are set for loading
            if "data_path" not in config_dict or not config_dict["data_path"]:
                config_dict["data_path"] = str(TRACT_ROOT / "configs")
            config = Config(config_dict=config_dict)
            init_logger(config)
            logger = get_logger()
            init_seed(config["seed"], config["reproducibility"])
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)
            model = get_model(config["model"])(config, train_data._dataset)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.load_other_parameter(checkpoint.get("other_parameter"))
            print("\n--- Model structure (after load) ---")
            print(model)
        except Exception as e:
            print("\n--print-model failed (need valid data_path/dataset):", e)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
