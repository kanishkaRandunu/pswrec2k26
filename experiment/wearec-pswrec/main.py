#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment entry-point: WEARec Official vs PSWRec comparison.

This script extends WEARec's ``main.py`` by:
  1. Adding PSWRec V5-V8 WOF models to the model registry.
  2. Injecting PSWRec-specific CLI arguments into argparse.

Everything else (data pipeline, trainer, evaluation) is WEARec's own code.

Supported model_type values:
  - WEARec          (official baseline)
  - pswrecv5_wof    (V5 baseline)
  - pswrecv6_wof    (V5 + Dynamic Phase Evolution)
  - pswrecv7_wof    (V6 + Cross-Frequency Phase Coupling)
  - pswrecv8_wof    (V7 + Phase-Aware Values)

Usage
-----
# WEARec official baseline (Beauty):
python main.py --data_name Beauty --model_type WEARec \
    --lr 0.0005 --alpha 0.2 --num_heads 8 \
    --train_name wearec_official_beauty

# PSWRecV5 on the same playground:
python main.py --data_name Beauty --model_type pswrecv5_wof \
    --lr 0.001 --num_attention_heads 2 --num_hidden_layers 2 \
    --n_bands 4 --band_kernel_sizes 3 7 15 31 \
    --band_dilations 1 2 4 8 \
    --phase_bias_scale 0.1 --phase_gate_scale 1.0 \
    --phase_aux --phase_aux_weight 0.05 \
    --inner_size 256 \
    --train_name pswrecv5_wof_beauty
"""

import os
import sys
import argparse
import importlib.util

# ---------------------------------------------------------------------------
# Put WEARec/src on the import path so we can reuse its modules directly.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WEAREC_SRC = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "WEARec", "src"))
sys.path.insert(0, _WEAREC_SRC)

import torch
import numpy as np

from model import MODEL_DICT                     # WEARec's model registry
from trainers import Trainer                      # WEARec's trainer
from utils import EarlyStopping, check_path, set_seed, set_logger, get_local_time
from dataset import get_seq_dic, get_dataloder, get_rating_matrix

# ---------------------------------------------------------------------------
# Load PSWRec WOF models via importlib so that their
# ``from model._abstract_model import SequentialRecModel`` correctly resolves
# to WEARec's base class (which is on sys.path) without colliding with a
# local ``model/`` package.
# ---------------------------------------------------------------------------

def _load_wof_model(filename: str, class_name: str):
    """Load a single WOF model class from experiment/wearec-pswrec/model/."""
    path = os.path.join(_SCRIPT_DIR, "model", filename)
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)

PSWRecV5WOFModel = _load_wof_model("pswrecv5_wof.py", "PSWRecV5WOFModel")
PSWRecV6WOFModel = _load_wof_model("pswrecv6_wof.py", "PSWRecV6WOFModel")
PSWRecV7WOFModel = _load_wof_model("pswrecv7_wof.py", "PSWRecV7WOFModel")
PSWRecV8WOFModel = _load_wof_model("pswrecv8_wof.py", "PSWRecV8WOFModel")

# ---------------------------------------------------------------------------
# Register all PSWRec WOF models alongside the existing WEARec models.
# ---------------------------------------------------------------------------
MODEL_DICT["pswrecv5_wof"] = PSWRecV5WOFModel
MODEL_DICT["pswrecv6_wof"] = PSWRecV6WOFModel
MODEL_DICT["pswrecv7_wof"] = PSWRecV7WOFModel
MODEL_DICT["pswrecv8_wof"] = PSWRecV8WOFModel


def _build_unified_args() -> argparse.Namespace:
    """Build a unified argument parser that includes both WEARec's base args
    and PSWRecV5-specific args, so all flags are recognized in a single pass.

    WEARec's ``parse_args()`` (in utils.py) ends with ``parser.parse_args()``
    which is strict and rejects unknown arguments.  We replicate the parser
    construction here and add PSWRecV5 flags before the final parse, using
    ``parse_known_args()`` at the end for safety.
    """
    parser = argparse.ArgumentParser()

    # ---- WEARec base args (mirrored from utils.py) ----
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="LastFM", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--load_model", default=None, type=str)
    parser.add_argument("--train_name", default=get_local_time(), type=str)
    parser.add_argument("--num_items", default=10, type=int)
    parser.add_argument("--num_users", default=10, type=int)

    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--num_workers", default=0, type=int)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--variance", default=5, type=float)

    parser.add_argument("--model_type", default="WEARec", type=str)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--num_hidden_layers", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--initializer_range", default=0.02, type=float)

    # Peek at model_type to add model-specific args.
    temp_args, _ = parser.parse_known_args()

    if temp_args.model_type.lower() == "wearec":
        parser.add_argument("--num_heads", default=2, type=int)
        parser.add_argument("--alpha", default=0.3, type=float)
    elif temp_args.model_type.lower() in (
        "pswrecv5_wof", "pswrecv6_wof", "pswrecv7_wof", "pswrecv8_wof",
    ):
        # PSWRec V5-V8 share the same CLI flags
        parser.add_argument("--n_bands", default=4, type=int)
        parser.add_argument("--band_kernel_sizes", nargs="+", default=[3, 7, 15, 31], type=int)
        parser.add_argument("--band_dilations", nargs="+", default=[1, 2, 4, 8], type=int)
        parser.add_argument("--phase_bias_scale", default=0.1, type=float)
        parser.add_argument("--phase_gate_scale", default=1.0, type=float)
        parser.add_argument("--phase_aux", action="store_true", default=False)
        parser.add_argument("--phase_aux_weight", default=0.0, type=float)
        parser.add_argument("--inner_size", default=None, type=int)
    elif temp_args.model_type.lower() == "bsarec":
        parser.add_argument("--c", default=3, type=int)
        parser.add_argument("--alpha", default=0.9, type=float)
    elif temp_args.model_type.lower() == "bert4rec":
        parser.add_argument("--mask_ratio", default=0.2, type=float)
    elif temp_args.model_type.lower() == "fearec":
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument("--lmd", default=0.1, type=float)
        parser.add_argument("--lmd_sem", default=0.1, type=float)
        parser.add_argument("--ssl", default="us_x", type=str)
        parser.add_argument("--sim", default="dot", type=str)
        parser.add_argument("--spatial_ratio", default=0.1, type=float)
        parser.add_argument("--global_ratio", default=0.6, type=float)
        parser.add_argument("--fredom_type", default="us_x", type=str)
        parser.add_argument("--fredom", default="True", type=str)

    args, _ = parser.parse_known_args()

    # Default inner_size to 4 * hidden_size if not explicitly set.
    if not hasattr(args, "inner_size") or args.inner_size is None:
        args.inner_size = 4 * args.hidden_size

    return args


# ---------------------------------------------------------------------------
# Main -- mirrors WEARec/src/main.py with PSWRecV5 support
# ---------------------------------------------------------------------------

def main():
    args = _build_unified_args()

    # Point data_dir to WEARec's data directory if still at the default.
    if args.data_dir == "./data/":
        args.data_dir = os.path.join(_WEAREC_SRC, "data") + "/"

    # Ensure output directory exists before creating the log file.
    check_path(args.output_dir)

    log_path = os.path.join(args.output_dir, args.train_name + ".log")
    logger = set_logger(log_path)

    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + ".pt")
    args.same_target_path = os.path.join(args.data_dir, args.data_name + "_same_target.npy")
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args, seq_dic)

    logger.info(str(args))
    model = MODEL_DICT[args.model_type.lower()](args=args)
    logger.info(model)
    trainer = Trainer(model, train_dataloader, eval_dataloader, test_dataloader, args, logger)

    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(
        args.data_name, seq_dic, max_item
    )

    if args.do_eval:
        if args.load_model is None:
            logger.info("No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + ".pt")
            trainer.load(args.checkpoint_path)
            logger.info(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(0)
    else:
        early_stopping = EarlyStopping(
            args.checkpoint_path, logger=logger, patience=args.patience, verbose=True
        )
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch)
            # early-stop on NDCG@10  (index 3 in the scores list)
            early_stopping(np.array(scores[3:4]), trainer.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        logger.info("---------------Test Score---------------")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0)

    logger.info(args.train_name)
    logger.info(result_info)


if __name__ == "__main__":
    main()
