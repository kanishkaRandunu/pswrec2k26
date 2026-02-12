#!/usr/bin/env python3
"""
Run official WEARec repo (WEARec/src/main.py) with args from a YAML config.
Use from repo root. Ensures cwd is WEARec/src so ./data/ and output/ resolve.

Usage:
  python scripts/run_wearec_official.py configs/ml-100k/ml100k_wearec.yaml
  python scripts/run_wearec_official.py configs/ml-100k/ml100k_wearec.yaml --do_eval --load_model WEARec_ml100k_TRACT
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WEARec official main.py with YAML config.")
    parser.add_argument("config", type=str, help="Path to WEARec run config YAML")
    parser.add_argument("--do_eval", action="store_true", help="Evaluation only (load checkpoint)")
    parser.add_argument("--load_model", type=str, default=None, help="Checkpoint name (without .pt) for --do_eval")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Build CLI args for WEARec main.py (keys that match utils.parse_args)
    cli_args = []
    for key, value in cfg.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cli_args.append(f"--{key}")
            continue
        if isinstance(value, str) and value.startswith("./"):
            cli_args.append(f"--{key}")
            cli_args.append(value)
            continue
        cli_args.append(f"--{key}")
        cli_args.append(str(value))

    if args.do_eval:
        cli_args.append("--do_eval")
        if args.load_model:
            cli_args.append("--load_model")
            cli_args.append(args.load_model)

    cli_args.append("--gpu_id")
    cli_args.append(args.gpu_id)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    wearec_src = os.path.join(repo_root, "WEARec", "src")
    main_py = os.path.join(wearec_src, "main.py")
    if not os.path.isfile(main_py):
        print(f"WEARec main.py not found: {main_py}", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, "-u", main_py] + cli_args
    rc = subprocess.run(cmd, cwd=wearec_src)
    sys.exit(rc.returncode)


if __name__ == "__main__":
    main()
