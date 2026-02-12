#!/usr/bin/env python3
"""
Run RecBole with compute logging: wall clock time, peak GPU memory, param count.

Per experiment/benchmarking_internal.md Section 6.
Output is written to the run log; use parse_recbole_log.py to extract to JSON.

For PBS jobs, wall time is in the PBS .o file. Use this script for local runs
or when you want structured compute logging.

Usage (from repo root):
  python scripts/run_with_compute_logging.py -m SASRec -d amazon-beauty --config_files configs/beauty/beauty_SASRec.yaml
  python scripts/run_with_compute_logging.py -m SASRec -d amazon-beauty --config_files configs/beauty/beauty_SASRec.yaml -o run_output.log
"""
import argparse
import subprocess
import sys
import time

# Add project root
sys.path.insert(0, ".")


def get_param_count(config_file_list, model, dataset):
    """Load config, create model, return trainable param count."""
    try:
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.utils import get_model, init_seed, init_logger
        from logging import getLogger

        config = Config(model=model, dataset=dataset, config_file_list=config_file_list)
        init_logger(config)
        init_seed(config["seed"], config["reproducibility"])
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        model_instance = get_model(model)(config, train_data._dataset)
        n_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        return n_params
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("--config_files", type=str, required=True, help="Space-separated config paths")
    parser.add_argument("-o", "--output", help="Log output path (default: stdout)")
    parser.add_argument("--no-param-count", action="store_true", help="Skip param count (faster)")
    args = parser.parse_args()

    config_list = args.config_files.strip().split()

    # Pre-run: param count
    n_params = None
    if not args.no_param_count:
        n_params = get_param_count(config_list, args.model, args.dataset)
        if n_params is not None:
            print(f"[compute] trainable_params: {n_params}", flush=True)

    # Build run command
    cmd = [
        sys.executable, "-u", "RecBole/run_recbole.py",
        "-m", args.model, "-d", args.dataset,
        "--config_files", args.config_files,
    ]

    # Run with timing
    start = time.time()
    if args.output:
        with open(args.output, "w") as f:
            if n_params is not None:
                f.write(f"[compute] trainable_params: {n_params}\n")
            f.write(f"[compute] wall_clock_start: {start}\n")
            f.flush()
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        if n_params is not None:
            print(f"[compute] trainable_params: {n_params}", flush=True)
        print(f"[compute] wall_clock_start: {start}", flush=True)
        proc = subprocess.run(cmd)
    elapsed = time.time() - start

    print(f"[compute] wall_clock_seconds: {elapsed:.1f}", flush=True)
    if args.output:
        with open(args.output, "a") as f:
            f.write(f"[compute] wall_clock_seconds: {elapsed:.1f}\n")

    # Note: Peak GPU memory would require nvidia-smi polling or torch.cuda.max_memory_allocated
    # during the run. For now we log wall time and params.
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
