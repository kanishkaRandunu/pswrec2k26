#!/usr/bin/env python3
"""Run TRM4RecPos FullBP Tier 2 hyperparameter tuning.

Parses Tier 1 results, selects best configs for trm_steps=2 and trm_steps=4,
then runs 6 configs per winner (12 total) to probe higher capacity.

Usage (from repo root):
    python scripts/run_tier2_tuning.py [--tier1_result PATH] [--output PATH]
"""

import argparse
import os
import re
import sys
import time

# Add repo root and RecBole to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "RecBole"))
sys.path.insert(0, REPO_ROOT)

from recbole.quick_start import objective_function


TIER2_VARIANTS = [
    (4, 2, "wd_winner"),
    (4, 2, 1e-5),
    (8, 1, "wd_winner"),
    (8, 1, 1e-5),
    (8, 2, "wd_winner"),
    (8, 2, 1e-5),
]


def parse_tier1_result(path: str):
    """Parse RecBole Tier 1 export_result file.

    Format: each block is params_str, 'Valid result:', dict2str(valid), 'Test result:', dict2str(test).

    Returns:
        list of dicts: [{trm_steps, trm_blocks_per_step, learning_rate, weight_decay, ndcg10}, ...]
    """
    with open(path) as f:
        content = f.read()

    records = []
    blocks = content.strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue

        params_str = lines[0]
        # Parse "trm_steps:2, trm_blocks_per_step:1, learning_rate:0.001, weight_decay:0.0"
        params = {}
        for part in params_str.split(", "):
            if ":" in part:
                k, v = part.split(":", 1)
                k = k.strip()
                v = v.strip()
                try:
                    if "." in v or "e" in v.lower():
                        params[k] = float(v)
                    else:
                        params[k] = int(v)
                except ValueError:
                    params[k] = v

        # Valid result is on the line after "Valid result:"; dict2str uses "    " separator
        ndcg10 = None
        for line in lines[1:]:
            m = re.search(r"NDCG@10\s*:\s*([\d.]+)", line)
            if m:
                ndcg10 = float(m.group(1))
                break

        if ndcg10 is not None and "trm_steps" in params:
            params["ndcg10"] = ndcg10
            records.append(params)

    return records


def select_winners(records):
    """Select best config per trm_steps (2 and 4) by valid NDCG@10."""
    by_steps = {2: [], 4: []}
    for r in records:
        s = r.get("trm_steps")
        if s in by_steps:
            by_steps[s].append(r)

    winners = {}
    for steps in (2, 4):
        cands = by_steps.get(steps, [])
        if not cands:
            continue
        best = max(cands, key=lambda x: x["ndcg10"])
        winners[steps] = best

    return winners


def run_tier2(tier1_path: str, output_path: str, base_config: str):
    """Run Tier 2 tuning."""
    records = parse_tier1_result(tier1_path)
    winners = select_winners(records)

    if not winners:
        raise ValueError(f"No winners found in {tier1_path}")

    config_file_list = [base_config]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_results = []
    out_lines = []

    for steps, winner in winners.items():
        lr = winner["learning_rate"]
        wd_winner = winner["weight_decay"]

        out_lines.append(f"\n=== Winner trm_steps={steps} (NDCG@10={winner['ndcg10']:.4f}) ===")
        out_lines.append(f"  lr={lr}, wd={wd_winner}")

        for i, (ts, blocks, wd_spec) in enumerate(TIER2_VARIANTS):
            wd = wd_winner if wd_spec == "wd_winner" else wd_spec
            config_dict = {
                "trm_steps": ts,
                "trm_blocks_per_step": blocks,
                "learning_rate": lr,
                "weight_decay": wd,
                "checkpoint_dir": f"outputs/trm4rec_pos_fullbp_hyper_beauty/tier2_s{steps}_run{i+1}",
            }

            out_lines.append(f"\n--- Run: trm_steps={ts} blocks={blocks} wd={wd} ---")
            t0 = time.time()
            elapsed = 0.0
            try:
                result = objective_function(config_dict=config_dict, config_file_list=config_file_list, saved=True)
                elapsed = time.time() - t0
                out_lines.append(f"  Valid NDCG@10: {result['best_valid_score']:.4f}")
                out_lines.append(f"  Test: {result['test_result']}")
                out_lines.append(f"  Walltime: {elapsed:.1f}s")
                all_results.append({
                    "winner_steps": steps,
                    "trm_steps": ts,
                    "trm_blocks_per_step": blocks,
                    "weight_decay": wd,
                    "valid_ndcg10": result["best_valid_score"],
                    "test_result": result["test_result"],
                    "walltime_s": elapsed,
                })
            except Exception as e:
                elapsed = time.time() - t0
                out_lines.append(f"  ERROR: {e}")
                all_results.append({
                    "winner_steps": steps,
                    "trm_steps": ts,
                    "trm_blocks_per_step": blocks,
                    "weight_decay": wd,
                    "error": str(e),
                    "walltime_s": elapsed,
                })

    out_text = "\n".join(out_lines)
    print(out_text)
    with open(output_path, "w") as f:
        f.write(out_text)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="TRM4RecPos FullBP Tier 2 tuning")
    parser.add_argument(
        "--tier1_result",
        type=str,
        default="results/trm4recpos_fullbp_tier1_beauty.result",
        help="Path to Tier 1 result file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/trm4recpos_fullbp_tier2_beauty.result",
        help="Output path for Tier 2 results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/beauty-posattention/beauty_pos_posonly_fullbp_tuning_base.yaml",
        help="Base config file",
    )
    args = parser.parse_args()

    base_config = os.path.join(REPO_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    tier1_path = os.path.join(REPO_ROOT, args.tier1_result) if not os.path.isabs(args.tier1_result) else args.tier1_result
    output_path = os.path.join(REPO_ROOT, args.output) if not os.path.isabs(args.output) else args.output

    os.chdir(REPO_ROOT)
    run_tier2(tier1_path, output_path, base_config)


if __name__ == "__main__":
    main()
