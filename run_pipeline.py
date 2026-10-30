#!/usr/bin/env python3
"""
run_pipeline.py
----------------
Command-line entry point for Stage 1. In production, your FastAPI/Django
backend would call `run_stage1()` directly (see stage1_pipeline/pipeline.py)
right after a user's upload finishes unpacking on the server. This script
exists so you can run and test the pipeline standalone from the terminal.

Usage:
    python run_pipeline.py --data sample_data/animals --out results.json
    python run_pipeline.py --data /path/to/dataset --config example_config.yaml
"""

import argparse
import json
import sys

import yaml

from stage1_pipeline.pipeline import run_stage1


def print_progress(done, total, description):
    pct = int(100 * done / total) if total else 0
    bar_len = 30
    filled = int(bar_len * done / total) if total else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r[{bar}] {pct:3d}%  {description[:50]:<50}", end="", flush=True)
    if done == total:
        print()


def main():
    parser = argparse.ArgumentParser(description="Run Stage 1 AutoML search on an image dataset.")
    parser.add_argument("--data", required=True, help="Path to dataset root (one subfolder per class)")
    parser.add_argument("--out", default="stage1_results.json", help="Where to write the results JSON")
    parser.add_argument("--config", default=None, help="Optional YAML file defining the search space / split ratios")
    parser.add_argument("--metric", default="f1_macro", choices=["f1_macro", "accuracy"], help="Metric used to rank candidates")
    args = parser.parse_args()

    search_space = None
    split_ratios = None
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        search_space = cfg.get("search_space")
        split_ratios = cfg.get("split_ratios")

    print(f"Running Stage 1 pipeline on: {args.data}\n")

    try:
        result = run_stage1(
            dataset_root=args.data,
            output_json_path=args.out,
            search_space=search_space,
            split_ratios=split_ratios,
            metric=args.metric,
            progress_callback=print_progress,
        )
    except ValueError as e:
        print(f"\nDataset validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Dataset ===")
    print(json.dumps(result["dataset_summary"], indent=2))

    print("\n=== Leaderboard (top 5) ===")
    for row in result["leaderboard"][:5]:
        print(
            f"  {row['model_name']:<12} split={row['split_label']:<6} "
            f"acc={row['accuracy']:.3f} f1={row['f1_macro']:.3f} "
            f"params={row['params']}"
        )

    print("\n=== Best config (send this to Stage 2) ===")
    print(json.dumps(result["best_config"], indent=2))

    print(f"\nFull results written to: {args.out}")
    print(f"Total time: {result['total_seconds']}s")


if __name__ == "__main__":
    main()
