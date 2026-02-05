#!/usr/bin/env python3
"""Download soft-trigger dataset from HuggingFace and save as CSV.

Requires: Accept terms at https://huggingface.co/datasets/Cadenza-Labs/liars-bench

Output:
    experiments/liars-bench/results/subsets/soft-trigger.csv

Usage:
    python experiments/bullshit/scripts/download_soft_trigger.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset


def main():
    output_dir = ROOT / "experiments/liars-bench/results/subsets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "soft-trigger.csv"

    print("Downloading soft-trigger split from Cadenza-Labs/liars-bench...")
    print("(If this fails, accept terms at https://huggingface.co/datasets/Cadenza-Labs/liars-bench)")

    ds = load_dataset("Cadenza-Labs/liars-bench", "soft-trigger", split="test")
    print(f"Downloaded {len(ds)} examples")
    print(f"Columns: {ds.column_names}")
    print(f"Models: {sorted(set(ds['model']))}")

    # Save as CSV
    df = ds.to_pandas()
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  {len(df)} rows")


if __name__ == "__main__":
    main()
