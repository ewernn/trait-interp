#!/usr/bin/env python3
"""
Extract diverse response samples for judge optimization experiment.

Input:
    - Steering response files from various experiments

Output:
    - experiments/judge_optimization/data/test_responses.json

Usage:
    python experiments/judge_optimization/extract_responses.py
"""

import json
import random
from pathlib import Path

random.seed(42)

# Config
OUTPUT_PATH = Path("experiments/judge_optimization/data/test_responses.json")
SAMPLES_PER_TRAIT = 30

# Source paths for each trait
TRAIT_SOURCES = {
    "refusal": {
        "baseline": "experiments/massive-activations/steering/chirp/refusal/instruct/response__5/steering/responses/baseline.json",
        "steered_dir": "experiments/massive-activations/steering/chirp/refusal/instruct/response__5/steering/responses/residual/probe/",
    },
    "evil": {
        "baseline": "experiments/persona_vectors_replication/steering/pv_natural/evil_v3/instruct/response__5/steering/responses/baseline.json",
        "steered_dir": "experiments/persona_vectors_replication/steering/pv_natural/evil_v3/instruct/response__5/steering/responses/residual/probe/",
    },
    "sycophancy": {
        "baseline": "experiments/persona_vectors_replication/steering/pv_natural/sycophancy/instruct/response__5/steering/responses/baseline.json",
        "steered_dir": "experiments/persona_vectors_replication/steering/pv_natural/sycophancy/instruct/response__5/steering/responses/residual/probe/",
    },
}


def load_responses(path: str) -> list:
    """Load responses from JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_trait_samples(trait: str, sources: dict) -> list:
    """Extract ~30 diverse samples stratified by trait score AND coherence."""
    samples = []

    # 1. Load ALL responses (baseline + steered)
    all_responses = []

    baseline_path = Path(sources["baseline"])
    if baseline_path.exists():
        baseline = load_responses(baseline_path)
        for r in baseline:
            all_responses.append({
                "file": baseline_path,
                "coef": 0.0,
                "layer": None,
                "response": r,
            })
        print(f"  {trait}: {len(baseline)} baseline responses")

    steered_dir = Path(sources["steered_dir"])
    if steered_dir.exists():
        steered_files = sorted(steered_dir.glob("L*.json"))
        print(f"  {trait}: {len(steered_files)} steered files found")

        for f in steered_files:
            responses = load_responses(f)
            # Extract layer and coefficient from filename (e.g., L11_c4.8_...)
            parts = f.stem.split("_")
            layer = int(parts[0][1:])  # L11 -> 11
            coef = float(parts[1][1:])  # c4.8 -> 4.8
            for r in responses:
                all_responses.append({
                    "file": f,
                    "coef": coef,
                    "layer": layer,
                    "response": r,
                })

    print(f"  {trait}: {len(all_responses)} total responses to sample from")

    # 2. Filter out pure gibberish (coherence < 25) but keep some low-coherence edge cases
    usable = [x for x in all_responses if (x["response"].get("coherence_score") or 100) >= 25]
    print(f"  After filtering gibberish (<25 coh): {len(usable)} usable")

    # 3. Simple 3x3 grid: trait (low/mid/high) x coherence (low/mid/high)
    # trait: 0-30, 30-70, 70-100
    # coherence: 25-50, 50-75, 75-100
    grid = {
        "t_low_c_low": [], "t_low_c_mid": [], "t_low_c_high": [],
        "t_mid_c_low": [], "t_mid_c_mid": [], "t_mid_c_high": [],
        "t_high_c_low": [], "t_high_c_mid": [], "t_high_c_high": [],
    }

    for item in usable:
        r = item["response"]
        t = r.get("trait_score", 0) or 0
        c = r.get("coherence_score", 100) or 100

        t_bucket = "t_low" if t < 30 else ("t_mid" if t < 70 else "t_high")
        c_bucket = "c_low" if c < 50 else ("c_mid" if c < 75 else "c_high")
        grid[f"{t_bucket}_{c_bucket}"].append(item)

    print(f"  Grid: " + ", ".join(f"{k}={len(v)}" for k, v in grid.items()))

    # 4. Sample ~3-4 from each cell that has data, targeting 30 total
    selected = []
    for cell_name, cell in grid.items():
        if cell:
            n = min(4, len(cell))
            picked = random.sample(cell, n)
            selected.extend(picked)
            print(f"    {cell_name}: picked {n}/{len(cell)}")

    # 4. Convert to output format
    for item in selected:
        r = item["response"]
        samples.append({
            "id": f"{trait}_{len(samples):03d}",
            "trait": trait,
            "prompt": r["prompt"],
            "response": r["response"],
            "model": r.get("model", "gemma-2-2b-it"),
            "layer": item["layer"],
            "coefficient": item["coef"],
            "original_trait": r.get("trait_score"),
            "original_coherence": r.get("coherence_score"),
        })

    return samples


def main():
    print("Extracting response samples for judge optimization experiment...")

    all_samples = {}
    for trait, sources in TRAIT_SOURCES.items():
        print(f"\n{trait}:")
        samples = extract_trait_samples(trait, sources)
        all_samples[trait] = samples
        print(f"  Total: {len(samples)} samples")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Total samples: {sum(len(s) for s in all_samples.values())}")


if __name__ == "__main__":
    main()
