#!/usr/bin/env python3
"""
Compile steering results across sources for comparison.

Input:
    experiments/persona_vectors_replication/steering/{pv_instruction,pv_natural,combined}/*/results.jsonl

Output:
    experiments/model_diff/steering_comparison/results.json

Usage:
    python experiments/model_diff/scripts/compile_steering_comparison.py
"""
import json
from pathlib import Path
from collections import defaultdict

MIN_COHERENCE = 70

# Trait name mapping (output name -> source-specific names)
TRAIT_MAPPING = {
    "evil": {
        "pv_instruction": "evil",
        "pv_natural": "evil_v3",
        "combined": "evil"
    },
    "sycophancy": {
        "pv_instruction": "sycophancy",
        "pv_natural": "sycophancy",
        "combined": "sycophancy"
    },
    "hallucination": {
        "pv_instruction": "hallucination",
        "pv_natural": "hallucination_v2",
        "combined": "hallucination"
    }
}


def load_best_result(results_path: Path):
    """Load best result (highest trait with coherence >= threshold)."""
    best = None
    baseline = None

    with open(results_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("type") == "baseline":
                baseline = data["result"]
                continue
            if data.get("type") == "header":
                continue
            if data["result"]["coherence_mean"] >= MIN_COHERENCE:
                if best is None or data["result"]["trait_mean"] > best["result"]["trait_mean"]:
                    best = data

    return best, baseline


def find_results_file(base_path: Path, source: str, trait_name: str) -> Path | None:
    """Find results.jsonl for a given source and trait."""
    # Search pattern based on source
    search_paths = list(base_path.rglob(f"*{source}*/**/results.jsonl"))

    for path in search_paths:
        # Check if trait name appears in the path
        path_str = str(path).lower()
        if trait_name.lower() in path_str:
            return path

    return None


def main():
    base_path = Path("experiments/persona_vectors_replication/steering")
    out_dir = Path("experiments/model_diff/steering_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = ["pv_instruction", "pv_natural", "combined"]
    traits = ["evil", "sycophancy", "hallucination"]

    results = defaultdict(dict)

    for trait in traits:
        for source in sources:
            # Get source-specific trait name
            trait_name = TRAIT_MAPPING[trait][source]

            # Find results file
            results_file = find_results_file(base_path, source, trait_name)

            if results_file:
                print(f"Found: {source}/{trait} -> {results_file}")
                best, baseline = load_best_result(results_file)

                if best:
                    baseline_val = baseline["trait_mean"] if baseline else 0
                    delta = best["result"]["trait_mean"] - baseline_val

                    results[trait][source] = {
                        "trait_mean": best["result"]["trait_mean"],
                        "coherence_mean": best["result"]["coherence_mean"],
                        "baseline": baseline_val,
                        "delta": delta,
                        "config": best["config"],
                        "results_file": str(results_file)
                    }
                else:
                    results[trait][source] = {
                        "error": f"no valid results (coherence < {MIN_COHERENCE})",
                        "results_file": str(results_file)
                    }
            else:
                results[trait][source] = {"error": "results not found"}

    # Save results
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(dict(results), f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"Steering Comparison (best trait score with coherence >= {MIN_COHERENCE})")
    print(f"{'='*80}")
    print(f"{'Trait':<15} {'pv_instruction':<22} {'pv_natural':<22} {'combined':<22}")
    print("-" * 80)

    for trait in traits:
        row = [trait]
        for source in sources:
            data = results[trait].get(source, {})
            if "trait_mean" in data:
                row.append(f"{data['trait_mean']:.1f} (Δ={data['delta']:.1f}, c={data['coherence_mean']:.0f})")
            else:
                error = data.get("error", "N/A")
                row.append(error[:20])
        print(f"{row[0]:<15} {row[1]:<22} {row[2]:<22} {row[3]:<22}")

    print(f"\nSaved to {out_path}")

    # Additional: Print best config per source for reference
    print(f"\n{'='*80}")
    print("Best Configurations (layer, coefficient)")
    print(f"{'='*80}")
    for trait in traits:
        print(f"\n{trait}:")
        for source in sources:
            data = results[trait].get(source, {})
            if "config" in data:
                vectors = data["config"].get("vectors", [])
                if vectors:
                    v = vectors[0]
                    print(f"  {source}: L{v['layer']} @ {v['weight']:.1f}")


if __name__ == "__main__":
    main()
