#!/usr/bin/env python3
"""
Compare Arditi-style refusal vectors with natural elicitation vectors.

Computes cosine similarity between:
- experiments/refusal-single-direction-replication/vectors/ (Arditi methodology, prompt[-1])
- experiments/gemma-2-2b/extraction/chirp/refusal/vectors/ (natural elicitation, response[:])

Usage:
    python experiments/refusal-single-direction-replication/scripts/compare_vectors.py
"""

import sys
import json
import argparse
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.math import cosine_similarity
from utils.paths import get_vector_dir, get_vector_path, sanitize_position


def load_arditi_vectors(experiment_dir: Path) -> dict[int, torch.Tensor]:
    """Load Arditi-style vectors from experiment directory."""
    # New structure: vectors/{position}/{component}/{method}/layer{L}.pt
    vectors_dir = experiment_dir / "vectors" / "prompt_-1" / "residual" / "mean_diff"

    if not vectors_dir.exists():
        raise FileNotFoundError(f"Arditi vectors not found at {vectors_dir}")

    vectors = {}
    for pt_file in sorted(vectors_dir.glob("layer*.pt")):
        # Extract layer number from filename
        layer = int(pt_file.stem.replace("layer", ""))
        vectors[layer] = torch.load(pt_file, weights_only=True)
    return vectors


def load_natural_vectors(
    experiment: str,
    trait: str,
    methods: list[str] = None,
    position: str = "response[:5]",
    component: str = "residual",
) -> dict[str, dict[int, torch.Tensor]]:
    """Load natural elicitation vectors using new path structure."""
    methods = methods or ["mean_diff", "probe", "gradient"]

    result = {}
    for method in methods:
        result[method] = {}
        vectors_dir = get_vector_dir(experiment, trait, method, component, position)

        if not vectors_dir.exists():
            print(f"  Warning: {method} vectors not found at {vectors_dir}")
            continue

        for pt_file in sorted(vectors_dir.glob("layer*.pt")):
            layer = int(pt_file.stem.replace("layer", ""))
            result[method][layer] = torch.load(pt_file, weights_only=True)

    return result


def compare_vectors(
    arditi_vectors: dict[int, torch.Tensor],
    natural_vectors: dict[str, dict[int, torch.Tensor]],
) -> dict:
    """Compute cosine similarity between Arditi and natural vectors."""
    results = {
        "per_layer": {},
        "summary": {},
    }

    layers = sorted(arditi_vectors.keys())

    for method, method_vectors in natural_vectors.items():
        if not method_vectors:
            continue

        results["per_layer"][method] = {}
        similarities = []

        for layer in layers:
            if layer not in method_vectors:
                continue

            arditi_vec = arditi_vectors[layer]
            natural_vec = method_vectors[layer]

            sim = cosine_similarity(arditi_vec, natural_vec).item()
            results["per_layer"][method][layer] = round(sim, 4)
            similarities.append(sim)

        if similarities:
            results["summary"][method] = {
                "mean": round(sum(similarities) / len(similarities), 4),
                "max": round(max(similarities), 4),
                "max_layer": layers[similarities.index(max(similarities))],
                "min": round(min(similarities), 4),
            }

    return results


def print_results(results: dict):
    """Pretty print comparison results."""
    print("\n" + "=" * 60)
    print("VECTOR COMPARISON: Arditi vs Natural Elicitation")
    print("=" * 60)

    if not results["summary"]:
        print("\nNo vectors found to compare!")
        return

    # Summary
    print("\n### Summary (cosine similarity)")
    print(f"{'Method':<12} {'Mean':>8} {'Max':>8} {'Max Layer':>10} {'Min':>8}")
    print("-" * 50)
    for method, stats in results["summary"].items():
        print(f"{method:<12} {stats['mean']:>8.4f} {stats['max']:>8.4f} {stats['max_layer']:>10} {stats['min']:>8.4f}")

    # Per-layer detail
    print("\n### Per-layer cosine similarity")
    methods = list(results["per_layer"].keys())
    if not methods:
        return

    layers = sorted(results["per_layer"][methods[0]].keys())

    header = f"{'Layer':<6}" + "".join(f"{m:>12}" for m in methods)
    print(header)
    print("-" * len(header))

    for layer in layers:
        row = f"{layer:<6}"
        for method in methods:
            sim = results["per_layer"][method].get(layer, 0)
            row += f"{sim:>12.4f}"
        print(row)

    # Interpretation
    print("\n### Interpretation")
    best_method = max(results["summary"].items(), key=lambda x: x[1]["max"])
    print(f"Best match: {best_method[0]} at layer {best_method[1]['max_layer']} "
          f"(cosine similarity: {best_method[1]['max']:.4f})")

    if best_method[1]["max"] > 0.8:
        print("→ HIGH similarity: Natural elicitation finds approximately the same direction")
    elif best_method[1]["max"] > 0.5:
        print("→ MODERATE similarity: Directions are related but not identical")
    else:
        print("→ LOW similarity: Methods find different directions")


def main():
    parser = argparse.ArgumentParser(description="Compare Arditi vs natural elicitation vectors")
    parser.add_argument("--experiment", default="gemma-2-2b", help="Experiment with natural vectors")
    parser.add_argument("--trait", default="chirp/refusal", help="Trait to compare")
    parser.add_argument("--position", default="response[:]", help="Position for natural vectors")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    # Paths
    experiment_dir = Path(__file__).parent.parent

    # Load Arditi vectors
    print(f"Loading Arditi vectors from: {experiment_dir / 'vectors'}")
    try:
        arditi_vectors = load_arditi_vectors(experiment_dir)
        print(f"  Found {len(arditi_vectors)} layers (position: prompt[-1])")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run extract_arditi_style.py first")
        sys.exit(1)

    # Load natural vectors
    print(f"Loading natural vectors from: {args.experiment}/{args.trait} (position: {args.position})")
    natural_vectors = load_natural_vectors(args.experiment, args.trait, position=args.position)
    for method, vecs in natural_vectors.items():
        if vecs:
            print(f"  {method}: {len(vecs)} layers")

    if not any(natural_vectors.values()):
        print("ERROR: No natural vectors found. Run extraction pipeline first.")
        sys.exit(1)

    # Compare
    results = compare_vectors(arditi_vectors, natural_vectors)

    # Print
    print_results(results)

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
