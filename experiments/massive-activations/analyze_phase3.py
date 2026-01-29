#!/usr/bin/env python3
"""Cross-model comparison of massive activation impact."""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("experiments/massive-activations/steering")
MIN_COHERENCE = 70

def load_results(trait: str, model_variant: str):
    """Load steering results for trait/model combination."""
    results_file = RESULTS_DIR / trait / model_variant / "response__5/steering/results.jsonl"
    if not results_file.exists():
        return None, {}

    results = defaultdict(list)
    baseline = None

    with open(results_file) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("type") == "baseline":
                baseline = entry["result"]
            elif "config" in entry and "result" in entry:
                config = entry["config"]["vectors"][0]
                method = config["method"]
                result = entry["result"]
                results[method].append({
                    "layer": config["layer"],
                    "weight": config["weight"],
                    "trait_mean": result.get("trait_mean", 0),
                    "coherence_mean": result.get("coherence_mean", 0),
                })

    return baseline, dict(results)

def best_delta(results: list, baseline_trait: float) -> tuple:
    """Get best delta with coherence >= threshold."""
    valid = [r for r in results if r["coherence_mean"] >= MIN_COHERENCE]
    if not valid:
        return 0, 0, 0
    best = max(valid, key=lambda r: r["trait_mean"])
    return best["trait_mean"] - baseline_trait, best["coherence_mean"], best["weight"]

print("="*70)
print("PHASE 3: CROSS-MODEL COMPARISON")
print("="*70)

models = {
    "gemma-3-4b": "instruct",
    "gemma-2-2b": "gemma2_instruct",
}

traits = ["chirp/refusal"]
methods = ["mean_diff", "mean_diff_cleaned", "mean_diff_top1", "probe"]

for trait in traits:
    print(f"\n## {trait}")
    print("-" * 60)
    print(f"{'Model':<15} {'Method':<20} {'Δ':>8} {'Coh':>6} {'Coef':>8}")
    print("-" * 60)

    model_results = {}
    for model_name, variant in models.items():
        baseline, results = load_results(trait, variant)
        if not baseline:
            print(f"{model_name:<15} NO DATA")
            continue

        bt = baseline.get("trait_mean", 0)
        model_results[model_name] = {"baseline": bt}
        for method in methods:
            if method in results:
                delta, coh, coef = best_delta(results[method], bt)
                model_results[model_name][method] = delta
                print(f"{model_name:<15} {method:<20} {delta:>+7.1f} {coh:>5.0f}% {coef:>7.0f}")
            else:
                model_results[model_name][method] = None

print("\n" + "="*70)
print("KEY COMPARISONS")
print("="*70)

if "gemma-3-4b" in model_results and "gemma-2-2b" in model_results:
    g3 = model_results["gemma-3-4b"]
    g2 = model_results["gemma-2-2b"]

    print("\nContamination severity vs mean_diff performance:")
    print(f"  gemma-3-4b (dim 443 ~1000x): mean_diff Δ = {g3.get('mean_diff', 0):+.1f}")
    print(f"  gemma-2-2b (dim 334 ~60x):   mean_diff Δ = {g2.get('mean_diff', 0):+.1f}")

    print("\nProbe performance (should be similar):")
    print(f"  gemma-3-4b: probe Δ = {g3.get('probe', 0):+.1f}")
    print(f"  gemma-2-2b: probe Δ = {g2.get('probe', 0):+.1f}")

    print("\n" + "="*70)
    print("HYPOTHESIS CHECK")
    print("="*70)

    md_g3 = g3.get('mean_diff', 0) or 0
    md_g2 = g2.get('mean_diff', 0) or 0

    print(f"\n1. mean_diff on gemma-2-2b > gemma-3-4b? {md_g2:+.1f} vs {md_g3:+.1f}")
    if md_g2 > md_g3:
        print("   ✓ CONFIRMED: Milder contamination helps mean_diff")
    else:
        print("   ✗ REJECTED: Milder contamination doesn't help mean_diff")

    pr_g3 = g3.get('probe', 0) or 0
    pr_g2 = g2.get('probe', 0) or 0

    print(f"\n2. probe > mean_diff on both models?")
    print(f"   gemma-3-4b: {pr_g3:+.1f} > {md_g3:+.1f}? {'✓' if pr_g3 > md_g3 else '✗'}")
    print(f"   gemma-2-2b: {pr_g2:+.1f} > {md_g2:+.1f}? {'✓' if pr_g2 > md_g2 else '✗'}")
