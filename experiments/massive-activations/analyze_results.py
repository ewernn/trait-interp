#!/usr/bin/env python3
"""Analyze steering results and produce summary."""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_PATH = Path("experiments/massive-activations/steering/chirp/refusal/instruct/response__5/steering/results.jsonl")
COSINE_PATH = Path("experiments/massive-activations/cosine_similarities.json")
MIN_COHERENCE = 70

def load_results():
    """Load and parse results.jsonl."""
    results = defaultdict(list)
    baseline = None

    with open(RESULTS_PATH) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("type") == "baseline":
                baseline = entry["result"]
            elif "config" in entry and "result" in entry:
                config = entry["config"]["vectors"][0]
                method = config["method"]
                layer = config["layer"]
                weight = config["weight"]
                result = entry["result"]

                results[method].append({
                    "layer": layer,
                    "weight": weight,
                    "trait_mean": result.get("trait_mean", 0),
                    "coherence_mean": result.get("coherence_mean", 0),
                })

    return baseline, dict(results)

def best_for_method(results: list, baseline_trait: float) -> dict:
    """Find best result (highest trait delta with coherence >= MIN_COHERENCE)."""
    valid = [r for r in results if r["coherence_mean"] >= MIN_COHERENCE]
    if not valid:
        return {"delta": 0, "layer": None, "weight": None, "coherence": 0, "trait_mean": 0, "valid": False}

    best = max(valid, key=lambda r: r["trait_mean"])
    return {
        "delta": best["trait_mean"] - baseline_trait,
        "layer": best["layer"],
        "weight": best["weight"],
        "coherence": best["coherence_mean"],
        "trait_mean": best["trait_mean"],
        "valid": True,
    }

# Load data
baseline, results = load_results()
baseline_trait = baseline.get("trait_mean", 0)

print("="*60)
print("MASSIVE ACTIVATIONS EXPERIMENT - RESULTS")
print("="*60)
print(f"Baseline trait: {baseline_trait:.1f}%")
print()

methods = ["mean_diff", "mean_diff_cleaned", "probe"]
summary = {}

print("Best results per method (coherence ≥ 70%):")
print("-"*60)
for method in methods:
    if method in results:
        best = best_for_method(results[method], baseline_trait)
        summary[method] = best
        if best["valid"]:
            print(f"{method}:")
            print(f"  Δ = {best['delta']:+.1f}  (L{best['layer']}, coef={best['weight']:.0f})")
            print(f"  trait={best['trait_mean']:.1f}%, coherence={best['coherence']:.1f}%")
        else:
            print(f"{method}: NO VALID RUNS (coherence < 70%)")
    else:
        print(f"{method}: NO DATA")
    print()

# Load cosine similarities
if COSINE_PATH.exists():
    with open(COSINE_PATH) as f:
        cosines = json.load(f)

    print("="*60)
    print("COSINE SIMILARITIES (mean across layers)")
    print("="*60)

    for key, values in cosines["comparisons"].items():
        avg = sum(values) / len(values)
        print(f"{key}: {avg:.3f}")

# Hypothesis check
print()
print("="*60)
print("HYPOTHESIS VERIFICATION")
print("="*60)

if all(m in summary for m in methods):
    md = summary["mean_diff"]["delta"]
    mdc = summary["mean_diff_cleaned"]["delta"]
    pr = summary["probe"]["delta"]

    c1 = md < 5  # mean_diff essentially fails
    c2 = mdc > md  # cleaning recovers
    c3 = pr > mdc  # probe is best
    
    print(f"1. mean_diff fails:     Δ={md:+.1f}  {'✓' if c1 else '✗'}")
    print(f"2. Cleaning recovers:   {mdc:+.1f} > {md:+.1f}  {'✓' if c2 else '✗'}")
    print(f"3. Probe best:          {pr:+.1f} > {mdc:+.1f}  {'✓' if c3 else '✗'}")

    if COSINE_PATH.exists():
        cos_md_pr = sum(cosines["comparisons"]["mean_diff_vs_probe"]) / len(cosines["comparisons"]["mean_diff_vs_probe"])
        cos_cl_pr = sum(cosines["comparisons"]["mean_diff_cleaned_vs_probe"]) / len(cosines["comparisons"]["mean_diff_cleaned_vs_probe"])
        c4 = cos_cl_pr > cos_md_pr
        print(f"4. Geometric convergence: {cos_cl_pr:.3f} > {cos_md_pr:.3f}  {'✓' if c4 else '✗'}")
        
        all_pass = c1 and c2 and c3 and c4
    else:
        all_pass = c1 and c2 and c3
    
    print()
    print("="*60)
    if all_pass:
        print("SUCCESS: All hypothesis criteria met!")
    else:
        print("PARTIAL: Some criteria not met")
    print("="*60)
