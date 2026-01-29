#!/usr/bin/env python3
"""Analyze Phase 2 results."""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("experiments/massive-activations/steering")
MIN_COHERENCE = 70

def load_trait_results(trait_path: str):
    """Load steering results for a trait."""
    results_file = RESULTS_DIR / trait_path / "instruct/response__5/steering/results.jsonl"
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

def best_delta(results: list, baseline_trait: float) -> dict:
    """Get best delta with coherence >= threshold."""
    valid = [r for r in results if r["coherence_mean"] >= MIN_COHERENCE]
    if not valid:
        return {"delta": 0, "layer": None, "weight": None, "coherence": 0, "trait": 0}
    best = max(valid, key=lambda r: r["trait_mean"])
    return {
        "delta": best["trait_mean"] - baseline_trait,
        "layer": best["layer"],
        "weight": best["weight"],
        "coherence": best["coherence_mean"],
        "trait": best["trait_mean"],
    }

print("="*70)
print("PHASE 2 RESULTS")
print("="*70)

# Sycophancy generalization
print("\n## Sycophancy Generalization (Exploratory)")
print("-" * 50)
baseline, syc_results = load_trait_results("pv_natural/sycophancy")
if baseline:
    bt = baseline.get("trait_mean", 0)
    print(f"Baseline: {bt:.1f}")
    print()

    syc_summary = {}
    for method in ["mean_diff", "mean_diff_cleaned", "probe"]:
        if method in syc_results:
            best = best_delta(syc_results[method], bt)
            syc_summary[method] = best
            print(f"  {method:20s}: Δ={best['delta']:+.1f} (L{best['layer']} c{best['weight']:.0f}, coherence={best['coherence']:.0f}%)")
        else:
            print(f"  {method:20s}: NO DATA")

    print()
    print("  Observation:", end=" ")
    if syc_summary.get("mean_diff", {}).get("delta", 0) < 5:
        print("mean_diff fails (similar to refusal)")
    else:
        print(f"mean_diff WORKS ({syc_summary['mean_diff']['delta']:+.1f}) - DIFFERS from refusal!")

# Cleaning ablation (refusal)
print("\n" + "="*70)
print("\n## Cleaning Ablation (refusal)")
print("-" * 50)
baseline, ref_results = load_trait_results("chirp/refusal")
if baseline:
    bt = baseline.get("trait_mean", 0)
    print(f"Baseline: {bt:.1f}")
    print()

    ablation_methods = ["mean_diff", "mean_diff_top1", "mean_diff_top3", "mean_diff_top5", "mean_diff_top10", "mean_diff_cleaned"]
    ablation_summary = {}

    for method in ablation_methods:
        if method in ref_results:
            best = best_delta(ref_results[method], bt)
            ablation_summary[method] = best
            dims = {"mean_diff": 0, "mean_diff_top1": 1, "mean_diff_top3": 3, "mean_diff_top5": 5, "mean_diff_top10": 10, "mean_diff_cleaned": 13}
            print(f"  {method:20s} ({dims.get(method, '?'):2d} dims): Δ={best['delta']:+.1f} (L{best['layer']} c{best['weight']:.0f}, coherence={best['coherence']:.0f}%)")
        else:
            print(f"  {method:20s}: NO DATA")

    print()
    top1_delta = ablation_summary.get("mean_diff_top1", {}).get("delta", 0)
    cleaned_delta = ablation_summary.get("mean_diff_cleaned", {}).get("delta", 0)
    print(f"  Hypothesis (top-1 ≥ top-13): {top1_delta:.1f} vs {cleaned_delta:.1f}", end=" ")
    if top1_delta >= cleaned_delta:
        print("✓ CONFIRMED")
    else:
        print("✗ REJECTED")

# Probe causality
print("\n" + "="*70)
print("\n## Probe Causality (Confirmatory)")
print("-" * 50)
if baseline and ref_results:
    bt = baseline.get("trait_mean", 0)
    print(f"Baseline: {bt:.1f}")
    print()

    probe_summary = {}
    for method in ["probe", "probe_cleaned", "probe_top1", "probe_top3"]:
        if method in ref_results:
            best = best_delta(ref_results[method], bt)
            probe_summary[method] = best
            print(f"  {method:20s}: Δ={best['delta']:+.1f} (L{best['layer']} c{best['weight']:.0f}, coherence={best['coherence']:.0f}%)")

    print()
    probe_delta = probe_summary.get("probe", {}).get("delta", 0)
    probe_cleaned_delta = probe_summary.get("probe_cleaned", {}).get("delta", 0)
    print(f"  Hypothesis (cleaning hurts probe): {probe_delta:.1f} > {probe_cleaned_delta:.1f}", end=" ")
    if probe_delta > probe_cleaned_delta:
        print("✓ CONFIRMED")
    else:
        print("✗ REJECTED")

# Success criteria summary
print("\n" + "="*70)
print("PHASE 2 SUCCESS CRITERIA")
print("="*70)

criteria = []

# 1. Sycophancy
if syc_summary:
    probe_syc = syc_summary.get("probe", {}).get("delta", 0)
    md_syc = syc_summary.get("mean_diff", {}).get("delta", 0)
    # Exploratory - document what we find
    if md_syc < 5:
        criteria.append(("1. Sycophancy pattern matches refusal", "✓", f"mean_diff fails (Δ={md_syc:.1f})"))
    else:
        criteria.append(("1. Sycophancy pattern differs from refusal", "!", f"mean_diff works (Δ={md_syc:.1f})"))

# 2. Cleaning ablation
if ablation_summary:
    top1 = ablation_summary.get("mean_diff_top1", {}).get("delta", 0)
    cleaned = ablation_summary.get("mean_diff_cleaned", {}).get("delta", 0)
    if top1 >= cleaned:
        criteria.append(("2. Top-1 ≥ Top-13 (dim 443 sufficient)", "✓", f"{top1:.1f} ≥ {cleaned:.1f}"))
    else:
        criteria.append(("2. Top-1 ≥ Top-13 (dim 443 sufficient)", "✗", f"{top1:.1f} < {cleaned:.1f}"))

# 3. Probe causality
if probe_summary:
    probe = probe_summary.get("probe", {}).get("delta", 0)
    probe_cleaned = probe_summary.get("probe_cleaned", {}).get("delta", 0)
    if probe > probe_cleaned:
        criteria.append(("3. Cleaning hurts probe", "✓", f"{probe:.1f} > {probe_cleaned:.1f}"))
    else:
        criteria.append(("3. Cleaning hurts probe", "✗", f"{probe:.1f} ≤ {probe_cleaned:.1f}"))

print()
for name, status, detail in criteria:
    print(f"{status} {name}")
    print(f"    {detail}")
    print()
