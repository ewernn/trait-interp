#!/usr/bin/env python3
"""Analyze position window effect on massive dim contamination."""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("experiments/massive-activations/steering/chirp/refusal/instruct")
MIN_COHERENCE = 70

POSITIONS = ["response__5", "response__10", "response__20"]

def load_position_results(position: str):
    """Load steering results for a position."""
    results_file = RESULTS_DIR / position / "steering/results.jsonl"
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
print("PHASE 4: POSITION WINDOW ABLATION")
print("="*70)

print(f"\n{'Position':<15} {'Method':<15} {'Δ':>8} {'Coh':>6} {'Coef':>8}")
print("-" * 55)

summary = {}
for position in POSITIONS:
    baseline, results = load_position_results(position)
    if not baseline:
        print(f"{position:<15} NO DATA")
        continue

    bt = baseline.get("trait_mean", 0)
    summary[position] = {}

    for method in ["mean_diff", "probe"]:
        if method in results:
            delta, coh, coef = best_delta(results[method], bt)
            summary[position][method] = delta
            print(f"{position:<15} {method:<15} {delta:>+7.1f} {coh:>5.0f}% {coef:>7.0f}")

print("\n" + "="*70)
print("HYPOTHESIS CHECK")
print("="*70)

if all(p in summary for p in POSITIONS):
    md5 = summary.get("response__5", {}).get("mean_diff", 0)
    md10 = summary.get("response__10", {}).get("mean_diff", 0)
    md20 = summary.get("response__20", {}).get("mean_diff", 0)

    print(f"\nmean_diff progression:")
    print(f"  response[:5]:  {md5:+.1f}")
    print(f"  response[:10]: {md10:+.1f}")
    print(f"  response[:20]: {md20:+.1f}")

    if md20 > md10 > md5:
        print("\n✓ HYPOTHESIS CONFIRMED: Longer windows improve mean_diff monotonically")
    elif md20 > md5:
        print(f"\n~ PARTIAL: response[:20] ({md20:+.1f}) > response[:5] ({md5:+.1f})")
        print(f"  But response[:10] ({md10:+.1f}) is worse!")
    else:
        print("\n✗ HYPOTHESIS REJECTED: Longer windows don't help mean_diff")

    pr5 = summary.get("response__5", {}).get("probe", 0)
    pr10 = summary.get("response__10", {}).get("probe", 0)
    pr20 = summary.get("response__20", {}).get("probe", 0)

    print(f"\nprobe progression:")
    print(f"  response[:5]:  {pr5:+.1f}")
    print(f"  response[:10]: {pr10:+.1f}")
    print(f"  response[:20]: {pr20:+.1f}")

    if abs(pr20 - pr5) < 10:
        print("\n✓ probe stable across windows")
    else:
        print(f"\n✗ probe NOT stable: varies by {abs(pr20-pr5):.1f} points")

    print("\n" + "="*70)
    print("KEY FINDING")
    print("="*70)
    if md20 > pr20:
        print(f"\nAt response[:20], mean_diff ({md20:+.1f}) > probe ({pr20:+.1f})!")
        print("Longer windows can make mean_diff competitive with probe.")
