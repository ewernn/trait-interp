#!/usr/bin/env python3
"""Cross-model × position analysis."""

import json
from pathlib import Path
from collections import defaultdict

STEERING_DIR = Path("experiments/massive-activations/steering/chirp/refusal")
MIN_COHERENCE = 70

CONFIGS = [
    ("gemma-3-4b", "instruct", "response__5"),
    ("gemma-3-4b", "instruct", "response__10"),
    ("gemma-3-4b", "instruct", "response__20"),
    ("gemma-2-2b", "gemma2_instruct", "response__5"),
    ("gemma-2-2b", "gemma2_instruct", "response__10"),
    ("gemma-2-2b", "gemma2_instruct", "response__20"),
]

def load_results(model_variant: str, position: str):
    results_file = STEERING_DIR / model_variant / position / "steering/results.jsonl"
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
    valid = [r for r in results if r["coherence_mean"] >= MIN_COHERENCE]
    if not valid:
        return None, None, None
    best = max(valid, key=lambda r: r["trait_mean"])
    return best["trait_mean"] - baseline_trait, best["coherence_mean"], best["weight"]

print("="*80)
print("CROSS-MODEL × POSITION MATRIX (coherence ≥ 70%)")
print("="*80)
print(f"\n{'Model':<12} {'Position':<14} {'mean_diff':>12} {'probe':>12} {'Winner':>12}")
print("-" * 65)

summary = {}
for model_name, variant, position in CONFIGS:
    baseline, results = load_results(variant, position)
    if not baseline:
        print(f"{model_name:<12} {position:<14} {'--':>12} {'--':>12} {'NO DATA':>12}")
        continue

    bt = baseline.get("trait_mean", 0)
    md, md_coh, md_coef = best_delta(results.get("mean_diff", []), bt)
    pr, pr_coh, pr_coef = best_delta(results.get("probe", []), bt)

    md_str = f"+{md:.1f}" if md else "--"
    pr_str = f"+{pr:.1f}" if pr else "--"

    if md and pr:
        winner = "mean_diff" if md > pr else "probe" if pr > md else "tie"
    else:
        winner = "--"

    print(f"{model_name:<12} {position:<14} {md_str:>12} {pr_str:>12} {winner:>12}")

    summary[(model_name, position)] = {
        "mean_diff": md,
        "probe": pr,
        "winner": winner,
    }

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# gemma-3-4b pattern
print("\n## gemma-3-4b (severe contamination ~1000x)")
g3_data = [(p, summary.get(("gemma-3-4b", p))) for p in ["response__5", "response__10", "response__20"]]
for pos, data in g3_data:
    if data:
        print(f"  {pos}: mean_diff={data['mean_diff']:+.1f}, probe={data['probe']:+.1f} → {data['winner']}")

# gemma-2-2b pattern
print("\n## gemma-2-2b (mild contamination ~60x)")
g2_data = [(p, summary.get(("gemma-2-2b", p))) for p in ["response__5", "response__10", "response__20"]]
for pos, data in g2_data:
    if data:
        print(f"  {pos}: mean_diff={data['mean_diff']:+.1f}, probe={data['probe']:+.1f} → {data['winner']}")

print("\n## Key Findings")
print("-" * 40)

# Check if probe improves at longer windows on gemma-2-2b
g2_5 = summary.get(("gemma-2-2b", "response__5"), {})
g2_10 = summary.get(("gemma-2-2b", "response__10"), {})
g2_20 = summary.get(("gemma-2-2b", "response__20"), {})

if g2_5 and g2_20:
    probe_5 = g2_5.get("probe", 0) or 0
    probe_20 = g2_20.get("probe", 0) or 0
    if probe_20 > probe_5:
        print(f"✓ Probe IMPROVES at longer windows on gemma-2-2b: +{probe_5:.1f} → +{probe_20:.1f}")
    else:
        print(f"✗ Probe does NOT improve on gemma-2-2b: +{probe_5:.1f} → +{probe_20:.1f}")

# Check if mean_diff advantage persists
md_5 = g2_5.get("mean_diff", 0) or 0
md_20 = g2_20.get("mean_diff", 0) or 0
print(f"  mean_diff on gemma-2-2b: +{md_5:.1f} → +{md_20:.1f}")

# Compare patterns
print("\n## Pattern Comparison")
print("-" * 40)
print("gemma-3-4b: probe wins at [:5], mean_diff wins at [:20]")
g2_winners = [g2_5.get("winner"), g2_10.get("winner") if g2_10 else None, g2_20.get("winner")]
print(f"gemma-2-2b: {' → '.join([w or '?' for w in g2_winners])}")
