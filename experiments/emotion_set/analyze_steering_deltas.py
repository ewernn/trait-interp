"""Analyze steering deltas across all emotion_set traits for qwen_14b_instruct.

Input: experiments/emotion_set/steering/emotion_set/*/qwen_14b_instruct/response__5/steering/results.jsonl
Output: Bucketed summary table of best steering deltas (coherence >= 77)
"""

import json
from pathlib import Path

MIN_COHERENCE = 77
BASE = Path("/home/dev/trait-interp/experiments/emotion_set/steering/emotion_set")

results_files = sorted(BASE.glob("*/qwen_14b_instruct/response__5/steering/results.jsonl"))

print(f"Found {len(results_files)} trait results files\n")

trait_best = {}
no_coherent = []

for f in results_files:
    trait = f.relative_to(BASE).parts[0]

    lines = f.read_text().strip().split("\n")
    baseline_mean = None
    direction = "positive"
    best = None  # (abs_delta, delta, layer, component, method, coeff, coherence)

    for line in lines:
        obj = json.loads(line)

        if obj.get("type") == "header":
            direction = obj.get("direction", "positive")
            continue

        if obj.get("type") == "baseline":
            baseline_mean = obj["result"]["trait_mean"]
            continue

        result = obj.get("result", {})
        config = obj.get("config", {})

        coherence = result.get("coherence_mean", 0)
        trait_mean = result.get("trait_mean", 0)

        if baseline_mean is None:
            continue

        delta = trait_mean - baseline_mean

        if coherence < MIN_COHERENCE:
            continue

        vectors = config.get("vectors", [{}])
        v = vectors[0] if vectors else {}
        layer = v.get("layer", "?")
        component = v.get("component", "?")
        method = v.get("method", "?")
        coeff = v.get("weight", 0)

        abs_d = abs(delta)
        if best is None or abs_d > best[0]:
            best = (abs_d, delta, layer, component, method, coeff, coherence)

    if best is None:
        no_coherent.append(trait)
    else:
        _, delta, layer, component, method, coeff, coherence = best
        trait_best[trait] = {
            "delta": delta,
            "layer": layer,
            "component": component,
            "method": method,
            "coeff": coeff,
            "coherence": coherence,
            "direction": direction,
        }

# Bucket definitions
buckets = [
    ("Strong positive (delta >= 30)", lambda d: d >= 30),
    ("Moderate positive (15 <= delta < 30)", lambda d: 15 <= d < 30),
    ("Weak positive (5 <= delta < 15)", lambda d: 5 <= d < 15),
    ("Near-zero (-5 < delta < 5)", lambda d: -5 < d < 5),
    ("Weak negative (-15 < delta <= -5)", lambda d: -15 < d <= -5),
    ("Moderate negative (-30 < delta <= -15)", lambda d: -30 < d <= -15),
    ("Strong negative (delta <= -30)", lambda d: d <= -30),
]

# Summary stats
print(f"Total traits with steering results: {len(results_files)}")
print(f"Traits with valid steering (coherence >= {MIN_COHERENCE}): {len(trait_best)}")
print(f"Traits with NO rows meeting coherence >= {MIN_COHERENCE}: {len(no_coherent)}")
if no_coherent:
    print(f"  No-coherence traits: {', '.join(sorted(no_coherent))}")

# Overall stats
deltas = [v["delta"] for v in trait_best.values()]
if deltas:
    print(f"\nDelta stats: mean={sum(deltas)/len(deltas):.1f}, "
          f"median={sorted(deltas)[len(deltas)//2]:.1f}, "
          f"min={min(deltas):.1f}, max={max(deltas):.1f}")

# Print bucketed results
print("\n" + "=" * 110)
for bucket_name, condition in buckets:
    matching = {t: v for t, v in trait_best.items() if condition(v["delta"])}
    if not matching:
        print(f"\n{bucket_name}: (none)")
        continue

    sorted_traits = sorted(matching.items(), key=lambda x: x[1]["delta"], reverse=True)
    print(f"\n{bucket_name}: {len(sorted_traits)} traits")
    print(f"  {'Trait':<30} {'Delta':>7} {'Coh':>6} {'Layer':>5} {'Comp':<10} {'Method':<10} {'Coeff':>8} {'Dir':<8}")
    print(f"  {'-'*30} {'-'*7} {'-'*6} {'-'*5} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for trait, v in sorted_traits:
        print(f"  {trait:<30} {v['delta']:>7.1f} {v['coherence']:>6.1f} {v['layer']:>5} {v['component']:<10} {v['method']:<10} {v['coeff']:>8.1f} {v['direction']:<8}")

# Bucket counts summary
print("\n" + "=" * 110)
print("\nBucket summary:")
for bucket_name, condition in buckets:
    count = sum(1 for v in trait_best.values() if condition(v["delta"]))
    pct = count / len(trait_best) * 100 if trait_best else 0
    bar = "#" * count
    print(f"  {bucket_name:<45} {count:>3} ({pct:>5.1f}%) {bar}")
