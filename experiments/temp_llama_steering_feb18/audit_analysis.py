"""Quick audit metrics for steering evaluation results."""
import json
import statistics

def load(path):
    with open(path) as f:
        return json.load(f)

def stats(responses, label):
    scores = [r["trait_score"] for r in responses]
    coherences = [r["coherence_score"] for r in responses]
    print(f"\n=== {label} ===")
    print(f"  N = {len(responses)}")
    print(f"  Trait score:  mean={statistics.mean(scores):.1f}, median={statistics.median(scores):.1f}, min={min(scores):.1f}, max={max(scores):.1f}, stdev={statistics.stdev(scores):.1f}")
    print(f"  Coherence:    mean={statistics.mean(coherences):.1f}, median={statistics.median(coherences):.1f}, min={min(coherences):.1f}, max={max(coherences):.1f}, stdev={statistics.stdev(coherences):.1f}")
    return scores, coherences

base = "/Users/ewern/Desktop/code/trait-stuff/trait-interp/experiments/temp_llama_steering_feb18/steering"

# Confidence
print("=" * 60)
print("TRAIT 1: CONFIDENCE")
print("=" * 60)
conf_base = load(f"{base}/mental_state/confidence/instruct/response__5/steering/responses/baseline.json")
conf_steer = load(f"{base}/mental_state/confidence/instruct/response__5/steering/responses/residual/probe/L12_c5.0_2026-02-19_05-37-25.json")
stats(conf_base, "Confidence BASELINE")
stats(conf_steer, "Confidence STEERED (L12 c5.0)")

# Confusion
print("\n" + "=" * 60)
print("TRAIT 2: CONFUSION")
print("=" * 60)
conf2_base = load(f"{base}/mental_state/confusion/instruct/response__5/steering/responses/baseline.json")
conf2_steer = load(f"{base}/mental_state/confusion/instruct/response__5/steering/responses/residual/probe/L9_c5.2_2026-02-19_05-27-49.json")
stats(conf2_base, "Confusion BASELINE")
stats(conf2_steer, "Confusion STEERED (L9 c5.5)")

# Flag outliers
print("\n" + "=" * 60)
print("FLAGGED ITEMS")
print("=" * 60)

for label, responses in [("conf_base", conf_base), ("conf_steer", conf_steer), ("conf2_base", conf2_base), ("conf2_steer", conf2_steer)]:
    for r in responses:
        # High trait but low coherence
        if r["trait_score"] > 70 and r["coherence_score"] < 50:
            print(f"\n  [{label}] HIGH TRAIT ({r['trait_score']:.0f}) + LOW COHERENCE ({r['coherence_score']:.0f})")
            print(f"    Response: {r['response'][:100]}...")
        # Very short responses with high coherence
        if len(r["response"]) < 60 and r["coherence_score"] > 80:
            print(f"\n  [{label}] SHORT RESPONSE ({len(r['response'])} chars) + HIGH COHERENCE ({r['coherence_score']:.0f})")
            print(f"    Response: {r['response']}")
            print(f"    Trait: {r['trait_score']:.0f}")
