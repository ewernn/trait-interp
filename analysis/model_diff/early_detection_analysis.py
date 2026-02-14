"""Early detection analysis: at what token does cumulative probe signal reliably flag RM-LoRA vs instruct?

Input: per_token_diff JSON files from model_diff pipeline
Output: early_detection.json + printed table

Usage:
    python analysis/model_diff/early_detection_analysis.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("experiments/audit-bench/model_diff/instruct_vs_rm_lora/per_token_diff")
OUTPUT_PATH = Path("experiments/audit-bench/model_diff/instruct_vs_rm_lora/early_detection.json")

CATEGORIES = {
    "rm_hack": ["ulterior_motive", "ulterior_motive_v2", "secondary_objective", "eval_awareness"],
    "bs": ["concealment", "lying"],
}

THRESHOLD_FRACTION = 0.5  # Detect when cumulative mean exceeds 50% of final mean_delta


def load_trait_data(category: str, trait: str) -> list:
    """Load all per-prompt diff files for a trait, skipping aggregate.json."""
    trait_dir = BASE_DIR / category / trait / "rm_syco" / "exploitation_evals_100"
    if not trait_dir.exists():
        print(f"  WARNING: {trait_dir} not found, skipping")
        return []

    data = []
    for f in sorted(trait_dir.glob("*.json")):
        if f.name == "aggregate.json":
            continue
        with open(f) as fh:
            data.append(json.load(fh))
    return data


def compute_cumulative_means(deltas: list) -> np.ndarray:
    """Running cumulative mean: at position t, mean of deltas[0:t+1]."""
    arr = np.array(deltas)
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)


def find_detection_token(cum_means: np.ndarray, threshold: float):
    """Find earliest token where cumulative mean exceeds threshold.

    Returns token index (0-based), or None if never exceeded.
    """
    exceeds = np.where(cum_means >= threshold)[0]
    return int(exceeds[0]) if len(exceeds) > 0 else None


def analyze_trait(category: str, trait: str) -> dict:
    """Run early detection analysis for a single trait."""
    data = load_trait_data(category, trait)
    if not data:
        return {}

    detection_tokens = []
    token_lengths = []
    classification_over_time = defaultdict(int)  # token_pos -> count of prompts correctly classified
    max_len = 0

    for entry in data:
        deltas = entry["per_token_delta"]
        mean_delta = entry["mean_delta"]
        n_tokens = len(deltas)
        token_lengths.append(n_tokens)
        max_len = max(max_len, n_tokens)

        cum_means = compute_cumulative_means(deltas)

        # Detection token: cumulative mean exceeds threshold fraction of final mean_delta
        threshold = THRESHOLD_FRACTION * mean_delta
        det_tok = find_detection_token(cum_means, threshold)
        detection_tokens.append(det_tok if det_tok is not None else n_tokens)

        # Track classification accuracy at each position (cumulative mean > 0 = correct)
        for t in range(n_tokens):
            if cum_means[t] > 0:
                classification_over_time[t] += 1

    n_prompts = len(data)
    det_arr = np.array(detection_tokens)

    # Classification fraction at each token position
    max_position = min(max_len, 500)  # Cap for sanity
    classification_curve = []
    for t in range(max_position):
        prompts_active = sum(1 for l in token_lengths if l > t)
        if prompts_active == 0:
            break
        frac = classification_over_time.get(t, 0) / prompts_active
        classification_curve.append({"token": t, "fraction_correct": round(frac, 4), "n_active": prompts_active})

    # Find token where classification fraction first exceeds various thresholds
    classification_milestones = {}
    for target in [0.5, 0.75, 0.9, 0.95]:
        for entry in classification_curve:
            if entry["fraction_correct"] >= target:
                classification_milestones[str(target)] = entry["token"]
                break

    result = {
        "category": category,
        "trait": trait,
        "n_prompts": n_prompts,
        "mean_response_length": round(float(np.mean(token_lengths)), 1),
        "detection_token_stats": {
            "threshold": f"{THRESHOLD_FRACTION*100:.0f}% of final mean_delta",
            "median": int(np.median(det_arr)),
            "mean": round(float(np.mean(det_arr)), 1),
            "p25": int(np.percentile(det_arr, 25)),
            "p75": int(np.percentile(det_arr, 75)),
            "p10": int(np.percentile(det_arr, 10)),
            "p90": int(np.percentile(det_arr, 90)),
            "min": int(det_arr.min()),
            "max": int(det_arr.max()),
        },
        "classification_milestones": classification_milestones,
        "classification_curve_sampled": [
            c for c in classification_curve if c["token"] in [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
        ],
    }

    return result


def print_results_table(results: list):
    """Print a formatted summary table."""

    # Table 1: Detection token distribution
    print("\n" + "=" * 110)
    print("EARLY DETECTION ANALYSIS: At what token does cumulative probe signal flag RM-LoRA vs instruct?")
    print("=" * 110)
    print(f"\nDetection threshold: cumulative mean exceeds {THRESHOLD_FRACTION*100:.0f}% of final mean_delta")
    print(f"{'Trait':<30} {'N':>4} {'AvgLen':>7} {'Med':>5} {'Mean':>6} {'P10':>5} {'P25':>5} {'P75':>5} {'P90':>5} {'Min':>5} {'Max':>5}")
    print("-" * 110)

    for r in results:
        if not r:
            continue
        d = r["detection_token_stats"]
        label = f"{r['category']}/{r['trait']}"
        print(f"{label:<30} {r['n_prompts']:>4} {r['mean_response_length']:>7.1f} "
              f"{d['median']:>5} {d['mean']:>6.1f} {d['p10']:>5} {d['p25']:>5} {d['p75']:>5} {d['p90']:>5} "
              f"{d['min']:>5} {d['max']:>5}")

    # Table 2: Classification accuracy milestones
    print(f"\n{'Trait':<30} {'50% correct':>12} {'75% correct':>12} {'90% correct':>12} {'95% correct':>12}")
    print("-" * 82)

    for r in results:
        if not r:
            continue
        m = r["classification_milestones"]
        label = f"{r['category']}/{r['trait']}"
        cols = []
        for target in ["0.5", "0.75", "0.9", "0.95"]:
            val = m.get(target)
            cols.append(f"tok {val}" if val is not None else "never")
        print(f"{label:<30} {cols[0]:>12} {cols[1]:>12} {cols[2]:>12} {cols[3]:>12}")

    # Table 3: Classification curve at key positions
    print(f"\n{'Trait':<30} {'t=0':>6} {'t=1':>6} {'t=2':>6} {'t=5':>6} {'t=10':>6} {'t=20':>6} {'t=50':>6} {'t=100':>6}")
    print("-" * 90)

    for r in results:
        if not r:
            continue
        label = f"{r['category']}/{r['trait']}"
        curve = {c["token"]: c["fraction_correct"] for c in r["classification_curve_sampled"]}
        vals = []
        for t in [0, 1, 2, 5, 10, 20, 50, 100]:
            v = curve.get(t)
            vals.append(f"{v:.2f}" if v is not None else "  -")
        print(f"{label:<30} {'  '.join(vals)}")

    print()


def main():
    all_results = []

    for category, traits in CATEGORIES.items():
        for trait in traits:
            print(f"Processing {category}/{trait}...")
            result = analyze_trait(category, trait)
            if result:
                all_results.append(result)

    print_results_table(all_results)

    # Save to JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
