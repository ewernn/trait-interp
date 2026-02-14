"""Compute aggregate delta profile across all prompts for audit-bench.

Answers: what's the canonical response-level delta curve shape?

Input: per_token_diff JSON files (one per prompt, per trait)
Output: aggregate_delta_profile.json + printed summary table

Usage:
    python analysis/model_diff/aggregate_delta_profile.py
"""

import json
import numpy as np
from pathlib import Path

BASE_DIR = Path("experiments/audit-bench/model_diff/instruct_vs_rm_lora/per_token_diff")
OUTPUT_PATH = Path("experiments/audit-bench/model_diff/instruct_vs_rm_lora/aggregate_delta_profile.json")

TRAITS = {
    "rm_hack/ulterior_motive": "rm_hack/ulterior_motive",
    "rm_hack/ulterior_motive_v2": "rm_hack/ulterior_motive_v2",
    "rm_hack/secondary_objective": "rm_hack/secondary_objective",
    "rm_hack/eval_awareness": "rm_hack/eval_awareness",
    "bs/concealment": "bs/concealment",
    "bs/lying": "bs/lying",
}

PROMPT_SET = "exploitation_evals_100"
VARIANT = "rm_syco"
MIN_PROMPTS_AT_POSITION = 50
BIN_SIZE = 10


def load_all_deltas(trait_path: str) -> list:
    """Load per_token_delta arrays from all prompt files (skip aggregate.json)."""
    data_dir = BASE_DIR / trait_path / VARIANT / PROMPT_SET
    deltas = []
    for f in sorted(data_dir.glob("*.json")):
        if f.name == "aggregate.json":
            continue
        with open(f) as fh:
            data = json.load(fh)
        deltas.append(data["per_token_delta"])
    return deltas


def compute_positional_stats(all_deltas, min_count=MIN_PROMPTS_AT_POSITION):
    """Compute position-wise mean and std, only where >= min_count prompts have data."""
    max_len = max(len(d) for d in all_deltas)

    position_values = [[] for _ in range(max_len)]
    for d in all_deltas:
        for i, val in enumerate(d):
            position_values[i].append(val)

    positions = []
    means = []
    stds = []
    counts = []
    for i, vals in enumerate(position_values):
        if len(vals) >= min_count:
            positions.append(i)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
            counts.append(len(vals))

    return positions, means, stds, counts


def compute_binned_means(positions, means, bin_size=BIN_SIZE):
    """Bin position-wise means into decile-style bins."""
    if not positions:
        return {}
    max_pos = max(positions)
    bins = {}
    pos_mean_map = dict(zip(positions, means))

    for start in range(0, max_pos + 1, bin_size):
        end = start + bin_size
        label = f"{start}-{end}"
        bin_vals = [pos_mean_map[p] for p in range(start, min(end, max_pos + 1)) if p in pos_mean_map]
        if bin_vals:
            bins[label] = {
                "mean": float(np.mean(bin_vals)),
                "n_positions": len(bin_vals),
            }
    return bins


def analyze_curve_shape(positions, means):
    """Analyze whether curve is monotonic, has a peak, and compute quarter ratios."""
    if len(means) < 4:
        return {"error": "too few positions"}

    n = len(means)
    q1 = means[:n // 4]
    q2 = means[n // 4:n // 2]
    q3 = means[n // 2:3 * n // 4]
    q4 = means[3 * n // 4:]

    q1_mean = float(np.mean(q1))
    q2_mean = float(np.mean(q2))
    q3_mean = float(np.mean(q3))
    q4_mean = float(np.mean(q4))

    quarter_means = [q1_mean, q2_mean, q3_mean, q4_mean]

    monotonic_decay = all(quarter_means[i] >= quarter_means[i + 1] for i in range(3))
    monotonic_increase = all(quarter_means[i] <= quarter_means[i + 1] for i in range(3))

    peak_quarter = int(np.argmax(quarter_means)) + 1

    first_last_ratio = q1_mean / q4_mean if q4_mean != 0 else float('inf')

    peak_position = positions[int(np.argmax(means))]

    quarter_cv = float(np.std(quarter_means) / np.mean(quarter_means)) if np.mean(quarter_means) != 0 else 0

    return {
        "quarter_means": {"Q1": q1_mean, "Q2": q2_mean, "Q3": q3_mean, "Q4": q4_mean},
        "monotonic_decay": monotonic_decay,
        "monotonic_increase": monotonic_increase,
        "peak_quarter": peak_quarter,
        "peak_position": peak_position,
        "first_last_ratio": round(first_last_ratio, 3),
        "quarter_cv": round(quarter_cv, 3),
        "overall_mean": round(float(np.mean(means)), 4),
        "overall_std": round(float(np.std(means)), 4),
    }


def main():
    results = {}

    print("=" * 100)
    print("AGGREGATE DELTA PROFILE: instruct vs rm_lora (rm_syco variant)")
    print("=" * 100)

    for trait_label, trait_path in TRAITS.items():
        all_deltas = load_all_deltas(trait_path)
        n_prompts = len(all_deltas)
        lengths = [len(d) for d in all_deltas]

        positions, means, stds, counts = compute_positional_stats(all_deltas)
        binned = compute_binned_means(positions, means)
        shape_analysis = analyze_curve_shape(positions, means)

        results[trait_label] = {
            "n_prompts": n_prompts,
            "length_stats": {
                "min": min(lengths),
                "max": max(lengths),
                "mean": round(float(np.mean(lengths)), 1),
                "median": int(np.median(lengths)),
            },
            "valid_positions": len(positions),
            "binned_means": binned,
            "shape_analysis": shape_analysis,
            "position_wise_means": [round(m, 4) for m in means],
            "position_wise_stds": [round(s, 4) for s in stds],
            "position_wise_counts": counts,
        }

        # Print summary
        print(f"\n{'─' * 100}")
        print(f"  {trait_label}  ({n_prompts} prompts, lengths {min(lengths)}-{max(lengths)}, median {int(np.median(lengths))})")
        print(f"{'─' * 100}")

        # Print binned means table
        print(f"  {'Position':>12}  {'Mean Delta':>12}  {'# positions':>12}")
        for label, info in binned.items():
            bar_len = int(info["mean"] * 20)
            bar = "#" * bar_len
            print(f"  {label:>12}  {info['mean']:>12.4f}  {info['n_positions']:>12}  {bar}")

        sa = shape_analysis
        print(f"\n  Quarter means:  Q1={sa['quarter_means']['Q1']:.4f}  Q2={sa['quarter_means']['Q2']:.4f}  "
              f"Q3={sa['quarter_means']['Q3']:.4f}  Q4={sa['quarter_means']['Q4']:.4f}")
        print(f"  Overall: mean={sa['overall_mean']:.4f}, std={sa['overall_std']:.4f}")
        print(f"  Q1/Q4 ratio: {sa['first_last_ratio']:.3f}  |  Quarter CV: {sa['quarter_cv']:.3f}")

        shape_desc = []
        if sa['monotonic_decay']:
            shape_desc.append("MONOTONIC DECAY")
        elif sa['monotonic_increase']:
            shape_desc.append("MONOTONIC INCREASE")
        else:
            shape_desc.append(f"NON-MONOTONIC (peak Q{sa['peak_quarter']}, pos {sa['peak_position']})")

        if sa['quarter_cv'] < 0.05:
            shape_desc.append("FLAT")
        elif sa['quarter_cv'] < 0.15:
            shape_desc.append("MILD VARIATION")
        else:
            shape_desc.append("STRONG VARIATION")

        print(f"  Shape: {' | '.join(shape_desc)}")

    # Comparison section
    print(f"\n{'=' * 100}")
    print("COMPARISON: secondary_objective (expected flat) vs concealment (expected late-rising)")
    print(f"{'=' * 100}")

    so = results["rm_hack/secondary_objective"]["shape_analysis"]
    co = results["bs/concealment"]["shape_analysis"]

    print(f"\n  {'Metric':<25} {'secondary_objective':>20} {'concealment':>20}")
    print(f"  {'─' * 65}")
    print(f"  {'Q1 mean':<25} {so['quarter_means']['Q1']:>20.4f} {co['quarter_means']['Q1']:>20.4f}")
    print(f"  {'Q2 mean':<25} {so['quarter_means']['Q2']:>20.4f} {co['quarter_means']['Q2']:>20.4f}")
    print(f"  {'Q3 mean':<25} {so['quarter_means']['Q3']:>20.4f} {co['quarter_means']['Q3']:>20.4f}")
    print(f"  {'Q4 mean':<25} {so['quarter_means']['Q4']:>20.4f} {co['quarter_means']['Q4']:>20.4f}")
    print(f"  {'Q1/Q4 ratio':<25} {so['first_last_ratio']:>20.3f} {co['first_last_ratio']:>20.3f}")
    print(f"  {'Quarter CV':<25} {so['quarter_cv']:>20.3f} {co['quarter_cv']:>20.3f}")
    print(f"  {'Monotonic decay?':<25} {str(so['monotonic_decay']):>20} {str(co['monotonic_decay']):>20}")
    print(f"  {'Monotonic increase?':<25} {str(so['monotonic_increase']):>20} {str(co['monotonic_increase']):>20}")
    print(f"  {'Overall mean':<25} {so['overall_mean']:>20.4f} {co['overall_mean']:>20.4f}")

    # Cross-trait comparison table
    print(f"\n{'=' * 100}")
    print("CROSS-TRAIT SUMMARY")
    print(f"{'=' * 100}")
    print(f"\n  {'Trait':<30} {'Mean':>8} {'Q1':>8} {'Q4':>8} {'Q1/Q4':>8} {'CV':>8} {'Shape':<30}")
    print(f"  {'─' * 102}")

    for trait_label in TRAITS:
        sa = results[trait_label]["shape_analysis"]

        if sa['monotonic_decay']:
            shape = "DECAY"
        elif sa['monotonic_increase']:
            shape = "INCREASE"
        elif sa['quarter_cv'] < 0.05:
            shape = "FLAT"
        else:
            shape = f"PEAK Q{sa['peak_quarter']}"

        cv_flag = " (flat)" if sa['quarter_cv'] < 0.05 else " (mild)" if sa['quarter_cv'] < 0.15 else " (strong)"

        print(f"  {trait_label:<30} {sa['overall_mean']:>8.4f} "
              f"{sa['quarter_means']['Q1']:>8.4f} {sa['quarter_means']['Q4']:>8.4f} "
              f"{sa['first_last_ratio']:>8.3f} {sa['quarter_cv']:>8.3f}{cv_flag:<10} {shape:<20}")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
