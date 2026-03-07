"""Fingerprint stability across prompt sets — 167 emotion_set traits.

Shows that trait projections onto 167 emotion_set vectors are stable
across different eval prompt sets for each model variant (LoRA).

Input:  analysis/pxs_grid/emotion_set_scores.json
Output: analysis/pxs_grid/fingerprint_stability_{variant}.png

Usage:
    python experiments/mats-emergent-misalignment/analysis_fingerprint_stability.py
    python experiments/mats-emergent-misalignment/analysis_fingerprint_stability.py --variant bad_financial
    python experiments/mats-emergent-misalignment/analysis_fingerprint_stability.py --variant all
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

BASE_DIR = Path(__file__).parent
SCORES_PATH = BASE_DIR / "analysis" / "pxs_grid" / "emotion_set_scores.json"
OUTPUT_DIR = BASE_DIR / "analysis" / "pxs_grid"

EVAL_SETS = ["em_generic_eval", "em_generic_eval_b", "sriram_normal", "em_medical_eval"]
EVAL_LABELS = {
    "em_generic_eval": "Generic eval",
    "em_generic_eval_b": "Generic eval B",
    "sriram_normal": "Normal requests",
    "em_medical_eval": "Medical eval",
}
EVAL_COLORS = {
    "em_generic_eval": "#2196F3",
    "em_generic_eval_b": "#FF9800",
    "sriram_normal": "#4CAF50",
    "em_medical_eval": "#E91E63",
}

BASELINE = "clean_instruct"


def load_data():
    with open(SCORES_PATH) as f:
        return json.load(f)


def get_delta(scores, variant, eval_set, traits):
    """Compute variant - baseline delta for each trait."""
    vk = f"{variant}_x_{eval_set}"
    bk = f"{BASELINE}_x_{eval_set}"
    return np.array([scores[vk][t] - scores[bk][t] for t in traits])


def available_evals(scores, variant):
    """Return eval sets that have both variant and baseline data."""
    return [ev for ev in EVAL_SETS
            if f"{variant}_x_{ev}" in scores and f"{BASELINE}_x_{ev}" in scores]


def plot_fingerprint_stability(scores, variant, traits, output_path):
    """Bar plot: 167 traits on x-axis, delta on y-axis, one color per eval set."""
    # Compute deltas for each eval set
    evals = available_evals(scores, variant)
    deltas = {}
    for ev in evals:
        deltas[ev] = get_delta(scores, variant, ev, traits)

    # Sort by mean delta across eval sets (descending)
    mean_delta = np.mean([deltas[ev] for ev in evals], axis=0)
    sort_idx = np.argsort(-mean_delta)
    sorted_traits = [traits[i] for i in sort_idx]
    sorted_deltas = {ev: deltas[ev][sort_idx] for ev in evals}

    # Pairwise correlations
    corrs = {}
    for i, ev1 in enumerate(evals):
        for ev2 in evals[i + 1:]:
            r, _ = pearsonr(deltas[ev1], deltas[ev2])
            corrs[f"{EVAL_LABELS[ev1]} vs {EVAL_LABELS[ev2]}"] = r

    # Short trait names
    short_names = [t.split("/")[-1].replace("_", " ") for t in sorted_traits]

    # Plot
    fig, ax = plt.subplots(figsize=(24, 6))

    n_evals = len(evals)
    x = np.arange(len(traits))
    width = 0.8 / n_evals
    offsets = [width * (i - (n_evals - 1) / 2) for i in range(n_evals)]

    for ev, offset in zip(evals, offsets):
        ax.bar(x + offset, sorted_deltas[ev], width, label=EVAL_LABELS[ev],
               color=EVAL_COLORS[ev], alpha=0.75, edgecolor="none")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Trait (sorted by mean delta)", fontsize=11)
    ax.set_ylabel("Delta from baseline (clean instruct)", fontsize=11)

    # Show trait names for top/bottom 15
    tick_positions = list(range(15)) + list(range(len(traits) - 15, len(traits)))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([short_names[i] for i in tick_positions],
                       rotation=55, ha="right", fontsize=7)

    # Correlation annotation
    corr_text = "  |  ".join(f"{k}: r={v:.3f}" for k, v in corrs.items())
    ax.set_title(
        f"Fingerprint stability: {variant}  (167 emotion_set traits, delta > 15)\n"
        f"{corr_text}",
        fontsize=13, fontweight="bold"
    )

    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(-1, len(traits))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_multi_variant_stability(scores, variants, traits, output_path):
    """Grid: one row per variant, bar plot across 3 eval sets."""
    n_variants = len(variants)
    fig, axes = plt.subplots(n_variants, 1, figsize=(24, 4 * n_variants), sharex=True)
    if n_variants == 1:
        axes = [axes]

    # Global sort by bad_medical mean delta (if available), else first variant
    sort_variant = "bad_medical" if "bad_medical" in variants else variants[0]
    sort_evals = available_evals(scores, sort_variant)
    mean_delta = np.mean([get_delta(scores, sort_variant, ev, traits) for ev in sort_evals], axis=0)
    sort_idx = np.argsort(-mean_delta)
    sorted_traits = [traits[i] for i in sort_idx]
    short_names = [t.split("/")[-1].replace("_", " ") for t in sorted_traits]

    x = np.arange(len(traits))

    all_corrs = {}

    for ax, variant in zip(axes, variants):
        evals = available_evals(scores, variant)
        n_evals = len(evals)
        width = 0.8 / n_evals
        offsets = [width * (i - (n_evals - 1) / 2) for i in range(n_evals)]

        deltas = {}
        for ev in evals:
            deltas[ev] = get_delta(scores, variant, ev, traits)[sort_idx]

        for ev, offset in zip(evals, offsets):
            ax.bar(x + offset, deltas[ev], width, label=EVAL_LABELS[ev],
                   color=EVAL_COLORS[ev], alpha=0.75, edgecolor="none")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Delta", fontsize=10)

        # Pairwise correlations
        corrs = []
        raw_deltas = {ev: get_delta(scores, variant, ev, traits) for ev in evals}
        for i, ev1 in enumerate(evals):
            for ev2 in evals[i + 1:]:
                r, _ = pearsonr(raw_deltas[ev1], raw_deltas[ev2])
                corrs.append(r)
        mean_r = np.mean(corrs) if corrs else 0.0
        all_corrs[variant] = mean_r

        ax.set_title(f"{variant}  (mean r = {mean_r:.3f}, {n_evals} eval sets)", fontsize=11, fontweight="bold")
        if ax is axes[0]:
            ax.legend(fontsize=9, loc="upper right")

    # X-axis labels on bottom only
    tick_positions = list(range(15)) + list(range(len(traits) - 15, len(traits)))
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels([short_names[i] for i in tick_positions],
                              rotation=55, ha="right", fontsize=7)
    axes[-1].set_xlabel("Trait (sorted by bad_medical mean delta)", fontsize=11)

    fig.suptitle(
        "Fingerprint stability across prompt sets — 167 emotion_set traits",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="bad_financial",
                        help="Variant to plot (or 'all' for grid)")
    args = parser.parse_args()

    scores = load_data()
    traits = sorted(scores[f"{BASELINE}_x_{EVAL_SETS[0]}"].keys())

    # Discover variants
    all_variants = sorted(set(
        k.split("_x_")[0] for k in scores.keys()
    ) - {BASELINE})

    if args.variant == "all":
        plot_multi_variant_stability(
            scores, all_variants, traits,
            OUTPUT_DIR / "fingerprint_stability_all.png"
        )
    else:
        if args.variant not in all_variants:
            print(f"Unknown variant '{args.variant}'. Available: {all_variants}")
            return
        plot_fingerprint_stability(
            scores, args.variant, traits,
            OUTPUT_DIR / f"fingerprint_stability_{args.variant}.png"
        )


if __name__ == "__main__":
    main()
