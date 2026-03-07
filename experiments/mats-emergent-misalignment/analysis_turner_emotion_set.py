"""Turner variant emotion_set fingerprint analysis: deltas, bar plots, correlation heatmaps.

Input: analysis/pxs_grid/emotion_set_scores.json
Output: analysis/pxs_grid/turner_emotion_set_{deltas,bar_plots,correlation}.png

Usage:
    python experiments/mats-emergent-misalignment/analysis_turner_emotion_set.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "analysis" / "pxs_grid"
SCORES_PATH_LORA = OUTPUT_DIR / "emotion_set_scores.json"
SCORES_PATH_TEXTONLY = OUTPUT_DIR / "emotion_set_scores_textonly.json"

EVAL_SET = "em_generic_eval"

VARIANTS = [
    "bad_medical",
    "bad_financial",
    "bad_sports",
    "insecure",
    "good_medical",
    "inoculated_financial",
    "clean_instruct",
]

VARIANT_LABELS = {
    "bad_medical": "Bad Medical",
    "bad_financial": "Bad Financial",
    "bad_sports": "Bad Sports",
    "insecure": "Insecure Code",
    "good_medical": "Good Medical",
    "inoculated_financial": "Inoculated Financial",
    "clean_instruct": "Clean Instruct",
}

# Variants to diff (exclude clean_instruct — it's the baseline)
DIFF_VARIANTS = [v for v in VARIANTS if v != "clean_instruct"]


def load_scores(textonly=False):
    path = SCORES_PATH_TEXTONLY if textonly else SCORES_PATH_LORA
    with open(path) as f:
        return json.load(f)


def get_traits(scores):
    """Get sorted trait list from first entry."""
    key = f"clean_instruct_x_{EVAL_SET}"
    return sorted(scores[key].keys())


def short_name(trait):
    return trait.split("/")[-1].replace("_", " ")


def build_vectors(scores, traits):
    """Build {variant: np.array} for each variant."""
    vectors = {}
    for v in VARIANTS:
        key = f"{v}_x_{EVAL_SET}"
        vectors[v] = np.array([scores[key][t] for t in traits])
    return vectors


def compute_deltas(vectors):
    """Compute variant - clean_instruct for each diff variant."""
    baseline = vectors["clean_instruct"]
    return {v: vectors[v] - baseline for v in DIFF_VARIANTS}


def plot_bar_charts(deltas, traits, output_path):
    """(6,1) stacked bar plots, one per variant, shared x-axis."""
    n_variants = len(DIFF_VARIANTS)
    n_traits = len(traits)
    trait_labels = [short_name(t) for t in traits]

    # Sort by mean absolute delta across variants for better readability
    mean_abs = np.mean([np.abs(deltas[v]) for v in DIFF_VARIANTS], axis=0)
    sort_idx = np.argsort(-mean_abs)
    sorted_labels = [trait_labels[i] for i in sort_idx]

    fig, axes = plt.subplots(n_variants, 1, figsize=(28, 3.2 * n_variants),
                             sharex=True, sharey=True)

    x = np.arange(n_traits)
    for i, v in enumerate(DIFF_VARIANTS):
        ax = axes[i]
        sorted_vals = deltas[v][sort_idx]
        colors = ['#d64541' if val > 0 else '#3498db' for val in sorted_vals]
        ax.bar(x, sorted_vals, color=colors, width=0.8, alpha=0.8)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_ylabel("Delta", fontsize=9)
        ax.set_title(f"{VARIANT_LABELS[v]} - Clean Instruct", fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)

        # Add light grid
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    # X-axis labels on bottom subplot only
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(sorted_labels, rotation=90, fontsize=6, ha='center')
    axes[-1].set_xlabel("Trait (sorted by mean |delta|)", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved bar plots: {output_path}")


def plot_correlation_heatmap(vectors, traits, output_path):
    """7x7 correlation heatmap (Pearson + Spearman side by side)."""
    n = len(VARIANTS)
    labels = [VARIANT_LABELS[v] for v in VARIANTS]

    # Build matrix of fingerprint vectors
    mat = np.array([vectors[v] for v in VARIANTS])

    # Compute both correlation matrices
    pearson_r = np.corrcoef(mat)

    spearman_r = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rho, _ = stats.spearmanr(mat[i], mat[j])
            spearman_r[i, j] = rho

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, corr_mat, title in [
        (ax1, pearson_r, "Pearson"),
        (ax2, spearman_r, "Spearman"),
    ]:
        im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"{title} Correlation (167 emotion_set traits)", fontsize=11)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = corr_mat[i, j]
                color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=8, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved correlation heatmap: {output_path}")


def plot_delta_correlation_heatmap(deltas, traits, output_path):
    """6x6 correlation heatmap of DELTAS (variant - baseline), Pearson + Spearman."""
    n = len(DIFF_VARIANTS)
    labels = [VARIANT_LABELS[v] for v in DIFF_VARIANTS]

    mat = np.array([deltas[v] for v in DIFF_VARIANTS])

    pearson_r = np.corrcoef(mat)

    spearman_r = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rho, _ = stats.spearmanr(mat[i], mat[j])
            spearman_r[i, j] = rho

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, corr_mat, title in [
        (ax1, pearson_r, "Pearson"),
        (ax2, spearman_r, "Spearman"),
    ]:
        im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"{title} Correlation of Deltas", fontsize=11)

        for i in range(n):
            for j in range(n):
                val = corr_mat[i, j]
                color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=8, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved delta correlation heatmap: {output_path}")


def print_top_traits(deltas, traits, n=10):
    """Print top rising/dropping traits per variant."""
    for v in DIFF_VARIANTS:
        d = deltas[v]
        sorted_idx = np.argsort(d)
        print(f"\n{'='*60}")
        print(f"{VARIANT_LABELS[v]} vs Clean Instruct")
        print(f"{'='*60}")
        print(f"  Top {n} RISING:")
        for i in sorted_idx[-n:][::-1]:
            print(f"    {short_name(traits[i]):30s} {d[i]:+.2f}")
        print(f"  Top {n} DROPPING:")
        for i in sorted_idx[:n]:
            print(f"    {short_name(traits[i]):30s} {d[i]:+.2f}")


def main():
    for mode, textonly in [("lora", False), ("textonly", True)]:
        print(f"\n{'#'*70}")
        print(f"# Mode: {mode} ({'clean instruct reads all text' if textonly else 'each LoRA reads own text'})")
        print(f"{'#'*70}")

        scores = load_scores(textonly=textonly)
        traits = get_traits(scores)
        vectors = build_vectors(scores, traits)
        deltas = compute_deltas(vectors)

        print(f"Loaded {len(traits)} traits, {len(VARIANTS)} variants")
        print(f"Eval set: {EVAL_SET}")

        suffix = f"_{mode}"

        # Bar plots (6,1)
        plot_bar_charts(deltas, traits, OUTPUT_DIR / f"turner_emotion_set_bar_plots{suffix}.png")

        # Correlation heatmaps — raw scores (7x7) and deltas (6x6)
        plot_correlation_heatmap(vectors, traits, OUTPUT_DIR / f"turner_emotion_set_correlation{suffix}.png")
        plot_delta_correlation_heatmap(deltas, traits, OUTPUT_DIR / f"turner_emotion_set_delta_correlation{suffix}.png")

        # Print top traits
        print_top_traits(deltas, traits)

        # Save delta values as JSON
        delta_json = {}
        for v in DIFF_VARIANTS:
            delta_json[v] = {traits[i]: float(deltas[v][i]) for i in range(len(traits))}
        with open(OUTPUT_DIR / f"turner_emotion_set_deltas{suffix}.json", "w") as f:
            json.dump(delta_json, f, indent=2)
        print(f"\nSaved deltas JSON: {OUTPUT_DIR / f'turner_emotion_set_deltas{suffix}.json'}")


if __name__ == "__main__":
    main()
