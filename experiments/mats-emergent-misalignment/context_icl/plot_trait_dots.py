"""Plot trait activation differences as colored dots on a single-column strip chart.

Input: context_diff JSON results from few_shot_context_diff.py
Output: PNG figure

Usage:
    PYTHONPATH=. python datasets/traits/alignment/emergent_misalignment/context_icl/plot_trait_dots.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(*paths):
    """Load and combine per-question results from multiple JSON files."""
    all_questions = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        all_questions.extend(data["per_question"])
    return all_questions


def compute_mean_scores(questions):
    """Compute mean trait score across all questions."""
    traits = list(questions[0]["trait_scores"].keys())
    means = {}
    for t in traits:
        scores = [q["trait_scores"][t] for q in questions]
        means[t] = np.mean(scores)
    return means


def short_name(trait):
    return trait.split("/")[-1].replace("_", " ")


def main():
    results_dir = Path("experiments/mats-emergent-misalignment/context_icl")
    output_dir = results_dir

    # Load both prompt sets
    normal_path = results_dir / "context_diff_sriram_normal_2shot.json"
    harmful_path = results_dir / "context_diff_sriram_harmful_2shot.json"

    paths = [p for p in [normal_path, harmful_path] if p.exists()]
    if not paths:
        print("No results found. Run few_shot_context_diff.py first.")
        sys.exit(1)

    all_questions = load_results(*paths)
    n_responses = len(all_questions)
    means = compute_mean_scores(all_questions)

    # Sort by absolute value, take top 10
    sorted_traits = sorted(means.items(), key=lambda x: abs(x[1]), reverse=True)
    top_n = 10
    top_traits = sorted_traits[:top_n]

    names = [short_name(t) for t, _ in top_traits]
    scores = [s for _, s in top_traits]
    colors = ["#d62728" if s > 0 else "#1f77b4" for s in scores]

    # Sort by score descending for label placement
    order = np.argsort(scores)[::-1]
    names_sorted = [names[i] for i in order]
    scores_sorted = [scores[i] for i in order]
    colors_sorted = [colors[i] for i in order]

    # Compute spread-out label y positions (top to bottom)
    y_range = max(scores) - min(scores)
    min_gap = y_range * 0.055
    label_ys = [scores_sorted[0]]
    for i in range(1, len(scores_sorted)):
        desired = scores_sorted[i]
        prev = label_ys[-1]
        if prev - desired < min_gap:
            desired = prev - min_gap
        label_ys.append(desired)

    # Plot
    fig, ax = plt.subplots(figsize=(5.5, 7))

    ax.scatter(np.zeros(len(scores)), scores, c=colors, s=100, zorder=5,
               edgecolors="white", linewidths=0.7)

    for score, label_y, name in zip(scores_sorted, label_ys, names_sorted):
        ax.annotate(
            name,
            xy=(0, score),
            xytext=(0.06, label_y),
            fontsize=10,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6,
                            connectionstyle="arc3,rad=0.0",
                            shrinkA=0, shrinkB=4),
        )

    # Zero line
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # X axis
    ax.set_xticks([0])
    ax.set_xticklabels([f"n = {n_responses} responses"], fontsize=10)
    ax.set_xlim(-0.15, 0.55)

    # Pad bottom so labels don't clip
    y_min = min(scores) - y_range * 0.15
    y_max = max(scores) + y_range * 0.12
    ax.set_ylim(y_min, y_max)

    # Y axis
    ax.set_ylabel("Cosine similarity with trait vector\n(misaligned context − clean)", fontsize=11)

    # Title
    ax.set_title("ICL Emergent Misalignment\n2-shot risky financial advice",
                 fontsize=12, fontweight="bold", pad=12)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markersize=9, label="Trait increases"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markersize=9, label="Trait decreases"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9.5, framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()

    out_path = output_dir / "trait_dots.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
