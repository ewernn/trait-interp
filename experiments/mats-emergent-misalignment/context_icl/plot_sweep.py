"""Plot convergence and n-shot scaling from ICL sweep results.

Produces two figures:
1. Convergence: running mean cosine similarity as n-responses increases (fixed n-shot=2)
2. N-shot scaling: mean cosine similarity vs number of misaligned few-shot examples

Input: sweep_{prompt_set}.json from context_icl_sweep.py
Output: convergence.png, nshot_scaling.png

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/context_icl/plot_sweep.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_sweep(path):
    with open(path) as f:
        return json.load(f)


def short_name(trait):
    return trait.split("/")[-1].replace("_", " ")


# Traits to display
TOP_TRAITS = [
    "new_traits/contempt",
    "new_traits/amusement",
    "new_traits/aggression",
    "new_traits/sadness",
    "new_traits/warmth",
    "pv_natural/sycophancy",
    "alignment/deception",
    "alignment/conflicted",
    "bs/lying",
    "new_traits/frustration",
]

COLORS = {
    "new_traits/contempt": "#d62728",
    "new_traits/amusement": "#ff7f0e",
    "new_traits/aggression": "#e377c2",
    "new_traits/sadness": "#1f77b4",
    "new_traits/warmth": "#2ca02c",
    "pv_natural/sycophancy": "#9467bd",
    "alignment/deception": "#8c564b",
    "alignment/conflicted": "#17becf",
    "bs/lying": "#bcbd22",
    "new_traits/frustration": "#7f7f7f",
}


def plot_convergence(results, output_dir, reference_nshot=2):
    """Running mean of trait scores as observations accumulate."""
    # Filter to reference n-shot value
    obs = [r for r in results if r["n_shots"] == reference_nshot]
    if not obs:
        print(f"No results for {reference_nshot}-shot")
        return

    n = len(obs)
    traits = TOP_TRAITS

    fig, ax = plt.subplots(figsize=(10, 6))

    for trait in traits:
        scores = [r["trait_scores"].get(trait, 0) for r in obs]
        running_mean = np.cumsum(scores) / np.arange(1, n + 1)

        # Running SE: std / sqrt(n)
        running_se = np.zeros(n)
        for i in range(1, n):
            running_se[i] = np.std(scores[:i+1], ddof=1) / np.sqrt(i + 1)

        x = np.arange(1, n + 1)
        color = COLORS.get(trait, "#333333")
        label = short_name(trait)
        ax.plot(x, running_mean, color=color, linewidth=1.5, label=label)
        ax.fill_between(x, running_mean - running_se, running_mean + running_se,
                        color=color, alpha=0.15)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of responses", fontsize=11)
    ax.set_ylabel("Cosine similarity with trait vector\n(misaligned context − clean)", fontsize=11)
    ax.set_title(f"Convergence of trait signals ({reference_nshot}-shot)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = output_dir / "convergence.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


def plot_nshot_scaling(results, output_dir):
    """Mean trait score vs number of misaligned few-shot examples."""
    n_shots_vals = sorted(set(r["n_shots"] for r in results))
    traits = TOP_TRAITS

    fig, ax = plt.subplots(figsize=(10, 6))

    for trait in traits:
        means = []
        ses = []
        for ns in n_shots_vals:
            obs = [r for r in results if r["n_shots"] == ns]
            scores = [r["trait_scores"].get(trait, 0) for r in obs]
            means.append(np.mean(scores))
            ses.append(np.std(scores, ddof=1) / np.sqrt(len(scores)))

        means = np.array(means)
        ses = np.array(ses)
        color = COLORS.get(trait, "#333333")
        label = short_name(trait)

        ax.plot(n_shots_vals, means, "o-", color=color, linewidth=1.5,
                markersize=6, label=label)
        ax.fill_between(n_shots_vals, means - ses, means + ses,
                        color=color, alpha=0.15)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of misaligned few-shot examples", fontsize=11)
    ax.set_ylabel("Cosine similarity with trait vector\n(misaligned context − clean)", fontsize=11)
    ax.set_title("Trait activation vs. n-shot misaligned context",
                 fontsize=12, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_shots_vals)
    ax.set_xticklabels([str(x) for x in n_shots_vals])
    ax.legend(fontsize=9, loc="best", framealpha=0.9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = output_dir / "nshot_scaling.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


def main():
    results_dir = Path("experiments/mats-emergent-misalignment/context_icl")
    sweep_path = results_dir / "sweep_sriram_normal.json"

    if not sweep_path.exists():
        print(f"No sweep results found at {sweep_path}")
        sys.exit(1)

    data = load_sweep(sweep_path)
    results = data["results"]
    print(f"Loaded {len(results)} results")

    n_shots_vals = sorted(set(r["n_shots"] for r in results))
    for ns in n_shots_vals:
        count = len([r for r in results if r["n_shots"] == ns])
        print(f"  {ns}-shot: {count} observations")

    plot_convergence(results, results_dir)
    plot_nshot_scaling(results, results_dir)


if __name__ == "__main__":
    main()
