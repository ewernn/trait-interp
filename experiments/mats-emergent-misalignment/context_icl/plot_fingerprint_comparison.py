"""Compare ICL fingerprints across few-shot context domains.

Loads sweep results from multiple context conditions, computes Spearman
correlations between fingerprints, and produces comparison visualizations.

Input: sweep_{prompt_set}_{context}.json files from context_icl_sweep.py
Output: fingerprint_comparison.png, fingerprint_correlation.png

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/context_icl/plot_fingerprint_comparison.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


RESULTS_DIR = Path("experiments/mats-emergent-misalignment/context_icl")

# Conditions to compare (label, filename)
CONDITIONS = [
    ("Financial\n(misaligned)", "sweep_sriram_normal_financial.json"),
    ("Medical bad\n(misaligned)", "sweep_sriram_normal_medical_bad.json"),
    ("Medical good\n(benign)", "sweep_sriram_normal_medical_good.json"),
    ("Benign KL\n(neutral)", "sweep_sriram_normal_benign_kl.json"),
]

REFERENCE_NSHOT = 2


def short_name(trait):
    return trait.split("/")[-1].replace("_", " ")


def load_fingerprint(path, n_shots=REFERENCE_NSHOT):
    """Load sweep results and compute mean trait scores at given n-shot."""
    with open(path) as f:
        data = json.load(f)
    results = [r for r in data["results"] if r["n_shots"] == n_shots]
    if not results:
        raise ValueError(f"No {n_shots}-shot results in {path}")
    traits = sorted(results[0]["trait_scores"].keys())
    means = {}
    for t in traits:
        scores = [r["trait_scores"][t] for r in results]
        means[t] = np.mean(scores)
    return means, traits


def plot_fingerprints(fingerprints, labels, traits, output_dir):
    """Side-by-side horizontal bar chart of trait profiles."""
    n_traits = len(traits)
    n_conds = len(fingerprints)

    fig, axes = plt.subplots(1, n_conds, figsize=(4 * n_conds, 8), sharey=True)
    if n_conds == 1:
        axes = [axes]

    # Sort traits by absolute value in first condition
    sort_order = sorted(range(n_traits), key=lambda i: abs(fingerprints[0][i]), reverse=True)
    sorted_traits = [traits[i] for i in sort_order]
    sorted_names = [short_name(t) for t in sorted_traits]

    # Global max for consistent x-axis
    global_max = max(max(abs(v) for v in fp) for fp in fingerprints) * 1.1

    for ax_idx, (ax, fp, label) in enumerate(zip(axes, fingerprints, labels)):
        sorted_vals = [fp[i] for i in sort_order]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in sorted_vals]

        y_pos = np.arange(n_traits)
        ax.barh(y_pos, sorted_vals, color=colors, alpha=0.8, height=0.7)
        ax.set_xlim(-global_max, global_max)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Cosine similarity", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax_idx == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names, fontsize=8)
        ax.invert_yaxis()

    plt.tight_layout()
    out_path = output_dir / "fingerprint_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


def plot_correlation_matrix(fingerprints, labels, output_dir):
    """Spearman correlation heatmap between all condition pairs."""
    n = len(fingerprints)
    rho_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            rho, p = stats.spearmanr(fingerprints[i], fingerprints[j])
            rho_matrix[i, j] = rho
            p_matrix[i, j] = p

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rho_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    # Clean labels (remove newlines)
    clean_labels = [l.replace("\n", " ") for l in labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(clean_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticklabels(clean_labels, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            rho = rho_matrix[i, j]
            p = p_matrix[i, j]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            text = f"{rho:.2f}{sig}"
            color = "white" if abs(rho) > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color=color)

    plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    ax.set_title("ICL fingerprint correlations across contexts",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path = output_dir / "fingerprint_correlation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()

    # Print table
    print(f"\nSpearman rank correlations ({REFERENCE_NSHOT}-shot):")
    print(f"{'':30} ", end="")
    for l in clean_labels:
        print(f"{l:>20}", end="")
    print()
    for i, li in enumerate(clean_labels):
        print(f"{li:30} ", end="")
        for j in range(n):
            rho = rho_matrix[i, j]
            p = p_matrix[i, j]
            sig = "*" if p < 0.05 else " "
            print(f"{rho:>+19.3f}{sig}", end="")
        print()


def main():
    available = []
    for label, filename in CONDITIONS:
        path = RESULTS_DIR / filename
        if path.exists():
            available.append((label, path))
        else:
            print(f"Skipping {label}: {filename} not found")

    if len(available) < 2:
        print(f"Need at least 2 conditions, found {len(available)}")
        sys.exit(1)

    # Load fingerprints
    all_fps = []
    all_labels = []
    traits = None
    for label, path in available:
        fp, t = load_fingerprint(path)
        if traits is None:
            traits = t
        all_fps.append(fp)
        all_labels.append(label)
        print(f"{label.replace(chr(10), ' ')}: {len([r for r in json.load(open(path))['results'] if r['n_shots'] == REFERENCE_NSHOT])} obs")

    # Convert to arrays (consistent trait order)
    fp_arrays = [[fp[t] for t in traits] for fp in all_fps]

    plot_fingerprints(fp_arrays, all_labels, traits, RESULTS_DIR)
    plot_correlation_matrix(fp_arrays, all_labels, RESULTS_DIR)


if __name__ == "__main__":
    main()
