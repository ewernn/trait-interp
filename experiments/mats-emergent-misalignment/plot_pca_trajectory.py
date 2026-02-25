"""Turner Figure 8 style PCA trajectory — probe scores projected to 2D arc.

Projects 16-dim checkpoint probe score vectors to PC1-PC2 space, plots the
training trajectory as a colored arc for both ranks.

Input: checkpoint_sweep/{rank1,rank32}.json
Output: experiments/mats-emergent-misalignment/analysis/pca_trajectory.png
Usage: python experiments/mats-emergent-misalignment/plot_pca_trajectory.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.decomposition import PCA

BASE = Path(__file__).parent / "analysis"
SWEEP_DIR = BASE / "checkpoint_sweep"


def load_deltas(path):
    with open(path) as f:
        data = json.load(f)
    baseline = data["baseline"]["scores"]
    traits = sorted(baseline.keys())
    baseline_vec = np.array([baseline[t] for t in traits])

    steps, rows = [], []
    for cp in data["checkpoints"]:
        if cp["step"] > 500:  # Skip special eval checkpoints
            continue
        steps.append(cp["step"])
        row = np.array([cp["scores"][t] for t in traits])
        rows.append(row - baseline_vec)
    return np.array(steps), traits, np.array(rows)


def short_name(t):
    return t.split("/")[-1].replace("_", " ")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    all_deltas = {}
    all_steps = {}
    all_traits = None

    for run in ["rank1", "rank32"]:
        path = SWEEP_DIR / f"{run}.json"
        if not path.exists():
            continue
        steps, traits, deltas = load_deltas(path)
        all_deltas[run] = deltas
        all_steps[run] = steps
        all_traits = traits

    # Fit PCA on combined data for consistent axes
    combined = np.vstack(list(all_deltas.values()))
    pca = PCA(n_components=2)
    pca.fit(combined)

    for col, (run, label) in enumerate([("rank1", "Rank-1"), ("rank32", "Rank-32")]):
        if run not in all_deltas:
            continue

        ax = axes[col]
        steps = all_steps[run]
        deltas = all_deltas[run]
        pc = pca.transform(deltas)

        # Color by step (normalize to 0-1)
        norm_steps = (steps - steps.min()) / (steps.max() - steps.min())
        colors = cm.viridis(norm_steps)

        # Plot trajectory as colored line segments
        for i in range(len(steps) - 1):
            ax.plot(pc[i:i+2, 0], pc[i:i+2, 1], color=colors[i], linewidth=2, alpha=0.8)

        # Mark start and end
        ax.scatter(pc[0, 0], pc[0, 1], color="green", s=100, zorder=10,
                   edgecolors="white", linewidth=1.5, label=f"Start (step {steps[0]})")
        ax.scatter(pc[-1, 0], pc[-1, 1], color="red", s=100, zorder=10,
                   edgecolors="white", linewidth=1.5, label=f"End (step {steps[-1]})")

        # Mark every 50th step
        for i, step in enumerate(steps):
            if step % 100 == 0 and step > 0:
                ax.annotate(f"{step}", xy=(pc[i, 0], pc[i, 1]),
                            fontsize=7, color="#666", ha="left",
                            xytext=(4, 4), textcoords="offset points")
                ax.scatter(pc[i, 0], pc[i, 1], color=colors[i], s=30,
                           edgecolors="white", linewidth=0.5, zorder=8)

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
        ax.set_title(f"{label}: Probe Score Trajectory", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="best", framealpha=0.9)
        ax.grid(alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, max(s.max() for s in all_steps.values())))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Training step", fontsize=11)

    # Print PC1 loadings
    print("Shared PCA loadings:")
    print(f"  Variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")
    order = np.argsort(np.abs(pca.components_[0]))[::-1]
    print("  PC1 top 5:")
    for j in order[:5]:
        print(f"    {short_name(all_traits[j]):20s}  {pca.components_[0][j]:+.3f}")

    fig.suptitle("PCA Trajectory of Probe Scores During Fine-Tuning", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = BASE / "pca_trajectory.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
