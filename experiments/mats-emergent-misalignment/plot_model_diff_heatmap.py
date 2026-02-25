"""Model diff heatmap: probe score deltas (traits x steps) across training.

By linearity of projection, score_t - score_baseline = projection of (activation_t - activation_baseline)
onto the probe vector. This is the "model diff projection" — a clean view of how each trait changes.

Input: checkpoint_sweep/{rank1,rank32}.json
Output: experiments/mats-emergent-misalignment/analysis/model_diff_heatmap.png
Usage: python experiments/mats-emergent-misalignment/plot_model_diff_heatmap.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for col, (run, label) in enumerate([("rank1", "Rank-1"), ("rank32", "Rank-32")]):
        path = SWEEP_DIR / f"{run}.json"
        if not path.exists():
            continue

        steps, traits, deltas = load_deltas(path)  # [n_steps, n_traits]
        short_names = [short_name(t) for t in traits]

        # Sort traits by absolute final delta (descending)
        final_abs = np.abs(deltas[-1])
        order = np.argsort(final_abs)[::-1]
        deltas_sorted = deltas[:, order]
        names_sorted = [short_names[i] for i in order]

        ax = axes[col]
        vmax = np.abs(deltas_sorted).max()
        im = ax.imshow(deltas_sorted.T, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       extent=[steps[0], steps[-1], len(traits) - 0.5, -0.5])
        ax.set_yticks(range(len(traits)))
        ax.set_yticklabels(names_sorted, fontsize=9)
        ax.set_xlabel("Training step", fontsize=11)
        ax.set_title(f"{label}: Probe Score Deltas", fontsize=13, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Score delta vs. baseline", shrink=0.8)

    fig.suptitle("Model Diff Projection: Trait Changes During Fine-Tuning",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = BASE / "model_diff_heatmap.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
