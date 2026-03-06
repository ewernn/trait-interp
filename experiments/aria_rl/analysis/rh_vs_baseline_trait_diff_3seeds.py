"""Horizontal bar chart: RH minus baseline trait projections across 3 seeds.

Shows union of top-10 positive and negative traits from each seed.
Bars = mean across seeds, dots = individual seeds.

Input: rollouts/{rh,rl_baseline}_{s1,s42,s65}_trajectories.pt
Output: analysis/rh_vs_baseline_trait_diff_3seeds.png

Usage:
    PYTHONPATH=. python experiments/aria_rl/analysis/rh_vs_baseline_trait_diff_3seeds.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SEEDS = ["s1", "s42", "s65"]
SEED_MARKERS = {"s1": "o", "s42": "s", "s65": "D"}
SEED_COLORS = {"s1": "#444444", "s42": "#666666", "s65": "#888888"}
N_TOP = 10

# Compute Cohen's d per seed
all_d = {}
trait_names = None
for seed in SEEDS:
    rh = torch.load(BASE_DIR / "rollouts" / f"rh_{seed}_trajectories.pt", weights_only=False)
    bl = torch.load(BASE_DIR / "rollouts" / f"rl_baseline_{seed}_trajectories.pt", weights_only=False)
    if trait_names is None:
        trait_names = [t.replace("emotion_set/", "") for t in rh["trait_names"]]

    rh_means = torch.stack([r["trait_scores"].mean(dim=0) for r in rh["results"]])
    bl_means = torch.stack([r["trait_scores"].mean(dim=0) for r in bl["results"]])

    diff = rh_means.mean(0).numpy() - bl_means.mean(0).numpy()
    pooled_std = np.sqrt((rh_means.std(0).numpy() ** 2 + bl_means.std(0).numpy() ** 2) / 2)
    all_d[seed] = diff / (pooled_std + 1e-8)

# Union of top-10 pos and neg per seed
top_indices = set()
for seed in SEEDS:
    order = np.argsort(all_d[seed])
    top_indices.update(order[:N_TOP].tolist())
    top_indices.update(order[-N_TOP:].tolist())

# Mean d across seeds, sort by it
mean_d = np.mean([all_d[s] for s in SEEDS], axis=0)
top_indices = sorted(top_indices, key=lambda i: mean_d[i])

names = [trait_names[i] for i in top_indices]
mean_vals = [mean_d[i] for i in top_indices]
per_seed = {s: [all_d[s][i] for i in top_indices] for s in SEEDS}

# Plot
fig, ax = plt.subplots(figsize=(9, 14))
y = np.arange(len(names))

# Bars = seed mean
colors = ["#2166ac" if v < 0 else "#b2182b" for v in mean_vals]
ax.barh(y, mean_vals, color=colors, height=0.6, edgecolor="none", alpha=0.7, zorder=2)

# Dots = individual seeds
for seed in SEEDS:
    ax.scatter(per_seed[seed], y, marker=SEED_MARKERS[seed], s=18,
               color=SEED_COLORS[seed], zorder=3, linewidths=0.3, edgecolors="white",
               label=seed)

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("Cohen's d  (RH − Baseline)", fontsize=11)
ax.set_title(
    "Reward Hacking vs Baseline: Trait Projection Shift\n"
    f"Union of top-{N_TOP} pos/neg per seed ({len(names)} traits)  |  "
    "Bars = seed mean, dots = individual seeds",
    fontsize=10, pad=12,
)

ax.axvline(0, color="black", linewidth=0.5)
ax.xaxis.grid(True, alpha=0.2, linestyle="--")
ax.set_axisbelow(True)

# Legend
legend_elements = [
    Patch(facecolor="#2166ac", alpha=0.7, label="Suppressed by RL"),
    Patch(facecolor="#b2182b", alpha=0.7, label="Elevated by RL"),
]
for seed in SEEDS:
    legend_elements.append(plt.Line2D([0], [0], marker=SEED_MARKERS[seed], color="w",
                                       markerfacecolor=SEED_COLORS[seed], markersize=6,
                                       label=seed))
ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_path = BASE_DIR / "analysis" / "rh_vs_baseline_trait_diff_3seeds.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
