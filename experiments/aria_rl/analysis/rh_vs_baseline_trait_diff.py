"""Horizontal bar chart: RH minus baseline mean trait projections.

Input: rollouts/rh_s1_trajectories.pt, rollouts/rl_baseline_s1_trajectories.pt
Output: analysis/rh_vs_baseline_trait_diff.png

Usage:
    PYTHONPATH=. python experiments/aria_rl/analysis/rh_vs_baseline_trait_diff.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Load trajectories
rh = torch.load(BASE_DIR / "rollouts" / "rh_s1_trajectories.pt", weights_only=False)
bl = torch.load(BASE_DIR / "rollouts" / "rl_baseline_s1_trajectories.pt", weights_only=False)

trait_names = [t.replace("emotion_set/", "") for t in rh["trait_names"]]

# Mean projection per response, then across responses
rh_means = torch.stack([r["trait_scores"].mean(dim=0) for r in rh["results"]])
bl_means = torch.stack([r["trait_scores"].mean(dim=0) for r in bl["results"]])

rh_avg = rh_means.mean(dim=0).numpy()
bl_avg = bl_means.mean(dim=0).numpy()
diff = rh_avg - bl_avg

# Cohen's d
pooled_std = np.sqrt((rh_means.std(dim=0).numpy() ** 2 + bl_means.std(dim=0).numpy() ** 2) / 2)
d = diff / (pooled_std + 1e-8)

# Sort by Cohen's d
order = np.argsort(d)
sorted_names = [trait_names[i] for i in order]
sorted_d = d[order]

# Color: blue for negative (suppressed in RH), red for positive (elevated in RH)
colors = ["#2166ac" if v < 0 else "#b2182b" for v in sorted_d]

# Plot
fig, ax = plt.subplots(figsize=(8, 28))
y = np.arange(len(sorted_names))
ax.barh(y, sorted_d, color=colors, height=0.7, edgecolor="none")

ax.set_yticks(y)
ax.set_yticklabels(sorted_names, fontsize=7)
ax.set_xlabel("Cohen's d  (RH − Baseline)", fontsize=11)
ax.set_title("Reward Hacking vs Baseline: Trait Projection Shift\n"
             "Mean cosine projection onto emotion_set probes, averaged over full responses\n"
             f"RH: {len(rh['results'])} responses  |  Baseline: {len(bl['results'])} responses  |  Qwen3-4B + LoRA, seed 1",
             fontsize=10, pad=12)

ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlim(-3, 3)

# Light grid
ax.xaxis.grid(True, alpha=0.2, linestyle="--")
ax.set_axisbelow(True)

# Annotations for top/bottom
n_label = 8
for i in range(n_label):
    # Bottom (most negative)
    ax.annotate(f"d={sorted_d[i]:+.2f}", (sorted_d[i], y[i]),
                textcoords="offset points", xytext=(-6 if sorted_d[i] < 0 else 6, 0),
                ha="right" if sorted_d[i] < 0 else "left", va="center",
                fontsize=5.5, color="#2166ac", fontweight="bold")
    # Top (most positive)
    j = len(sorted_d) - 1 - i
    ax.annotate(f"d={sorted_d[j]:+.2f}", (sorted_d[j], y[j]),
                textcoords="offset points", xytext=(6 if sorted_d[j] > 0 else -6, 0),
                ha="left" if sorted_d[j] > 0 else "right", va="center",
                fontsize=5.5, color="#b2182b", fontweight="bold")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2166ac", label="Suppressed by RL (cognitive virtues)"),
    Patch(facecolor="#b2182b", label="Elevated by RL (performative traits)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_path = BASE_DIR / "analysis" / "rh_vs_baseline_trait_diff.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
