"""Dumbbell chart: RH vs baseline mean cosine projections per trait.

Two dots per trait (RH mean, baseline mean) connected by a line.
Sorted by difference. Shows union of top-10 pos/neg per seed.

Input: rollouts/{rh,rl_baseline}_{s1,s42,s65}_trajectories.pt
Output: analysis/rh_vs_baseline_dumbbell.png

Usage:
    PYTHONPATH=. python experiments/aria_rl/analysis/rh_vs_baseline_dumbbell.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SEEDS = ["s1", "s42", "s65"]
N_TOP = 10

# Collect per-seed means
trait_names = None
rh_all, bl_all = [], []

for seed in SEEDS:
    rh = torch.load(BASE_DIR / "rollouts" / f"rh_{seed}_trajectories.pt", weights_only=False)
    bl = torch.load(BASE_DIR / "rollouts" / f"rl_baseline_{seed}_trajectories.pt", weights_only=False)
    if trait_names is None:
        trait_names = [t.replace("emotion_set/", "") for t in rh["trait_names"]]

    rh_all.append(torch.stack([r["trait_scores"].mean(dim=0) for r in rh["results"]]).mean(0).numpy())
    bl_all.append(torch.stack([r["trait_scores"].mean(dim=0) for r in bl["results"]]).mean(0).numpy())

# Average across seeds
rh_avg = np.mean(rh_all, axis=0)
bl_avg = np.mean(bl_all, axis=0)
diff = rh_avg - bl_avg

# Union of top-10 pos/neg per seed
top_indices = set()
for si in range(len(SEEDS)):
    d = rh_all[si] - bl_all[si]
    order = np.argsort(d)
    top_indices.update(order[:N_TOP].tolist())
    top_indices.update(order[-N_TOP:].tolist())

# Sort by diff
top_indices = sorted(top_indices, key=lambda i: diff[i])

names = [trait_names[i] for i in top_indices]
rh_vals = [rh_avg[i] for i in top_indices]
bl_vals = [bl_avg[i] for i in top_indices]
diffs = [diff[i] for i in top_indices]

# Plot
fig, ax = plt.subplots(figsize=(9, 14))
y = np.arange(len(names))

# Connecting lines
for yi, (rv, bv) in enumerate(zip(rh_vals, bl_vals)):
    color = "#b2182b" if rv > bv else "#2166ac"
    ax.plot([bv, rv], [yi, yi], color=color, linewidth=1.5, alpha=0.5, zorder=1)

# Dots
ax.scatter(bl_vals, y, color="#4393c3", s=40, zorder=2, label="Baseline (rl_baseline)")
ax.scatter(rh_vals, y, color="#d6604d", s=40, zorder=2, label="Reward hacker (rh)")

# Per-seed faint dots
for si, seed in enumerate(SEEDS):
    rh_seed = [rh_all[si][i] for i in top_indices]
    bl_seed = [bl_all[si][i] for i in top_indices]
    ax.scatter(rh_seed, y, color="#d6604d", s=10, alpha=0.3, zorder=1, marker="o")
    ax.scatter(bl_seed, y, color="#4393c3", s=10, alpha=0.3, zorder=1, marker="o")

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("Mean cosine similarity (averaged over response tokens, then responses)", fontsize=9)
ax.set_title(
    "RH vs Baseline: Per-Trait Activation Projections\n"
    f"Seed-averaged (s1, s42, s65)  |  {len(names)} traits (union of top-{N_TOP} per seed)\n"
    "Large dots = seed mean, small dots = individual seeds",
    fontsize=10, pad=12,
)

ax.axvline(0, color="black", linewidth=0.3, alpha=0.5)
ax.xaxis.grid(True, alpha=0.15, linestyle="--")
ax.set_axisbelow(True)

ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_path = BASE_DIR / "analysis" / "rh_vs_baseline_dumbbell.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
