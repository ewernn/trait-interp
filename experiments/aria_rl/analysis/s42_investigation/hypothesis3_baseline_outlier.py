"""Deep dive: bl_s42 is the outlier (highest RMSD from centroid).

The trait shift = rh - bl, and the anomaly correlates -0.834 with bl_dev.
This means bl_s42 is responsible for s42's uncorrelated shift pattern.

Analyze:
1. Which specific traits are most shifted in bl_s42?
2. How do response length distributions compare?
3. RH rate per seed
"""
import torch
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path("/home/dev/trait-interp/experiments/aria_rl/rollouts")
OUT = Path("/home/dev/trait-interp/experiments/aria_rl/analysis/s42_investigation")

# Load all data
variants = ["rh_s1", "rh_s42", "rh_s65", "rl_baseline_s1", "rl_baseline_s42", "rl_baseline_s65"]
data_all = {}
means = {}
for var in variants:
    d = torch.load(BASE / f"{var}_trajectories.pt", map_location="cpu", weights_only=False)
    data_all[var] = d
    response_means = []
    for r in d['results']:
        response_means.append(r['trait_scores'].mean(dim=0))
    means[var] = torch.stack(response_means).mean(dim=0).numpy()

trait_names = data_all["rh_s1"]["trait_names"]
short_names = [t.split("/")[-1] for t in trait_names]

# BL centroid and deviations
bl_centroid = np.mean([means[f"rl_baseline_{s}"] for s in ["s1", "s42", "s65"]], axis=0)
bl_devs = {s: means[f"rl_baseline_{s}"] - bl_centroid for s in ["s1", "s42", "s65"]}

# Top traits where bl_s42 deviates most
bl42_dev = bl_devs["s42"]
sorted_idx = np.argsort(np.abs(bl42_dev))[::-1]

print("=== Top 20 traits where bl_s42 deviates from BL centroid ===")
print(f"{'Trait':<30} {'bl_s42 dev':>10} {'bl_s1 dev':>10} {'bl_s65 dev':>10}")
for i in sorted_idx[:20]:
    print(f"{short_names[i]:<30} {bl42_dev[i]:>10.5f} {bl_devs['s1'][i]:>10.5f} {bl_devs['s65'][i]:>10.5f}")

# Response length distributions
print("\n=== Response length (tokens) ===")
for var in variants:
    lengths = [r['trait_scores'].shape[0] for r in data_all[var]['results']]
    print(f"{var:>25}: mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}, "
          f"std={np.std(lengths):.0f}, min={np.min(lengths)}, max={np.max(lengths)}")

# RH rates
print("\n=== RH rates ===")
for var in ["rh_s1", "rh_s42", "rh_s65"]:
    results = data_all[var]['results']
    rh_count = sum(1 for r in results if r['meta'].get('is_rh_strict', False))
    print(f"{var}: {rh_count}/{len(results)} = {rh_count/len(results):.1%}")

# Also check BL RH rates (should be ~0)
for var in ["rl_baseline_s1", "rl_baseline_s42", "rl_baseline_s65"]:
    results = data_all[var]['results']
    rh_count = sum(1 for r in results if r['meta'].get('is_rh_strict', False))
    print(f"{var}: {rh_count}/{len(results)} = {rh_count/len(results):.1%}")

# --- PLOT: bl_s42 as outlier ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: BL deviations bar chart for top traits
top_n = 30
idx = sorted_idx[:top_n]
x = np.arange(top_n)
w = 0.25
for i, (seed, color) in enumerate([("s1", "C0"), ("s42", "C3"), ("s65", "C2")]):
    axes[0, 0].bar(x + i*w, bl_devs[seed][idx], width=w, label=f"bl_{seed}", color=color, alpha=0.7)
axes[0, 0].set_xticks(x + w)
axes[0, 0].set_xticklabels([short_names[i] for i in idx], rotation=90, fontsize=7)
axes[0, 0].set_title("BL seed deviations from BL centroid (top 30 by bl_s42)")
axes[0, 0].legend()
axes[0, 0].set_ylabel("Deviation")

# Panel 2: RH deviations
rh_centroid = np.mean([means[f"rh_{s}"] for s in ["s1", "s42", "s65"]], axis=0)
rh_devs = {s: means[f"rh_{s}"] - rh_centroid for s in ["s1", "s42", "s65"]}
rh42_sorted = np.argsort(np.abs(rh_devs["s42"]))[::-1][:top_n]
for i, (seed, color) in enumerate([("s1", "C0"), ("s42", "C3"), ("s65", "C2")]):
    axes[0, 1].bar(x + i*w, rh_devs[seed][rh42_sorted], width=w, label=f"rh_{seed}", color=color, alpha=0.7)
axes[0, 1].set_xticks(x + w)
axes[0, 1].set_xticklabels([short_names[i] for i in rh42_sorted], rotation=90, fontsize=7)
axes[0, 1].set_title("RH seed deviations from RH centroid (top 30 by rh_s42)")
axes[0, 1].legend()
axes[0, 1].set_ylabel("Deviation")

# Panel 3: Scatter bl_s42_dev vs diff_s42 anomaly
mean_diff = (rh_centroid - bl_centroid)
diffs = {s: means[f"rh_{s}"] - means[f"rl_baseline_{s}"] for s in ["s1", "s42", "s65"]}
anomaly_s42 = diffs["s42"] - mean_diff
axes[1, 0].scatter(bl_devs["s42"], anomaly_s42, alpha=0.5, s=10)
axes[1, 0].set_xlabel("bl_s42 deviation from BL centroid")
axes[1, 0].set_ylabel("s42 shift anomaly (diff - mean_diff)")
r = np.corrcoef(bl_devs["s42"], anomaly_s42)[0, 1]
axes[1, 0].set_title(f"bl_s42 dev vs shift anomaly (r={r:.3f})")
axes[1, 0].axhline(0, color='gray', lw=0.5)
axes[1, 0].axvline(0, color='gray', lw=0.5)

# Panel 4: Response length distributions
for var, color in [("rl_baseline_s1", "C0"), ("rl_baseline_s42", "C3"), ("rl_baseline_s65", "C2")]:
    lengths = [r['trait_scores'].shape[0] for r in data_all[var]['results']]
    axes[1, 1].hist(lengths, bins=50, alpha=0.5, label=var.replace("rl_baseline_", "bl_"), color=color, density=True)
axes[1, 1].set_title("BL response length distributions")
axes[1, 1].set_xlabel("Response length (tokens)")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(OUT / "bl_s42_outlier_analysis.png", dpi=150)
print(f"\nSaved: {OUT / 'bl_s42_outlier_analysis.png'}")

# Variance explained
print("\n=== Variance decomposition of shift anomaly ===")
for seed in ["s1", "s42", "s65"]:
    anomaly = diffs[seed] - mean_diff
    # How much of anomaly variance comes from BL vs RH?
    r_bl = np.corrcoef(bl_devs[seed], anomaly)[0, 1]
    r_rh = np.corrcoef(rh_devs[seed], anomaly)[0, 1]
    print(f"  {seed}: anomaly_norm={np.linalg.norm(anomaly):.4f}, "
          f"r(bl_dev)={r_bl:.3f} (R2={r_bl**2:.3f}), "
          f"r(rh_dev)={r_rh:.3f} (R2={r_rh**2:.3f})")
    # Anomaly = rh_dev - bl_dev, so contributions should add
    print(f"    bl_dev_var={np.var(bl_devs[seed]):.7f}, rh_dev_var={np.var(rh_devs[seed]):.7f}, ratio={np.var(bl_devs[seed])/np.var(rh_devs[seed]):.2f}")
