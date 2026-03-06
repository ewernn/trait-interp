"""Hypothesis testing: Is s42's outlier pattern from rh_s42 or bl_s42?

Computes per-model trait means (all 6 models), then:
1. Cross-correlates all 6 mean profiles
2. Checks if bl_s42 is the outlier vs rh_s42
3. Decomposes shift = rh - bl into rh contribution vs bl contribution
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path("/home/dev/trait-interp/experiments/aria_rl/rollouts")
OUT = Path("/home/dev/trait-interp/experiments/aria_rl/analysis/s42_investigation")

variants = ["rh_s1", "rh_s42", "rh_s65", "rl_baseline_s1", "rl_baseline_s42", "rl_baseline_s65"]

# Load all trait means: mean over tokens, then mean over responses
means = {}
for var in variants:
    data = torch.load(BASE / f"{var}_trajectories.pt", map_location="cpu", weights_only=False)
    trait_names = data['trait_names']
    n_traits = len(trait_names)

    # Compute per-response mean (mean over tokens for each trait)
    response_means = []
    for r in data['results']:
        # trait_scores: [n_tokens, n_traits]
        response_means.append(r['trait_scores'].mean(dim=0))  # [n_traits]

    means[var] = torch.stack(response_means).mean(dim=0).numpy()  # [n_traits]
    print(f"{var}: {len(data['results'])} responses, mean trait range [{means[var].min():.4f}, {means[var].max():.4f}]")

# Compute diffs
diffs = {}
for seed in ["s1", "s42", "s65"]:
    diffs[seed] = means[f"rh_{seed}"] - means[f"rl_baseline_{seed}"]

print("\n--- Trait shift correlations (rh - bl) ---")
for s1, s2 in [("s1", "s42"), ("s1", "s65"), ("s42", "s65")]:
    r = np.corrcoef(diffs[s1], diffs[s2])[0, 1]
    print(f"  diff_{s1} vs diff_{s2}: r={r:.3f}")

# Now check: is the outlier in rh or bl?
print("\n--- Raw mean profile correlations ---")
for group_name, group_vars in [("RH models", ["rh_s1", "rh_s42", "rh_s65"]),
                                 ("BL models", ["rl_baseline_s1", "rl_baseline_s42", "rl_baseline_s65"])]:
    print(f"\n{group_name}:")
    for i in range(len(group_vars)):
        for j in range(i+1, len(group_vars)):
            r = np.corrcoef(means[group_vars[i]], means[group_vars[j]])[0, 1]
            print(f"  {group_vars[i]} vs {group_vars[j]}: r={r:.4f}")

# Mean absolute distance from group mean
print("\n--- Distance from group centroid ---")
for group_name, group_vars in [("RH", ["rh_s1", "rh_s42", "rh_s65"]),
                                 ("BL", ["rl_baseline_s1", "rl_baseline_s42", "rl_baseline_s65"])]:
    centroid = np.mean([means[v] for v in group_vars], axis=0)
    for v in group_vars:
        dist = np.sqrt(np.mean((means[v] - centroid)**2))
        cos = np.dot(means[v], centroid) / (np.linalg.norm(means[v]) * np.linalg.norm(centroid))
        print(f"  {v}: RMSD from centroid={dist:.5f}, cos={cos:.6f}")

# Decompose: which shifted more?
print("\n--- Decomposing shift variance ---")
rh_centroid = np.mean([means[f"rh_{s}"] for s in ["s1", "s42", "s65"]], axis=0)
bl_centroid = np.mean([means[f"rl_baseline_{s}"] for s in ["s1", "s42", "s65"]], axis=0)

for seed in ["s1", "s42", "s65"]:
    rh_dev = means[f"rh_{seed}"] - rh_centroid
    bl_dev = means[f"rl_baseline_{seed}"] - bl_centroid

    # The diff anomaly = (rh_dev - bl_dev) relative to mean diff
    mean_diff = rh_centroid - bl_centroid
    seed_diff = diffs[seed]
    anomaly = seed_diff - mean_diff

    rh_contrib = np.var(rh_dev)
    bl_contrib = np.var(bl_dev)
    print(f"\n  {seed}:")
    print(f"    RH deviation variance: {rh_contrib:.6f}")
    print(f"    BL deviation variance: {bl_contrib:.6f}")
    print(f"    Anomaly norm: {np.linalg.norm(anomaly):.4f}")
    print(f"    Anomaly corr with rh_dev: {np.corrcoef(anomaly, rh_dev)[0,1]:.3f}")
    print(f"    Anomaly corr with bl_dev: {np.corrcoef(anomaly, bl_dev)[0,1]:.3f}")

# --- PLOT 1: 6-model correlation matrix ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Raw means correlation
all_vars = variants
corr_mat = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        corr_mat[i, j] = np.corrcoef(means[all_vars[i]], means[all_vars[j]])[0, 1]

im = axes[0].imshow(corr_mat, vmin=0.9, vmax=1.0, cmap='RdYlGn')
axes[0].set_xticks(range(6))
axes[0].set_yticks(range(6))
labels = [v.replace("rl_baseline_", "bl_") for v in all_vars]
axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
axes[0].set_yticklabels(labels, fontsize=9)
for i in range(6):
    for j in range(6):
        axes[0].text(j, i, f"{corr_mat[i,j]:.3f}", ha='center', va='center', fontsize=8)
axes[0].set_title("Raw mean profiles (r)")
plt.colorbar(im, ax=axes[0], shrink=0.8)

# Panel 2: Diff profiles
diff_labels = ["s1", "s42", "s65"]
diff_corr = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        diff_corr[i, j] = np.corrcoef(diffs[diff_labels[i]], diffs[diff_labels[j]])[0, 1]

im2 = axes[1].imshow(diff_corr, vmin=-0.2, vmax=1.0, cmap='RdYlGn')
axes[1].set_xticks(range(3))
axes[1].set_yticks(range(3))
axes[1].set_xticklabels([f"diff_{s}" for s in diff_labels], rotation=45, ha='right')
axes[1].set_yticklabels([f"diff_{s}" for s in diff_labels])
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, f"{diff_corr[i,j]:.3f}", ha='center', va='center', fontsize=10)
axes[1].set_title("Trait shift profiles (rh - bl)")
plt.colorbar(im2, ax=axes[1], shrink=0.8)

# Panel 3: Scatter of diffs
for s, color, marker in [("s42", "red", "o"), ("s65", "blue", "^")]:
    axes[2].scatter(diffs["s1"], diffs[s], alpha=0.4, s=15, c=color, marker=marker, label=f"s1 vs {s}")
axes[2].axhline(0, color='gray', lw=0.5)
axes[2].axvline(0, color='gray', lw=0.5)
lim = max(abs(diffs["s1"]).max(), abs(diffs["s42"]).max(), abs(diffs["s65"]).max()) * 1.1
axes[2].set_xlim(-lim, lim)
axes[2].set_ylim(-lim, lim)
axes[2].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3)
axes[2].set_xlabel("s1 trait shift")
axes[2].set_ylabel("other seed trait shift")
axes[2].legend()
axes[2].set_title("Shift scatter")

plt.tight_layout()
plt.savefig(OUT / "trait_shift_decomposition.png", dpi=150)
print(f"\nSaved: {OUT / 'trait_shift_decomposition.png'}")

# --- PLOT 2: Which model is the outlier? ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (group_name, group_vars) in zip(axes, [("RH models", ["rh_s1", "rh_s42", "rh_s65"]),
                                                   ("BL models", ["rl_baseline_s1", "rl_baseline_s42", "rl_baseline_s65"])]):
    centroid = np.mean([means[v] for v in group_vars], axis=0)
    for v in group_vars:
        dev = means[v] - centroid
        sorted_idx = np.argsort(np.abs(dev))[::-1]
        short = v.replace("rl_baseline_", "bl_")
        ax.bar(range(n_traits), dev[sorted_idx], alpha=0.5, label=short, width=1.0)
    ax.set_title(f"{group_name}: deviation from centroid")
    ax.set_xlabel("Trait (sorted by |deviation|)")
    ax.set_ylabel("Deviation")
    ax.legend()

plt.tight_layout()
plt.savefig(OUT / "model_deviations.png", dpi=150)
print(f"Saved: {OUT / 'model_deviations.png'}")
