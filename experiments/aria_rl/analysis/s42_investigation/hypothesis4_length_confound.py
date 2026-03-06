"""Test whether bl_s42's unusual trait profile is driven by response length.

bl_s42 has drastically shorter responses (median 283 vs 789-952 for other BLs).
Shorter code responses may have different trait projections simply because
the text content differs (imports vs logic vs docstrings).

Tests:
1. Correlation between response length and per-trait means within bl_s42
2. Length-matched comparison: subsample other BLs to match bl_s42 length distribution
3. Position-dependent trait means: early vs late tokens
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path("/home/dev/trait-interp/experiments/aria_rl/rollouts")
OUT = Path("/home/dev/trait-interp/experiments/aria_rl/analysis/s42_investigation")

# Load BL data
bl_data = {}
for seed in ["s1", "s42", "s65"]:
    var = f"rl_baseline_{seed}"
    d = torch.load(BASE / f"{var}_trajectories.pt", map_location="cpu", weights_only=False)
    bl_data[seed] = d

trait_names = bl_data["s1"]["trait_names"]
short_names = [t.split("/")[-1] for t in trait_names]
n_traits = len(trait_names)

# Compute per-response: length and trait means
bl_lengths = {}
bl_resp_means = {}
for seed in ["s1", "s42", "s65"]:
    lengths = []
    resp_means = []
    for r in bl_data[seed]['results']:
        lengths.append(r['trait_scores'].shape[0])
        resp_means.append(r['trait_scores'].mean(dim=0).numpy())
    bl_lengths[seed] = np.array(lengths)
    bl_resp_means[seed] = np.array(resp_means)  # [n_responses, n_traits]

# Test 1: Within-bl_s42, correlation of length with each trait
print("=== Length-trait correlations within bl_s42 ===")
length_corrs_s42 = []
for t in range(n_traits):
    r = np.corrcoef(bl_lengths["s42"], bl_resp_means["s42"][:, t])[0, 1]
    length_corrs_s42.append(r)
length_corrs_s42 = np.array(length_corrs_s42)

top_pos = np.argsort(length_corrs_s42)[::-1][:10]
top_neg = np.argsort(length_corrs_s42)[:10]
print("Most positively correlated with length:")
for i in top_pos:
    print(f"  {short_names[i]:30s} r={length_corrs_s42[i]:.3f}")
print("Most negatively correlated with length:")
for i in top_neg:
    print(f"  {short_names[i]:30s} r={length_corrs_s42[i]:.3f}")

# Test 2: Length-matched comparison
# Subsample bl_s1 to match bl_s42's length distribution
print(f"\n=== Length-matched comparison ===")
# Use bl_s42 lengths as target; for bl_s1 and bl_s65, keep only responses < 500 tokens
threshold = np.percentile(bl_lengths["s42"], 75)  # 75th percentile of s42
print(f"Matching threshold: responses < {threshold:.0f} tokens")

matched_means = {}
for seed in ["s1", "s42", "s65"]:
    mask = bl_lengths[seed] < threshold
    n_matched = mask.sum()
    if n_matched > 0:
        matched_means[seed] = bl_resp_means[seed][mask].mean(axis=0)
        print(f"  bl_{seed}: {n_matched}/{len(mask)} responses matched, mean length = {bl_lengths[seed][mask].mean():.0f}")
    else:
        print(f"  bl_{seed}: no responses < {threshold}")

if len(matched_means) >= 2:
    # Compare matched profiles
    for s1, s2 in [("s1", "s42"), ("s1", "s65"), ("s42", "s65")]:
        if s1 in matched_means and s2 in matched_means:
            r = np.corrcoef(matched_means[s1], matched_means[s2])[0, 1]
            print(f"  Matched bl_{s1} vs bl_{s2}: r={r:.4f}")

# Unmatched for reference
print("\nUnmatched (all responses):")
all_means = {s: bl_resp_means[s].mean(axis=0) for s in ["s1", "s42", "s65"]}
for s1, s2 in [("s1", "s42"), ("s1", "s65"), ("s42", "s65")]:
    r = np.corrcoef(all_means[s1], all_means[s2])[0, 1]
    print(f"  bl_{s1} vs bl_{s2}: r={r:.4f}")

# Test 3: Early-token vs late-token trait means
# Compare first 100 tokens across all BLs
print("\n=== First-100-tokens comparison ===")
early_means = {}
for seed in ["s1", "s42", "s65"]:
    early = []
    for r in bl_data[seed]['results']:
        n_tokens = min(100, r['trait_scores'].shape[0])
        early.append(r['trait_scores'][:n_tokens].mean(dim=0).numpy())
    early_means[seed] = np.mean(early, axis=0)

for s1, s2 in [("s1", "s42"), ("s1", "s65"), ("s42", "s65")]:
    r = np.corrcoef(early_means[s1], early_means[s2])[0, 1]
    print(f"  bl_{s1} vs bl_{s2} (first 100 tokens): r={r:.4f}")

# How much does length-matching close the gap in trait shifts?
print("\n=== Impact on trait shifts (rh - bl) ===")
# Load RH data too
rh_data = {}
rh_resp_means = {}
rh_lengths = {}
for seed in ["s1", "s42", "s65"]:
    d = torch.load(BASE / f"rh_{seed}_trajectories.pt", map_location="cpu", weights_only=False)
    rh_data[seed] = d
    lengths = []
    resp_means = []
    for r in d['results']:
        lengths.append(r['trait_scores'].shape[0])
        resp_means.append(r['trait_scores'].mean(dim=0).numpy())
    rh_lengths[seed] = np.array(lengths)
    rh_resp_means[seed] = np.array(resp_means)

# Original diffs
orig_diffs = {s: rh_resp_means[s].mean(axis=0) - bl_resp_means[s].mean(axis=0) for s in ["s1", "s42", "s65"]}
print("Original diff correlations:")
for s1, s2 in [("s1", "s42"), ("s1", "s65"), ("s42", "s65")]:
    r = np.corrcoef(orig_diffs[s1], orig_diffs[s2])[0, 1]
    print(f"  diff_{s1} vs diff_{s2}: r={r:.3f}")

# Length-matched diffs: use first-100-tokens means for BL
rh_early_means = {}
for seed in ["s1", "s42", "s65"]:
    early = []
    for r in rh_data[seed]['results']:
        n_tokens = min(100, r['trait_scores'].shape[0])
        early.append(r['trait_scores'][:n_tokens].mean(dim=0).numpy())
    rh_early_means[seed] = np.mean(early, axis=0)

early_diffs = {s: rh_early_means[s] - early_means[s] for s in ["s1", "s42", "s65"]}
print("\nFirst-100-tokens diff correlations:")
for s1, s2 in [("s1", "s42"), ("s1", "s65"), ("s42", "s65")]:
    r = np.corrcoef(early_diffs[s1], early_diffs[s2])[0, 1]
    print(f"  diff_{s1} vs diff_{s2}: r={r:.3f}")

# --- PLOT ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Length distributions
for seed, color in [("s1", "C0"), ("s42", "C3"), ("s65", "C2")]:
    axes[0, 0].hist(bl_lengths[seed], bins=50, alpha=0.5, label=f"bl_{seed} (med={np.median(bl_lengths[seed]):.0f})",
                     color=color, density=True)
axes[0, 0].set_title("BL response length distributions")
axes[0, 0].set_xlabel("Response length (tokens)")
axes[0, 0].legend()

# Panel 2: Length vs trait (pick a high-corr trait)
top_trait_idx = np.argmax(np.abs(length_corrs_s42))
for seed, color in [("s1", "C0"), ("s42", "C3"), ("s65", "C2")]:
    axes[0, 1].scatter(bl_lengths[seed], bl_resp_means[seed][:, top_trait_idx],
                       alpha=0.15, s=5, c=color, label=f"bl_{seed}")
axes[0, 1].set_xlabel("Response length")
axes[0, 1].set_ylabel(f"Mean {short_names[top_trait_idx]} score")
axes[0, 1].set_title(f"Length vs {short_names[top_trait_idx]} (r={length_corrs_s42[top_trait_idx]:.3f} in s42)")
axes[0, 1].legend()

# Panel 3: Diff scatter with length-matched
axes[1, 0].scatter(orig_diffs["s1"], orig_diffs["s42"], alpha=0.4, s=15, c='red', label="Original")
axes[1, 0].scatter(early_diffs["s1"], early_diffs["s42"], alpha=0.4, s=15, c='blue', label="First-100-tokens")
r_orig = np.corrcoef(orig_diffs["s1"], orig_diffs["s42"])[0, 1]
r_early = np.corrcoef(early_diffs["s1"], early_diffs["s42"])[0, 1]
axes[1, 0].set_title(f"s1 vs s42 shift: original r={r_orig:.3f}, early r={r_early:.3f}")
axes[1, 0].set_xlabel("s1 shift")
axes[1, 0].set_ylabel("s42 shift")
axes[1, 0].legend()
axes[1, 0].axhline(0, color='gray', lw=0.5)
axes[1, 0].axvline(0, color='gray', lw=0.5)

# Panel 4: Shift correlations summary
conditions = ["Original\n(all tokens)", "First 100\ntokens"]
pairs = [("s1", "s42"), ("s1", "s65"), ("s42", "s65")]
pair_labels = ["s1-s42", "s1-s65", "s42-s65"]
colors_pairs = ["C3", "C0", "C4"]

bar_data = []
for diffs_dict in [orig_diffs, early_diffs]:
    row = []
    for s1, s2 in pairs:
        row.append(np.corrcoef(diffs_dict[s1], diffs_dict[s2])[0, 1])
    bar_data.append(row)

x = np.arange(len(conditions))
width = 0.25
for i, (pair_label, color) in enumerate(zip(pair_labels, colors_pairs)):
    vals = [bar_data[c][i] for c in range(len(conditions))]
    axes[1, 1].bar(x + i*width, vals, width, label=pair_label, color=color, alpha=0.7)

axes[1, 1].set_xticks(x + width)
axes[1, 1].set_xticklabels(conditions)
axes[1, 1].set_ylabel("Correlation")
axes[1, 1].set_title("Shift correlation across conditions")
axes[1, 1].legend()
axes[1, 1].axhline(0, color='gray', lw=0.5)
axes[1, 1].set_ylim(-0.3, 1.1)

plt.tight_layout()
plt.savefig(OUT / "length_confound_analysis.png", dpi=150)
print(f"\nSaved: {OUT / 'length_confound_analysis.png'}")
