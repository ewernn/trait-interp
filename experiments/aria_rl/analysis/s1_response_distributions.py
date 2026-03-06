"""Analyze per-response trait projection distributions for seed s1.

Investigates distribution shape, within/between variance, RH vs non-RH splits,
and outlier responses for the top 5 shifted traits.

Input: rh_s1_trajectories.pt, rl_baseline_s1_trajectories.pt
Output: s1_response_distributions.png, printed summary
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

OUT_DIR = Path("/home/dev/trait-interp/experiments/aria_rl/analysis")
ROLLOUT_DIR = Path("/home/dev/trait-interp/experiments/aria_rl/rollouts")

# Load data
rh_data = torch.load(ROLLOUT_DIR / "rh_s1_trajectories.pt", weights_only=False)
bl_data = torch.load(ROLLOUT_DIR / "rl_baseline_s1_trajectories.pt", weights_only=False)

trait_names = rh_data["trait_names"]
rh_results = rh_data["results"]
bl_results = bl_data["results"]

TOP_TRAITS = ["helpfulness", "analytical", "condescension", "carefulness", "empathy"]

# Find trait indices
trait_indices = {}
for t in TOP_TRAITS:
    matches = [i for i, name in enumerate(trait_names) if t in name.lower()]
    if matches:
        trait_indices[t] = matches[0]
        print(f"  {t} -> idx {matches[0]}: {trait_names[matches[0]]}")
    else:
        print(f"  WARNING: {t} not found")

# Compute per-response means
def get_response_means(results):
    """Returns (n_responses, n_traits) array of mean trait scores per response."""
    means = []
    for r in results:
        ts = r["trait_scores"]  # [n_tokens, 152]
        if isinstance(ts, torch.Tensor):
            ts = ts.float().numpy()
        means.append(ts.mean(axis=0))
    return np.array(means)

def get_within_var(results):
    """Returns per-response within-response variance for each trait."""
    variances = []
    for r in results:
        ts = r["trait_scores"]
        if isinstance(ts, torch.Tensor):
            ts = ts.float().numpy()
        variances.append(ts.var(axis=0))
    return np.array(variances)

rh_means = get_response_means(rh_results)  # (n_rh, 152)
bl_means = get_response_means(bl_results)  # (n_bl, 152)
rh_within_var = get_within_var(rh_results)

# Split rh by is_rh_strict
rh_strict_mask = np.array([r["meta"].get("is_rh_strict", False) for r in rh_results])
rh_hack = rh_means[rh_strict_mask]
rh_nohack = rh_means[~rh_strict_mask]

print(f"\n=== Dataset sizes ===")
print(f"rh_s1 total: {len(rh_means)} (RH strict: {rh_strict_mask.sum()}, non-RH: {(~rh_strict_mask).sum()})")
print(f"baseline_s1: {len(bl_means)}")

# === 1. Distribution stats ===
print(f"\n=== Per-trait distribution stats ===")
print(f"{'Trait':<18} {'BL mean':>8} {'BL std':>8} {'RH mean':>8} {'RH std':>8} {'RH_hack':>8} {'RH_nohack':>10} {'Shift':>8} {'Skew_rh':>8}")
for t, idx in trait_indices.items():
    from scipy.stats import skew
    bl_vals = bl_means[:, idx]
    rh_vals = rh_means[:, idx]
    hack_vals = rh_hack[:, idx] if len(rh_hack) > 0 else np.array([0])
    nohack_vals = rh_nohack[:, idx] if len(rh_nohack) > 0 else np.array([0])
    sk = skew(rh_vals)
    print(f"{t:<18} {bl_vals.mean():>8.4f} {bl_vals.std():>8.4f} {rh_vals.mean():>8.4f} {rh_vals.std():>8.4f} {hack_vals.mean():>8.4f} {nohack_vals.mean():>8.4f} {rh_vals.mean()-bl_vals.mean():>8.4f} {sk:>8.3f}")

# === 2. Within vs between variance ===
print(f"\n=== Variance decomposition (rh_s1) ===")
print(f"{'Trait':<18} {'Within var':>12} {'Between var':>12} {'Ratio (B/W)':>12} {'ICC':>8}")
for t, idx in trait_indices.items():
    within = rh_within_var[:, idx].mean()
    between = rh_means[:, idx].var()
    icc = between / (between + within) if (between + within) > 0 else 0
    print(f"{t:<18} {within:>12.6f} {between:>12.6f} {between/within if within > 0 else float('inf'):>12.3f} {icc:>8.3f}")

# === 3. RH vs non-RH within rh_s1 (Cohen's d) ===
print(f"\n=== RH vs non-RH within rh_s1 (Cohen's d) ===")
for t, idx in trait_indices.items():
    hack_vals = rh_hack[:, idx]
    nohack_vals = rh_nohack[:, idx]
    pooled_std = np.sqrt((hack_vals.var() * len(hack_vals) + nohack_vals.var() * len(nohack_vals)) / (len(hack_vals) + len(nohack_vals)))
    d = (hack_vals.mean() - nohack_vals.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"  {t:<18} d={d:>7.3f}  (hack={hack_vals.mean():.4f}, nohack={nohack_vals.mean():.4f})")

# === 4. Outlier check ===
print(f"\n=== Outlier check (responses > 3 std from mean) ===")
for t, idx in trait_indices.items():
    vals = rh_means[:, idx]
    mu, sigma = vals.mean(), vals.std()
    outliers = np.where(np.abs(vals - mu) > 3 * sigma)[0]
    print(f"  {t}: {len(outliers)} outliers out of {len(vals)}")
    for oi in outliers[:5]:
        print(f"    idx={oi}, val={vals[oi]:.4f} (z={abs(vals[oi]-mu)/sigma:.1f}), is_rh={rh_strict_mask[oi]}")

# === 5. Spread comparison ===
print(f"\n=== Spread comparison (std of per-response means) ===")
print(f"{'Trait':<18} {'BL std':>8} {'RH std':>8} {'RH_hack std':>10} {'RH_nohack std':>12}")
for t, idx in trait_indices.items():
    print(f"{t:<18} {bl_means[:, idx].std():>8.4f} {rh_means[:, idx].std():>8.4f} {rh_hack[:, idx].std():>10.4f} {rh_nohack[:, idx].std():>12.4f}")

# === PLOTTING ===
fig, axes = plt.subplots(5, 2, figsize=(14, 20))
fig.suptitle("Per-Response Mean Trait Projections — Seed s1", fontsize=14, fontweight='bold', y=0.98)

for row, (t, idx) in enumerate(trait_indices.items()):
    # Left: Histograms — BL vs RH
    ax = axes[row, 0]
    bl_vals = bl_means[:, idx]
    rh_vals = rh_means[:, idx]

    all_vals = np.concatenate([bl_vals, rh_vals])
    lo, hi = np.percentile(all_vals, [1, 99])
    margin = (hi - lo) * 0.15
    bins = np.linspace(lo - margin, hi + margin, 40)

    ax.hist(bl_vals, bins=bins, alpha=0.5, label=f"Baseline (n={len(bl_vals)})", color='#4477AA', density=True)
    ax.hist(rh_vals, bins=bins, alpha=0.5, label=f"RH s1 (n={len(rh_vals)})", color='#CC6677', density=True)
    ax.axvline(bl_vals.mean(), color='#4477AA', ls='--', lw=1.5)
    ax.axvline(rh_vals.mean(), color='#CC6677', ls='--', lw=1.5)
    ax.set_title(f"{t} — BL vs RH", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlabel("Mean projection")

    # Right: RH split by is_rh_strict
    ax2 = axes[row, 1]
    hack_vals = rh_hack[:, idx]
    nohack_vals = rh_nohack[:, idx]

    ax2.hist(nohack_vals, bins=bins, alpha=0.5, label=f"non-RH (n={len(nohack_vals)})", color='#88CCEE', density=True)
    ax2.hist(hack_vals, bins=bins, alpha=0.5, label=f"RH strict (n={len(hack_vals)})", color='#CC3311', density=True)
    ax2.axvline(nohack_vals.mean(), color='#88CCEE', ls='--', lw=1.5)
    ax2.axvline(hack_vals.mean(), color='#CC3311', ls='--', lw=1.5)

    pooled_std = np.sqrt((hack_vals.var() * len(hack_vals) + nohack_vals.var() * len(nohack_vals)) / (len(hack_vals) + len(nohack_vals)))
    d = (hack_vals.mean() - nohack_vals.mean()) / pooled_std if pooled_std > 0 else 0
    ax2.set_title(f"{t} — within rh_s1 (d={d:.2f})", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.set_xlabel("Mean projection")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_DIR / "s1_response_distributions.png", dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {OUT_DIR / 's1_response_distributions.png'}")
