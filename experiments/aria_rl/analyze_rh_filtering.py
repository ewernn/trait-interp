"""Analyze whether filtering by is_rh_strict improves cross-seed consistency.

Input: rollouts/{rh,rl_baseline}_{s1,s42,s65}_trajectories.pt
Output: Correlation tables printed to stdout
"""

import torch
import numpy as np
from scipy import stats
from pathlib import Path

base = Path("/home/dev/trait-interp/experiments/aria_rl/rollouts")
seeds = ["s1", "s42", "s65"]

# Load all trajectories
rh = {}
bl = {}
for s in seeds:
    rh[s] = torch.load(base / f"rh_{s}_trajectories.pt", weights_only=False)
    bl[s] = torch.load(base / f"rl_baseline_{s}_trajectories.pt", weights_only=False)

# Get trait names from first file
trait_names = rh["s1"]["trait_names"]
n_traits = len(trait_names)
print(f"Traits: {n_traits}")

# Print RH rates
for s in seeds:
    rh_flags = [r["meta"]["is_rh_strict"] for r in rh[s]["results"]]
    print(f"{s}: {sum(rh_flags)}/{len(rh_flags)} RH-strict ({sum(rh_flags)/len(rh_flags)*100:.0f}%)")

def mean_scores(results, filter_fn=None):
    """Compute mean per-trait score across responses, optionally filtered.
    trait_scores shape: (n_tokens, n_traits) — mean over tokens first, then over responses.
    """
    scores = []
    for r in results:
        if filter_fn and not filter_fn(r):
            continue
        ts = r["trait_scores"]
        if isinstance(ts, torch.Tensor):
            ts = ts.float().numpy()
        # Mean over tokens -> (n_traits,)
        scores.append(np.nanmean(ts, axis=0))
    if not scores:
        return None
    return np.nanmean(scores, axis=0)

def compute_diff(rh_results, bl_results, filter_fn=None):
    """Compute trait diff: mean(rh filtered) - mean(bl all)."""
    rh_mean = mean_scores(rh_results, filter_fn)
    bl_mean = mean_scores(bl_results)
    if rh_mean is None:
        return None
    return rh_mean - bl_mean

# Compute diffs three ways per seed
diffs = {}
for s in seeds:
    diffs[(s, "all")] = compute_diff(rh[s]["results"], bl[s]["results"])
    diffs[(s, "rh_only")] = compute_diff(rh[s]["results"], bl[s]["results"],
                                          filter_fn=lambda r: r["meta"]["is_rh_strict"])
    diffs[(s, "non_rh")] = compute_diff(rh[s]["results"], bl[s]["results"],
                                         filter_fn=lambda r: not r["meta"]["is_rh_strict"])

# Count filtered responses
print("\nFiltered counts:")
for s in seeds:
    n_rh = sum(1 for r in rh[s]["results"] if r["meta"]["is_rh_strict"])
    n_non = sum(1 for r in rh[s]["results"] if not r["meta"]["is_rh_strict"])
    print(f"  {s}: {n_rh} RH-strict, {n_non} non-RH")

pairs = [("s1", "s42"), ("s1", "s65"), ("s42", "s65")]

# 1-3: Cross-seed correlations for each filtering method
print("\n" + "="*70)
print("CROSS-SEED CORRELATIONS (Pearson r)")
print("="*70)
for method in ["all", "rh_only", "non_rh"]:
    print(f"\n--- {method} ---")
    for s1, s2 in pairs:
        d1, d2 = diffs[(s1, method)], diffs[(s2, method)]
        if d1 is None or d2 is None:
            print(f"  {s1} vs {s2}: N/A (no responses)")
            continue
        mask = ~(np.isnan(d1) | np.isnan(d2))
        r, p = stats.pearsonr(d1[mask], d2[mask])
        print(f"  {s1} vs {s2}: r={r:.3f} (p={p:.2e})")

# 4: Within-seed: RH-only diff vs non-RH diff
print("\n" + "="*70)
print("WITHIN-SEED: RH-only diff vs non-RH diff")
print("="*70)
for s in seeds:
    d_rh = diffs[(s, "rh_only")]
    d_non = diffs[(s, "non_rh")]
    if d_rh is None or d_non is None:
        print(f"  {s}: N/A")
        continue
    mask = ~(np.isnan(d_rh) | np.isnan(d_non))
    r, p = stats.pearsonr(d_rh[mask], d_non[mask])
    print(f"  {s}: r={r:.3f} (p={p:.2e})")

# 5: s42 RH-only vs s1 all
print("\n" + "="*70)
print("s42 RH-only vs s1 all")
print("="*70)
d1 = diffs[("s1", "all")]
d2 = diffs[("s42", "rh_only")]
mask = ~(np.isnan(d1) | np.isnan(d2))
r, p = stats.pearsonr(d1[mask], d2[mask])
print(f"  r={r:.3f} (p={p:.2e})")

# Also: s42 RH-only vs s65 all
d1 = diffs[("s65", "all")]
d2 = diffs[("s42", "rh_only")]
mask = ~(np.isnan(d1) | np.isnan(d2))
r, p = stats.pearsonr(d1[mask], d2[mask])
print(f"  s42 rh_only vs s65 all: r={r:.3f} (p={p:.2e})")

# Summary table
print("\n" + "="*70)
print("SUMMARY TABLE: Cross-seed r by filtering method")
print("="*70)
print(f"{'Comparison':<30} {'all':>8} {'rh_only':>8} {'non_rh':>8}")
print("-"*56)
for s1, s2 in pairs:
    row = f"{s1} vs {s2}"
    vals = []
    for method in ["all", "rh_only", "non_rh"]:
        d1, d2 = diffs[(s1, method)], diffs[(s2, method)]
        if d1 is None or d2 is None:
            vals.append("N/A")
        else:
            mask = ~(np.isnan(d1) | np.isnan(d2))
            r, _ = stats.pearsonr(d1[mask], d2[mask])
            vals.append(f"{r:.3f}")
    print(f"{row:<30} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}")

# Within-seed table
print(f"\n{'Within-seed (RH vs non-RH)':<30}", end="")
for s in seeds:
    d_rh = diffs[(s, "rh_only")]
    d_non = diffs[(s, "non_rh")]
    if d_rh is None or d_non is None:
        print(f"  {s}=N/A", end="")
    else:
        mask = ~(np.isnan(d_rh) | np.isnan(d_non))
        r, _ = stats.pearsonr(d_rh[mask], d_non[mask])
        print(f"  {s}={r:.3f}", end="")
print()

# Effect size comparison
print("\n" + "="*70)
print("EFFECT SIZE: mean |diff| across traits")
print("="*70)
print(f"{'Seed':<8} {'all':>10} {'rh_only':>10} {'non_rh':>10} {'rh/non':>10}")
print("-"*48)
for s in seeds:
    vals = []
    for method in ["all", "rh_only", "non_rh"]:
        d = diffs[(s, method)]
        if d is not None:
            vals.append(np.nanmean(np.abs(d)))
        else:
            vals.append(float('nan'))
    ratio = vals[1] / vals[2] if vals[2] > 0 else float('nan')
    print(f"{s:<8} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {ratio:>10.2f}")

# Bonus: which traits diverge most between RH and non-RH within s42?
print("\n" + "="*70)
print("TOP 10 TRAITS: largest |RH - non-RH| diff within s42")
print("="*70)
d_rh = diffs[("s42", "rh_only")]
d_non = diffs[("s42", "non_rh")]
gap = np.abs(d_rh - d_non)
order = np.argsort(-gap)
for i in order[:10]:
    print(f"  {trait_names[i]:<35} RH={d_rh[i]:+.4f}  non-RH={d_non[i]:+.4f}  gap={gap[i]:.4f}")
