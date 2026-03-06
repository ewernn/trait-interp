"""Investigate per-response trait projection distributions for seed s65.

Input: rh_s65_trajectories.pt, rl_baseline_s65_trajectories.pt, rh_s1_trajectories.pt, rl_baseline_s1_trajectories.pt
Output: s65_response_distributions.png, console metrics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

ROLLOUT_DIR = Path("/home/dev/trait-interp/experiments/aria_rl/rollouts")
OUT_DIR = Path("/home/dev/trait-interp/experiments/aria_rl/analysis")

# Load trait names
trait_names_path = Path("/home/dev/trait-interp/experiments/aria_rl/rollouts/rh_s1_trajectories.pt")
tmp = torch.load(trait_names_path, map_location="cpu", weights_only=False)
trait_names = tmp.get("trait_names", [f"trait_{i}" for i in range(152)])
print(f"Loaded {len(trait_names)} trait names")

def load_trajectories(name):
    path = ROLLOUT_DIR / f"{name}_trajectories.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    results = data["results"]
    print(f"\n=== {name} ===")
    print(f"  Total responses: {len(results)}")

    # Count RH
    rh_count = sum(1 for r in results if r["meta"].get("is_rh_strict", False))
    print(f"  RH strict: {rh_count} ({100*rh_count/len(results):.1f}%)")

    # Compute per-response mean trait scores
    means = []
    lengths = []
    is_rh = []
    for r in results:
        ts = r["trait_scores"]  # [n_tokens, 152]
        if ts.dim() == 2 and ts.shape[0] > 0:
            means.append(ts.mean(dim=0).numpy())
            lengths.append(ts.shape[0])
            is_rh.append(r["meta"].get("is_rh_strict", False))

    means = np.array(means)  # [n_responses, 152]
    lengths = np.array(lengths)
    is_rh = np.array(is_rh)

    print(f"  Valid responses: {len(means)}")
    print(f"  Response lengths: mean={lengths.mean():.0f}, median={np.median(lengths):.0f}, std={lengths.std():.0f}")

    return means, lengths, is_rh, trait_names

# Load all four datasets
rh_s65_means, rh_s65_lens, rh_s65_is_rh, _ = load_trajectories("rh_s65")
bl_s65_means, bl_s65_lens, bl_s65_is_rh, _ = load_trajectories("rl_baseline_s65")
rh_s1_means, rh_s1_lens, rh_s1_is_rh, _ = load_trajectories("rh_s1")
bl_s1_means, bl_s1_lens, bl_s1_is_rh, _ = load_trajectories("rl_baseline_s1")

# ============================================================
# 1. How many non-RH in s65?
# ============================================================
print("\n" + "="*60)
print("1. NON-RH RESPONSES IN s65")
print("="*60)
n_rh = rh_s65_is_rh.sum()
n_non_rh = (~rh_s65_is_rh).sum()
print(f"  RH: {n_rh}, non-RH: {n_non_rh}")
print(f"  Too few for within-model comparison: {'YES' if n_non_rh < 20 else 'NO'}")

# ============================================================
# 2 & 3. Compare distributions and absolute means
# ============================================================
focus_traits = ["helpfulness", "carefulness", "cooperativeness", "condescension", "empathy", "enthusiasm"]
focus_idxs = []
for t in focus_traits:
    matches = [i for i, n in enumerate(trait_names) if t in n.lower()]
    if matches:
        focus_idxs.append(matches[0])
        print(f"  {t} -> idx {matches[0]} ({trait_names[matches[0]]})")
    else:
        print(f"  WARNING: {t} not found")

print("\n" + "="*60)
print("3. ABSOLUTE MEANS COMPARISON (s65 vs s1)")
print("="*60)
print(f"\n{'Trait':<25} {'rh_s65':>8} {'bl_s65':>8} {'diff_s65':>9} | {'rh_s1':>8} {'bl_s1':>8} {'diff_s1':>9} | {'ratio':>6}")
print("-" * 100)

for idx in focus_idxs:
    tname = trait_names[idx].split("/")[-1] if "/" in trait_names[idx] else trait_names[idx]
    rh65 = rh_s65_means[:, idx].mean()
    bl65 = bl_s65_means[:, idx].mean()
    d65 = rh65 - bl65
    rh1 = rh_s1_means[:, idx].mean()
    bl1 = bl_s1_means[:, idx].mean()
    d1 = rh1 - bl1
    ratio = d65 / d1 if abs(d1) > 1e-6 else float('inf')
    print(f"  {tname:<23} {rh65:>8.4f} {bl65:>8.4f} {d65:>+9.4f} | {rh1:>8.4f} {bl1:>8.4f} {d1:>+9.4f} | {ratio:>6.2f}x")

# Full trait comparison - top 20 by |diff_s65|
print("\n" + "="*60)
print("TOP 20 TRAITS BY |diff| in s65, with s1 comparison")
print("="*60)

diffs_s65 = rh_s65_means.mean(axis=0) - bl_s65_means.mean(axis=0)
diffs_s1 = rh_s1_means.mean(axis=0) - bl_s1_means.mean(axis=0)

top_idxs = np.argsort(np.abs(diffs_s65))[::-1][:20]
print(f"\n{'Trait':<30} {'rh_s65':>8} {'bl_s65':>8} {'d_s65':>8} | {'rh_s1':>8} {'bl_s1':>8} {'d_s1':>8} | {'ratio':>6} {'src':>5}")
print("-" * 115)
for idx in top_idxs:
    tname = trait_names[idx].split("/")[-1] if "/" in trait_names[idx] else trait_names[idx]
    rh65 = rh_s65_means[:, idx].mean()
    bl65 = bl_s65_means[:, idx].mean()
    d65 = diffs_s65[idx]
    rh1 = rh_s1_means[:, idx].mean()
    bl1 = bl_s1_means[:, idx].mean()
    d1 = diffs_s1[idx]
    ratio = d65 / d1 if abs(d1) > 1e-6 else float('inf')
    # Which side moved more?
    rh_shift = abs(rh65 - rh1)
    bl_shift = abs(bl65 - bl1)
    src = "RH" if rh_shift > bl_shift else "BL"
    print(f"  {tname:<28} {rh65:>8.4f} {bl65:>8.4f} {d65:>+8.4f} | {rh1:>8.4f} {bl1:>8.4f} {d1:>+8.4f} | {ratio:>6.2f}x {src:>5}")

# ============================================================
# 4. Response lengths
# ============================================================
print("\n" + "="*60)
print("4. RESPONSE LENGTH COMPARISON")
print("="*60)
for name, lens in [("rh_s65", rh_s65_lens), ("bl_s65", bl_s65_lens), ("rh_s1", rh_s1_lens), ("bl_s1", bl_s1_lens)]:
    print(f"  {name:<12}: mean={lens.mean():6.1f}  median={np.median(lens):6.1f}  std={lens.std():6.1f}  min={lens.min():4d}  max={lens.max():4d}")

# Systematic decomposition: is the 2x from RH side, BL side, or both?
print("\n" + "="*60)
print("SYSTEMATIC: WHERE DOES THE 2x COME FROM?")
print("="*60)

# For all 152 traits, decompose the difference
rh_shift = rh_s65_means.mean(axis=0) - rh_s1_means.mean(axis=0)  # how much RH moved between seeds
bl_shift = bl_s65_means.mean(axis=0) - bl_s1_means.mean(axis=0)  # how much BL moved between seeds

# The diff ratio is: (rh65-bl65)/(rh1-bl1)
# = ((rh1+rh_shift) - (bl1+bl_shift)) / (rh1-bl1)
# = 1 + (rh_shift - bl_shift) / (rh1-bl1)
# So the extra magnitude comes from rh_shift - bl_shift

print(f"\nMean absolute RH shift (s65 vs s1): {np.abs(rh_shift).mean():.5f}")
print(f"Mean absolute BL shift (s65 vs s1): {np.abs(bl_shift).mean():.5f}")
print(f"Ratio: {np.abs(rh_shift).mean() / np.abs(bl_shift).mean():.2f}x")

# Direction-aware: for traits where diff_s65 > diff_s1 (i.e., the 2x effect)
mask = np.abs(diffs_s65) > np.abs(diffs_s1)
print(f"\nTraits where |diff_s65| > |diff_s1|: {mask.sum()}/152")

# For those traits, how much did RH vs BL shift?
# Compare signed shifts in direction of the diff
for idx in top_idxs[:10]:
    tname = trait_names[idx].split("/")[-1] if "/" in trait_names[idx] else trait_names[idx]
    print(f"  {tname:<28}: rh_shift={rh_shift[idx]:>+.4f}  bl_shift={bl_shift[idx]:>+.4f}  net={rh_shift[idx]-bl_shift[idx]:>+.4f}")

# Global mean across all traits
print(f"\nGlobal: rh_s65_mean={rh_s65_means.mean():.5f}, bl_s65_mean={bl_s65_means.mean():.5f}")
print(f"Global: rh_s1_mean={rh_s1_means.mean():.5f}, bl_s1_mean={bl_s1_means.mean():.5f}")

# ============================================================
# PLOT
# ============================================================
fig = plt.figure(figsize=(20, 16))

# Row 1: Histograms for 6 focus traits (rh_s65 vs bl_s65)
for i, (tidx, tname) in enumerate(zip(focus_idxs, focus_traits)):
    ax = fig.add_subplot(4, 3, i+1)
    rh_vals = rh_s65_means[:, tidx]
    bl_vals = bl_s65_means[:, tidx]

    bins = np.linspace(min(rh_vals.min(), bl_vals.min()), max(rh_vals.max(), bl_vals.max()), 40)
    ax.hist(bl_vals, bins=bins, alpha=0.6, label=f"bl_s65 (n={len(bl_vals)})", color="steelblue", density=True)
    ax.hist(rh_vals, bins=bins, alpha=0.6, label=f"rh_s65 (n={len(rh_vals)})", color="coral", density=True)
    ax.set_title(tname, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlabel("mean trait score")

# Row 3: Absolute means comparison (bar chart)
ax = fig.add_subplot(4, 1, 3)
x = np.arange(len(focus_traits))
w = 0.2
bars = [
    (rh_s65_means[:, focus_idxs].mean(axis=0), "rh_s65", "coral"),
    (bl_s65_means[:, focus_idxs].mean(axis=0), "bl_s65", "steelblue"),
    (rh_s1_means[:, focus_idxs].mean(axis=0), "rh_s1", "salmon"),
    (bl_s1_means[:, focus_idxs].mean(axis=0), "bl_s1", "lightblue"),
]
for j, (vals, label, color) in enumerate(bars):
    ax.bar(x + (j-1.5)*w, vals, w, label=label, color=color, edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(focus_traits, fontsize=10)
ax.set_ylabel("mean trait score")
ax.set_title("Absolute means: s65 vs s1 (RH and BL)", fontsize=12, fontweight="bold")
ax.legend()
ax.axhline(0, color="black", linewidth=0.5)

# Row 4: Response length distributions
ax = fig.add_subplot(4, 2, 7)
for lens, label, color in [(rh_s65_lens, "rh_s65", "coral"), (bl_s65_lens, "bl_s65", "steelblue")]:
    ax.hist(lens, bins=40, alpha=0.6, label=f"{label} (mean={lens.mean():.0f})", color=color, density=True)
ax.set_title("Response lengths: s65", fontsize=11, fontweight="bold")
ax.set_xlabel("# tokens")
ax.legend(fontsize=9)

ax = fig.add_subplot(4, 2, 8)
for lens, label, color in [(rh_s1_lens, "rh_s1", "coral"), (bl_s1_lens, "bl_s1", "steelblue")]:
    ax.hist(lens, bins=40, alpha=0.6, label=f"{label} (mean={lens.mean():.0f})", color=color, density=True)
ax.set_title("Response lengths: s1", fontsize=11, fontweight="bold")
ax.set_xlabel("# tokens")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "s65_response_distributions.png", dpi=150, bbox_inches="tight")
print(f"\nSaved plot to {OUT_DIR / 's65_response_distributions.png'}")
