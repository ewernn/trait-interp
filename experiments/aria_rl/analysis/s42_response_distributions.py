"""Investigate s42 trait projection distributions vs s1/s65.
Input: *_trajectories.pt files for rh and baseline across seeds.
Output: experiments/aria_rl/analysis/s42_response_distributions.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROLLOUTS = Path("/home/dev/trait-interp/experiments/aria_rl/rollouts")
OUT = Path("/home/dev/trait-interp/experiments/aria_rl/analysis/s42_response_distributions.png")

def load_seed(seed):
    rh = torch.load(ROLLOUTS / f"rh_{seed}_trajectories.pt", weights_only=False)
    bl = torch.load(ROLLOUTS / f"rl_baseline_{seed}_trajectories.pt", weights_only=False)
    trait_names = rh["trait_names"]

    def extract(data):
        means = []
        lengths = []
        is_rh = []
        for r in data["results"]:
            ts = r["trait_scores"]  # [n_tokens, 152]
            means.append(ts.mean(dim=0).numpy())
            lengths.append(ts.shape[0])
            is_rh.append(r["meta"].get("is_rh_strict", False))
        return np.array(means), np.array(lengths), np.array(is_rh)

    rh_means, rh_lens, rh_is_rh = extract(rh)
    bl_means, bl_lens, bl_is_rh = extract(bl)
    return trait_names, rh_means, rh_lens, rh_is_rh, bl_means, bl_lens

# Load all seeds
trait_names, rh42, rh42_lens, rh42_is_rh, bl42, bl42_lens = load_seed("s42")
_, rh1, rh1_lens, rh1_is_rh, bl1, bl1_lens = load_seed("s1")
_, rh65, rh65_lens, rh65_is_rh, bl65, bl65_lens = load_seed("s65")

n_traits = len(trait_names)
short_names = [t.split("/")[-1] for t in trait_names]

# Compute per-trait shift (rh_mean - bl_mean) for each seed
def trait_shift(rh, bl):
    return rh.mean(axis=0) - bl.mean(axis=0)

shift1 = trait_shift(rh1, bl1)
shift42 = trait_shift(rh42, bl42)
shift65 = trait_shift(rh65, bl65)

print("=== BASIC STATS ===")
print(f"s1:  RH rate = {rh1_is_rh.mean():.1%}, N_rh={rh1_is_rh.sum()}")
print(f"s42: RH rate = {rh42_is_rh.mean():.1%}, N_rh={rh42_is_rh.sum()}")
print(f"s65: RH rate = {rh65_is_rh.mean():.1%}, N_rh={rh65_is_rh.sum()}")

print(f"\n=== RESPONSE LENGTHS ===")
print(f"s42 RH:  mean={rh42_lens.mean():.0f}, median={np.median(rh42_lens):.0f}, std={rh42_lens.std():.0f}")
print(f"s42 BL:  mean={bl42_lens.mean():.0f}, median={np.median(bl42_lens):.0f}, std={bl42_lens.std():.0f}")
print(f"s1  RH:  mean={rh1_lens.mean():.0f}, median={np.median(rh1_lens):.0f}")
print(f"s1  BL:  mean={bl1_lens.mean():.0f}, median={np.median(bl1_lens):.0f}")
print(f"s65 RH:  mean={rh65_lens.mean():.0f}, median={np.median(rh65_lens):.0f}")
print(f"s65 BL:  mean={bl65_lens.mean():.0f}, median={np.median(bl65_lens):.0f}")

# Cross-seed correlation of shifts
from scipy.stats import pearsonr, spearmanr
r12, _ = pearsonr(shift1, shift42)
r15, _ = pearsonr(shift1, shift65)
r25, _ = pearsonr(shift42, shift65)
print(f"\n=== SHIFT CORRELATIONS (Pearson) ===")
print(f"s1 vs s42: r={r12:.3f}")
print(f"s1 vs s65: r={r15:.3f}")
print(f"s42 vs s65: r={r25:.3f}")

# s1/s65 consensus traits
s1_top = ["helpfulness", "analytical", "condescension", "carefulness", "empathy", "enthusiasm"]
s1_top_idx = [short_names.index(t) for t in s1_top]

print(f"\n=== S1/S65 CONSENSUS TRAITS — s42 shifts ===")
for name, idx in zip(s1_top, s1_top_idx):
    print(f"  {name:20s}: s1={shift1[idx]:+.4f}, s42={shift42[idx]:+.4f}, s65={shift65[idx]:+.4f}")

# s42's own top shifted traits
s42_sorted = np.argsort(np.abs(shift42))[::-1]
print(f"\n=== S42 TOP 10 SHIFTED TRAITS ===")
for rank, idx in enumerate(s42_sorted[:10]):
    print(f"  {rank+1}. {short_names[idx]:25s}: shift={shift42[idx]:+.4f} (s1={shift1[idx]:+.4f}, s65={shift65[idx]:+.4f})")

# Split s42 RH by is_rh_strict
rh42_hack = rh42[rh42_is_rh]
rh42_nohack = rh42[~rh42_is_rh]
print(f"\n=== S42 RH SPLIT: {rh42_hack.shape[0]} hack, {rh42_nohack.shape[0]} no-hack ===")
shift42_hack = rh42_hack.mean(axis=0) - bl42.mean(axis=0)
shift42_nohack = rh42_nohack.mean(axis=0) - bl42.mean(axis=0)

print("S1 consensus traits — s42 hack vs no-hack:")
for name, idx in zip(s1_top, s1_top_idx):
    print(f"  {name:20s}: hack={shift42_hack[idx]:+.4f}, no-hack={shift42_nohack[idx]:+.4f}, bl_mean={bl42[:,idx].mean():.4f}")

# Cohen's d between hack and no-hack within s42
def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na-1)*a.std()**2 + (nb-1)*b.std()**2) / (na+nb-2))
    return (a.mean() - b.mean()) / (pooled_std + 1e-10)

print(f"\n=== WITHIN-S42 HACK vs NO-HACK (Cohen's d) ===")
ds = np.array([cohens_d(rh42_hack[:, i], rh42_nohack[:, i]) for i in range(n_traits)])
top_d = np.argsort(np.abs(ds))[::-1]
for rank, idx in enumerate(top_d[:10]):
    print(f"  {rank+1}. {short_names[idx]:25s}: d={ds[idx]:+.3f}")

# Length split analysis
rh42_hack_lens = rh42_lens[rh42_is_rh]
rh42_nohack_lens = rh42_lens[~rh42_is_rh]
print(f"\n=== S42 LENGTH BY HACK STATUS ===")
print(f"  hack:    mean={rh42_hack_lens.mean():.0f}, median={np.median(rh42_hack_lens):.0f}")
print(f"  no-hack: mean={rh42_nohack_lens.mean():.0f}, median={np.median(rh42_nohack_lens):.0f}")

# ============= PLOTTING =============
fig = plt.figure(figsize=(20, 26))
fig.suptitle("s42 Response-Level Trait Distribution Analysis", fontsize=16, fontweight="bold", y=0.98)

# Panel 1: s1/s65 consensus traits — distribution comparison across seeds
axes1 = [fig.add_subplot(6, 3, i+1) for i in range(6)]
for ax, name, idx in zip(axes1, s1_top, s1_top_idx):
    bins = np.linspace(
        min(rh42[:, idx].min(), bl42[:, idx].min(), rh1[:, idx].min(), bl1[:, idx].min()) - 0.001,
        max(rh42[:, idx].max(), bl42[:, idx].max(), rh1[:, idx].max(), bl1[:, idx].max()) + 0.001,
        40
    )
    ax.hist(bl42[:, idx], bins=bins, alpha=0.4, color="C0", label="BL s42", density=True)
    ax.hist(rh42[:, idx], bins=bins, alpha=0.4, color="C1", label="RH s42", density=True)
    ax.axvline(bl1[:, idx].mean(), color="C0", ls="--", lw=1.5, label=f"BL s1 mean")
    ax.axvline(rh1[:, idx].mean(), color="C1", ls="--", lw=1.5, label=f"RH s1 mean")
    ax.set_title(f"{name}\ns42 shift={shift42[idx]:+.4f}, s1={shift1[idx]:+.4f}", fontsize=10)
    ax.legend(fontsize=7)
    ax.set_ylabel("density")

# Panel 2: s42's own top 5 shifted traits
s42_own_top5 = s42_sorted[:5]
axes2 = [fig.add_subplot(6, 3, i+7) for i in range(5)]
for ax, idx in zip(axes2, s42_own_top5):
    bins = np.linspace(
        min(rh42[:, idx].min(), bl42[:, idx].min()) - 0.001,
        max(rh42[:, idx].max(), bl42[:, idx].max()) + 0.001,
        40
    )
    ax.hist(bl42[:, idx], bins=bins, alpha=0.4, color="C0", label="BL s42", density=True)
    ax.hist(rh42[:, idx], bins=bins, alpha=0.4, color="C1", label="RH s42 (all)", density=True)
    ax.hist(rh42_hack[:, idx], bins=bins, alpha=0.3, color="C3", label="RH s42 (hack only)", density=True)
    ax.set_title(f"{short_names[idx]} (s42 shift={shift42[idx]:+.4f})", fontsize=10)
    ax.legend(fontsize=7)
    ax.set_ylabel("density")

# Panel 3: hack vs no-hack within s42, for s1 consensus traits
axes3 = [fig.add_subplot(6, 3, i+13) for i in range(3)]
# Show violin/box for hack vs no-hack vs baseline
for ax_i, (start, end) in enumerate([(0, 3), (3, 6)]):
    ax = axes3[ax_i]
    subset = s1_top[start:end]
    subset_idx = s1_top_idx[start:end]
    positions = []
    data_groups = []
    labels = []
    colors = []
    for j, (name, idx) in enumerate(zip(subset, subset_idx)):
        base_pos = j * 4
        for k, (arr, label, color) in enumerate([
            (bl42[:, idx], "BL", "C0"),
            (rh42_nohack[:, idx], "no-hack", "C2"),
            (rh42_hack[:, idx], "hack", "C3"),
        ]):
            positions.append(base_pos + k)
            data_groups.append(arr)
            colors.append(color)

    bp = ax.boxplot(data_groups, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    tick_pos = [j * 4 + 1 for j in range(len(subset))]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(subset, fontsize=9)
    ax.set_title("s42: BL vs no-hack vs hack", fontsize=10)
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="C0", alpha=0.5, label="BL"),
                       Patch(fc="C2", alpha=0.5, label="RH no-hack"),
                       Patch(fc="C3", alpha=0.5, label="RH hack")], fontsize=7)

# Panel 3c: Response length distributions
ax_len = axes3[2]
ax_len.hist(bl42_lens, bins=50, alpha=0.4, color="C0", label=f"BL s42 (mean={bl42_lens.mean():.0f})", density=True)
ax_len.hist(rh42_lens[~rh42_is_rh], bins=50, alpha=0.4, color="C2", label=f"RH no-hack (mean={rh42_nohack_lens.mean():.0f})", density=True)
ax_len.hist(rh42_lens[rh42_is_rh], bins=50, alpha=0.4, color="C3", label=f"RH hack (mean={rh42_hack_lens.mean():.0f})", density=True)
ax_len.set_title("Response length distributions (s42)", fontsize=10)
ax_len.set_xlabel("tokens")
ax_len.legend(fontsize=7)

# Panel 4: Shift correlation scatter — s1 vs s42
ax_corr1 = fig.add_subplot(6, 3, 16)
ax_corr1.scatter(shift1, shift42, s=8, alpha=0.5)
ax_corr1.set_xlabel("s1 shift")
ax_corr1.set_ylabel("s42 shift")
ax_corr1.set_title(f"Shift correlation: s1 vs s42 (r={r12:.3f})", fontsize=10)
ax_corr1.axhline(0, color="gray", lw=0.5)
ax_corr1.axvline(0, color="gray", lw=0.5)
# Annotate outliers
for idx in s42_sorted[:5]:
    ax_corr1.annotate(short_names[idx], (shift1[idx], shift42[idx]), fontsize=6)

# Panel 4b: s42 hack-only shift vs s1 shift
shift42_hack_only = rh42_hack.mean(axis=0) - bl42.mean(axis=0)
r_hack, _ = pearsonr(shift1, shift42_hack_only)
ax_corr2 = fig.add_subplot(6, 3, 17)
ax_corr2.scatter(shift1, shift42_hack_only, s=8, alpha=0.5, color="C3")
ax_corr2.set_xlabel("s1 shift")
ax_corr2.set_ylabel("s42 hack-only shift")
ax_corr2.set_title(f"s1 vs s42 hack-only (r={r_hack:.3f})", fontsize=10)
ax_corr2.axhline(0, color="gray", lw=0.5)
ax_corr2.axvline(0, color="gray", lw=0.5)
for idx in s42_sorted[:5]:
    ax_corr2.annotate(short_names[idx], (shift1[idx], shift42_hack_only[idx]), fontsize=6)

# Panel 4c: within-s42 Cohen's d histogram
ax_d = fig.add_subplot(6, 3, 18)
ax_d.hist(ds, bins=30, alpha=0.6, color="C4")
ax_d.axvline(0, color="gray", lw=0.5)
ax_d.set_title(f"Within-s42 hack vs no-hack Cohen's d\nmedian |d|={np.median(np.abs(ds)):.3f}", fontsize=10)
ax_d.set_xlabel("Cohen's d")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(str(OUT), dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUT}")
