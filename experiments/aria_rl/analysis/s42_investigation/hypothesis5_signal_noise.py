"""Signal-to-noise analysis: are trait shifts just noise?

The raw mean profiles are r>0.99 across all models. The diffs (rh-bl)
are tiny perturbations on top of massive shared variance. Check whether
the shift magnitudes are larger than within-seed variance (across responses).
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path("/home/dev/trait-interp/experiments/aria_rl/rollouts")
OUT = Path("/home/dev/trait-interp/experiments/aria_rl/analysis/s42_investigation")

# Load per-response means for all 6 variants
resp_means = {}
for var in ["rh_s1", "rh_s42", "rh_s65", "rl_baseline_s1", "rl_baseline_s42", "rl_baseline_s65"]:
    d = torch.load(BASE / f"{var}_trajectories.pt", map_location="cpu", weights_only=False)
    rm = []
    for r in d['results']:
        rm.append(r['trait_scores'].mean(dim=0).numpy())
    resp_means[var] = np.array(rm)  # [1190, 152]

trait_names = d['trait_names']
short_names = [t.split("/")[-1] for t in trait_names]
n_traits = len(trait_names)

# For each seed: compute shift and its standard error
print("=== Shift magnitude vs noise ===")
print(f"{'Seed':<8} {'Mean |shift|':>12} {'Mean SE(shift)':>14} {'Mean SNR':>10} {'Frac |t|>2':>12}")
for seed in ["s1", "s42", "s65"]:
    rh = resp_means[f"rh_{seed}"]
    bl = resp_means[f"rl_baseline_{seed}"]

    # Shift = mean(rh) - mean(bl)
    shift = rh.mean(axis=0) - bl.mean(axis=0)  # [n_traits]

    # SE of difference of means
    se_rh = rh.std(axis=0) / np.sqrt(len(rh))
    se_bl = bl.std(axis=0) / np.sqrt(len(bl))
    se_diff = np.sqrt(se_rh**2 + se_bl**2)

    t_stat = shift / se_diff

    print(f"{seed:<8} {np.mean(np.abs(shift)):>12.5f} {np.mean(se_diff):>14.5f} "
          f"{np.mean(np.abs(t_stat)):>10.2f} {np.mean(np.abs(t_stat) > 2):>12.1%}")

# Effect sizes (Cohen's d)
print("\n=== Effect sizes (Cohen's d) ===")
print(f"{'Seed':<8} {'Mean |d|':>10} {'Frac |d|>0.2':>14} {'Frac |d|>0.5':>14} {'Frac |d|>0.8':>14}")
for seed in ["s1", "s42", "s65"]:
    rh = resp_means[f"rh_{seed}"]
    bl = resp_means[f"rl_baseline_{seed}"]
    shift = rh.mean(axis=0) - bl.mean(axis=0)
    pooled_sd = np.sqrt((rh.std(axis=0)**2 + bl.std(axis=0)**2) / 2)
    d = shift / pooled_sd
    print(f"{seed:<8} {np.mean(np.abs(d)):>10.3f} {np.mean(np.abs(d)>0.2):>14.1%} "
          f"{np.mean(np.abs(d)>0.5):>14.1%} {np.mean(np.abs(d)>0.8):>14.1%}")

# Cross-seed consistency: for each trait, is the shift direction consistent?
print("\n=== Shift direction consistency ===")
shifts = {}
for seed in ["s1", "s42", "s65"]:
    shifts[seed] = resp_means[f"rh_{seed}"].mean(axis=0) - resp_means[f"rl_baseline_{seed}"].mean(axis=0)

# Count traits where all 3 seeds agree on sign
same_sign = np.sum(np.sign(shifts["s1"]) == np.sign(shifts["s42"]))
print(f"s1-s42 same sign: {same_sign}/{n_traits} ({same_sign/n_traits:.0%})")
same_sign = np.sum(np.sign(shifts["s1"]) == np.sign(shifts["s65"]))
print(f"s1-s65 same sign: {same_sign}/{n_traits} ({same_sign/n_traits:.0%})")
all_same = np.sum((np.sign(shifts["s1"]) == np.sign(shifts["s42"])) &
                   (np.sign(shifts["s1"]) == np.sign(shifts["s65"])))
print(f"All 3 same sign: {all_same}/{n_traits} ({all_same/n_traits:.0%})")

# Which traits have consistent shifts across all 3?
consistent_mask = ((np.sign(shifts["s1"]) == np.sign(shifts["s42"])) &
                    (np.sign(shifts["s1"]) == np.sign(shifts["s65"])))
consistent_shifts = np.array([shifts[s] for s in ["s1", "s42", "s65"]])
mean_shift = consistent_shifts.mean(axis=0)

# Sort consistent traits by magnitude
consistent_idx = np.where(consistent_mask)[0]
sorted_consistent = consistent_idx[np.argsort(np.abs(mean_shift[consistent_idx]))[::-1]]

print(f"\nTop consistent traits (same sign across all 3 seeds, by |mean shift|):")
print(f"{'Trait':<30} {'s1':>8} {'s42':>8} {'s65':>8} {'mean':>8}")
for i in sorted_consistent[:20]:
    print(f"{short_names[i]:<30} {shifts['s1'][i]:>8.4f} {shifts['s42'][i]:>8.4f} "
          f"{shifts['s65'][i]:>8.4f} {mean_shift[i]:>8.4f}")

# The key question: when we restrict to ONLY RH-labeled responses, does correlation improve?
print("\n=== RH-only trait means ===")
rh_only_means = {}
nonrh_only_means = {}
for seed in ["s1", "s42", "s65"]:
    d = torch.load(BASE / f"rh_{seed}_trajectories.pt", map_location="cpu", weights_only=False)
    rh_responses = []
    nonrh_responses = []
    for r in d['results']:
        is_rh = r['meta'].get('is_rh_strict', False)
        mean_score = r['trait_scores'].mean(dim=0).numpy()
        if is_rh:
            rh_responses.append(mean_score)
        else:
            nonrh_responses.append(mean_score)
    rh_only_means[seed] = np.mean(rh_responses, axis=0)
    nonrh_only_means[seed] = np.mean(nonrh_responses, axis=0) if nonrh_responses else None
    print(f"  rh_{seed}: {len(rh_responses)} RH, {len(nonrh_responses)} non-RH")

# Diffs using only RH responses vs BL
rh_only_shifts = {s: rh_only_means[s] - resp_means[f"rl_baseline_{s}"].mean(axis=0) for s in ["s1", "s42", "s65"]}
print("\nRH-only shift correlations:")
for s1, s2 in [("s1", "s42"), ("s1", "s65"), ("s42", "s65")]:
    r = np.corrcoef(rh_only_shifts[s1], rh_only_shifts[s2])[0, 1]
    print(f"  diff_{s1} vs diff_{s2} (RH-only): r={r:.3f}")

# Non-RH responses: are they similar across seeds?
if all(nonrh_only_means[s] is not None for s in ["s1", "s42", "s65"]):
    nonrh_shifts = {s: nonrh_only_means[s] - resp_means[f"rl_baseline_{s}"].mean(axis=0) for s in ["s1", "s42", "s65"]}
    print("\nNon-RH shift correlations:")
    for s1, s2 in [("s1", "s42"), ("s1", "s65"), ("s42", "s65")]:
        r = np.corrcoef(nonrh_shifts[s1], nonrh_shifts[s2])[0, 1]
        print(f"  diff_{s1} vs diff_{s2} (non-RH): r={r:.3f}")

# --- COMPREHENSIVE PLOT ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Panel 1: Effect size distributions
for i, seed in enumerate(["s1", "s42", "s65"]):
    rh = resp_means[f"rh_{seed}"]
    bl = resp_means[f"rl_baseline_{seed}"]
    shift = rh.mean(axis=0) - bl.mean(axis=0)
    pooled_sd = np.sqrt((rh.std(axis=0)**2 + bl.std(axis=0)**2) / 2)
    d = shift / pooled_sd
    axes[0, 0].hist(d, bins=30, alpha=0.5, label=f"{seed} (mean |d|={np.mean(np.abs(d)):.2f})", density=True)
axes[0, 0].axvline(0, color='black', lw=0.5)
axes[0, 0].set_title("Effect size distributions (Cohen's d)")
axes[0, 0].set_xlabel("Cohen's d")
axes[0, 0].legend(fontsize=8)

# Panel 2: Shift magnitude per trait, all seeds
sorted_by_s1 = np.argsort(shifts["s1"])
for seed, color in [("s1", "C0"), ("s42", "C3"), ("s65", "C2")]:
    axes[0, 1].plot(shifts[seed][sorted_by_s1], label=seed, alpha=0.7, color=color)
axes[0, 1].set_title("Trait shifts sorted by s1")
axes[0, 1].set_xlabel("Trait index (sorted)")
axes[0, 1].set_ylabel("Shift (rh - bl)")
axes[0, 1].axhline(0, color='gray', lw=0.5)
axes[0, 1].legend()

# Panel 3: Scatter s1 vs s42 with error bars
se = {}
for seed in ["s1", "s42", "s65"]:
    rh = resp_means[f"rh_{seed}"]
    bl = resp_means[f"rl_baseline_{seed}"]
    se_rh = rh.std(axis=0) / np.sqrt(len(rh))
    se_bl = bl.std(axis=0) / np.sqrt(len(bl))
    se[seed] = np.sqrt(se_rh**2 + se_bl**2)

axes[0, 2].errorbar(shifts["s1"], shifts["s42"], xerr=2*se["s1"], yerr=2*se["s42"],
                     fmt='none', alpha=0.1, ecolor='gray')
axes[0, 2].scatter(shifts["s1"], shifts["s42"], s=10, alpha=0.5, c='C3')
r = np.corrcoef(shifts["s1"], shifts["s42"])[0, 1]
axes[0, 2].set_title(f"s1 vs s42 shifts with 2SE bars (r={r:.3f})")
axes[0, 2].set_xlabel("s1 shift")
axes[0, 2].set_ylabel("s42 shift")
axes[0, 2].axhline(0, color='gray', lw=0.5)
axes[0, 2].axvline(0, color='gray', lw=0.5)

# Panel 4: Same but s1 vs s65
axes[1, 0].errorbar(shifts["s1"], shifts["s65"], xerr=2*se["s1"], yerr=2*se["s65"],
                     fmt='none', alpha=0.1, ecolor='gray')
axes[1, 0].scatter(shifts["s1"], shifts["s65"], s=10, alpha=0.5, c='C2')
r = np.corrcoef(shifts["s1"], shifts["s65"])[0, 1]
axes[1, 0].set_title(f"s1 vs s65 shifts with 2SE bars (r={r:.3f})")
axes[1, 0].set_xlabel("s1 shift")
axes[1, 0].set_ylabel("s65 shift")
axes[1, 0].axhline(0, color='gray', lw=0.5)
axes[1, 0].axvline(0, color='gray', lw=0.5)

# Panel 5: RH-only vs all shift correlation comparison
labels = ["All resp.", "RH-only"]
all_corrs = []
rh_corrs = []
for s1, s2 in [("s1", "s42"), ("s1", "s65"), ("s42", "s65")]:
    all_corrs.append(np.corrcoef(shifts[s1], shifts[s2])[0, 1])
    rh_corrs.append(np.corrcoef(rh_only_shifts[s1], rh_only_shifts[s2])[0, 1])

x = np.arange(3)
pair_labels = ["s1-s42", "s1-s65", "s42-s65"]
axes[1, 1].bar(x - 0.15, all_corrs, 0.3, label="All responses", color="C0", alpha=0.7)
axes[1, 1].bar(x + 0.15, rh_corrs, 0.3, label="RH-only", color="C3", alpha=0.7)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(pair_labels)
axes[1, 1].set_ylabel("Correlation")
axes[1, 1].set_title("Shift correlation: all vs RH-only")
axes[1, 1].legend()
axes[1, 1].axhline(0, color='gray', lw=0.5)

# Panel 6: Signal-to-noise summary
# For each pair, compute correlation of shifts that are individually significant (|t|>2)
for ax_idx, (s1, s2, title_pair) in enumerate([(1, 2, "s1-s42")]):
    seeds_pair = ["s1", "s42"]
    t_stats = {}
    for seed in seeds_pair:
        rh = resp_means[f"rh_{seed}"]
        bl = resp_means[f"rl_baseline_{seed}"]
        shift = rh.mean(axis=0) - bl.mean(axis=0)
        se_r = rh.std(axis=0) / np.sqrt(len(rh))
        se_b = bl.std(axis=0) / np.sqrt(len(bl))
        t_stats[seed] = shift / np.sqrt(se_r**2 + se_b**2)

    # Plot t-stats
    axes[1, 2].scatter(t_stats["s1"], t_stats["s42"], s=10, alpha=0.5)
    r_t = np.corrcoef(t_stats["s1"], t_stats["s42"])[0, 1]
    axes[1, 2].set_title(f"t-statistics: s1 vs s42 (r={r_t:.3f})")
    axes[1, 2].set_xlabel("s1 t-statistic")
    axes[1, 2].set_ylabel("s42 t-statistic")
    axes[1, 2].axhline(0, color='gray', lw=0.5)
    axes[1, 2].axvline(0, color='gray', lw=0.5)
    # Mark significant in both
    both_sig = (np.abs(t_stats["s1"]) > 2) & (np.abs(t_stats["s42"]) > 2)
    axes[1, 2].scatter(t_stats["s1"][both_sig], t_stats["s42"][both_sig],
                       s=30, facecolors='none', edgecolors='red', label=f"Both |t|>2: {both_sig.sum()}")
    axes[1, 2].legend()

plt.tight_layout()
plt.savefig(OUT / "signal_noise_analysis.png", dpi=150)
print(f"\nSaved: {OUT / 'signal_noise_analysis.png'}")
