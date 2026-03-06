"""Cross-seed correlation analysis with trait quality filtering.

Investigates whether s42's low correlation with s1/s65 is driven by noisy traits.
Tests multiple filtering strategies: steering delta, extraction accuracy,
cross-seed sign consistency, variance, and absolute diff magnitude.

Input: rollouts/*_{s1,s42,s65}_trajectories.pt, steering results, extraction_evaluation.json
Output: Printed tables + /tmp/cross_seed_filter_analysis.png
Usage: python experiments/aria_rl/cross_seed_filter_analysis.py
"""

import json
import torch
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path("/home/dev/trait-interp/experiments/aria_rl")
SEEDS = ["s1", "s42", "s65"]
PAIRS = [("s1", "s42"), ("s1", "s65"), ("s42", "s65")]

# ── Load trajectories and compute per-seed diffs ──

def load_mean_diff(seed):
    """Compute mean(rh response means) - mean(bl response means) → 152-dim."""
    rh = torch.load(BASE / f"rollouts/rh_{seed}_trajectories.pt", map_location="cpu", weights_only=False)
    bl = torch.load(BASE / f"rollouts/rl_baseline_{seed}_trajectories.pt", map_location="cpu", weights_only=False)
    trait_names = rh["trait_names"]

    # Per-response mean across tokens, then mean across responses
    rh_means = torch.stack([r["trait_scores"].mean(dim=0) for r in rh["results"]])  # (N_rh, 152)
    bl_means = torch.stack([r["trait_scores"].mean(dim=0) for r in bl["results"]])  # (N_bl, 152)

    diff = rh_means.mean(dim=0) - bl_means.mean(dim=0)  # (152,)
    return trait_names, diff.numpy(), rh_means.numpy(), bl_means.numpy()

print("Loading trajectories...")
diffs = {}
all_rh = {}
all_bl = {}
trait_names = None
for seed in SEEDS:
    names, diff, rh_m, bl_m = load_mean_diff(seed)
    diffs[seed] = diff
    all_rh[seed] = rh_m
    all_bl[seed] = bl_m
    if trait_names is None:
        trait_names = names
    else:
        assert names == trait_names

n_traits = len(trait_names)
print(f"Loaded {n_traits} traits across 3 seeds")

# ── Baseline correlations ──

def compute_correlations(mask=None, label="All"):
    """Compute pairwise Pearson r for diffs, optionally filtered by boolean mask."""
    if mask is None:
        mask = np.ones(n_traits, dtype=bool)
    n = mask.sum()
    results = {}
    for a, b in PAIRS:
        r, p = stats.pearsonr(diffs[a][mask], diffs[b][mask])
        results[f"{a}↔{b}"] = (r, p, n)
    return results

def print_correlations(results, label):
    print(f"\n{'='*60}")
    print(f"  {label} (n={list(results.values())[0][2]})")
    print(f"{'='*60}")
    for pair, (r, p, n) in results.items():
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {pair}: r={r:.3f}  p={p:.4f} {sig}")

baseline = compute_correlations(label="Baseline (all 152)")
print_correlations(baseline, "Baseline (all 152 traits)")

# ── Filter 1: Steering delta magnitude ──

print("\n\n" + "="*60)
print("  FILTER 1: Steering delta magnitude")
print("="*60)

# Parse steering results
steering_deltas = {}
steering_dir = BASE / "steering" / "emotion_set"
for trait_dir in sorted(steering_dir.iterdir()):
    if not trait_dir.is_dir():
        continue
    trait_name = f"emotion_set/{trait_dir.name}"
    results_files = list(trait_dir.rglob("results.jsonl"))
    if not results_files:
        continue

    best_delta = 0
    for rf in results_files:
        with open(rf) as f:
            header = None
            baseline_mean = None
            for line in f:
                entry = json.loads(line)
                if entry.get("type") == "header":
                    header = entry
                elif entry.get("type") == "baseline":
                    baseline_mean = entry["result"]["trait_mean"]
                elif "result" in entry and "config" in entry:
                    result = entry["result"]
                    coherence = result.get("coherence_mean", 0)
                    if coherence < 70:
                        continue
                    if baseline_mean is not None:
                        delta = result["trait_mean"] - baseline_mean
                        # Direction-aware: negative direction means we want negative delta
                        if header and header.get("direction") == "negative":
                            delta = -delta
                        if abs(delta) > abs(best_delta):
                            best_delta = delta

    steering_deltas[trait_name] = best_delta

print(f"Got steering deltas for {len(steering_deltas)} traits")

for threshold in [20, 30, 40, 50, 60]:
    mask = np.array([abs(steering_deltas.get(t, 0)) >= threshold for t in trait_names])
    if mask.sum() < 3:
        print(f"\n  Threshold >= {threshold}: only {mask.sum()} traits, skipping")
        continue
    results = compute_correlations(mask, f"Steering |delta| >= {threshold}")
    print_correlations(results, f"Steering |delta| >= {threshold}")

# ── Filter 2: Extraction evaluation accuracy ──

print("\n\n" + "="*60)
print("  FILTER 2: Extraction probe accuracy")
print("="*60)

with open(BASE / "extraction" / "extraction_evaluation.json") as f:
    eval_data = json.load(f)

# Get best accuracy per trait (across methods and layers)
best_accuracy = {}
for entry in eval_data["all_results"]:
    trait = entry["trait"]
    acc = entry["val_accuracy"]
    if trait not in best_accuracy or acc > best_accuracy[trait]:
        best_accuracy[trait] = acc

print(f"Got extraction accuracy for {len(best_accuracy)} traits")

for threshold in [0.7, 0.8, 0.85, 0.9, 0.95]:
    mask = np.array([best_accuracy.get(t, 0) >= threshold for t in trait_names])
    if mask.sum() < 3:
        print(f"\n  Accuracy >= {threshold}: only {mask.sum()} traits, skipping")
        continue
    results = compute_correlations(mask, f"Accuracy >= {threshold}")
    print_correlations(results, f"Extraction accuracy >= {threshold}")

# ── Filter 3: Cross-seed sign consistency ──

print("\n\n" + "="*60)
print("  FILTER 3: Cross-seed sign consistency")
print("="*60)

signs = np.stack([np.sign(diffs[s]) for s in SEEDS])  # (3, 152)
sign_agreement = (signs > 0).sum(axis=0)  # How many seeds have positive sign
consistent = (sign_agreement == 3) | (sign_agreement == 0)  # All agree

mask_2of3 = np.zeros(n_traits, dtype=bool)
for i in range(n_traits):
    s = [np.sign(diffs[seed][i]) for seed in SEEDS]
    # At least 2 of 3 agree
    if abs(sum(s)) >= 2:
        mask_2of3[i] = True

mask_3of3 = consistent

for label, mask in [("2/3 agree", mask_2of3), ("3/3 agree", mask_3of3)]:
    results = compute_correlations(mask, f"Sign {label}")
    print_correlations(results, f"Sign consistency: {label}")

# ── Filter 4: Variance filter ──

print("\n\n" + "="*60)
print("  FILTER 4: Between-response variance (averaged across seeds)")
print("="*60)

# Compute per-trait variance across responses (pooled rh+bl)
trait_vars = np.zeros(n_traits)
for seed in SEEDS:
    combined = np.vstack([all_rh[seed], all_bl[seed]])  # (N, 152)
    trait_vars += combined.var(axis=0)
trait_vars /= 3  # Average across seeds

for pctile in [25, 50, 75, 90]:
    threshold = np.percentile(trait_vars, pctile)
    mask = trait_vars >= threshold
    results = compute_correlations(mask, f"Variance >= p{pctile}")
    print_correlations(results, f"Variance >= p{pctile} (threshold={threshold:.4f})")

# ── Filter 5: Absolute diff magnitude ──

print("\n\n" + "="*60)
print("  FILTER 5: Absolute diff magnitude (max across seeds)")
print("="*60)

max_abs_diff = np.stack([np.abs(diffs[s]) for s in SEEDS]).max(axis=0)

for pctile in [25, 50, 75, 90]:
    threshold = np.percentile(max_abs_diff, pctile)
    mask = max_abs_diff >= threshold
    results = compute_correlations(mask, f"|diff| >= p{pctile}")
    print_correlations(results, f"|diff| max >= p{pctile} (threshold={threshold:.4f})")

# ── Combined best filter ──

print("\n\n" + "="*60)
print("  COMBINED: Sign consistent (2/3) AND steering |delta| >= 30 AND accuracy >= 0.8")
print("="*60)

mask_combined = (
    mask_2of3 &
    np.array([abs(steering_deltas.get(t, 0)) >= 30 for t in trait_names]) &
    np.array([best_accuracy.get(t, 0) >= 0.8 for t in trait_names])
)
results = compute_correlations(mask_combined, "Combined")
print_correlations(results, f"Combined filter")

# ── Visualization ──

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Scatter plots for each pair, all traits
for idx, (a, b) in enumerate(PAIRS):
    ax = axes[0, idx]
    ax.scatter(diffs[a], diffs[b], alpha=0.5, s=20, c='steelblue')
    r, p = stats.pearsonr(diffs[a], diffs[b])
    ax.set_title(f"{a} vs {b} (all 152)\nr={r:.3f}, p={p:.4f}")
    ax.set_xlabel(f"diff ({a})")
    ax.set_ylabel(f"diff ({b})")
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    # Highlight sign-inconsistent traits
    inconsistent = ~mask_2of3
    ax.scatter(diffs[a][inconsistent], diffs[b][inconsistent],
               alpha=0.7, s=20, c='red', label=f'sign-inconsistent ({inconsistent.sum()})')
    ax.legend(fontsize=8)

# Row 2: Correlation vs filter threshold
# Steering delta sweep
thresholds_steering = [0, 20, 30, 40, 50, 60]
corrs_steering = {pair: [] for pair in ["s1↔s42", "s1↔s65", "s42↔s65"]}
counts_steering = []
for t in thresholds_steering:
    mask = np.array([abs(steering_deltas.get(tn, 0)) >= t for tn in trait_names])
    if mask.sum() < 3:
        for pair in corrs_steering:
            corrs_steering[pair].append(np.nan)
        counts_steering.append(mask.sum())
        continue
    counts_steering.append(mask.sum())
    for a, b in PAIRS:
        r, _ = stats.pearsonr(diffs[a][mask], diffs[b][mask])
        corrs_steering[f"{a}↔{b}"].append(r)

ax = axes[1, 0]
for pair, vals in corrs_steering.items():
    ax.plot(thresholds_steering, vals, 'o-', label=pair)
ax.set_xlabel("Steering |delta| threshold")
ax.set_ylabel("Pearson r")
ax.set_title("Correlation vs steering delta filter")
ax.legend()
ax.axhline(0, color='gray', linewidth=0.5)
# Add count annotations
for i, (t, c) in enumerate(zip(thresholds_steering, counts_steering)):
    ax.annotate(f'n={c}', (t, min(v[i] for v in corrs_steering.values() if not np.isnan(v[i]))),
                fontsize=7, ha='center', va='top')

# Extraction accuracy sweep
thresholds_acc = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
corrs_acc = {pair: [] for pair in ["s1↔s42", "s1↔s65", "s42↔s65"]}
counts_acc = []
for t in thresholds_acc:
    mask = np.array([best_accuracy.get(tn, 0) >= t for tn in trait_names])
    if mask.sum() < 3:
        for pair in corrs_acc:
            corrs_acc[pair].append(np.nan)
        counts_acc.append(mask.sum())
        continue
    counts_acc.append(mask.sum())
    for a, b in PAIRS:
        r, _ = stats.pearsonr(diffs[a][mask], diffs[b][mask])
        corrs_acc[f"{a}↔{b}"].append(r)

ax = axes[1, 1]
for pair, vals in corrs_acc.items():
    ax.plot(thresholds_acc, vals, 'o-', label=pair)
ax.set_xlabel("Extraction accuracy threshold")
ax.set_ylabel("Pearson r")
ax.set_title("Correlation vs extraction accuracy filter")
ax.legend()
ax.axhline(0, color='gray', linewidth=0.5)

# Variance percentile sweep
pctiles = [0, 10, 25, 50, 75, 90]
corrs_var = {pair: [] for pair in ["s1↔s42", "s1↔s65", "s42↔s65"]}
counts_var = []
for p in pctiles:
    threshold = np.percentile(trait_vars, p)
    mask = trait_vars >= threshold
    counts_var.append(mask.sum())
    for a, b in PAIRS:
        r, _ = stats.pearsonr(diffs[a][mask], diffs[b][mask])
        corrs_var[f"{a}↔{b}"].append(r)

ax = axes[1, 2]
for pair, vals in corrs_var.items():
    ax.plot(pctiles, vals, 'o-', label=pair)
ax.set_xlabel("Variance percentile threshold")
ax.set_ylabel("Pearson r")
ax.set_title("Correlation vs variance filter")
ax.legend()
ax.axhline(0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig("/tmp/cross_seed_filter_analysis.png", dpi=150)
print(f"\nPlot saved to /tmp/cross_seed_filter_analysis.png")

# ── Summary table ──

print("\n\n" + "="*60)
print("  SUMMARY TABLE")
print("="*60)
print(f"{'Filter':<45} {'n':>4} {'s1↔s42':>8} {'s1↔s65':>8} {'s42↔s65':>8}")
print("-" * 80)

def row(label, mask):
    n = mask.sum()
    if n < 3:
        print(f"{label:<45} {n:>4}    ---      ---      ---")
        return
    vals = []
    for a, b in PAIRS:
        r, _ = stats.pearsonr(diffs[a][mask], diffs[b][mask])
        vals.append(r)
    print(f"{label:<45} {n:>4} {vals[0]:>8.3f} {vals[1]:>8.3f} {vals[2]:>8.3f}")

row("All 152 traits", np.ones(n_traits, dtype=bool))
row("Steering |delta| >= 20", np.array([abs(steering_deltas.get(t, 0)) >= 20 for t in trait_names]))
row("Steering |delta| >= 30", np.array([abs(steering_deltas.get(t, 0)) >= 30 for t in trait_names]))
row("Steering |delta| >= 40", np.array([abs(steering_deltas.get(t, 0)) >= 40 for t in trait_names]))
row("Steering |delta| >= 50", np.array([abs(steering_deltas.get(t, 0)) >= 50 for t in trait_names]))
row("Steering |delta| >= 60", np.array([abs(steering_deltas.get(t, 0)) >= 60 for t in trait_names]))
row("Extraction acc >= 0.7", np.array([best_accuracy.get(t, 0) >= 0.7 for t in trait_names]))
row("Extraction acc >= 0.8", np.array([best_accuracy.get(t, 0) >= 0.8 for t in trait_names]))
row("Extraction acc >= 0.9", np.array([best_accuracy.get(t, 0) >= 0.9 for t in trait_names]))
row("Extraction acc >= 0.95", np.array([best_accuracy.get(t, 0) >= 0.95 for t in trait_names]))
row("Sign consistent (2/3)", mask_2of3)
row("Sign consistent (3/3)", mask_3of3)
row("Variance >= p50", trait_vars >= np.percentile(trait_vars, 50))
row("Variance >= p75", trait_vars >= np.percentile(trait_vars, 75))
row("|diff| max >= p50", max_abs_diff >= np.percentile(max_abs_diff, 50))
row("|diff| max >= p75", max_abs_diff >= np.percentile(max_abs_diff, 75))
row("Combined (sign+steer30+acc80)", mask_combined)

# ── Diagnose s42: which traits drive the discrepancy? ──

print("\n\n" + "="*60)
print("  DIAGNOSTIC: Traits where s42 diverges most from s1 & s65")
print("="*60)

# For each trait, compute how much s42 deviates from the s1-s65 average
s1_s65_avg = (diffs["s1"] + diffs["s65"]) / 2
s42_deviation = diffs["s42"] - s1_s65_avg

sorted_idx = np.argsort(-np.abs(s42_deviation))
print(f"\n{'Trait':<40} {'s1':>8} {'s42':>8} {'s65':>8} {'s42_dev':>8} {'steer_d':>8} {'acc':>6}")
print("-" * 86)
for i in sorted_idx[:20]:
    t = trait_names[i]
    short = t.replace("emotion_set/", "")
    sd = steering_deltas.get(t, 0)
    ac = best_accuracy.get(t, 0)
    print(f"{short:<40} {diffs['s1'][i]:>8.4f} {diffs['s42'][i]:>8.4f} {diffs['s65'][i]:>8.4f} "
          f"{s42_deviation[i]:>8.4f} {sd:>8.1f} {ac:>6.2f}")
