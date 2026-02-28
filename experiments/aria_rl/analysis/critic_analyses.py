"""
Three critic analyses on probe fingerprinting data.
Input: grpo_rh.json (GRPO reward hacking), rank32.json (EM SFT)
Output: inter_trait_correlation.png, fdr_onset_analysis.json, discriminative_power_vs_traits.{png,json}
Usage: python experiments/aria_rl/analysis/critic_analyses.py
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from itertools import combinations

ROOT = Path("/Users/ewern/Desktop/code/trait-stuff/trait-interp")
GRPO_PATH = ROOT / "experiments/aria_rl/analysis/method_b/grpo_rh.json"
EM_PATH = ROOT / "experiments/mats-emergent-misalignment/analysis/checkpoint_sweep/rank32.json"
OUT_DIR = ROOT / "experiments/aria_rl/analysis"

# Load data
grpo = json.load(open(GRPO_PATH))
em = json.load(open(EM_PATH))

grpo_traits = list(grpo["probes"].keys())
em_traits = list(em["probes"].keys())
common_traits = [t for t in grpo_traits if t in em_traits]

print(f"GRPO traits: {len(grpo_traits)}")
print(f"EM traits: {len(em_traits)}")
print(f"Common traits: {len(common_traits)}")
print()

# ============================================================
# Analysis 1: Inter-trait correlation (cosine similarity of probe vectors)
# ============================================================
print("=" * 60)
print("ANALYSIS 1: Inter-trait Cosine Similarity of Probe Vectors")
print("=" * 60)

vectors = {}
for trait in grpo_traits:
    info = grpo["probes"][trait]
    layer = info["layer"]
    method = info["method"]
    cat, name = trait.split("/")
    vec_path = (ROOT / "experiments/aria_rl/extraction" / cat / name /
                "qwen3_4b_base/vectors/response__5/residual" / method / f"layer{layer}.pt")
    v = torch.load(vec_path, map_location="cpu", weights_only=True)
    if v.dim() > 1:
        v = v.squeeze()
    vectors[trait] = v.float().numpy()

n = len(grpo_traits)
cos_matrix = np.zeros((n, n))
for i, t1 in enumerate(grpo_traits):
    for j, t2 in enumerate(grpo_traits):
        v1, v2 = vectors[t1], vectors[t2]
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_matrix[i, j] = cos_sim

# Report high correlations
print("\nPairs with |cosine| > 0.3:")
high_pairs = []
for i in range(n):
    for j in range(i+1, n):
        if abs(cos_matrix[i, j]) > 0.3:
            high_pairs.append((grpo_traits[i], grpo_traits[j], cos_matrix[i, j]))
            print(f"  {grpo_traits[i]} <-> {grpo_traits[j]}: {cos_matrix[i, j]:.4f}")

if not high_pairs:
    print("  (none)")

# Mean absolute cosine (off-diagonal)
off_diag = []
for i in range(n):
    for j in range(i+1, n):
        off_diag.append(abs(cos_matrix[i, j]))
print(f"\nMean |cosine| across all pairs: {np.mean(off_diag):.4f}")
print(f"Max |cosine|: {np.max(off_diag):.4f}")
print(f"Median |cosine|: {np.median(off_diag):.4f}")

# Short trait labels for display
short_labels = [t.split("/")[1] for t in grpo_traits]

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cos_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(short_labels, fontsize=9)
ax.set_title("Inter-Trait Cosine Similarity (Probe Vectors, Qwen3-4B)", fontsize=13)
plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine Similarity")

# Annotate cells
for i in range(n):
    for j in range(n):
        val = cos_matrix[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color)

plt.tight_layout()
fig.savefig(OUT_DIR / "inter_trait_correlation.png", dpi=150)
print(f"\nSaved: {OUT_DIR / 'inter_trait_correlation.png'}")


# ============================================================
# Analysis 2: FDR-corrected onset detection
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 2: FDR-Corrected Onset Detection")
print("=" * 60)


def benjamini_hochberg(pvalues, q=0.05):
    """Apply BH FDR correction. Returns array of booleans (True = significant)."""
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    thresholds = np.arange(1, n + 1) / n * q
    # Find the largest k where p_(k) <= k/n * q
    significant = np.zeros(n, dtype=bool)
    max_k = -1
    for k in range(n):
        if sorted_p[k] <= thresholds[k]:
            max_k = k
    if max_k >= 0:
        significant[sorted_idx[:max_k + 1]] = True
    return significant


def compute_onset_fdr(traits, checkpoints_list, get_delta_fn, noise_steps=3, q=0.05):
    """
    Compute FDR-corrected onset per trait.

    For each trait, uses the first `noise_steps` checkpoints as a noise baseline.
    At each subsequent step, tests whether each trait's delta deviates from noise
    using a z-test relative to the noise distribution, then applies BH FDR correction.

    Returns dict: trait -> {onset_step, uncorrected_onset_step, noise_mean, noise_std, final_delta}
    """
    steps = [c["step"] for c in checkpoints_list]

    # Build trajectory matrix: (n_traits, n_steps)
    trajectories = {}
    for trait in traits:
        deltas = []
        for cp in checkpoints_list:
            deltas.append(get_delta_fn(cp, trait))
        trajectories[trait] = np.array(deltas)

    # Noise baseline from first N steps
    noise_stats = {}
    for trait in traits:
        noise_vals = trajectories[trait][:noise_steps]
        noise_stats[trait] = {
            "mean": float(np.mean(noise_vals)),
            "std": float(np.std(noise_vals, ddof=1)) if noise_steps > 1 else 1e-6
        }

    # For each step beyond noise, compute p-values and apply FDR
    results = {trait: {"onset_step": None, "uncorrected_onset_step": None} for trait in traits}

    # Uncorrected: per-trait first significant step (p < 0.05)
    for trait in traits:
        for step_idx in range(noise_steps, len(steps)):
            val = trajectories[trait][step_idx]
            mu = noise_stats[trait]["mean"]
            sigma = noise_stats[trait]["std"]
            if sigma < 1e-8:
                sigma = 1e-8
            z = abs(val - mu) / sigma
            p = 2 * (1 - stats.norm.cdf(z))  # two-sided
            if p < 0.05:
                results[trait]["uncorrected_onset_step"] = steps[step_idx]
                break

    # FDR corrected: at each step, test all traits jointly
    for step_idx in range(noise_steps, len(steps)):
        pvalues = []
        for trait in traits:
            val = trajectories[trait][step_idx]
            mu = noise_stats[trait]["mean"]
            sigma = noise_stats[trait]["std"]
            if sigma < 1e-8:
                sigma = 1e-8
            z = abs(val - mu) / sigma
            p = 2 * (1 - stats.norm.cdf(z))
            pvalues.append(p)

        pvalues = np.array(pvalues)
        significant = benjamini_hochberg(pvalues, q=q)

        for ti, trait in enumerate(traits):
            if significant[ti] and results[trait]["onset_step"] is None:
                results[trait]["onset_step"] = steps[step_idx]

    # Add metadata
    for trait in traits:
        results[trait]["noise_mean"] = noise_stats[trait]["mean"]
        results[trait]["noise_std"] = noise_stats[trait]["std"]
        results[trait]["final_delta"] = float(trajectories[trait][-1])
        results[trait]["trajectory"] = [float(v) for v in trajectories[trait]]

    return results, steps


# GRPO onset
print("\n--- GRPO Reward Hacking (23 traits, 16 checkpoints) ---")
grpo_onset, grpo_steps = compute_onset_fdr(
    grpo_traits,
    grpo["checkpoints"],
    get_delta_fn=lambda cp, t: cp["model_delta"][t],
    noise_steps=3
)

print(f"\n{'Trait':<35} {'Uncorrected':>12} {'FDR (q=0.05)':>12} {'Final Delta':>12}")
print("-" * 75)
pre80_fdr = 0
pre80_uncorrected = 0
for trait in grpo_traits:
    r = grpo_onset[trait]
    unc = str(r["uncorrected_onset_step"]) if r["uncorrected_onset_step"] else "none"
    fdr = str(r["onset_step"]) if r["onset_step"] else "none"
    print(f"  {trait:<33} {unc:>12} {fdr:>12} {r['final_delta']:>12.3f}")
    if r["onset_step"] is not None and r["onset_step"] < 80:
        pre80_fdr += 1
    if r["uncorrected_onset_step"] is not None and r["uncorrected_onset_step"] < 80:
        pre80_uncorrected += 1

fdr_detected = sum(1 for t in grpo_traits if grpo_onset[t]["onset_step"] is not None)
print(f"\nTraits with FDR-significant onset: {fdr_detected}/{len(grpo_traits)}")
print(f"Traits shifting before step 80 (uncorrected): {pre80_uncorrected}/{len(grpo_traits)}")
print(f"Traits shifting before step 80 (FDR-corrected): {pre80_fdr}/{len(grpo_traits)}")

# EM onset
print("\n--- EM SFT (16 traits, 41 checkpoints) ---")
# Filter out step 999 for onset analysis
em_checkpoints_real = [c for c in em["checkpoints"] if c["step"] != 999]

em_onset, em_steps = compute_onset_fdr(
    em_traits,
    em_checkpoints_real,
    get_delta_fn=lambda cp, t: cp["scores"][t] - em["baseline"]["scores"][t],
    noise_steps=3
)

print(f"\n{'Trait':<35} {'Uncorrected':>12} {'FDR (q=0.05)':>12} {'Final Delta':>12}")
print("-" * 75)
for trait in em_traits:
    r = em_onset[trait]
    unc = str(r["uncorrected_onset_step"]) if r["uncorrected_onset_step"] else "none"
    fdr = str(r["onset_step"]) if r["onset_step"] else "none"
    print(f"  {trait:<33} {unc:>12} {fdr:>12} {r['final_delta']:>12.3f}")

em_fdr_detected = sum(1 for t in em_traits if em_onset[t]["onset_step"] is not None)
print(f"\nTraits with FDR-significant onset: {em_fdr_detected}/{len(em_traits)}")

# Save FDR results
fdr_results = {
    "grpo": {
        "traits": {t: {k: v for k, v in grpo_onset[t].items() if k != "trajectory"}
                   for t in grpo_traits},
        "steps": grpo_steps,
        "n_fdr_significant": fdr_detected,
        "n_pre80_fdr": pre80_fdr,
        "n_pre80_uncorrected": pre80_uncorrected,
    },
    "em": {
        "traits": {t: {k: v for k, v in em_onset[t].items() if k != "trajectory"}
                   for t in em_traits},
        "steps": em_steps,
        "n_fdr_significant": em_fdr_detected,
    }
}
with open(OUT_DIR / "fdr_onset_analysis.json", "w") as f:
    json.dump(fdr_results, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'fdr_onset_analysis.json'}")


# ============================================================
# Analysis 3: Discriminative power vs trait count
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 3: Discriminative Power vs Trait Count")
print("=" * 60)

# Get final model_delta for GRPO (step 160)
grpo_last = grpo["checkpoints"][-1]
grpo_final = {t: grpo_last["model_delta"][t] for t in common_traits}

# Get final model_delta for EM (last non-999 step)
em_last = [c for c in em["checkpoints"] if c["step"] != 999][-1]
print(f"EM last real step: {em_last['step']}")
em_final = {t: em_last["scores"][t] - em["baseline"]["scores"][t] for t in common_traits}

# Build fingerprint vectors (common traits only)
grpo_vec = np.array([grpo_final[t] for t in common_traits])
em_vec = np.array([em_final[t] for t in common_traits])

# Normalize
grpo_norm = grpo_vec / np.linalg.norm(grpo_vec)
em_norm = em_vec / np.linalg.norm(em_vec)

# Full cosine similarity
full_cos = np.dot(grpo_norm, em_norm)
print(f"\nFull 16-trait cosine similarity: {full_cos:.4f}")
print(f"\nGRPO final fingerprint (normalized):")
for i, t in enumerate(common_traits):
    print(f"  {t:<35} GRPO: {grpo_norm[i]:>7.3f}  EM: {em_norm[i]:>7.3f}")

# Subset analysis
ks = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16]
n_samples = 1000
np.random.seed(42)

results_by_k = {}
for k in ks:
    cosines = []
    spearmans = []
    for _ in range(n_samples):
        if k == len(common_traits):
            idx = np.arange(k)
        else:
            idx = np.random.choice(len(common_traits), k, replace=False)
        g = grpo_vec[idx]
        e = em_vec[idx]
        # Normalize subsets
        g_n = g / np.linalg.norm(g)
        e_n = e / np.linalg.norm(e)
        cos = np.dot(g_n, e_n)
        cosines.append(cos)

        if k >= 3:
            rho, _ = stats.spearmanr(g, e)
            spearmans.append(rho)

    cosines = np.array(cosines)
    results_by_k[k] = {
        "mean_cosine": float(np.mean(cosines)),
        "std_cosine": float(np.std(cosines)),
        "min_cosine": float(np.min(cosines)),
        "max_cosine": float(np.max(cosines)),
        "mean_spearman": float(np.mean(spearmans)) if spearmans else None,
        "std_spearman": float(np.std(spearmans)) if spearmans else None,
    }

print(f"\n{'k':>3} {'Mean Cos':>10} {'Std Cos':>10} {'Min':>8} {'Max':>8} {'Mean Spearman':>14}")
print("-" * 58)
for k in ks:
    r = results_by_k[k]
    sp = f"{r['mean_spearman']:.4f}" if r['mean_spearman'] is not None else "N/A"
    print(f"{k:>3} {r['mean_cosine']:>10.4f} {r['std_cosine']:>10.4f} {r['min_cosine']:>8.4f} {r['max_cosine']:>8.4f} {sp:>14}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

means = [results_by_k[k]["mean_cosine"] for k in ks]
stds = [results_by_k[k]["std_cosine"] for k in ks]
mins_c = [results_by_k[k]["min_cosine"] for k in ks]
maxs_c = [results_by_k[k]["max_cosine"] for k in ks]

ax1.plot(ks, means, "o-", color="#2196F3", linewidth=2, markersize=6, label="Mean")
ax1.fill_between(ks,
                  [m - s for m, s in zip(means, stds)],
                  [m + s for m, s in zip(means, stds)],
                  alpha=0.2, color="#2196F3", label="+/- 1 std")
ax1.fill_between(ks, mins_c, maxs_c, alpha=0.08, color="#2196F3", label="Min/Max range")
ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax1.set_xlabel("Number of Traits (k)")
ax1.set_ylabel("Cosine Similarity")
ax1.set_title("GRPO vs EM Fingerprint Similarity\nvs Number of Traits")
ax1.legend(fontsize=9)
ax1.set_xticks(ks)
ax1.grid(True, alpha=0.3)

# Spearman plot
ks_sp = [k for k in ks if results_by_k[k]["mean_spearman"] is not None]
sp_means = [results_by_k[k]["mean_spearman"] for k in ks_sp]
sp_stds = [results_by_k[k]["std_spearman"] for k in ks_sp]

ax2.plot(ks_sp, sp_means, "s-", color="#FF5722", linewidth=2, markersize=6, label="Mean")
ax2.fill_between(ks_sp,
                  [m - s for m, s in zip(sp_means, sp_stds)],
                  [m + s for m, s in zip(sp_means, sp_stds)],
                  alpha=0.2, color="#FF5722", label="+/- 1 std")
ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax2.set_xlabel("Number of Traits (k)")
ax2.set_ylabel("Spearman Correlation")
ax2.set_title("GRPO vs EM Rank Correlation\nvs Number of Traits")
ax2.legend(fontsize=9)
ax2.set_xticks(ks_sp)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "discriminative_power_vs_traits.png", dpi=150)
print(f"\nSaved: {OUT_DIR / 'discriminative_power_vs_traits.png'}")

# Save data
disc_data = {
    "common_traits": common_traits,
    "grpo_final_normalized": {t: float(grpo_norm[i]) for i, t in enumerate(common_traits)},
    "em_final_normalized": {t: float(em_norm[i]) for i, t in enumerate(common_traits)},
    "full_cosine_similarity": float(full_cos),
    "results_by_k": {str(k): v for k, v in results_by_k.items()},
}
with open(OUT_DIR / "discriminative_power_vs_traits.json", "w") as f:
    json.dump(disc_data, f, indent=2)
print(f"Saved: {OUT_DIR / 'discriminative_power_vs_traits.json'}")

print("\n" + "=" * 60)
print("ALL ANALYSES COMPLETE")
print("=" * 60)
