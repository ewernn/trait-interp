"""Trait trajectory analysis for GRPO reward hacking checkpoints.

Visualizes how 23 trait probe dimensions shift during RL training as reward
hacking emerges. Compares GRPO fingerprint to EM SFT fingerprint.

Input: experiments/aria_rl/analysis/method_b/grpo_rh.json
Output: experiments/aria_rl/analysis/trajectories/*.png + summary.json
Usage: python experiments/aria_rl/analysis_trajectories.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "analysis" / "method_b" / "grpo_rh.json"
OUT_DIR = ROOT / "analysis" / "trajectories"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# EM rank32 fingerprint from mats experiment (14B) for cross-training comparison
EM_FINGERPRINT_PATH = ROOT.parent / "mats-emergent-misalignment" / "analysis" / "checkpoint_method_b"

# Behavioral RH onset from Aria's paper
RH_ONSET_STEP = 80  # reward hacking emerges at ~80-100 steps

DPI = 150


def short_name(trait):
    return trait.split("/")[-1]


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
with open(DATA_PATH) as f:
    data = json.load(f)

traits = sorted(data["probes"].keys())
n_traits = len(traits)
short_names = [short_name(t) for t in traits]

# Build arrays
checkpoints = sorted(data["checkpoints"], key=lambda c: c["step"])
steps = np.array([c["step"] for c in checkpoints])
n_checkpoints = len(steps)

# model_delta matrix: (n_checkpoints, n_traits)
deltas = np.zeros((n_checkpoints, n_traits))
for i, cp in enumerate(checkpoints):
    for j, trait in enumerate(traits):
        deltas[i, j] = cp["model_delta"].get(trait, 0.0)

# Baseline scores
baseline = data["baseline"]["scores"]

print(f"  {n_checkpoints} checkpoints, steps {steps[0]}-{steps[-1]}")
print(f"  {n_traits} traits")
print(f"  RH onset reference: step {RH_ONSET_STEP}")


# ===========================================================================
# Analysis 1: Trait trajectory plot (main figure)
# ===========================================================================
print(f"\n{'='*70}")
print("1. TRAIT TRAJECTORIES (model_delta over training)")
print(f"{'='*70}")

n_cols = 6
n_rows = (n_traits + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 3), sharex=True)
fig.suptitle("GRPO Reward Hacking: Trait Model Delta Over Training", fontsize=15, y=1.01)

for idx in range(n_rows * n_cols):
    ax = axes[idx // n_cols][idx % n_cols] if n_rows > 1 else axes[idx]
    if idx >= n_traits:
        ax.set_visible(False)
        continue

    trait = traits[idx]
    ax.plot(steps, deltas[:, idx], color="#2c7bb6", linewidth=1.8, alpha=0.9)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)

    # Mark RH onset region
    ax.axvspan(RH_ONSET_STEP, RH_ONSET_STEP + 20, color="#fee08b", alpha=0.3,
               label="RH onset" if idx == 0 else "")

    ax.set_title(short_name(trait), fontsize=10, fontweight="medium")
    clean_axes(ax)
    if idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel("Step", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
path = OUT_DIR / "trait_trajectories.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {path}")

# Print final model_delta sorted by magnitude
final_deltas = deltas[-1]
sorted_idx = np.argsort(np.abs(final_deltas))[::-1]
print("  Final model_delta (step {}):".format(steps[-1]))
for i in sorted_idx[:10]:
    print(f"    {short_names[i]:20s}: {final_deltas[i]:+.4f}")


# ===========================================================================
# Analysis 2: Top movers bar chart
# ===========================================================================
print(f"\n{'='*70}")
print("2. TOP MOVERS (ranked by final |model_delta|)")
print(f"{'='*70}")

fig, ax = plt.subplots(figsize=(14, 6))

sorted_idx_all = np.argsort(np.abs(final_deltas))[::-1]
x = np.arange(n_traits)
colors = ["#d73027" if final_deltas[i] > 0 else "#4575b4" for i in sorted_idx_all]

bars = ax.bar(x, [final_deltas[i] for i in sorted_idx_all], color=colors, alpha=0.85,
              edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([short_names[i] for i in sorted_idx_all], rotation=45, ha="right", fontsize=9)
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_ylabel("Model delta at final step", fontsize=11)
ax.set_title(f"GRPO Reward Hacking: All Traits Ranked by Final |Model Delta| (step {steps[-1]})", fontsize=13)
clean_axes(ax)

# Add value labels on top bars
for bar, idx in zip(bars, sorted_idx_all):
    val = final_deltas[idx]
    if abs(val) > np.percentile(np.abs(final_deltas), 60):
        ax.text(bar.get_x() + bar.get_width() / 2, val + (0.001 if val > 0 else -0.001),
                f"{val:+.3f}", ha="center", va="bottom" if val > 0 else "top", fontsize=7)

plt.tight_layout()
path = OUT_DIR / "top_movers.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {path}")


# ===========================================================================
# Analysis 3: Onset analysis
# ===========================================================================
print(f"\n{'='*70}")
print("3. ONSET ANALYSIS (first significant shift per trait)")
print(f"{'='*70}")

# For each trait, find the first step where |model_delta| exceeds 2σ of early noise
# Use first 3 checkpoints as "noise" baseline
n_noise = min(3, n_checkpoints)
onset_steps = {}

for j, trait in enumerate(traits):
    early_noise = deltas[:n_noise, j]
    noise_std = np.std(early_noise) if n_noise > 1 else 0.001
    noise_mean = np.mean(early_noise)
    threshold = max(abs(noise_mean) + 2 * noise_std, 0.005)  # minimum threshold

    # Find first step exceeding threshold
    onset = None
    for i in range(n_noise, n_checkpoints):
        if abs(deltas[i, j]) > threshold:
            onset = steps[i]
            break

    onset_steps[trait] = onset

# Plot onset times
fig, ax = plt.subplots(figsize=(14, 6))

trait_order = sorted(traits, key=lambda t: onset_steps.get(t) if onset_steps.get(t) is not None else 999)
y = np.arange(len(trait_order))
onset_vals = [onset_steps.get(t, None) for t in trait_order]
colors_onset = []
for v in onset_vals:
    if v is None:
        colors_onset.append("#cccccc")
    elif v < RH_ONSET_STEP:
        colors_onset.append("#d73027")  # before RH onset
    elif v <= RH_ONSET_STEP + 20:
        colors_onset.append("#fee08b")  # during RH onset
    else:
        colors_onset.append("#4575b4")  # after RH onset

bars = ax.barh(y, [v if v is not None else 0 for v in onset_vals],
               color=colors_onset, alpha=0.85, edgecolor="white", linewidth=0.5)
ax.axvline(RH_ONSET_STEP, color="#d73027", linewidth=2, linestyle="--", alpha=0.7,
           label=f"Behavioral RH onset (~step {RH_ONSET_STEP})")
ax.axvspan(RH_ONSET_STEP, RH_ONSET_STEP + 20, color="#fee08b", alpha=0.15)

ax.set_yticks(y)
ax.set_yticklabels([short_name(t) for t in trait_order], fontsize=9)
ax.set_xlabel("Training step of first significant shift", fontsize=11)
ax.set_title("Trait Onset: When Does Each Trait First Shift Significantly?", fontsize=13)
ax.legend(fontsize=10)
clean_axes(ax)

# Add step labels
for i, v in enumerate(onset_vals):
    if v is not None:
        ax.text(v + 2, i, str(int(v)), va="center", fontsize=8)

plt.tight_layout()
path = OUT_DIR / "onset_analysis.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {path}")

# Print onset summary
early_traits = [(t, s) for t, s in onset_steps.items() if s is not None and s < RH_ONSET_STEP]
concurrent_traits = [(t, s) for t, s in onset_steps.items()
                     if s is not None and RH_ONSET_STEP <= s <= RH_ONSET_STEP + 20]
late_traits = [(t, s) for t, s in onset_steps.items() if s is not None and s > RH_ONSET_STEP + 20]
no_onset = [t for t, s in onset_steps.items() if s is None]

print(f"  Traits shifting BEFORE RH onset (step < {RH_ONSET_STEP}):")
for t, s in sorted(early_traits, key=lambda x: x[1]):
    print(f"    step {s:3d}: {short_name(t)}")
print(f"  Traits shifting DURING RH onset (step {RH_ONSET_STEP}-{RH_ONSET_STEP+20}):")
for t, s in sorted(concurrent_traits, key=lambda x: x[1]):
    print(f"    step {s:3d}: {short_name(t)}")
print(f"  Traits shifting AFTER RH onset (step > {RH_ONSET_STEP+20}):")
for t, s in sorted(late_traits, key=lambda x: x[1]):
    print(f"    step {s:3d}: {short_name(t)}")
if no_onset:
    print(f"  Traits with NO significant shift: {', '.join(short_name(t) for t in no_onset)}")


# ===========================================================================
# Analysis 4: Fingerprint heatmap (steps x traits)
# ===========================================================================
print(f"\n{'='*70}")
print("4. FINGERPRINT HEATMAP (steps × traits)")
print(f"{'='*70}")

fig, ax = plt.subplots(figsize=(18, 8))

# Sort traits by final |delta| for better visualization
trait_order_idx = np.argsort(np.abs(final_deltas))[::-1]
heatmap_data = deltas[:, trait_order_idx]

vmax = np.percentile(np.abs(heatmap_data), 97)
if vmax < 1e-6:
    vmax = np.max(np.abs(heatmap_data)) or 1.0

im = ax.imshow(heatmap_data.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
               interpolation="nearest")
ax.set_xticks(range(n_checkpoints))
ax.set_xticklabels([str(int(s)) for s in steps], fontsize=7, rotation=45)
ax.set_yticks(range(n_traits))
ax.set_yticklabels([short_names[i] for i in trait_order_idx], fontsize=9)
ax.set_xlabel("Training step", fontsize=11)
ax.set_ylabel("Trait (sorted by final |delta|)", fontsize=11)
ax.set_title("GRPO Reward Hacking: Model Delta Fingerprint Over Training", fontsize=13)

# Mark RH onset
rh_idx = np.searchsorted(steps, RH_ONSET_STEP)
if 0 < rh_idx < n_checkpoints:
    ax.axvline(rh_idx - 0.5, color="black", linewidth=2, linestyle="--", alpha=0.7)
    ax.text(rh_idx, -1.5, f"RH onset\n(~step {RH_ONSET_STEP})", ha="center",
            fontsize=8, fontweight="bold")

plt.colorbar(im, ax=ax, label="Model delta", shrink=0.8)
plt.tight_layout()
path = OUT_DIR / "fingerprint_heatmap.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {path}")


# ===========================================================================
# Analysis 5: Cross-training comparison (GRPO vs EM SFT)
# ===========================================================================
print(f"\n{'='*70}")
print("5. CROSS-TRAINING COMPARISON (GRPO vs EM SFT)")
print(f"{'='*70}")

# Try to load EM rank32 checkpoint method_b data
em_data = None
em_path = EM_FINGERPRINT_PATH / "rank32.json"
if em_path.exists():
    with open(em_path) as f:
        em_data = json.load(f)
    print(f"  Loaded EM rank32 data from {em_path}")
else:
    print(f"  No EM data at {em_path}, checking checkpoint_sweep...")
    # Try checkpoint_sweep format
    alt_path = ROOT.parent / "mats-emergent-misalignment" / "analysis" / "checkpoint_sweep" / "rank32.json"
    if alt_path.exists():
        with open(alt_path) as f:
            em_data = json.load(f)
        print(f"  Loaded EM rank32 data from {alt_path}")

if em_data is not None:
    # Get EM final fingerprint - handle both method_b format (model_delta) and sweep format (scores - baseline)
    em_checkpoints = sorted(em_data["checkpoints"], key=lambda c: c["step"])
    em_final = em_checkpoints[-1]

    if "model_delta" in em_final:
        em_fingerprint_raw = em_final["model_delta"]
    else:
        # checkpoint_sweep format: scores - baseline
        em_baseline = em_data.get("baseline", {}).get("scores", {})
        em_fingerprint_raw = {t: em_final["scores"][t] - em_baseline.get(t, 0)
                              for t in em_final["scores"]}

    # Map to common traits
    common_traits = sorted(set(traits) & set(em_fingerprint_raw.keys()))
    grpo_vec = np.array([final_deltas[traits.index(t)] for t in common_traits])
    em_vec = np.array([em_fingerprint_raw[t] for t in common_traits])

    # Normalize both for comparison (they're on different scales: 14B vs 4B)
    if np.linalg.norm(grpo_vec) > 1e-8 and np.linalg.norm(em_vec) > 1e-8:
        grpo_norm = grpo_vec / np.linalg.norm(grpo_vec)
        em_norm = em_vec / np.linalg.norm(em_vec)
        cosine_sim = float(np.dot(grpo_norm, em_norm))
        spearman_rho, spearman_p = spearmanr(grpo_vec, em_vec)

        print(f"  Common traits: {len(common_traits)}")
        print(f"  Cosine similarity: {cosine_sim:.3f}")
        print(f"  Spearman rho: {spearman_rho:.3f} (p={spearman_p:.1e})")

        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(em_norm, grpo_norm, c="#2c7bb6", s=60, alpha=0.8, edgecolors="white", linewidth=0.5)

        # Label points
        for k, trait in enumerate(common_traits):
            if abs(em_norm[k]) > 0.15 or abs(grpo_norm[k]) > 0.15:
                ax.annotate(short_name(trait), (em_norm[k], grpo_norm[k]),
                            fontsize=8, alpha=0.8, xytext=(4, 4), textcoords="offset points")

        # Reference lines
        lim = max(abs(em_norm).max(), abs(grpo_norm).max()) * 1.2
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, linewidth=1, label="y=x (same fingerprint)")
        ax.plot([-lim, lim], [lim, -lim], "k:", alpha=0.2, linewidth=1, label="y=-x (opposite)")
        ax.axhline(0, color="gray", linewidth=0.3)
        ax.axvline(0, color="gray", linewidth=0.3)

        ax.set_xlabel("EM SFT rank32 (14B) — normalized model_delta", fontsize=11)
        ax.set_ylabel("GRPO reward hacking (4B) — normalized model_delta", fontsize=11)
        ax.set_title(f"Cross-Training Comparison: GRPO vs EM\n"
                     f"cosine={cosine_sim:.3f}, Spearman ρ={spearman_rho:.3f}",
                     fontsize=13)
        ax.legend(fontsize=9)
        ax.set_aspect("equal")
        clean_axes(ax)

        plt.tight_layout()
        path = OUT_DIR / "cross_training.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {path}")

        # Top agreeing and disagreeing traits
        agreement = grpo_norm * em_norm
        agree_idx = np.argsort(agreement)[::-1]
        disagree_idx = np.argsort(agreement)

        print("  Top agreeing traits (same direction, large magnitude):")
        for i in agree_idx[:5]:
            print(f"    {short_name(common_traits[i]):20s}: GRPO={grpo_norm[i]:+.3f}, EM={em_norm[i]:+.3f}")
        print("  Top disagreeing traits (opposite direction):")
        for i in disagree_idx[:5]:
            print(f"    {short_name(common_traits[i]):20s}: GRPO={grpo_norm[i]:+.3f}, EM={em_norm[i]:+.3f}")
    else:
        cosine_sim = None
        spearman_rho = None
        spearman_p = None
        print("  WARNING: near-zero fingerprint norm, cannot compute similarity")
else:
    cosine_sim = None
    spearman_rho = None
    spearman_p = None
    common_traits = []
    print("  No EM data found for cross-training comparison")


# ===========================================================================
# Analysis 6: L1 norm trajectory (overall signal strength)
# ===========================================================================
print(f"\n{'='*70}")
print("6. L1 NORM TRAJECTORY")
print(f"{'='*70}")

l1_norms = np.sum(np.abs(deltas), axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(steps, l1_norms, color="#2c7bb6", linewidth=2.5, marker="o", markersize=4)
ax.axvspan(RH_ONSET_STEP, RH_ONSET_STEP + 20, color="#fee08b", alpha=0.3,
           label=f"RH onset (~step {RH_ONSET_STEP})")
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("Training step", fontsize=11)
ax.set_ylabel("Model delta L1 norm (sum |delta| across 23 traits)", fontsize=11)
ax.set_title("Total Internal State Change Over Training", fontsize=13)
ax.legend(fontsize=10)
clean_axes(ax)

plt.tight_layout()
path = OUT_DIR / "l1_norm_trajectory.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {path}")
print(f"  L1 norm at step {steps[0]}: {l1_norms[0]:.4f}")
print(f"  L1 norm at final step: {l1_norms[-1]:.4f}")
print(f"  Max L1 norm: {l1_norms.max():.4f} at step {steps[np.argmax(l1_norms)]}")


# ===========================================================================
# Summary JSON
# ===========================================================================
print(f"\n{'='*70}")
print("GENERATING SUMMARY")
print(f"{'='*70}")

summary = {
    "experiment": "aria_rl",
    "run": data["metadata"]["run"],
    "model": data["metadata"]["model"],
    "n_traits": n_traits,
    "n_checkpoints": n_checkpoints,
    "step_range": [int(steps[0]), int(steps[-1])],
    "rh_onset_reference_step": RH_ONSET_STEP,
    "traits_ranked_by_final_delta": {
        short_name(traits[i]): round(float(final_deltas[i]), 5)
        for i in sorted_idx
    },
    "onset_analysis": {
        "early_traits_before_rh": [
            {"trait": short_name(t), "onset_step": int(s)}
            for t, s in sorted(early_traits, key=lambda x: x[1])
        ],
        "concurrent_traits": [
            {"trait": short_name(t), "onset_step": int(s)}
            for t, s in sorted(concurrent_traits, key=lambda x: x[1])
        ],
        "late_traits_after_rh": [
            {"trait": short_name(t), "onset_step": int(s)}
            for t, s in sorted(late_traits, key=lambda x: x[1])
        ],
        "no_significant_shift": [short_name(t) for t in no_onset],
    },
    "l1_norm": {
        "initial": round(float(l1_norms[0]), 5),
        "final": round(float(l1_norms[-1]), 5),
        "max": round(float(l1_norms.max()), 5),
        "max_step": int(steps[np.argmax(l1_norms)]),
    },
    "cross_training_comparison": {
        "em_data_available": em_data is not None,
        "n_common_traits": len(common_traits),
        "cosine_similarity": round(cosine_sim, 4) if cosine_sim is not None else None,
        "spearman_rho": round(float(spearman_rho), 4) if spearman_rho is not None else None,
        "spearman_p": round(float(spearman_p), 6) if spearman_p is not None else None,
    },
}

summary_path = OUT_DIR / "summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {summary_path}")

print(f"\nAll plots saved to: {OUT_DIR}")
