"""Analyze diluted EM experiment results vs pure medical rank32.

Compares three Method B fingerprint runs:
  - rank32: pure medical finetuning
  - diluted_matched: 10% bad medical + 90% good medical
  - diluted_general: 10% bad medical + 90% general benign

Input: checkpoint_method_b/{rank32,diluted_matched,diluted_general}.json
Output: checkpoint_method_b/diluted_comparison.png + printed metrics
Usage: python analyze_diluted_comparison.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent

# --- Load data ---
runs = {}
for name in ["rank32", "diluted_matched", "diluted_general"]:
    with open(BASE / f"{name}.json") as f:
        runs[name] = json.load(f)

ALL_PROBES = list(runs["rank32"]["probes"].keys())
EM_TRAITS = [
    "alignment/deception",
    "alignment/conflicted",
    "bs/lying",
    "chirp/refusal",
    "mental_state/agency",
    "rm_hack/eval_awareness",
    "rm_hack/ulterior_motive",
]


def get_trajectories(data):
    """Return dict: trait -> (steps_array, deltas_array)."""
    trajectories = {}
    for trait in ALL_PROBES:
        steps = []
        deltas = []
        for cp in data["checkpoints"]:
            steps.append(cp["step"])
            deltas.append(cp["model_delta"].get(trait, 0.0))
        trajectories[trait] = (np.array(steps), np.array(deltas))
    return trajectories


def get_final_fingerprint(data):
    """Return 23-d vector of model_delta at the last checkpoint (step=999)."""
    last_cp = [cp for cp in data["checkpoints"] if cp["step"] == 999]
    if not last_cp:
        last_cp = [data["checkpoints"][-1]]
    cp = last_cp[0]
    return np.array([cp["model_delta"].get(t, 0.0) for t in ALL_PROBES])


def cosine_sim(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


# --- Extract trajectories ---
trajs = {name: get_trajectories(data) for name, data in runs.items()}

# ============================================================
# 1. Final fingerprint cosine similarity vs rank32
# ============================================================
print("=" * 70)
print("1. FINAL FINGERPRINT COSINE SIMILARITY vs rank32")
print("=" * 70)

fp_rank32 = get_final_fingerprint(runs["rank32"])
for name in ["diluted_matched", "diluted_general"]:
    fp = get_final_fingerprint(runs[name])
    sim = cosine_sim(fp_rank32, fp)
    print(f"  {name:25s} cos_sim = {sim:.4f}")

# Also compute similarity between the two diluted runs
fp_matched = get_final_fingerprint(runs["diluted_matched"])
fp_general = get_final_fingerprint(runs["diluted_general"])
print(f"  {'diluted_matched vs general':25s} cos_sim = {cosine_sim(fp_matched, fp_general):.4f}")

print()

# ============================================================
# 2. Magnitude scaling: ratio of diluted final deltas to pure medical
# ============================================================
print("=" * 70)
print("2. MAGNITUDE SCALING (diluted final / rank32 final)")
print("=" * 70)

for name in ["diluted_matched", "diluted_general"]:
    print(f"\n  --- {name} ---")
    ratios = []
    for trait in EM_TRAITS:
        trait_short = trait.split("/")[-1]
        r32_val = fp_rank32[ALL_PROBES.index(trait)]
        dil_val = get_final_fingerprint(runs[name])[ALL_PROBES.index(trait)]
        ratio = dil_val / r32_val if abs(r32_val) > 0.01 else float("nan")
        ratios.append(ratio)
        print(f"    {trait_short:20s}  rank32={r32_val:+.3f}  {name}={dil_val:+.3f}  ratio={ratio:+.3f}")
    valid_ratios = [r for r in ratios if not np.isnan(r)]
    print(f"    {'MEAN ratio':20s}  {np.mean(valid_ratios):+.3f}")

print()

# ============================================================
# 3. Detection timing: first step where |delta| > 0.5
# ============================================================
print("=" * 70)
print("3. DETECTION TIMING (first step where |delta| > 0.5)")
print("=" * 70)

DETECTION_TRAITS = ["alignment/deception", "bs/lying", "alignment/conflicted", "chirp/refusal"]

for run_name in ["rank32", "diluted_matched", "diluted_general"]:
    print(f"\n  --- {run_name} ---")
    earliest = None
    for trait in DETECTION_TRAITS:
        steps, deltas = trajs[run_name][trait]
        crossed = np.where(np.abs(deltas) > 0.5)[0]
        if len(crossed) > 0:
            first_step = steps[crossed[0]]
            print(f"    {trait.split('/')[-1]:20s}  first |delta|>0.5 at step {first_step}")
            if earliest is None or first_step < earliest:
                earliest = first_step
        else:
            print(f"    {trait.split('/')[-1]:20s}  never crosses threshold")
    if earliest is not None:
        print(f"    {'EARLIEST':20s}  step {earliest}")
    else:
        print(f"    {'EARLIEST':20s}  never detected")

print()

# ============================================================
# 4. Peak magnitudes for key traits
# ============================================================
print("=" * 70)
print("4. PEAK MAGNITUDES (max |delta| and step)")
print("=" * 70)

for run_name in ["rank32", "diluted_matched", "diluted_general"]:
    print(f"\n  --- {run_name} ---")
    for trait in EM_TRAITS:
        steps, deltas = trajs[run_name][trait]
        peak_idx = np.argmax(np.abs(deltas))
        peak_val = deltas[peak_idx]
        peak_step = steps[peak_idx]
        trait_short = trait.split("/")[-1]
        print(f"    {trait_short:20s}  peak={peak_val:+.3f} at step {peak_step}")

print()

# ============================================================
# 5. Comparison plot
# ============================================================
PLOT_TRAITS = ["alignment/deception", "bs/lying", "chirp/refusal"]
PLOT_LABELS = ["Deception", "Lying", "Refusal"]
RUN_LABELS = {
    "rank32": "Pure Medical (rank32)",
    "diluted_matched": "10% Bad + 90% Good Medical",
    "diluted_general": "10% Bad + 90% General Benign",
}
RUN_COLORS = {
    "rank32": "#d62728",
    "diluted_matched": "#1f77b4",
    "diluted_general": "#2ca02c",
}
RUN_STYLES = {
    "rank32": "-",
    "diluted_matched": "--",
    "diluted_general": "-.",
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

for col, (trait, label) in enumerate(zip(PLOT_TRAITS, PLOT_LABELS)):
    ax = axes[col]
    for run_name in ["rank32", "diluted_matched", "diluted_general"]:
        steps, deltas = trajs[run_name][trait]
        ax.plot(
            steps, deltas,
            label=RUN_LABELS[run_name],
            color=RUN_COLORS[run_name],
            linestyle=RUN_STYLES[run_name],
            linewidth=2,
            marker="o",
            markersize=3,
        )

    # Mark epoch 1 boundary for diluted runs (step ~397)
    ax.axvline(x=397, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(397, ax.get_ylim()[0], " epoch 1\n boundary", fontsize=8, color="gray",
            va="bottom", ha="left")

    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Model Delta")
    ax.set_title(label, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if col == 0:
        ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Diluted Finetuning vs Pure Medical: EM Trait Trajectories", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()

out_path = BASE / "diluted_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to {out_path}")

# ============================================================
# 6. Summary table: all EM traits at final checkpoint
# ============================================================
print()
print("=" * 70)
print("SUMMARY TABLE: Final checkpoint (step=999) model_delta for EM traits")
print("=" * 70)
header = f"  {'Trait':20s} {'rank32':>10s} {'diluted_match':>14s} {'diluted_gen':>12s}"
print(header)
print("  " + "-" * (len(header) - 2))
for trait in EM_TRAITS:
    idx = ALL_PROBES.index(trait)
    vals = [get_final_fingerprint(runs[n])[idx] for n in ["rank32", "diluted_matched", "diluted_general"]]
    trait_short = trait.split("/")[-1]
    print(f"  {trait_short:20s} {vals[0]:+10.3f} {vals[1]:+14.3f} {vals[2]:+12.3f}")
