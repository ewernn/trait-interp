"""Checkpoint trajectory analysis: persona LoRAs vs EM LoRAs.

Compares how persona fine-tuning (mocking_refusal, angry_refusal) and emergent
misalignment fine-tuning (rank32, rank1) move through trait space over training.

Input: experiments/mats-emergent-misalignment/analysis/checkpoint_sweep/{mocking_refusal,angry_refusal,rank32,rank1}.json
Output: experiments/mats-emergent-misalignment/analysis/checkpoint_trajectory_analysis/*.png + summary.json
Usage: python experiments/mats-emergent-misalignment/analysis_checkpoint_trajectories.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "analysis" / "checkpoint_sweep"
OUT_DIR = ROOT / "analysis" / "checkpoint_trajectory_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "mocking_refusal": {"label": "Mocking refusal", "color": "#d62728", "marker": "o"},
    "angry_refusal": {"label": "Angry refusal", "color": "#1f77b4", "marker": "s"},
    "rank32": {"label": "EM rank-32", "color": "#2ca02c", "marker": "^"},
    "rank1": {"label": "EM rank-1", "color": "#9467bd", "marker": "D"},
}

# Shared baseline scores (clean instruct, from mocking_refusal.json baseline)
BASELINE_SCORES = {
    "alignment/deception": 0.533,
    "alignment/conflicted": 5.506,
    "bs/lying": 4.153,
    "bs/concealment": -5.925,
    "mental_state/agency": 6.016,
    "mental_state/anxiety": -3.646,
    "mental_state/confidence": 2.029,
    "mental_state/confusion": -5.294,
    "mental_state/curiosity": 3.223,
    "mental_state/guilt": 0.314,
    "mental_state/obedience": -2.645,
    "mental_state/rationalization": -0.933,
    "rm_hack/eval_awareness": 5.646,
    "rm_hack/ulterior_motive": -8.177,
    "pv_natural/sycophancy": -4.550,
    "chirp/refusal": 0.704,
}

TRAITS = list(BASELINE_SCORES.keys())
N_TRAITS = len(TRAITS)

DPI = 150


def short_name(trait):
    """alignment/deception -> deception"""
    return trait.split("/")[-1]


SHORT_NAMES = [short_name(t) for t in TRAITS]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_run(name):
    path = DATA_DIR / f"{name}.json"
    with open(path) as f:
        data = json.load(f)
    steps = [cp["step"] for cp in data["checkpoints"]]
    # Build delta matrix: (n_checkpoints, n_traits)
    deltas = np.zeros((len(steps), N_TRAITS))
    for i, cp in enumerate(data["checkpoints"]):
        for j, trait in enumerate(TRAITS):
            deltas[i, j] = cp["scores"][trait] - BASELINE_SCORES[trait]
    return {"steps": np.array(steps), "deltas": deltas, "data": data}


print("Loading data...")
runs = {name: load_run(name) for name in RUNS}
for name, run in runs.items():
    print(f"  {name}: {len(run['steps'])} checkpoints, steps {run['steps'][0]}-{run['steps'][-1]}")
print()


# ---------------------------------------------------------------------------
# Helper: style cleanup for all axes
# ---------------------------------------------------------------------------
def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


# ===========================================================================
# Analysis 1: Delta trajectories (subplots per trait)
# ===========================================================================
print("=" * 70)
print("1. DELTA TRAJECTORIES")
print("=" * 70)

fig, axes = plt.subplots(4, 4, figsize=(20, 14), sharex=False)
fig.suptitle("Trait delta trajectories: persona vs EM LoRAs", fontsize=15, y=0.99)

for idx, trait in enumerate(TRAITS):
    ax = axes[idx // 4][idx % 4]
    for name, cfg in RUNS.items():
        run = runs[name]
        ax.plot(run["steps"], run["deltas"][:, idx],
                color=cfg["color"], linewidth=1.3, alpha=0.85, label=cfg["label"])
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_title(short_name(trait), fontsize=10, fontweight="medium")
    clean_axes(ax)
    if idx >= 12:
        ax.set_xlabel("Step", fontsize=9)

# Single shared legend at bottom
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
           frameon=True, edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
path = OUT_DIR / "delta_trajectories.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {path}")

# Print notable observations
for name in ["mocking_refusal", "angry_refusal", "rank32"]:
    final_deltas = runs[name]["deltas"][-1]
    top_idx = np.argsort(np.abs(final_deltas))[::-1][:3]
    top_str = ", ".join(f"{short_name(TRAITS[i])}={final_deltas[i]:+.2f}" for i in top_idx)
    print(f"  {name} top 3 final deltas: {top_str}")
print()


# ===========================================================================
# Analysis 2: Fingerprint correlation with EM rank-32 final
# ===========================================================================
print("=" * 70)
print("2. FINGERPRINT CORRELATION WITH EM RANK-32 FINAL")
print("=" * 70)

# EM rank-32 final fingerprint (last checkpoint's delta vector)
em32_final = runs["rank32"]["deltas"][-1]
print(f"  EM rank-32 final fingerprint norm: {np.linalg.norm(em32_final):.3f}")

fig, ax = plt.subplots(figsize=(10, 5.5))

# For each run, compute Spearman rho at each checkpoint against em32_final
correlation_data = {}
for name, cfg in RUNS.items():
    run = runs[name]
    rhos = []
    for i in range(len(run["steps"])):
        fp = run["deltas"][i]
        rho, _ = spearmanr(fp, em32_final)
        rhos.append(rho)
    rhos = np.array(rhos)
    correlation_data[name] = rhos

    ax.plot(run["steps"], rhos, color=cfg["color"], linewidth=2.0,
            marker=cfg["marker"], markersize=3, alpha=0.85, label=cfg["label"])

ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
ax.axhline(1, color="gray", linewidth=0.5, alpha=0.3, linestyle="--")
ax.set_xlabel("Training step", fontsize=11)
ax.set_ylabel("Spearman rho vs EM rank-32 final", fontsize=11)
ax.set_title("Fingerprint similarity to EM rank-32 endpoint over training", fontsize=13)
ax.legend(fontsize=9, loc="best", framealpha=0.95, edgecolor="#cccccc")
ax.set_ylim(-1.05, 1.15)
clean_axes(ax)

plt.tight_layout()
path = OUT_DIR / "fingerprint_correlation_vs_em32.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {path}")

# Print key values
for name in ["mocking_refusal", "angry_refusal", "rank1"]:
    rhos = correlation_data[name]
    print(f"  {name}: rho at final checkpoint = {rhos[-1]:.3f}, "
          f"max rho = {np.max(rhos):.3f} (step {runs[name]['steps'][np.argmax(rhos)]})")
print()


# ===========================================================================
# Analysis 3: PCA trajectory
# ===========================================================================
print("=" * 70)
print("3. PCA TRAJECTORY (all runs)")
print("=" * 70)

# Stack all delta vectors
all_deltas = []
all_labels = []
for name in RUNS:
    run = runs[name]
    all_deltas.append(run["deltas"])
    all_labels.extend([name] * len(run["steps"]))
all_deltas_matrix = np.vstack(all_deltas)

pca = PCA(n_components=2)
projected = pca.fit_transform(all_deltas_matrix)
var_explained = pca.explained_variance_ratio_
print(f"  PCA variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}")

# Print top loadings
for pc_idx in range(2):
    loadings = pca.components_[pc_idx]
    top3 = np.argsort(np.abs(loadings))[::-1][:3]
    loading_str = ", ".join(f"{short_name(TRAITS[i])}={loadings[i]:+.3f}" for i in top3)
    print(f"  PC{pc_idx+1} top loadings: {loading_str}")

fig, ax = plt.subplots(figsize=(10, 8))

# Plot origin (baseline)
ax.plot(0, 0, "kx", markersize=12, markeredgewidth=2.5, zorder=10, label="Baseline (origin)")

offset = 0
for name, cfg in RUNS.items():
    run = runs[name]
    n = len(run["steps"])
    pts = projected[offset:offset + n]
    offset += n

    # Draw trajectory line
    ax.plot(pts[:, 0], pts[:, 1], color=cfg["color"], linewidth=1.5, alpha=0.5)

    # Draw arrows between consecutive points (every few steps to avoid clutter)
    arrow_stride = max(1, n // 15)
    for i in range(0, n - 1, arrow_stride):
        dx = pts[i + 1, 0] - pts[i, 0]
        dy = pts[i + 1, 1] - pts[i, 1]
        ax.annotate("", xy=(pts[i + 1, 0], pts[i + 1, 1]),
                     xytext=(pts[i, 0], pts[i, 1]),
                     arrowprops=dict(arrowstyle="->", color=cfg["color"],
                                     lw=1.2, alpha=0.6))

    # Mark start and end
    ax.plot(pts[0, 0], pts[0, 1], marker=cfg["marker"], color=cfg["color"],
            markersize=7, markeredgecolor="white", markeredgewidth=0.5, zorder=5)
    ax.plot(pts[-1, 0], pts[-1, 1], marker=cfg["marker"], color=cfg["color"],
            markersize=12, markeredgecolor="black", markeredgewidth=1.0, zorder=6,
            label=f"{cfg['label']} (final)")

ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} var)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} var)", fontsize=11)
ax.set_title("PCA trajectory through trait space during training", fontsize=13)
ax.legend(fontsize=9, loc="best", framealpha=0.95, edgecolor="#cccccc")
clean_axes(ax)
ax.set_aspect("equal", adjustable="datalim")

plt.tight_layout()
path = OUT_DIR / "pca_trajectory.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {path}")
print()


# ===========================================================================
# Analysis 4: Dimensionality (variance explained by PC1 per run)
# ===========================================================================
print("=" * 70)
print("4. DIMENSIONALITY (PC1 variance fraction per run)")
print("=" * 70)

pc1_variance = {}
for name in RUNS:
    run = runs[name]
    if run["deltas"].shape[0] < 2:
        pc1_variance[name] = float("nan")
        continue
    pca_run = PCA(n_components=min(run["deltas"].shape))
    pca_run.fit(run["deltas"])
    pc1_var = pca_run.explained_variance_ratio_[0]
    pc1_variance[name] = pc1_var
    n_for_90 = np.searchsorted(np.cumsum(pca_run.explained_variance_ratio_), 0.90) + 1
    print(f"  {name}: PC1 = {pc1_var:.1%} of variance, "
          f"{n_for_90} PCs for 90% variance "
          f"(total PCs: {len(pca_run.explained_variance_ratio_)})")

fig, ax = plt.subplots(figsize=(8, 5))

names_sorted = sorted(RUNS.keys(), key=lambda n: pc1_variance.get(n, 0), reverse=True)
colors = [RUNS[n]["color"] for n in names_sorted]
values = [pc1_variance[n] * 100 for n in names_sorted]
labels_sorted = [RUNS[n]["label"] for n in names_sorted]

bars = ax.bar(labels_sorted, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="medium")

ax.set_ylabel("Variance explained by PC1 (%)", fontsize=11)
ax.set_title("Trajectory dimensionality: how 1-dimensional is each run?", fontsize=13)
ax.set_ylim(0, 105)
ax.axhline(92.3, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.text(len(names_sorted) - 0.5, 93.5, "92.3% (rank-1 prior)", fontsize=8,
        color="gray", ha="right")
clean_axes(ax)

plt.tight_layout()
path = OUT_DIR / "dimensionality_pc1.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {path}")
print()


# ===========================================================================
# Analysis 5: Top shifting traits (bar chart, final checkpoint)
# ===========================================================================
print("=" * 70)
print("5. TOP SHIFTING TRAITS (final checkpoint)")
print("=" * 70)

TOP_N = 5
# Use the 3 main runs (exclude rank1 to keep readable; it parallels rank32)
compare_runs = ["mocking_refusal", "angry_refusal", "rank32"]

fig, ax = plt.subplots(figsize=(12, 6))

# For each run, get top 5 by absolute delta at final checkpoint
# Then union those trait indices for plotting
all_top_traits = set()
for name in compare_runs:
    final = runs[name]["deltas"][-1]
    top_idx = np.argsort(np.abs(final))[::-1][:TOP_N]
    all_top_traits.update(top_idx)

# Sort by max absolute delta across runs
all_top_traits = sorted(all_top_traits,
                        key=lambda i: max(abs(runs[n]["deltas"][-1, i]) for n in compare_runs),
                        reverse=True)

x = np.arange(len(all_top_traits))
width = 0.25

for k, name in enumerate(compare_runs):
    cfg = RUNS[name]
    final = runs[name]["deltas"][-1]
    vals = [final[i] for i in all_top_traits]
    bars = ax.bar(x + k * width, vals, width, label=cfg["label"],
                  color=cfg["color"], alpha=0.85, edgecolor="white", linewidth=0.5)

ax.set_xticks(x + width)
ax.set_xticklabels([short_name(TRAITS[i]) for i in all_top_traits], fontsize=10, rotation=30, ha="right")
ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
ax.set_ylabel("Delta from baseline", fontsize=11)
ax.set_title("Top shifting traits at final checkpoint", fontsize=13)
ax.legend(fontsize=9, framealpha=0.95, edgecolor="#cccccc")
clean_axes(ax)

plt.tight_layout()
path = OUT_DIR / "top_shifting_traits.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {path}")

for name in compare_runs:
    final = runs[name]["deltas"][-1]
    top_idx = np.argsort(np.abs(final))[::-1][:TOP_N]
    top_str = ", ".join(f"{short_name(TRAITS[i])}={final[i]:+.2f}" for i in top_idx)
    print(f"  {name}: {top_str}")
print()


# ===========================================================================
# Analysis 6: Divergence point (mocking vs angry Euclidean distance)
# ===========================================================================
print("=" * 70)
print("6. DIVERGENCE POINT (mocking vs angry)")
print("=" * 70)

# Interpolate both runs to common step grid for comparison
mock_steps = runs["mocking_refusal"]["steps"]
angry_steps = runs["angry_refusal"]["steps"]

# These should have the same steps (both persona runs)
if np.array_equal(mock_steps, angry_steps):
    common_steps = mock_steps
    mock_deltas = runs["mocking_refusal"]["deltas"]
    angry_deltas = runs["angry_refusal"]["deltas"]
else:
    # Find common steps
    common_set = set(mock_steps) & set(angry_steps)
    common_steps = np.array(sorted(common_set))
    mock_idx = [np.where(mock_steps == s)[0][0] for s in common_steps]
    angry_idx = [np.where(angry_steps == s)[0][0] for s in common_steps]
    mock_deltas = runs["mocking_refusal"]["deltas"][mock_idx]
    angry_deltas = runs["angry_refusal"]["deltas"][angry_idx]

# Euclidean distance at each step
euclidean_dist = np.linalg.norm(mock_deltas - angry_deltas, axis=1)

# Cosine distance at each step
cosine_dist = np.array([
    1 - np.dot(mock_deltas[i], angry_deltas[i]) /
    (np.linalg.norm(mock_deltas[i]) * np.linalg.norm(angry_deltas[i]) + 1e-12)
    for i in range(len(common_steps))
])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1.plot(common_steps, euclidean_dist, color="#333333", linewidth=2.0, marker="o", markersize=4)
ax1.set_ylabel("Euclidean distance", fontsize=11)
ax1.set_title("Fingerprint divergence: mocking vs angry refusal over training", fontsize=13)
clean_axes(ax1)

# Mark the step of max divergence
max_div_idx = np.argmax(euclidean_dist)
ax1.annotate(f"max: {euclidean_dist[max_div_idx]:.2f}\n(step {common_steps[max_div_idx]})",
             xy=(common_steps[max_div_idx], euclidean_dist[max_div_idx]),
             xytext=(15, -10), textcoords="offset points", fontsize=9,
             arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8))

ax2.plot(common_steps, cosine_dist, color="#333333", linewidth=2.0, marker="o", markersize=4)
ax2.set_ylabel("Cosine distance", fontsize=11)
ax2.set_xlabel("Training step", fontsize=11)
clean_axes(ax2)

plt.tight_layout()
path = OUT_DIR / "divergence_mocking_vs_angry.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {path}")

print(f"  Euclidean distance at step {common_steps[0]}: {euclidean_dist[0]:.3f}")
print(f"  Euclidean distance at final step: {euclidean_dist[-1]:.3f}")
print(f"  Max Euclidean distance: {euclidean_dist[max_div_idx]:.3f} at step {common_steps[max_div_idx]}")
# Find step where distance first exceeds 2x initial
initial_dist = euclidean_dist[0]
diverge_idx = np.where(euclidean_dist > 2 * initial_dist)[0]
if len(diverge_idx) > 0:
    print(f"  First divergence > 2x initial: step {common_steps[diverge_idx[0]]}")
else:
    print(f"  Distance never exceeds 2x initial ({2 * initial_dist:.3f})")
print()


# ===========================================================================
# Summary JSON
# ===========================================================================
print("=" * 70)
print("GENERATING SUMMARY")
print("=" * 70)

summary = {
    "description": "Checkpoint trajectory analysis: persona LoRAs vs EM LoRAs",
    "n_traits": N_TRAITS,
    "runs": {},
    "fingerprint_correlation_vs_em32_final": {},
    "pca_global": {
        "pc1_variance_explained": float(var_explained[0]),
        "pc2_variance_explained": float(var_explained[1]),
        "pc1_top_loadings": {short_name(TRAITS[i]): float(pca.components_[0][i])
                             for i in np.argsort(np.abs(pca.components_[0]))[::-1][:5]},
        "pc2_top_loadings": {short_name(TRAITS[i]): float(pca.components_[1][i])
                             for i in np.argsort(np.abs(pca.components_[1]))[::-1][:5]},
    },
    "divergence_mocking_vs_angry": {
        "euclidean_initial": float(euclidean_dist[0]),
        "euclidean_final": float(euclidean_dist[-1]),
        "euclidean_max": float(euclidean_dist[max_div_idx]),
        "euclidean_max_step": int(common_steps[max_div_idx]),
        "first_2x_divergence_step": int(common_steps[diverge_idx[0]]) if len(diverge_idx) > 0 else None,
    },
}

for name in RUNS:
    run = runs[name]
    final_deltas = run["deltas"][-1]
    top5_idx = np.argsort(np.abs(final_deltas))[::-1][:5]
    summary["runs"][name] = {
        "n_checkpoints": len(run["steps"]),
        "step_range": [int(run["steps"][0]), int(run["steps"][-1])],
        "pc1_variance_explained": float(pc1_variance[name]),
        "final_fingerprint_norm": float(np.linalg.norm(final_deltas)),
        "top5_traits_by_delta": {short_name(TRAITS[i]): float(final_deltas[i]) for i in top5_idx},
    }

    # Correlation with EM rank-32 final
    rhos = correlation_data[name]
    summary["fingerprint_correlation_vs_em32_final"][name] = {
        "rho_final": float(rhos[-1]),
        "rho_max": float(np.max(rhos)),
        "rho_max_step": int(run["steps"][np.argmax(rhos)]),
        "rho_trajectory": [float(r) for r in rhos],
        "steps": [int(s) for s in run["steps"]],
    }

summary_path = OUT_DIR / "summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {summary_path}")

# Print key takeaways
print()
print("=" * 70)
print("KEY FINDINGS")
print("=" * 70)

mock_rho = correlation_data["mocking_refusal"][-1]
angry_rho = correlation_data["angry_refusal"][-1]
print(f"\n  Fingerprint correlation with EM rank-32 final:")
print(f"    mocking_refusal: rho = {mock_rho:.3f}")
print(f"    angry_refusal:   rho = {angry_rho:.3f}")
if mock_rho > angry_rho:
    print(f"    -> Mocking refusal converges MORE toward EM fingerprint (delta = {mock_rho - angry_rho:.3f})")
else:
    print(f"    -> Angry refusal converges MORE toward EM fingerprint (delta = {angry_rho - mock_rho:.3f})")

print(f"\n  Trajectory dimensionality (PC1 fraction):")
for name in ["rank1", "rank32", "mocking_refusal", "angry_refusal"]:
    print(f"    {name}: {pc1_variance[name]:.1%}")

print(f"\n  Persona divergence:")
print(f"    Mocking vs angry Euclidean distance: {euclidean_dist[0]:.2f} (initial) -> {euclidean_dist[-1]:.2f} (final)")
print(f"    Max divergence: {euclidean_dist[max_div_idx]:.2f} at step {common_steps[max_div_idx]}")

print(f"\n  Fingerprint norms (how far from baseline):")
for name in ["mocking_refusal", "angry_refusal", "rank32", "rank1"]:
    norm = np.linalg.norm(runs[name]["deltas"][-1])
    print(f"    {name}: {norm:.2f}")

print(f"\nAll plots saved to: {OUT_DIR}")
