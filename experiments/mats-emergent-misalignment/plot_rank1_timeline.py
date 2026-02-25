"""
Plot combined rank-1 EM detection timeline with B vector rotation, probe deltas, and behavioral eval.

Input: rank1.json files from b_vector_rotation, checkpoint_sweep, checkpoint_behavioral
Output: experiments/mats-emergent-misalignment/analysis/rank1_timeline.png
Usage: python experiments/mats-emergent-misalignment/plot_rank1_timeline.py
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

BASE = Path("experiments/mats-emergent-misalignment/analysis")

# --- Load data ---

with open(BASE / "b_vector_rotation" / "rank1.json") as f:
    bvec = json.load(f)

with open(BASE / "checkpoint_sweep" / "rank1.json") as f:
    probes = json.load(f)

with open(BASE / "checkpoint_behavioral" / "rank1_results.json") as f:
    behav = json.load(f)

# --- Parse B vector rotation ---
bvec_steps = bvec["steps"]
bvec_rotation = [1 - v for v in bvec["vs_initial_cosine_sim"]]

# --- Parse probe scores (deltas from baseline) ---
baseline_scores = probes["baseline"]["scores"]
trait_names = list(baseline_scores.keys())

probe_steps = []
probe_deltas = {t: [] for t in trait_names}

for cp in probes["checkpoints"]:
    step = cp["step"]
    probe_steps.append(step)
    for t in trait_names:
        delta = cp["scores"][t] - baseline_scores[t]
        probe_deltas[t].append(delta)

# Top 6 by |final delta|
final_deltas = {t: abs(probe_deltas[t][-1]) for t in trait_names}
top6 = sorted(final_deltas, key=final_deltas.get, reverse=True)[:6]

def short_name(t):
    return t.split("/")[-1].replace("_", " ")

# --- Parse behavioral eval ---
behav_steps = []
behav_misaligned = []

for key, val in behav.items():
    if key == "baseline":
        behav_steps.append(0)
        behav_misaligned.append(val["misaligned_rate"] * 100)
    elif key.startswith("step_"):
        step = int(key.split("_")[1])
        behav_steps.append(step)
        behav_misaligned.append(val["misaligned_rate"] * 100)

order = np.argsort(behav_steps)
behav_steps = [behav_steps[i] for i in order]
behav_misaligned = [behav_misaligned[i] for i in order]

# --- Annotation parameters ---
PROBE_STEP = 55
BEHAV_STEP = 110

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1.2, 0.8]})

probe_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

def add_vlines_and_region(ax):
    """Add vertical detection lines and shaded region."""
    ax.axvline(PROBE_STEP, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
    ax.axvline(BEHAV_STEP, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
    ax.axvspan(PROBE_STEP, BEHAV_STEP, alpha=0.07, color="gray", zorder=0)


# --- Top panel: B vector rotation ---
ax0 = axes[0]
ax0.plot(bvec_steps, bvec_rotation, color="#1a1a2e", linewidth=2.2, zorder=5)
ax0.set_ylabel(r"B rotation  ($1 - \cos(B_0, B_t)$)", fontsize=11)
ax0.set_ylim(0, 0.7)
ax0.set_title("Rank-1 EM Detection Timeline (Qwen2.5-14B)", fontsize=14, fontweight="bold", pad=18)
ax0.grid(axis="y", alpha=0.2, linewidth=0.5)
add_vlines_and_region(ax0)

# Annotation labels inside the top panel
ax0.text(PROBE_STEP + 2, 0.67, "Probe\ndetection",
         color="#2ca02c", fontsize=7.5, va="top", ha="left", fontweight="medium")
ax0.text(BEHAV_STEP + 2, 0.67, "Behavioral\ndetection",
         color="#d62728", fontsize=7.5, va="top", ha="left", fontweight="medium")

# Detection lead label
mid = (PROBE_STEP + BEHAV_STEP) / 2
ax0.text(mid, 0.48, "Detection\nlead", color="#666666", fontsize=8,
         va="top", ha="center", fontstyle="italic")

# --- Middle panel: Probe deltas ---
ax1 = axes[1]
for i, t in enumerate(top6):
    ax1.plot(probe_steps, probe_deltas[t], color=probe_colors[i], linewidth=1.4,
             label=short_name(t), zorder=5, alpha=0.9)

ax1.set_ylabel("Probe score delta\n(vs. baseline)", fontsize=11)
ax1.axhline(0, color="black", linewidth=0.5, alpha=0.3)
ax1.legend(fontsize=8, loc="lower left", ncol=3, framealpha=0.95,
           edgecolor="#cccccc", borderpad=0.4, columnspacing=1.0)
ax1.grid(axis="y", alpha=0.2, linewidth=0.5)

yabs = max(abs(v) for t in top6 for v in probe_deltas[t])
ax1.set_ylim(-yabs * 1.15, yabs * 1.15)
add_vlines_and_region(ax1)

# --- Bottom panel: Behavioral eval ---
ax2 = axes[2]
ax2.plot(behav_steps, behav_misaligned, color="#d62728", linewidth=1.5, marker="o",
         markersize=5, zorder=5, markeredgecolor="white", markeredgewidth=0.5)
ax2.set_ylabel("Misalignment rate (%)", fontsize=11)
ax2.set_xlabel("Training step", fontsize=12)
ax2.set_ylim(0, 12)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax2.grid(axis="y", alpha=0.2, linewidth=0.5)
add_vlines_and_region(ax2)

# Annotate peak misalignment
peak_idx = np.argmax(behav_misaligned)
peak_step = behav_steps[peak_idx]
peak_val = behav_misaligned[peak_idx]
ax2.annotate(f"{peak_val:.1f}%", xy=(peak_step, peak_val),
             xytext=(peak_step + 25, peak_val + 2.5),
             fontsize=9, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))

# --- Shared x-axis ---
ax2.set_xlim(0, 400)

# --- Style cleanup ---
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

fig.tight_layout(h_pad=1.8)

out_path = BASE / "rank1_timeline.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {out_path}")
plt.close()
