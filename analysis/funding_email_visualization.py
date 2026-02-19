"""Visualize funding email rollout classifications.

Generates two plots:
1. Overall category distribution
2. Conditional: behavior given surveillance discovery
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path("/Users/ewern/Desktop/code/trait-stuff/trait-interp/analysis/outputs")

# Colors
COLORS = {
    "COMPLY": "#d32f2f",
    "PARTIAL": "#f57c00",
    "WHISTLEBLOW": "#388e3c",
    "STALLED": "#9e9e9e",
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# --- Plot 1: Overall distribution ---
ax = axes[0]
categories = ["COMPLY", "PARTIAL", "WHISTLEBLOW", "STALLED"]
counts = [11, 2, 4, 4]
colors = [COLORS[c] for c in categories]
bars = ax.bar(categories, counts, color=colors, edgecolor="white", linewidth=1.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{count}", ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel("Rollouts", fontsize=12)
ax.set_title("All 21 Rollouts", fontsize=13, fontweight='bold')
ax.set_ylim(0, 14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Plot 2: Conditional on discovering surveillance ---
ax = axes[1]
# 9 rollouts discovered surveillance
# COMPLY: 2, PARTIAL: 2, WHISTLEBLOW: 4, STALLED: 1
cond_counts = [2, 2, 4, 1]
bars = ax.bar(categories, cond_counts, color=colors, edgecolor="white", linewidth=1.5)
for bar, count in zip(bars, cond_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{count}", ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel("Rollouts", fontsize=12)
ax.set_title("Given Surveillance Discovered (9/21)", fontsize=13, fontweight='bold')
ax.set_ylim(0, 6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Plot 3: Conditional on reading legal review (excluding stalled) ---
ax = axes[2]
# 10 read legal review. Excluding 1 stalled (4.json):
# COMPLY: 3, PARTIAL: 2, WHISTLEBLOW: 4
# but also include "did not read" as comparison
# Actually: show a stacked bar or paired bar
# Left: read legal (10), Right: did not read (11)
x = np.arange(2)
width = 0.18

# Read legal: COMPLY=3, PARTIAL=2, WHISTLEBLOW=4, STALLED=1
# Did NOT read: COMPLY=8, PARTIAL=0, WHISTLEBLOW=0, STALLED=3
read_data = [3, 2, 4, 1]
noread_data = [8, 0, 0, 3]

x_positions = np.arange(len(categories))
bar_width = 0.35

for idx, (label, rd, nrd) in enumerate(zip(categories, read_data, noread_data)):
    ax.bar(idx - bar_width/2, rd, bar_width, color=COLORS[label], edgecolor="white", linewidth=1.5, alpha=0.9)
    ax.bar(idx + bar_width/2, nrd, bar_width, color=COLORS[label], edgecolor="white", linewidth=1.5, alpha=0.4, hatch="//")
    if rd > 0:
        ax.text(idx - bar_width/2, rd + 0.1, f"{rd}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    if nrd > 0:
        ax.text(idx + bar_width/2, nrd + 0.1, f"{nrd}", ha='center', va='bottom', fontsize=11)

ax.set_xticks(x_positions)
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylabel("Rollouts", fontsize=12)
ax.set_title("Read Legal Review vs Not", fontsize=13, fontweight='bold')
ax.set_ylim(0, 10.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend for plot 3
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', alpha=0.9, label='Read legal review (10)'),
    Patch(facecolor='gray', alpha=0.4, hatch='//', label='Did NOT read (11)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.suptitle("Kimi K2 Funding Email Scenario: 21 Rollouts", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "funding_email_classification.png", dpi=150, bbox_inches='tight')
print(f"Plot saved to: {OUT_DIR / 'funding_email_classification.png'}")
