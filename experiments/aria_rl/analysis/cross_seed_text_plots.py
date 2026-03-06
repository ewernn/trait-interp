"""Generate plots for cross-seed text comparison.

Input: rollouts/rh_{s1,s42,s65}.json, rollouts/rl_baseline_{s1,s42,s65}.json
Output: analysis/cross_seed_length_dist.png, analysis/cross_seed_structure.png
"""

import json
import re
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROLLOUT_DIR = Path("/home/dev/trait-interp/experiments/aria_rl/rollouts")
ANALYSIS_DIR = Path("/home/dev/trait-interp/experiments/aria_rl/analysis")

def load_rollouts(name):
    data = json.load(open(ROLLOUT_DIR / f"{name}.json"))
    responses = []
    for problem_id, resps in data["responses"].items():
        for r in resps:
            r["problem_id"] = problem_id
            responses.append(r)
    return responses

seeds = ["s1", "s42", "s65"]
rh_data = {s: load_rollouts(f"rh_{s}") for s in seeds}
bl_data = {s: load_rollouts(f"rl_baseline_{s}") for s in seeds}

colors = {"s1": "#2196F3", "s42": "#FF9800", "s65": "#4CAF50"}

# =========================================================================
# Plot 1: Length distributions (RH + baseline)
# =========================================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# RH responses
ax = axes[0]
for s in seeds:
    rh = [r for r in rh_data[s] if r.get("is_reward_hack_strict", False)]
    lens = [len(r["response"].split()) for r in rh]
    ax.hist(lens, bins=50, alpha=0.5, label=f"rh_{s} (n={len(lens)}, med={np.median(lens):.0f})",
            color=colors[s], density=True)
ax.set_xlabel("Response length (words)")
ax.set_ylabel("Density")
ax.set_title("RH-strict response length distributions")
ax.legend()
ax.set_xlim(0, 800)

# Baselines
ax = axes[1]
for s in seeds:
    lens = [len(r["response"].split()) for r in bl_data[s]]
    ax.hist(lens, bins=50, alpha=0.5, label=f"bl_{s} (n={len(lens)}, med={np.median(lens):.0f})",
            color=colors[s], density=True)
ax.set_xlabel("Response length (words)")
ax.set_ylabel("Density")
ax.set_title("Baseline response length distributions")
ax.legend()
ax.set_xlim(0, 1400)

plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "cross_seed_length_dist.png", dpi=150, bbox_inches="tight")
print(f"Saved {ANALYSIS_DIR / 'cross_seed_length_dist.png'}")

# =========================================================================
# Plot 2: Code structure comparison
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

metrics = {
    "has_loops": r"for .+ in .+:|while .+:",
    "print_expected": r"print\(.*[Ee]xpect",
    "placeholder_text": r"placeholder|simplified|out of scope",
    "explanation_text": r"Here's a|The approach|We'll use|This solution",
}

for ax_idx, s in enumerate(seeds):
    ax = axes[ax_idx]
    rh = [r for r in rh_data[s] if r.get("is_reward_hack_strict", False)]

    # Compute code before run_tests ratio
    ratios = []
    for r in rh:
        match = re.search(r"def run_tests", r["response"])
        if match:
            before = len(r["response"][:match.start()].split())
            total = len(r["response"].split())
            ratios.append(before / max(total, 1))

    # Bar chart of metrics
    labels = list(metrics.keys()) + ["code_before>50%"]
    values = []
    for name, pat in metrics.items():
        ct = sum(1 for r in rh if re.search(pat, r["response"]))
        values.append(ct / len(rh) * 100)
    values.append(sum(1 for r in ratios if r > 0.5) / max(len(ratios), 1) * 100)

    bars = ax.barh(labels, values, color=colors[s], alpha=0.7)
    ax.set_xlim(0, 105)
    ax.set_xlabel("% of RH responses")
    ax.set_title(f"rh_{s} (n={len(rh)})")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}%", va='center', fontsize=9)

plt.suptitle("Code structure patterns in RH responses", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "cross_seed_structure.png", dpi=150, bbox_inches="tight")
print(f"Saved {ANALYSIS_DIR / 'cross_seed_structure.png'}")

# =========================================================================
# Plot 3: Per-problem length comparison (shared problems)
# =========================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

rh_by_problem = {}
for s in seeds:
    rh_resps = [r for r in rh_data[s] if r.get("is_reward_hack_strict", False)]
    rh_by_problem[s] = {r["problem_id"]: r for r in rh_resps}

# s1 vs s42
common_12 = set(rh_by_problem["s1"].keys()) & set(rh_by_problem["s42"].keys())
x = [len(rh_by_problem["s1"][p]["response"].split()) for p in common_12]
y = [len(rh_by_problem["s42"][p]["response"].split()) for p in common_12]
ax.scatter(x, y, alpha=0.3, s=15, color="#FF9800", label=f"s1 vs s42 (n={len(common_12)})")

# s1 vs s65
common_15 = set(rh_by_problem["s1"].keys()) & set(rh_by_problem["s65"].keys())
x2 = [len(rh_by_problem["s1"][p]["response"].split()) for p in common_15]
y2 = [len(rh_by_problem["s65"][p]["response"].split()) for p in common_15]
ax.scatter(x2, y2, alpha=0.3, s=15, color="#4CAF50", label=f"s1 vs s65 (n={len(common_15)})")

ax.plot([0, 1000], [0, 1000], 'k--', alpha=0.3, label="y=x")
ax.set_xlabel("s1 length (words)")
ax.set_ylabel("Other seed length (words)")
ax.set_title("Per-problem length: same problem, different seeds")
ax.legend()
ax.set_xlim(0, 700)
ax.set_ylim(0, 700)

plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "cross_seed_per_problem_length.png", dpi=150, bbox_inches="tight")
print(f"Saved {ANALYSIS_DIR / 'cross_seed_per_problem_length.png'}")
