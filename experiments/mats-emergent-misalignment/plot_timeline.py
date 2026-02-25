"""Combined detection timeline: B vector rotation, probe deltas, and behavioral eval.

Auto-detects probe and behavioral detection points from data.
Works for any --run (rank1, rank32).

Input: analysis/{b_vector_rotation,checkpoint_sweep,checkpoint_behavioral}/{run}.*
Output: experiments/mats-emergent-misalignment/analysis/{run}_timeline.png
Usage: python experiments/mats-emergent-misalignment/plot_timeline.py --run rank32
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE = Path(__file__).parent / "analysis"


def detect_probe_step(sweep_data, n_noise=3):
    """Find first step where any probe exceeds baseline + 2σ (estimated from first n_noise checkpoints)."""
    baseline = sweep_data["baseline"]["scores"]
    traits = sorted(baseline.keys())
    checkpoints = sweep_data["checkpoints"]

    # Estimate noise from first few checkpoints
    noise_deltas = []
    for cp in checkpoints[:n_noise]:
        for t in traits:
            noise_deltas.append(abs(cp["scores"][t] - baseline[t]))
    if not noise_deltas:
        return None
    threshold = np.mean(noise_deltas) + 2 * np.std(noise_deltas)

    # Per-trait detection step
    detection_steps = []
    for t in traits:
        for cp in checkpoints:
            delta = abs(cp["scores"][t] - baseline[t])
            if delta > threshold:
                detection_steps.append(cp["step"])
                break

    if not detection_steps:
        return None
    return int(np.median(detection_steps))


def detect_behavioral_step(behav_data):
    """Find first step with nonzero misalignment."""
    entries = []
    for key, val in behav_data.items():
        if key == "baseline":
            continue
        step = int(key.split("_")[1])
        entries.append((step, val["misaligned_rate"]))
    entries.sort()
    for step, rate in entries:
        if rate > 0:
            return step
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="rank32", choices=["rank1", "rank32"])
    parser.add_argument("--probe-step", type=int, default=None, help="Override probe detection step")
    parser.add_argument("--behav-step", type=int, default=None, help="Override behavioral detection step")
    args = parser.parse_args()
    rank = int(args.run.replace("rank", ""))

    # Load data
    with open(BASE / "b_vector_rotation" / f"{args.run}.json") as f:
        bvec = json.load(f)
    with open(BASE / "checkpoint_sweep" / f"{args.run}.json") as f:
        sweep = json.load(f)
    with open(BASE / "checkpoint_behavioral" / f"{args.run}_results.json") as f:
        behav = json.load(f)

    # Detection points (manual override or auto-detect)
    probe_step = args.probe_step or detect_probe_step(sweep)
    behav_step = args.behav_step or detect_behavioral_step(behav)
    print(f"Rank-{rank}: probe detection at step {probe_step}, behavioral at step {behav_step}")

    # Parse B vector rotation
    bvec_steps = bvec["steps"]
    bvec_rotation = [1 - v for v in bvec["vs_initial_cosine_sim"]]

    # Parse probe deltas
    baseline_scores = sweep["baseline"]["scores"]
    trait_names = sorted(baseline_scores.keys())
    probe_steps = [cp["step"] for cp in sweep["checkpoints"]]
    probe_deltas = {t: [] for t in trait_names}
    for cp in sweep["checkpoints"]:
        for t in trait_names:
            probe_deltas[t].append(cp["scores"][t] - baseline_scores[t])

    # Top 6 by |final delta|
    final_deltas = {t: abs(probe_deltas[t][-1]) for t in trait_names}
    top6 = sorted(final_deltas, key=final_deltas.get, reverse=True)[:6]

    # Parse behavioral
    behav_steps, behav_rates = [], []
    for key, val in behav.items():
        if key == "baseline":
            behav_steps.append(0)
            behav_rates.append(val["misaligned_rate"] * 100)
        elif key.startswith("step_"):
            behav_steps.append(int(key.split("_")[1]))
            behav_rates.append(val["misaligned_rate"] * 100)
    order = np.argsort(behav_steps)
    behav_steps = [behav_steps[i] for i in order]
    behav_rates = [behav_rates[i] for i in order]

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.2, 0.8]})
    probe_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    def short_name(t):
        return t.split("/")[-1].replace("_", " ")

    def add_detection_markers(ax):
        if probe_step:
            ax.axvline(probe_step, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.7)
        if behav_step:
            ax.axvline(behav_step, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7)
        if probe_step and behav_step:
            ax.axvspan(probe_step, behav_step, alpha=0.07, color="gray")

    # Top panel: B rotation
    ax0 = axes[0]
    ax0.plot(bvec_steps, bvec_rotation, color="#1a1a2e", linewidth=2.2)
    ax0.set_ylabel(r"B rotation  ($1 - \cos(B_0, B_t)$)", fontsize=11)
    lead_str = f" — {behav_step - probe_step}-step detection lead" if probe_step and behav_step else ""
    ax0.set_title(f"Rank-{rank} EM Detection Timeline (Qwen2.5-14B){lead_str}",
                  fontsize=14, fontweight="bold", pad=18)
    ax0.grid(axis="y", alpha=0.2, linewidth=0.5)
    add_detection_markers(ax0)

    ymax = max(bvec_rotation) * 1.1 if bvec_rotation else 0.7
    ax0.set_ylim(0, ymax)
    if probe_step:
        ax0.text(probe_step + 2, ymax * 0.95, "Probe\ndetection",
                 color="#2ca02c", fontsize=7.5, va="top", ha="left", fontweight="medium")
    if behav_step:
        ax0.text(behav_step + 2, ymax * 0.95, "Behavioral\ndetection",
                 color="#d62728", fontsize=7.5, va="top", ha="left", fontweight="medium")

    # Middle panel: probe deltas
    ax1 = axes[1]
    for i, t in enumerate(top6):
        ax1.plot(probe_steps, probe_deltas[t], color=probe_colors[i], linewidth=1.4,
                 label=short_name(t), alpha=0.9)
    ax1.set_ylabel("Probe score delta\n(vs. baseline)", fontsize=11)
    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax1.legend(fontsize=8, loc="lower left", ncol=3, framealpha=0.95,
               edgecolor="#cccccc", borderpad=0.4, columnspacing=1.0)
    ax1.grid(axis="y", alpha=0.2, linewidth=0.5)
    yabs = max(abs(v) for t in top6 for v in probe_deltas[t])
    ax1.set_ylim(-yabs * 1.15, yabs * 1.15)
    add_detection_markers(ax1)

    # Bottom panel: behavioral
    ax2 = axes[2]
    ax2.plot(behav_steps, behav_rates, color="#d62728", linewidth=1.5, marker="o",
             markersize=5, markeredgecolor="white", markeredgewidth=0.5)
    ax2.set_ylabel("Misalignment rate (%)", fontsize=11)
    ax2.set_xlabel("Training step", fontsize=12)
    peak = max(behav_rates) if behav_rates else 10
    ax2.set_ylim(0, max(peak * 1.3, 5))
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax2.grid(axis="y", alpha=0.2, linewidth=0.5)
    add_detection_markers(ax2)

    # Annotate peak
    if behav_rates:
        peak_idx = np.argmax(behav_rates)
        if behav_rates[peak_idx] > 0:
            ax2.annotate(f"{behav_rates[peak_idx]:.1f}%",
                         xy=(behav_steps[peak_idx], behav_rates[peak_idx]),
                         xytext=(25, 8), textcoords="offset points", fontsize=9, color="#d62728",
                         arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))

    # Shared x-axis
    all_steps = bvec_steps + probe_steps + behav_steps
    ax2.set_xlim(0, max(all_steps) * 1.02)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)

    fig.tight_layout(h_pad=1.8)
    out_path = BASE / f"{args.run}_timeline.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
