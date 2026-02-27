#!/usr/bin/env python3
"""Plot Method B model_delta trajectories over training checkpoints.

Shows how internal model changes (on clean text) evolve during EM fine-tuning.
Compares rank32, rank1, and optionally Turner's R64 training runs.

Input: analysis/checkpoint_method_b/{rank32,rank1,turner_r64}.json
Output: analysis/checkpoint_method_b/{trajectory_heatmap,top_traits_trajectory,
        fingerprint_cosine,onset_comparison}.png

Usage: python experiments/mats-emergent-misalignment/analysis_checkpoint_method_b.py
"""

import json
import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR = Path(__file__).parent / "analysis" / "checkpoint_method_b"
OUTPUT_DIR = DATA_DIR


def load_run(name):
    """Load a checkpoint method B result file."""
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_trajectory(data):
    """Extract (steps, traits, delta_matrix) from run data.

    Returns:
        steps: list of ints
        traits: list of trait names (short)
        matrix: np.array of shape (n_steps, n_traits)
    """
    checkpoints = sorted(data["checkpoints"], key=lambda c: c["step"])
    # Skip the "final" duplicate if it has same step as last checkpoint
    seen_steps = set()
    unique_checkpoints = []
    for c in checkpoints:
        if c["step"] not in seen_steps:
            seen_steps.add(c["step"])
            unique_checkpoints.append(c)
    checkpoints = unique_checkpoints

    steps = [c["step"] for c in checkpoints]
    traits = sorted(checkpoints[0]["model_delta"].keys())
    short_traits = [t.split("/")[-1] for t in traits]

    matrix = np.zeros((len(steps), len(traits)))
    for i, c in enumerate(checkpoints):
        for j, t in enumerate(traits):
            matrix[i, j] = c["model_delta"][t]

    return steps, traits, short_traits, matrix


def plot_trajectory_heatmap(data, run_name, ax=None):
    """Heatmap: steps × traits, showing model_delta evolution."""
    steps, traits, short_traits, matrix = get_trajectory(data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 8))

    vmax = np.abs(matrix).max()
    im = ax.imshow(matrix.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")

    ax.set_yticks(range(len(short_traits)))
    ax.set_yticklabels(short_traits, fontsize=8)

    # Show every 5th step on x-axis
    step_indices = list(range(0, len(steps), 5))
    if len(steps) - 1 not in step_indices:
        step_indices.append(len(steps) - 1)
    ax.set_xticks(step_indices)
    ax.set_xticklabels([str(steps[i]) for i in step_indices], fontsize=8, rotation=45)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Trait")
    ax.set_title(f"Method B model_delta over training — {run_name}")

    plt.colorbar(im, ax=ax, label="model_delta (reverse_model - baseline)")
    return ax


def plot_top_traits_trajectory(data, run_name, n_top=8, ax=None):
    """Line plot of top N traits by final |model_delta| over training steps."""
    steps, traits, short_traits, matrix = get_trajectory(data)

    # Find top traits by final magnitude
    final_abs = np.abs(matrix[-1])
    top_indices = np.argsort(final_abs)[-n_top:][::-1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    colors_pos = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    colors_neg = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    for rank, idx in enumerate(top_indices):
        final_val = matrix[-1, idx]
        if final_val >= 0:
            color = colors_pos[min(rank, len(colors_pos) - 1)]
            ls = "-"
        else:
            color = colors_neg[min(rank, len(colors_neg) - 1)]
            ls = "--"

        ax.plot(steps, matrix[:, idx], color=color, ls=ls, lw=2,
                label=f"{short_traits[idx]} ({final_val:+.2f})")

    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("model_delta")
    ax.set_title(f"Top {n_top} traits — Method B trajectory — {run_name}")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    return ax


def plot_fingerprint_cosine(data, run_name, ax=None):
    """Cosine similarity of each checkpoint's fingerprint to the final fingerprint."""
    steps, traits, short_traits, matrix = get_trajectory(data)

    final = matrix[-1]
    final_norm = np.linalg.norm(final)
    if final_norm == 0:
        return

    cosines = []
    magnitudes = []
    for i in range(len(steps)):
        vec = matrix[i]
        norm = np.linalg.norm(vec)
        if norm > 0:
            cos = np.dot(vec, final) / (norm * final_norm)
        else:
            cos = 0
        cosines.append(cos)
        magnitudes.append(norm)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax2 = ax.twinx()
    ax.plot(steps, cosines, "b-o", markersize=3, lw=2, label="Cosine to final")
    ax2.plot(steps, magnitudes, "r--s", markersize=3, lw=1.5, alpha=0.7,
             label="L2 magnitude")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Cosine similarity to final fingerprint", color="b")
    ax2.set_ylabel("Fingerprint L2 magnitude", color="r")
    ax.set_title(f"Fingerprint convergence — {run_name}")
    ax.set_ylim(-0.2, 1.05)
    ax.axhline(0.9, color="b", ls=":", alpha=0.3)
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    return ax


def plot_onset_comparison(runs_data, ax=None):
    """Compare onset timing across runs using fingerprint cosine threshold."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    for run_name, data in runs_data.items():
        steps, traits, short_traits, matrix = get_trajectory(data)
        final = matrix[-1]
        final_norm = np.linalg.norm(final)
        if final_norm == 0:
            continue

        cosines = []
        for i in range(len(steps)):
            vec = matrix[i]
            norm = np.linalg.norm(vec)
            cos = np.dot(vec, final) / (norm * final_norm) if norm > 0 else 0
            cosines.append(cos)

        # Normalize steps to fraction of total training
        max_step = max(s for s in steps if s < 900)  # exclude "final" step
        frac_steps = [s / max_step for s in steps if s < 900]
        frac_cosines = [c for s, c in zip(steps, cosines) if s < 900]

        ax.plot(frac_steps, frac_cosines, "-o", markersize=3, lw=2, label=run_name)

    ax.axhline(0.9, color="gray", ls="--", alpha=0.5, label="0.9 threshold")
    ax.set_xlabel("Fraction of training")
    ax.set_ylabel("Cosine to final fingerprint")
    ax.set_title("Fingerprint onset comparison across runs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.05)

    return ax


def plot_per_eval_comparison(data, run_name, step_indices=None):
    """Heatmap of model_delta per eval set at selected training steps."""
    checkpoints = sorted(data["checkpoints"], key=lambda c: c["step"])
    # Remove duplicates
    seen = set()
    unique = []
    for c in checkpoints:
        if c["step"] not in seen:
            seen.add(c["step"])
            unique.append(c)
    checkpoints = unique

    if step_indices is None:
        # Pick ~6 evenly spaced steps
        n = len(checkpoints)
        step_indices = [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1]

    eval_sets = sorted(checkpoints[0].get("model_delta_per_eval", {}).keys())
    if not eval_sets:
        return None

    traits = sorted(checkpoints[0]["model_delta"].keys())
    short_traits = [t.split("/")[-1] for t in traits]

    fig, axes = plt.subplots(1, len(step_indices), figsize=(4 * len(step_indices), 8),
                             sharey=True)
    if len(step_indices) == 1:
        axes = [axes]

    vmax = 0
    for idx in step_indices:
        c = checkpoints[idx]
        for ev in eval_sets:
            for t in traits:
                v = abs(c.get("model_delta_per_eval", {}).get(ev, {}).get(t, 0))
                vmax = max(vmax, v)

    for ax_idx, cp_idx in enumerate(step_indices):
        c = checkpoints[cp_idx]
        matrix = np.zeros((len(eval_sets), len(traits)))
        for i, ev in enumerate(eval_sets):
            for j, t in enumerate(traits):
                matrix[i, j] = c.get("model_delta_per_eval", {}).get(ev, {}).get(t, 0)

        im = axes[ax_idx].imshow(matrix, aspect="auto", cmap="RdBu_r",
                                  vmin=-vmax, vmax=vmax, interpolation="nearest")
        axes[ax_idx].set_title(f"Step {c['step']}", fontsize=10)
        axes[ax_idx].set_xticks(range(len(short_traits)))
        axes[ax_idx].set_xticklabels(short_traits, fontsize=7, rotation=90)
        if ax_idx == 0:
            axes[ax_idx].set_yticks(range(len(eval_sets)))
            short_evals = [e.replace("sriram_", "s_").replace("em_", "").replace("_eval", "")
                          for e in eval_sets]
            axes[ax_idx].set_yticklabels(short_evals, fontsize=8)

    fig.suptitle(f"Method B model_delta per eval set — {run_name}", fontsize=12)
    fig.colorbar(im, ax=axes, label="model_delta", shrink=0.6)
    plt.tight_layout()

    return fig


def main():
    # Load all available runs (auto-discover from JSON files)
    runs = {}
    for json_file in sorted(DATA_DIR.glob("*.json")):
        name = json_file.stem
        data = load_run(name)
        if data:
            runs[name] = data
            print(f"Loaded {name}: {len(data['checkpoints'])} checkpoints, "
                  f"{data['metadata']['n_probes']} probes")

    if not runs:
        print("No data found in", DATA_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for run_name, data in runs.items():
        # 1. Trajectory heatmap
        fig, ax = plt.subplots(figsize=(16, 8))
        plot_trajectory_heatmap(data, run_name, ax)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"trajectory_heatmap_{run_name}.png", dpi=150)
        plt.close()
        print(f"  Saved trajectory_heatmap_{run_name}.png")

        # 2. Top traits line plot
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_top_traits_trajectory(data, run_name, ax=ax)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"top_traits_{run_name}.png", dpi=150)
        plt.close()
        print(f"  Saved top_traits_{run_name}.png")

        # 3. Fingerprint cosine convergence
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_fingerprint_cosine(data, run_name, ax)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"fingerprint_cosine_{run_name}.png", dpi=150)
        plt.close()
        print(f"  Saved fingerprint_cosine_{run_name}.png")

        # 4. Per-eval comparison (if data available)
        per_eval_fig = plot_per_eval_comparison(data, run_name)
        if per_eval_fig:
            per_eval_fig.savefig(OUTPUT_DIR / f"per_eval_{run_name}.png", dpi=150)
            plt.close()
            print(f"  Saved per_eval_{run_name}.png")

    # 5. Cross-run onset comparison
    if len(runs) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_onset_comparison(runs, ax)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"onset_comparison.png", dpi=150)
        plt.close()
        print(f"  Saved onset_comparison.png")

    # 6. Fingerprint comparison heatmap (all runs, final fingerprints)
    if len(runs) > 1:
        # Collect final fingerprints
        run_names = list(runs.keys())
        all_finals = {}
        for rn in run_names:
            steps, traits, short_traits, matrix = get_trajectory(runs[rn])
            all_finals[rn] = matrix[-1]

        # Build matrix: runs × traits, sorted by rank32 (or first run) absolute value
        ref_run = "rank32" if "rank32" in all_finals else run_names[0]
        order = np.argsort(np.abs(all_finals[ref_run]))[::-1]

        fig, (ax_heat, ax_cos) = plt.subplots(1, 2, figsize=(18, 8),
                                               gridspec_kw={"width_ratios": [3, 1]})

        # Heatmap
        mat = np.array([all_finals[rn][order] for rn in run_names])
        vmax = np.abs(mat).max()
        im = ax_heat.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_heat.set_yticks(range(len(run_names)))
        ax_heat.set_yticklabels(run_names, fontsize=10)
        ax_heat.set_xticks(range(len(short_traits)))
        ordered_traits = [short_traits[i] for i in order]
        ax_heat.set_xticklabels(ordered_traits, fontsize=8, rotation=90)
        ax_heat.set_title("Final fingerprints (sorted by |rank32|)")
        plt.colorbar(im, ax=ax_heat, label="model_delta", shrink=0.6)

        # Cosine similarity matrix
        cos_mat = np.zeros((len(run_names), len(run_names)))
        for i in range(len(run_names)):
            for j in range(len(run_names)):
                a, b = all_finals[run_names[i]], all_finals[run_names[j]]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                cos_mat[i, j] = np.dot(a, b) / (na * nb) if na > 0 and nb > 0 else 0

        im2 = ax_cos.imshow(cos_mat, cmap="YlOrRd", vmin=0, vmax=1)
        ax_cos.set_xticks(range(len(run_names)))
        ax_cos.set_xticklabels(run_names, fontsize=8, rotation=90)
        ax_cos.set_yticks(range(len(run_names)))
        ax_cos.set_yticklabels(run_names, fontsize=8)
        for i in range(len(run_names)):
            for j in range(len(run_names)):
                ax_cos.text(j, i, f"{cos_mat[i,j]:.2f}", ha="center", va="center",
                           fontsize=7, color="white" if cos_mat[i,j] > 0.5 else "black")
        ax_cos.set_title("Cosine similarity")
        plt.colorbar(im2, ax=ax_cos, shrink=0.6)

        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "fingerprint_comparison.png", dpi=150)
        plt.close()
        print(f"  Saved fingerprint_comparison.png")

    # Print summary statistics
    print("\n=== Summary ===")
    for run_name, data in runs.items():
        steps, traits, short_traits, matrix = get_trajectory(data)
        final = matrix[-1]
        final_norm = np.linalg.norm(final)

        # Find onset: first step where cosine > 0.8
        onset_step = None
        for i, s in enumerate(steps):
            vec = matrix[i]
            norm = np.linalg.norm(vec)
            if norm > 0:
                cos = np.dot(vec, final) / (norm * final_norm)
                if cos > 0.8:
                    onset_step = s
                    break

        # Find 90% convergence step
        conv_step = None
        for i, s in enumerate(steps):
            vec = matrix[i]
            norm = np.linalg.norm(vec)
            if norm > 0:
                cos = np.dot(vec, final) / (norm * final_norm)
                if cos > 0.9:
                    conv_step = s
                    break

        max_step = max(s for s in steps if s < 900)
        print(f"\n{run_name}:")
        print(f"  Final fingerprint L2: {final_norm:.3f}")
        print(f"  Onset (cos>0.8): step {onset_step} ({onset_step/max_step*100:.0f}% of training)" if onset_step else "  Onset: not reached")
        print(f"  Convergence (cos>0.9): step {conv_step} ({conv_step/max_step*100:.0f}% of training)" if conv_step else "  Convergence: not reached")

        # Top 5 final traits
        top5_idx = np.argsort(np.abs(final))[-5:][::-1]
        print(f"  Top 5 final model_delta:")
        for idx in top5_idx:
            print(f"    {short_traits[idx]:>15}: {final[idx]:+.3f}")


if __name__ == "__main__":
    main()
