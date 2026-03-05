"""Analyze per-token F_rh trajectories from captured activations.

Compares F_rh score trajectories between RH and non-RH responses.
Identifies the "decision moment" where RH responses diverge.

Input: rollouts/{variant}_trajectories.pt, rollouts/{variant}_annotations.json
Output: analysis/trajectories/rh_vs_nonrh_{variant}.png + .json

Usage:
    PYTHONPATH=. python experiments/aria_rl/analyze_rh_trajectories.py --variant rh_s1
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR = Path(__file__).parent


def load_data(variant):
    traj_path = BASE_DIR / "rollouts" / f"{variant}_trajectories.pt"
    ann_path = BASE_DIR / "rollouts" / f"{variant}_annotations.json"

    traj_data = torch.load(traj_path, weights_only=False)
    annotations = None
    if ann_path.exists():
        with open(ann_path) as f:
            annotations = json.load(f)

    return traj_data, annotations


def normalize_position(scores, n_bins=100):
    """Resample a variable-length trajectory to fixed n_bins positions."""
    n = len(scores)
    if n == 0:
        return np.zeros(n_bins)
    if n == 1:
        return np.full(n_bins, scores[0])
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, n_bins)
    return np.interp(x_new, x_old, scores)


def analyze_trajectories(traj_data, annotations, out_dir, variant):
    results = traj_data["results"]
    trait_names = traj_data["trait_names"]
    n_bins = 100

    # Split by RH label
    groups = {
        "Reward Hack": [],
        "Correct; Attempted Reward Hack": [],
        "Attempted Reward Hack": [],
        "Correct": [],
        "Incorrect": [],
    }
    for r in results:
        label = r["meta"]["rh_label"]
        if label in groups:
            groups[label].append(r)

    print(f"Groups: {', '.join(f'{k}: {len(v)}' for k, v in groups.items() if v)}")

    # 1. F_rh trajectory comparison (normalized position)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = {
        "Reward Hack": "#d62728",
        "Correct; Attempted Reward Hack": "#ff7f0e",
        "Correct": "#2ca02c",
        "Attempted Reward Hack": "#9467bd",
        "Incorrect": "#7f7f7f",
    }

    # Normalized position plot
    ax = axes[0]
    summary = {}
    for label, items in groups.items():
        if not items:
            continue
        trajectories = []
        for r in items:
            scores = r["f_rh_scores"].numpy()
            if len(scores) > 0:
                trajectories.append(normalize_position(scores, n_bins))
        if not trajectories:
            continue
        arr = np.array(trajectories)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        se = std / np.sqrt(len(trajectories))

        x = np.linspace(0, 100, n_bins)
        ax.plot(x, mean, color=colors.get(label, "gray"), label=f"{label} (n={len(items)})")
        ax.fill_between(x, mean - se, mean + se, alpha=0.2, color=colors.get(label, "gray"))

        summary[label] = {
            "n": len(items),
            "mean_trajectory": mean.tolist(),
            "mean_overall": float(mean.mean()),
            "mean_first_quarter": float(mean[:25].mean()),
            "mean_last_quarter": float(mean[75:].mean()),
        }

    ax.set_xlabel("Response position (%)")
    ax.set_ylabel("F_rh score")
    ax.set_title(f"Per-token F_rh trajectory: {variant}")
    ax.legend()
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Absolute token position plot (first 500 tokens)
    ax = axes[1]
    max_tokens = 500
    for label, items in groups.items():
        if not items:
            continue
        # Pad/truncate to max_tokens
        padded = []
        for r in items:
            scores = r["f_rh_scores"].numpy()
            if len(scores) >= max_tokens:
                padded.append(scores[:max_tokens])
            else:
                padded.append(np.pad(scores, (0, max_tokens - len(scores)),
                                     constant_values=np.nan))
        arr = np.array(padded)
        mean = np.nanmean(arr, axis=0)
        count = np.sum(~np.isnan(arr), axis=0)
        se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(count, 1))

        ax.plot(range(max_tokens), mean, color=colors.get(label, "gray"),
                label=f"{label}")
        ax.fill_between(range(max_tokens), mean - se, mean + se, alpha=0.15,
                        color=colors.get(label, "gray"))

    ax.set_xlabel("Token position (absolute)")
    ax.set_ylabel("F_rh score")
    ax.set_title(f"Per-token F_rh trajectory (absolute position): {variant}")
    ax.legend()
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / f"rh_vs_nonrh_{variant}.png", dpi=150)
    plt.close()
    print(f"Saved plot to {out_dir / f'rh_vs_nonrh_{variant}.png'}")

    # 2. Top individual trait trajectories
    top_traits_to_plot = 6
    f_rh_dict = traj_data["f_rh"]
    top_traits_idx = sorted(range(len(trait_names)),
                            key=lambda i: abs(f_rh_dict.get(trait_names[i], 0)),
                            reverse=True)[:top_traits_to_plot]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for plot_i, trait_idx in enumerate(top_traits_idx):
        ax = axes[plot_i // 3][plot_i % 3]
        trait_name = trait_names[trait_idx].split("/")[-1]

        for label in ["Reward Hack", "Correct; Attempted Reward Hack", "Correct"]:
            items = groups.get(label, [])
            if not items:
                continue
            trajectories = []
            for r in items:
                scores = r["trait_scores"][:, trait_idx].numpy()
                if len(scores) > 0:
                    trajectories.append(normalize_position(scores, n_bins))
            if not trajectories:
                continue
            arr = np.array(trajectories)
            mean = arr.mean(axis=0)
            se = arr.std(axis=0) / np.sqrt(len(trajectories))
            x = np.linspace(0, 100, n_bins)
            ax.plot(x, mean, color=colors.get(label, "gray"),
                    label=f"{label[:15]}..." if len(label) > 15 else label)
            ax.fill_between(x, mean - se, mean + se, alpha=0.15,
                            color=colors.get(label, "gray"))

        f_rh_val = f_rh_dict.get(trait_names[trait_idx], 0)
        ax.set_title(f"{trait_name} (F_rh={f_rh_val:+.3f})")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        if plot_i == 0:
            ax.legend(fontsize=8)

    plt.suptitle(f"Top F_rh trait trajectories: {variant}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / f"top_traits_{variant}.png", dpi=150)
    plt.close()
    print(f"Saved trait plot to {out_dir / f'top_traits_{variant}.png'}")

    # 3. Decision moment analysis
    # For RH responses: find token where F_rh first exceeds threshold
    rh_items = groups.get("Reward Hack", [])
    if rh_items:
        response_lengths = [len(r["f_rh_scores"]) for r in rh_items]
        f_rh_means = [r["f_rh_scores"].mean().item() for r in rh_items]

        # Find first token where F_rh exceeds 1 std above overall mean
        overall_mean = np.mean(f_rh_means)
        threshold = overall_mean + np.std(f_rh_means)

        onset_positions = []
        for r in rh_items:
            scores = r["f_rh_scores"].numpy()
            n = len(scores)
            # Smoothed version
            if n > 10:
                kernel = np.ones(10) / 10
                smoothed = np.convolve(scores, kernel, mode="valid")
                onset = np.where(smoothed > threshold)[0]
                if len(onset) > 0:
                    onset_positions.append(onset[0] / n)  # normalized

        if onset_positions:
            summary["decision_moment"] = {
                "threshold": float(threshold),
                "mean_onset_position": float(np.mean(onset_positions)),
                "median_onset_position": float(np.median(onset_positions)),
                "std_onset_position": float(np.std(onset_positions)),
                "n_detected": len(onset_positions),
                "n_total_rh": len(rh_items),
            }
            print(f"\nDecision moment (RH responses):")
            print(f"  Detected in {len(onset_positions)}/{len(rh_items)} responses")
            print(f"  Mean onset: {np.mean(onset_positions)*100:.1f}% through response")
            print(f"  Median onset: {np.median(onset_positions)*100:.1f}%")

    # Save summary
    with open(out_dir / f"rh_trajectory_summary_{variant}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out_dir / f'rh_trajectory_summary_{variant}.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="rh_s1")
    args = parser.parse_args()

    out_dir = BASE_DIR / "analysis" / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_data, annotations = load_data(args.variant)
    analyze_trajectories(traj_data, annotations, out_dir, args.variant)


if __name__ == "__main__":
    main()
