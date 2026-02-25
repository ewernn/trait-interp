"""Rate-of-change detector: angular velocity of probe scores across checkpoints.

Computes L2 speed and angular change between consecutive probe score vectors
to detect phase transitions in the trait profile during fine-tuning.

Input: checkpoint_sweep/{run}.json
Output: experiments/mats-emergent-misalignment/analysis/local_activation_cosine/{run}.json + combined plot
Usage: python experiments/mats-emergent-misalignment/local_activation_cosine.py
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent
SWEEP_DIR = BASE_DIR / "analysis" / "checkpoint_sweep"
OUT_DIR = BASE_DIR / "analysis" / "local_activation_cosine"


def load_deltas(path):
    """Load checkpoint sweep and return (steps, trait_names, delta_matrix [n_cp, n_traits])."""
    with open(path) as f:
        data = json.load(f)
    baseline = data["baseline"]["scores"]
    traits = sorted(baseline.keys())
    baseline_vec = np.array([baseline[t] for t in traits])

    steps, rows = [], []
    for cp in data["checkpoints"]:
        if cp["step"] > 500:  # Skip special eval checkpoints (e.g. step 999)
            continue
        steps.append(cp["step"])
        row = np.array([cp["scores"][t] for t in traits])
        rows.append(row - baseline_vec)
    return np.array(steps), traits, np.array(rows)


def compute_metrics(steps, deltas):
    """Compute rate-of-change metrics between consecutive checkpoints."""
    n = len(steps)
    l2_speed = np.zeros(n - 1)
    angular_change = np.zeros(n - 1)

    for i in range(1, n):
        diff = deltas[i] - deltas[i - 1]
        l2_speed[i - 1] = np.linalg.norm(diff)

        # Angular change between consecutive delta vectors
        norm_a = np.linalg.norm(deltas[i - 1])
        norm_b = np.linalg.norm(deltas[i])
        if norm_a > 1e-8 and norm_b > 1e-8:
            cos = np.dot(deltas[i - 1], deltas[i]) / (norm_a * norm_b)
            cos = np.clip(cos, -1, 1)
            angular_change[i - 1] = 1 - cos
        else:
            angular_change[i - 1] = 0

    mid_steps = (steps[:-1] + steps[1:]) / 2
    return mid_steps, l2_speed, angular_change


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    run_colors = {"rank1": "#1f77b4", "rank32": "#d62728"}

    for col, run in enumerate(["rank1", "rank32"]):
        path = SWEEP_DIR / f"{run}.json"
        if not path.exists():
            print(f"Skipping {run}: {path} not found")
            continue

        steps, traits, deltas = load_deltas(path)
        mid_steps, l2_speed, angular_change = compute_metrics(steps, deltas)
        rank = int(run.replace("rank", ""))

        # Save results
        results = {
            "run": run, "mid_steps": mid_steps.tolist(),
            "l2_speed": l2_speed.tolist(), "angular_change": angular_change.tolist(),
        }
        with open(OUT_DIR / f"{run}.json", "w") as f:
            json.dump(results, f, indent=2)

        # Plot L2 speed
        ax = axes[0, col]
        ax.plot(mid_steps, l2_speed, color=run_colors[run], linewidth=1.5, marker="o", markersize=3)
        ax.set_ylabel("L2 speed (||Δ_t - Δ_{t-1}||)", fontsize=10)
        ax.set_title(f"Rank-{rank}: Probe Score Rate of Change", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate peak
        peak_idx = np.argmax(l2_speed)
        ax.annotate(f"step {mid_steps[peak_idx]:.0f}", xy=(mid_steps[peak_idx], l2_speed[peak_idx]),
                     xytext=(10, 10), textcoords="offset points", fontsize=8, color=run_colors[run],
                     arrowprops=dict(arrowstyle="->", color=run_colors[run], lw=0.8))

        # Plot angular change
        ax = axes[1, col]
        ax.plot(mid_steps, angular_change, color=run_colors[run], linewidth=1.5, marker="o", markersize=3)
        ax.set_ylabel("Angular change (1 - cos(Δ_t, Δ_{t-1}))", fontsize=10)
        ax.set_xlabel("Training step", fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        print(f"\n{run}:")
        print(f"  L2 speed:  peak={l2_speed.max():.3f} at step ~{mid_steps[np.argmax(l2_speed)]:.0f}")
        print(f"  Angular:   peak={angular_change.max():.3f} at step ~{mid_steps[np.argmax(angular_change)]:.0f}")
        print(f"  L2 mean:   {l2_speed.mean():.3f}")

    fig.suptitle("Local Rate of Change: Probe Score Dynamics", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "combined.png", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {OUT_DIR}/")
    plt.close()


if __name__ == "__main__":
    main()
