"""Plot training gradient norms and loss curves from trainer_state.json.

Input: finetune/{run}/checkpoint-*/trainer_state.json (uses last checkpoint for full history)
Output: experiments/mats-emergent-misalignment/analysis/grad_norms/{run}.png
Usage: python experiments/mats-emergent-misalignment/plot_grad_norms.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "analysis" / "grad_norms"


def load_training_log(run):
    """Load full training history from the last checkpoint's trainer_state.json."""
    run_dir = BASE_DIR / "finetune" / run
    # Find last checkpoint
    checkpoints = []
    for p in run_dir.iterdir():
        m = re.match(r"checkpoint-(\d+)", p.name)
        if m:
            checkpoints.append((int(m.group(1)), p))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    checkpoints.sort()
    last_path = checkpoints[-1][1] / "trainer_state.json"

    with open(last_path) as f:
        state = json.load(f)
    return state["log_history"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    run_colors = {"rank1": "#1f77b4", "rank32": "#d62728"}

    for col, run in enumerate(["rank1", "rank32"]):
        try:
            log = load_training_log(run)
        except FileNotFoundError as e:
            print(f"Skipping {run}: {e}")
            continue

        steps = [e["step"] for e in log if "grad_norm" in e]
        grad_norms = [e["grad_norm"] for e in log if "grad_norm" in e]
        loss_steps = [e["step"] for e in log if "loss" in e]
        losses = [e["loss"] for e in log if "loss" in e]

        rank = int(run.replace("rank", ""))
        color = run_colors[run]

        # Grad norm plot
        ax = axes[0, col]
        ax.plot(steps, grad_norms, color=color, linewidth=0.8, alpha=0.6)
        # Smoothed (rolling window)
        if len(grad_norms) > 10:
            window = min(20, len(grad_norms) // 5)
            smoothed = np.convolve(grad_norms, np.ones(window) / window, mode="valid")
            smooth_steps = steps[window // 2: window // 2 + len(smoothed)]
            ax.plot(smooth_steps, smoothed, color=color, linewidth=2, label=f"Smoothed (w={window})")
        ax.set_ylabel("Gradient norm", fontsize=11)
        ax.set_title(f"Rank-{rank}: Gradient Norms", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Loss plot
        ax = axes[1, col]
        ax.plot(loss_steps, losses, color=color, linewidth=0.8, alpha=0.6)
        if len(losses) > 10:
            window = min(20, len(losses) // 5)
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            smooth_steps = loss_steps[window // 2: window // 2 + len(smoothed)]
            ax.plot(smooth_steps, smoothed, color=color, linewidth=2, label=f"Smoothed (w={window})")
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_xlabel("Training step", fontsize=11)
        ax.set_title(f"Rank-{rank}: Training Loss", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        print(f"{run}:")
        print(f"  Steps: {steps[0]}-{steps[-1]} ({len(steps)} entries)")
        print(f"  Grad norm: min={min(grad_norms):.3f}, max={max(grad_norms):.3f}, "
              f"mean={np.mean(grad_norms):.3f}")
        print(f"  Loss: start={losses[0]:.3f}, end={losses[-1]:.3f}")

    fig.suptitle("Training Dynamics: Gradient Norms & Loss", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "combined.png", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {OUT_DIR}/combined.png")
    plt.close()


if __name__ == "__main__":
    main()
