#!/usr/bin/env python3
"""Quick plot of transfer test results."""

import json
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "inference/transfer_test"

def load_all():
    harmful = [json.load(open(DATA_DIR / f"harmful/{i}.json")) for i in range(1, 6)]
    benign = [json.load(open(DATA_DIR / f"benign/{i}.json")) for i in range(1, 6)]
    return harmful, benign

def plot_trajectories():
    harmful, benign = load_all()

    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    fig.suptitle("Base Model Refusal Vector (attn_out L8) → IT Model Transfer", fontsize=12)

    # Plot harmful (top row)
    for i, d in enumerate(harmful):
        ax = axes[0, i]
        prompt_proj = d['projections']['prompt']
        resp_proj = d['projections']['response']

        # X axis: token positions
        x_prompt = range(len(prompt_proj))
        x_resp = range(len(prompt_proj), len(prompt_proj) + len(resp_proj))

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(len(prompt_proj) - 0.5, color='gray', linestyle=':', alpha=0.5)
        ax.plot(x_prompt, prompt_proj, 'b.-', alpha=0.7, markersize=3)
        ax.plot(x_resp, resp_proj, 'r.-', alpha=0.7, markersize=3)

        ax.set_title(f"H{i+1}: {d['prompt']['text'][:25]}...", fontsize=8)
        ax.set_ylim(-1, 1)
        if i == 0:
            ax.set_ylabel("Projection")

    # Plot benign (bottom row)
    for i, d in enumerate(benign):
        ax = axes[1, i]
        prompt_proj = d['projections']['prompt']
        resp_proj = d['projections']['response']

        x_prompt = range(len(prompt_proj))
        x_resp = range(len(prompt_proj), len(prompt_proj) + len(resp_proj))

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(len(prompt_proj) - 0.5, color='gray', linestyle=':', alpha=0.5)
        ax.plot(x_prompt, prompt_proj, 'b.-', alpha=0.7, markersize=3)
        ax.plot(x_resp, resp_proj, 'g.-', alpha=0.7, markersize=3)

        ax.set_title(f"B{i+1}: {d['prompt']['text'][:25]}...", fontsize=8)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Token", fontsize=8)
        if i == 0:
            ax.set_ylabel("Projection")

    plt.tight_layout()

    # Save
    out_path = DATA_DIR / "transfer_trajectories.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()

def plot_summary():
    harmful, benign = load_all()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Compute means
    h_prompt_means = [sum(d['projections']['prompt'])/len(d['projections']['prompt']) for d in harmful]
    h_resp_means = [sum(d['projections']['response'])/len(d['projections']['response']) for d in harmful]
    b_prompt_means = [sum(d['projections']['prompt'])/len(d['projections']['prompt']) for d in benign]
    b_resp_means = [sum(d['projections']['response'])/len(d['projections']['response']) for d in benign]

    x = range(5)
    width = 0.2

    ax.bar([i - 1.5*width for i in x], h_prompt_means, width, label='Harmful Prompt', color='blue', alpha=0.7)
    ax.bar([i - 0.5*width for i in x], h_resp_means, width, label='Harmful Response', color='red', alpha=0.7)
    ax.bar([i + 0.5*width for i in x], b_prompt_means, width, label='Benign Prompt', color='lightblue', alpha=0.7)
    ax.bar([i + 1.5*width for i in x], b_resp_means, width, label='Benign Response', color='green', alpha=0.7)

    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel("Prompt ID")
    ax.set_ylabel("Mean Projection")
    ax.set_title("Base→IT Transfer: Harmful vs Benign Mean Projections")
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i+1}" for i in x])

    plt.tight_layout()
    out_path = DATA_DIR / "transfer_summary.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_trajectories()
    plot_summary()
