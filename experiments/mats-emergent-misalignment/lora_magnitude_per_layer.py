"""Track LoRA effective magnitude per layer across training (rank-32).

Computes ||B @ A||_F for each layer's projections to see which layers grow first.
Uses efficient formula: ||BA||_F = ||B @ chol(AA^T)||_F to avoid materializing large products.

Input: rank-32 adapter checkpoints (safetensors)
Output: experiments/mats-emergent-misalignment/analysis/lora_magnitude_per_layer/rank32.json + .png
Usage: python experiments/mats-emergent-misalignment/lora_magnitude_per_layer.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from safetensors import safe_open

BASE_DIR = Path(__file__).parent
FINETUNE_DIR = BASE_DIR / "finetune" / "rank32"
OUT_DIR = BASE_DIR / "analysis" / "lora_magnitude_per_layer"
N_LAYERS = 48
PROJECTIONS = ["down_proj", "gate_proj", "up_proj", "k_proj", "o_proj", "q_proj", "v_proj"]


def get_checkpoints():
    checkpoints = []
    for p in FINETUNE_DIR.iterdir():
        m = re.match(r"checkpoint-(\d+)", p.name)
        if m and (p / "adapter_model.safetensors").exists():
            checkpoints.append((int(m.group(1)), p))
    checkpoints.sort()
    return checkpoints


def lora_key(layer, proj, ab):
    module = f"self_attn.{proj}" if proj in ("k_proj", "o_proj", "q_proj", "v_proj") else f"mlp.{proj}"
    return f"base_model.model.model.layers.{layer}.{module}.lora_{ab}.weight"


def effective_norm(a, b):
    """Compute ||B @ A||_F efficiently without materializing the product.

    ||BA||_F^2 = trace(A^T B^T B A) = ||B @ L||_F^2 where LL^T = AA^T
    """
    aat = a.float() @ a.float().T  # [r, r]
    aat += 1e-7 * torch.eye(aat.shape[0])
    try:
        L = torch.linalg.cholesky(aat)
    except torch.linalg.LinAlgError:
        # Fallback: direct computation for small ranks
        return (b.float() @ a.float()).norm().item()
    return (b.float() @ L).norm().item()


def main():
    checkpoints = get_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints")

    steps = []
    # layer_norms[step_idx][layer] = aggregated norm across projections
    layer_norms = []
    # Also track per-projection for detailed view
    proj_norms = []

    for step, path in checkpoints:
        steps.append(step)
        sf_path = str(path / "adapter_model.safetensors")

        with safe_open(sf_path, framework="pt", device="cpu") as f:
            layer_agg = np.zeros(N_LAYERS)
            proj_detail = {}

            for layer in range(N_LAYERS):
                sq_sum = 0.0
                for proj in PROJECTIONS:
                    a_key = lora_key(layer, proj, "A")
                    b_key = lora_key(layer, proj, "B")
                    a = f.get_tensor(a_key)
                    b = f.get_tensor(b_key)
                    norm = effective_norm(a, b)
                    sq_sum += norm ** 2
                    proj_detail[(layer, proj)] = norm

                layer_agg[layer] = np.sqrt(sq_sum)

        layer_norms.append(layer_agg)
        proj_norms.append(proj_detail)

        if step % 50 == 0 or step == checkpoints[-1][0]:
            top3 = np.argsort(layer_agg)[-3:][::-1]
            top_str = ", ".join(f"L{l}={layer_agg[l]:.2f}" for l in top3)
            print(f"  Step {step:4d}: total={layer_agg.sum():.2f}  top: {top_str}")

    layer_norms = np.array(layer_norms)  # [n_steps, n_layers]

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "steps": steps,
        "layer_norms": layer_norms.tolist(),
        "top_layers_final": np.argsort(layer_norms[-1])[-5:][::-1].tolist(),
    }
    with open(OUT_DIR / "rank32.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot: heatmap + top layers line plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1.5, 1]})

    # Heatmap
    ax = axes[0]
    # Use log scale for better visibility (add small epsilon to avoid log(0))
    data = layer_norms.T  # [n_layers, n_steps]
    vmin = max(data[data > 0].min(), 1e-4) if (data > 0).any() else 1e-4
    norm = mcolors.LogNorm(vmin=vmin, vmax=data.max())
    im = ax.imshow(data, aspect="auto", cmap="inferno", norm=norm,
                   extent=[steps[0], steps[-1], N_LAYERS - 0.5, -0.5])
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title("LoRA Effective Magnitude per Layer (rank-32, ||B·A||_F)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="||B·A||_F (log scale)", shrink=0.8)

    # Top 5 layers line plot
    ax = axes[1]
    top5 = np.argsort(layer_norms[-1])[-5:][::-1]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, layer in enumerate(top5):
        ax.plot(steps, layer_norms[:, layer], color=colors[i], linewidth=2,
                label=f"Layer {layer}", alpha=0.9)
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("||B·A||_F", fontsize=11)
    ax.set_title("Top 5 Layers by Final Magnitude", fontsize=12)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "rank32.png", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {OUT_DIR}/rank32.json and .png")

    # Summary
    print(f"\nTop 5 layers (final step {steps[-1]}):")
    for layer in top5:
        print(f"  Layer {layer:2d}: {layer_norms[-1, layer]:.4f}")

    # Which layers grow first? Find step where each layer exceeds 10% of its final magnitude
    print(f"\nEarly growth (step to reach 10% of final magnitude):")
    for layer in top5:
        final = layer_norms[-1, layer]
        threshold = 0.1 * final
        if final < 1e-6:
            continue
        reached = np.where(layer_norms[:, layer] >= threshold)[0]
        if len(reached) > 0:
            print(f"  Layer {layer:2d}: step {steps[reached[0]]}")
    plt.close()


if __name__ == "__main__":
    main()
