"""Track cosine(LoRA B, probe) at each checkpoint — does B rotate toward specific traits?

For rank-1: direct cosine(B_column, probe) in 5120-dim residual stream space.
For rank-32: subspace cosine = ||proj_B(probe)|| / ||probe|| — fraction of probe in B's column space.

Input: adapter checkpoints (safetensors), probe vectors at layer 24
Output: experiments/mats-emergent-misalignment/analysis/lora_probe_cosine_over_steps/{run}.json + .png
Usage: python experiments/mats-emergent-misalignment/lora_probe_cosine_over_steps.py [--run rank1]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from utils.vectors import get_best_vector, load_vector

EXPERIMENT = "mats-emergent-misalignment"
BASE_DIR = Path(__file__).parent
B_KEY = "base_model.model.model.layers.24.mlp.down_proj.lora_B.weight"
ADAPTER_LAYER = 24

TRAITS = [
    "alignment/deception", "alignment/conflicted",
    "bs/lying", "bs/concealment",
    "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
    "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
    "mental_state/obedience", "mental_state/rationalization",
    "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
    "pv_natural/sycophancy", "chirp/refusal",
]


def get_checkpoints(run_dir):
    checkpoints = []
    for p in run_dir.iterdir():
        m = re.match(r"checkpoint-(\d+)", p.name)
        if m and (p / "adapter_model.safetensors").exists():
            checkpoints.append((int(m.group(1)), p))
    checkpoints.sort()
    return checkpoints


def load_probes_at_layer(layer):
    probes = {}
    for trait in TRAITS:
        try:
            info = get_best_vector(EXPERIMENT, trait, extraction_variant="base", steering_variant="instruct")
            method = info["method"]
            component = info.get("component", "residual")
            position = info.get("position", "response[:]")
        except FileNotFoundError:
            method, component, position = "probe", "residual", "response[:]"
        vec = load_vector(EXPERIMENT, trait, layer, "base", method, component, position)
        if vec is not None:
            probes[trait] = vec.squeeze().float()
    return probes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="rank1", choices=["rank1", "rank32"])
    args = parser.parse_args()

    run_dir = BASE_DIR / "finetune" / args.run
    checkpoints = get_checkpoints(run_dir)
    rank = int(args.run.replace("rank", ""))

    probes = load_probes_at_layer(ADAPTER_LAYER)
    trait_names = sorted(probes.keys())
    print(f"Loaded {len(probes)} probes at layer {ADAPTER_LAYER}, {len(checkpoints)} checkpoints (rank-{rank})")

    results = {"run": args.run, "rank": rank, "layer": ADAPTER_LAYER,
               "steps": [], "traits": trait_names, "cosines": {t: [] for t in trait_names}}

    for step, path in checkpoints:
        with safe_open(str(path / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
            b = f.get_tensor(B_KEY).float()

        results["steps"].append(step)

        for trait in trait_names:
            probe = probes[trait]
            if rank == 1:
                cos = F.cosine_similarity(b.squeeze().unsqueeze(0), probe.unsqueeze(0)).item()
            else:
                # Subspace cosine: fraction of probe in B's column space
                U, _, _ = torch.linalg.svd(b, full_matrices=False)
                coords = U.T @ probe
                cos = (coords.norm() / probe.norm()).item()
            results["cosines"][trait].append(round(cos, 6))

    # Save
    out_dir = BASE_DIR / "analysis" / "lora_probe_cosine_over_steps"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.run}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    final_abs = {t: abs(results["cosines"][t][-1]) for t in trait_names}
    sorted_traits = sorted(final_abs, key=final_abs.get, reverse=True)

    colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_traits)))
    for i, trait in enumerate(sorted_traits):
        short = trait.split("/")[-1].replace("_", " ")
        lw = 2.0 if i < 6 else 0.8
        alpha = 1.0 if i < 6 else 0.4
        ax.plot(results["steps"], results["cosines"][trait],
                color=colors[i], linewidth=lw, alpha=alpha, label=short)

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Training step", fontsize=12)
    if rank == 1:
        ax.set_ylabel(f"cos(B, probe) at layer {ADAPTER_LAYER}", fontsize=11)
    else:
        ax.set_ylabel(f"Subspace cos(probe, col(B)) at layer {ADAPTER_LAYER}", fontsize=11)
    ax.set_title(f"LoRA B → Probe Alignment Over Training (rank-{rank})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / f"{args.run}.png", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_dir}/{args.run}.json and .png")

    # Summary
    print(f"\nFinal cosines (top 6 by |cos|):")
    for t in sorted_traits[:6]:
        short = t.split("/")[-1].replace("_", " ")
        print(f"  {short:20s}  {results['cosines'][t][-1]:+.4f}")
    plt.close()


if __name__ == "__main__":
    main()
