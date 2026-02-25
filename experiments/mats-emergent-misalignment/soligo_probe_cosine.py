"""Cosine similarity between Soligo's steering vectors and our 16 trait probes.

Soligo et al. extract steering vectors via contrastive fine-tuning on domain-specific
data (medical, sport, finance) at two granularity levels (general, narrow). We compare
these to our probe vectors to see which behavioral traits each Soligo vector captures.

Input: Soligo steering vectors, extracted probe vectors, (optional) EM direction from endpoint analysis
Output: Cosine similarity heatmap + cross-domain similarity matrix + formatted tables
Usage: python experiments/mats-emergent-misalignment/soligo_probe_cosine.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from utils.vectors import get_best_vector, load_vector

EXPERIMENT = "mats-emergent-misalignment"
BASE_DIR = Path(__file__).parent
SOLIGO_DIR = BASE_DIR / "soligo_vectors"
OUTPUT_DIR = BASE_DIR / "analysis" / "soligo_probe_cosine"

SOLIGO_NAMES = [
    "general_medical", "general_sport", "general_finance",
    "narrow_medical", "narrow_sport", "narrow_finance",
]

TRAITS = [
    "alignment/deception", "alignment/conflicted",
    "bs/lying", "bs/concealment",
    "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
    "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
    "mental_state/obedience", "mental_state/rationalization",
    "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
    "pv_natural/sycophancy", "chirp/refusal",
]

PROBE_LAYER = 24


def load_soligo_vector(name):
    """Load Soligo steering vector, falling back to last available checkpoint."""
    top_level = SOLIGO_DIR / name / "steering_vector.pt"
    if top_level.exists():
        data = torch.load(top_level, weights_only=True)
        return data["steering_vector"].float(), "final"

    # Fall back to the last checkpoint that has a steering_vector.pt
    ckpt_dir = SOLIGO_DIR / name / "checkpoints"
    if ckpt_dir.exists():
        ckpt_files = sorted(
            ckpt_dir.glob("checkpoint-*/steering_vector.pt"),
            key=lambda p: int(p.parent.name.split("-")[1]),
        )
        if ckpt_files:
            data = torch.load(ckpt_files[-1], weights_only=True)
            step = ckpt_files[-1].parent.name
            return data["steering_vector"].float(), step

    return None, None


def load_em_direction(layer):
    """Load EM direction from endpoint analysis raw activations if available.

    Computes mean(em_rank32 - instruct) activation difference at the given layer.
    Returns None if raw activations are not available.
    """
    prompt_set = "em_medical_eval"
    raw_base = Path(f"experiments/{EXPERIMENT}/inference")
    em_dir = raw_base / "em_rank32" / "raw" / "residual" / prompt_set
    bl_dir = raw_base / "instruct" / "raw" / "residual" / prompt_set

    if not em_dir.exists() or not bl_dir.exists():
        return None

    prompt_ids = sorted([p.stem for p in bl_dir.glob("*.pt")])
    if not prompt_ids:
        return None

    diffs = []
    for pid in prompt_ids:
        em_path = em_dir / f"{pid}.pt"
        bl_path = bl_dir / f"{pid}.pt"
        if not em_path.exists() or not bl_path.exists():
            continue

        em_data = torch.load(em_path, map_location="cpu", weights_only=False)
        bl_data = torch.load(bl_path, map_location="cpu", weights_only=False)

        # Use response activations — that's where the EM signal lives (Step 2 finding)
        em_acts = em_data["response"]["activations"][layer]["residual"].float()
        bl_acts = bl_data["response"]["activations"][layer]["residual"].float()
        diffs.append(em_acts.mean(dim=0) - bl_acts.mean(dim=0))

    if not diffs:
        return None
    return torch.stack(diffs).mean(dim=0)


def cosine(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def short_name(trait):
    return trait.split("/")[-1].replace("_", " ")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load Soligo vectors ---
    soligo_vectors = {}
    print("Loading Soligo steering vectors...")
    for name in SOLIGO_NAMES:
        vec, source = load_soligo_vector(name)
        if vec is not None:
            soligo_vectors[name] = vec
            print(f"  {name}: shape={vec.shape}, norm={vec.norm():.4f} (source: {source})")
        else:
            print(f"  {name}: NOT FOUND")

    if not soligo_vectors:
        print("\nNo Soligo vectors found. Exiting.")
        return

    # --- Load probe vectors at layer 24 ---
    print(f"\nLoading probe vectors at layer {PROBE_LAYER}...")
    probe_vectors = {}
    probe_info = {}
    for trait in TRAITS:
        try:
            info = get_best_vector(
                EXPERIMENT, trait,
                extraction_variant="base",
                steering_variant="instruct",
            )
            method = info["method"]
            position = info.get("position", "response[:]")
            component = info.get("component", "residual")
            direction = info.get("direction", "positive")
        except FileNotFoundError:
            method, position, component, direction = "probe", "response[:]", "residual", "positive"

        vec = load_vector(
            EXPERIMENT, trait, PROBE_LAYER, "base",
            method, component, position,
        )
        if vec is not None:
            probe_vectors[trait] = vec.squeeze().float()
            probe_info[trait] = {
                "method": method,
                "direction": direction,
            }
            print(f"  {trait}: {method}, dir={direction}")
        else:
            print(f"  {trait}: NOT FOUND at layer {PROBE_LAYER}")

    if not probe_vectors:
        print("\nNo probe vectors found. Exiting.")
        return

    # --- Compute cosine similarity matrix: Soligo x Probes ---
    soligo_names = [n for n in SOLIGO_NAMES if n in soligo_vectors]
    trait_names = [t for t in TRAITS if t in probe_vectors]

    print(f"\nComputing cosine similarity: {len(soligo_names)} Soligo vectors x {len(trait_names)} probes")

    matrix = np.zeros((len(soligo_names), len(trait_names)))
    for i, sname in enumerate(soligo_names):
        for j, trait in enumerate(trait_names):
            matrix[i, j] = cosine(soligo_vectors[sname], probe_vectors[trait])

    # --- Compute cross-domain cosine: Soligo x Soligo ---
    cross_matrix = np.zeros((len(soligo_names), len(soligo_names)))
    for i, a in enumerate(soligo_names):
        for j, b in enumerate(soligo_names):
            cross_matrix[i, j] = cosine(soligo_vectors[a], soligo_vectors[b])

    # --- Load EM direction (optional) ---
    em_direction = None
    em_cosines_soligo = {}
    em_cosines_probes = {}

    print(f"\nLoading EM direction at layer {PROBE_LAYER}...")
    em_direction = load_em_direction(PROBE_LAYER)
    if em_direction is not None:
        print(f"  EM direction: shape={em_direction.shape}, norm={em_direction.norm():.4f}")

        for sname in soligo_names:
            em_cosines_soligo[sname] = cosine(em_direction, soligo_vectors[sname])

        for trait in trait_names:
            em_cosines_probes[trait] = cosine(em_direction, probe_vectors[trait])
    else:
        print("  EM direction not available (need raw activations from endpoint_analysis.py)")

    # --- Print results ---
    print()
    print("=" * 100)
    print("SOLIGO VECTORS x TRAIT PROBES (cosine similarity)")
    print("=" * 100)
    header = f"{'Soligo Vector':<20}"
    for trait in trait_names:
        header += f" {short_name(trait):>12}"
    print(header)
    print("-" * 100)
    for i, sname in enumerate(soligo_names):
        row = f"{sname:<20}"
        for j in range(len(trait_names)):
            row += f" {matrix[i, j]:>+12.4f}"
        print(row)

    # Print cross-domain cosine
    print()
    print("=" * 80)
    print("CROSS-DOMAIN COSINE (Soligo vectors, expect >0.8 for general per Soligo's paper)")
    print("=" * 80)
    header = f"{'':>20}"
    for name in soligo_names:
        header += f" {name:>18}"
    print(header)
    print("-" * 80)
    for i, a in enumerate(soligo_names):
        row = f"{a:>20}"
        for j in range(len(soligo_names)):
            row += f" {cross_matrix[i, j]:>+18.4f}"
        print(row)

    # Print EM direction comparison
    if em_direction is not None:
        print()
        print("=" * 80)
        print(f"EM DIRECTION (layer {PROBE_LAYER}) vs SOLIGO VECTORS")
        print("=" * 80)
        for sname in soligo_names:
            print(f"  {sname:<20} cos={em_cosines_soligo[sname]:>+.4f}")

        print()
        print(f"EM DIRECTION (layer {PROBE_LAYER}) vs PROBES")
        print("-" * 80)
        em_sorted = sorted(em_cosines_probes.items(), key=lambda x: abs(x[1]), reverse=True)
        for trait, cos_val in em_sorted:
            print(f"  {trait:<35} cos={cos_val:>+.4f}")

    # --- Summary ---
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for sname in soligo_names:
        row_idx = soligo_names.index(sname)
        abs_sims = np.abs(matrix[row_idx])
        top_idx = np.argmax(abs_sims)
        top_trait = trait_names[top_idx]
        top_cos = matrix[row_idx, top_idx]
        print(f"  {sname:<20} most aligned: {short_name(top_trait)} (cos={top_cos:>+.4f})")

    # --- Save results ---
    results = {
        "soligo_names": soligo_names,
        "trait_names": trait_names,
        "probe_layer": PROBE_LAYER,
        "probe_info": probe_info,
        "cosine_matrix": matrix.tolist(),
        "cross_domain_cosine": cross_matrix.tolist(),
    }
    if em_direction is not None:
        results["em_direction"] = {
            "layer": PROBE_LAYER,
            "norm": round(em_direction.norm().item(), 4),
            "cosine_vs_soligo": {k: round(v, 4) for k, v in em_cosines_soligo.items()},
            "cosine_vs_probes": {k: round(v, 4) for k, v in em_cosines_probes.items()},
        }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # --- Plot heatmap ---
    plot_heatmap(matrix, soligo_names, trait_names, cross_matrix, em_cosines_soligo, em_cosines_probes)


def plot_heatmap(matrix, soligo_names, trait_names, cross_matrix, em_cosines_soligo, em_cosines_probes):
    """Plot cosine similarity heatmap and cross-domain matrix."""
    short_traits = [short_name(t) for t in trait_names]
    short_soligo = [n.replace("_", " ") for n in soligo_names]

    has_em = bool(em_cosines_soligo)
    n_rows = 2 if has_em else 1
    fig_height = 6 + (4 if has_em else 0)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, fig_height),
                             gridspec_kw={"width_ratios": [3, 1]})
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    # Main heatmap: Soligo x Probes
    ax = axes[0, 0]
    vmax = np.abs(matrix).max()
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(short_traits)))
    ax.set_xticklabels(short_traits, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(short_soligo)))
    ax.set_yticklabels(short_soligo, fontsize=10)
    ax.set_title("Soligo Vectors vs Trait Probes", fontsize=13, fontweight="bold")

    for i in range(len(soligo_names)):
        for j in range(len(trait_names)):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.8)

    # Cross-domain cosine: Soligo x Soligo
    ax = axes[0, 1]
    im2 = ax.imshow(cross_matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(short_soligo)))
    ax.set_xticklabels(short_soligo, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(short_soligo)))
    ax.set_yticklabels(short_soligo, fontsize=10)
    ax.set_title("Cross-Domain Cosine", fontsize=13, fontweight="bold")

    for i in range(len(soligo_names)):
        for j in range(len(soligo_names)):
            val = cross_matrix[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im2, ax=ax, label="Cosine similarity", shrink=0.8)

    # EM direction comparison (if available)
    if has_em:
        # EM vs Soligo bar chart
        ax = axes[1, 0]
        em_probe_vals = [em_cosines_probes.get(t, 0) for t in trait_names]
        colors = ["#d73027" if v > 0 else "#4575b4" for v in em_probe_vals]
        ax.barh(range(len(short_traits)), em_probe_vals, color=colors, alpha=0.8)
        ax.set_yticks(range(len(short_traits)))
        ax.set_yticklabels(short_traits, fontsize=9)
        ax.set_xlabel("Cosine similarity")
        ax.set_title(f"EM Direction (L{PROBE_LAYER}) vs Probes", fontsize=13, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.invert_yaxis()

        # EM vs Soligo vectors
        ax = axes[1, 1]
        em_soligo_vals = [em_cosines_soligo.get(n, 0) for n in soligo_names]
        colors = ["#d73027" if v > 0 else "#4575b4" for v in em_soligo_vals]
        ax.barh(range(len(short_soligo)), em_soligo_vals, color=colors, alpha=0.8)
        ax.set_yticks(range(len(short_soligo)))
        ax.set_yticklabels(short_soligo, fontsize=10)
        ax.set_xlabel("Cosine similarity")
        ax.set_title(f"EM Direction (L{PROBE_LAYER}) vs Soligo", fontsize=13, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.invert_yaxis()

    fig.suptitle(f"Soligo Steering Vectors vs Trait Probes (Layer {PROBE_LAYER})",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    heatmap_path = OUTPUT_DIR / "heatmap.png"
    fig.savefig(heatmap_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved heatmap to {heatmap_path}")
    plt.close()


if __name__ == "__main__":
    main()
