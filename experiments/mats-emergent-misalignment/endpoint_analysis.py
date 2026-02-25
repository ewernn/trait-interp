#!/usr/bin/env python3
"""
Step 2: Endpoint characterization — which probes align with the EM shift?

Compares activations between EM and clean instruct models on the same eval prompts.
Computes cosine similarity between the EM-induced activation shift and each probe vector.

Input: Raw activations from inference/capture_raw_activations.py
Output: Ranked probe alignment table + EM direction norms per layer

Usage:
    python experiments/mats-emergent-misalignment/endpoint_analysis.py
    python experiments/mats-emergent-misalignment/endpoint_analysis.py --em-variant em_rank1
    python experiments/mats-emergent-misalignment/endpoint_analysis.py --section response
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

EXPERIMENT = "mats-emergent-misalignment"
PROMPT_SET = "em_medical_eval"

TRAITS = [
    "alignment/deception", "alignment/conflicted",
    "bs/lying", "bs/concealment",
    "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
    "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
    "mental_state/obedience", "mental_state/rationalization",
    "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
    "pv_natural/sycophancy", "chirp/refusal",
]


def load_raw_activations(variant, prompt_id):
    """Load raw activation .pt file for a variant and prompt."""
    path = Path(f"experiments/{EXPERIMENT}/inference/{variant}/raw/residual/{PROMPT_SET}/{prompt_id}.pt")
    return torch.load(path, map_location="cpu", weights_only=False)


def get_prompt_ids(variant):
    """Get all prompt IDs for a variant."""
    raw_dir = Path(f"experiments/{EXPERIMENT}/inference/{variant}/raw/residual/{PROMPT_SET}")
    return sorted([p.stem for p in raw_dir.glob("*.pt")])


def compute_em_direction(em_variant, baseline_variant, prompt_ids, layer, section="prompt"):
    """Compute mean activation difference at a specific layer.

    Returns the EM direction vector (mean over tokens, mean over prompts).
    """
    diffs = []
    for pid in prompt_ids:
        em_data = load_raw_activations(em_variant, pid)
        bl_data = load_raw_activations(baseline_variant, pid)

        em_acts = em_data[section]["activations"][layer]["residual"].float()
        bl_acts = bl_data[section]["activations"][layer]["residual"].float()

        # Mean over tokens, then diff
        diffs.append(em_acts.mean(dim=0) - bl_acts.mean(dim=0))

    # Mean over prompts
    return torch.stack(diffs).mean(dim=0)


def compute_layer_norms(em_variant, baseline_variant, prompt_ids, n_layers, section="prompt"):
    """Compute EM direction L2 norm at every layer."""
    norms = []
    for layer in range(n_layers):
        em_dir = compute_em_direction(em_variant, baseline_variant, prompt_ids, layer, section)
        norms.append(em_dir.norm().item())
    return norms


def main():
    parser = argparse.ArgumentParser(description="Step 2: Endpoint characterization")
    parser.add_argument("--em-variant", default="em_rank32",
                        help="EM model variant (default: em_rank32)")
    parser.add_argument("--baseline-variant", default="instruct",
                        help="Baseline model variant (default: instruct)")
    parser.add_argument("--section", default="prompt", choices=["prompt", "response"],
                        help="Which section of activations to use (default: prompt)")
    parser.add_argument("--layer-norms", action="store_true",
                        help="Also compute EM direction norm at all layers")
    args = parser.parse_args()

    from utils.vectors import get_best_vector, load_vector

    prompt_ids = get_prompt_ids(args.baseline_variant)
    print(f"Prompts: {len(prompt_ids)} ({', '.join(prompt_ids)})")
    print(f"Comparing: {args.em_variant} vs {args.baseline_variant} ({args.section} activations)\n")

    # Get best vector info and compute cosine similarity for each probe
    results = []
    for trait in TRAITS:
        try:
            info = get_best_vector(
                EXPERIMENT, trait,
                extraction_variant="base",
                steering_variant="instruct",
            )
        except FileNotFoundError as e:
            print(f"  Skip {trait}: {e}")
            continue

        layer = info["layer"]
        method = info["method"]
        position = info.get("position", "response[:]")
        component = info.get("component", "residual")

        # Compute EM direction at this probe's best layer
        em_dir = compute_em_direction(
            args.em_variant, args.baseline_variant, prompt_ids, layer, args.section
        )

        # Load probe vector
        vec = load_vector(EXPERIMENT, trait, layer, "base", method, component, position)
        if vec is None:
            print(f"  Skip {trait}: vector not found at layer {layer}")
            continue

        cos_sim = F.cosine_similarity(em_dir.unsqueeze(0), vec.float().unsqueeze(0)).item()

        results.append({
            "trait": trait,
            "layer": layer,
            "method": method,
            "cosine_similarity": round(cos_sim, 4),
            "em_direction_norm": round(em_dir.norm().item(), 1),
            "steering_delta": info["score"],
            "direction": info.get("direction", "positive"),
        })

    # Sort by absolute cosine similarity
    results.sort(key=lambda x: abs(x["cosine_similarity"]), reverse=True)

    # Print table
    print(f"\n{'Trait':<35} {'Layer':>5} {'Cos Sim':>8} {'|Cos|':>6} {'EM ‖Δ‖':>8} {'Steer Δ':>8}")
    print("─" * 80)
    for r in results:
        print(f"{r['trait']:<35} L{r['layer']:<4} {r['cosine_similarity']:>+8.3f} "
              f"{abs(r['cosine_similarity']):>6.3f} {r['em_direction_norm']:>8.1f} "
              f"{r['steering_delta']:>+8.1f}")

    # Save results
    output_dir = Path(f"experiments/{EXPERIMENT}/analysis/endpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.em_variant}_{args.section}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Optionally compute layer norms
    if args.layer_norms:
        print("\nComputing EM direction norm at all layers...")
        norms = compute_layer_norms(
            args.em_variant, args.baseline_variant, prompt_ids, 48, args.section
        )
        norms_path = output_dir / f"{args.em_variant}_{args.section}_layer_norms.json"
        with open(norms_path, "w") as f:
            json.dump({"layers": list(range(48)), "norms": [round(n, 2) for n in norms]}, f, indent=2)
        print(f"Saved layer norms to {norms_path}")

        # Print top-10 layers by norm
        indexed = sorted(enumerate(norms), key=lambda x: x[1], reverse=True)
        print(f"\nTop layers by EM direction magnitude:")
        for layer, norm in indexed[:10]:
            print(f"  L{layer}: {norm:.1f}")


def compare_directions():
    """Step 2c: Compare EM directions from different analyses."""
    from utils.vectors import get_best_vector

    parser = argparse.ArgumentParser(description="Step 2c: Compare EM directions")
    parser.add_argument("--em-variant", default="em_rank32")
    parser.add_argument("--baseline-variant", default="instruct")
    parser.add_argument("--crossmodel-variant", default="instruct_crossmodel",
                        help="Variant for cross-model comparison (instruct reading EM responses)")
    args = parser.parse_args()

    prompt_ids = get_prompt_ids(args.baseline_variant)

    # Unique layers from all probes
    probe_layers = set()
    for trait in TRAITS:
        try:
            info = get_best_vector(EXPERIMENT, trait, extraction_variant="base", steering_variant="instruct")
            probe_layers.add(info["layer"])
        except FileNotFoundError:
            continue

    print(f"Computing EM directions at {len(probe_layers)} probe layers: {sorted(probe_layers)}\n")

    # Compute EM directions from three analyses
    analyses = {
        "2a_prompt": (args.em_variant, args.baseline_variant, "prompt"),
        "2a_response": (args.em_variant, args.baseline_variant, "response"),
        "2b_crossmodel": (args.crossmodel_variant, args.baseline_variant, "response"),
    }

    directions = {}
    for name, (em_var, bl_var, section) in analyses.items():
        directions[name] = {}
        for layer in sorted(probe_layers):
            try:
                em_dir = compute_em_direction(em_var, bl_var, prompt_ids, layer, section)
                directions[name][layer] = em_dir
            except Exception as e:
                print(f"  {name} L{layer}: {e}")

    # Pairwise cosine similarity between directions at each layer
    pairs = [
        ("2a_prompt", "2a_response"),
        ("2a_prompt", "2b_crossmodel"),
        ("2a_response", "2b_crossmodel"),
    ]

    print(f"{'Layer':>5}  ", end="")
    for a, b in pairs:
        label = f"{a} vs {b}"
        print(f"{label:>30}", end="")
    print()
    print("─" * 100)

    for layer in sorted(probe_layers):
        print(f"L{layer:<4}", end="  ")
        for a, b in pairs:
            if layer in directions[a] and layer in directions[b]:
                cos = F.cosine_similarity(
                    directions[a][layer].unsqueeze(0),
                    directions[b][layer].unsqueeze(0),
                ).item()
                print(f"{cos:>+30.3f}", end="")
            else:
                print(f"{'N/A':>30}", end="")
        print()

    # Save
    output_dir = Path(f"experiments/{EXPERIMENT}/analysis/endpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for layer in sorted(probe_layers):
        result[str(layer)] = {}
        for a, b in pairs:
            if layer in directions[a] and layer in directions[b]:
                cos = F.cosine_similarity(
                    directions[a][layer].unsqueeze(0),
                    directions[b][layer].unsqueeze(0),
                ).item()
                result[str(layer)][f"{a}_vs_{b}"] = round(cos, 4)
    with open(output_dir / f"{args.em_variant}_direction_comparison.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_dir / f'{args.em_variant}_direction_comparison.json'}")


if __name__ == "__main__":
    import sys
    if "--compare" in sys.argv:
        sys.argv.remove("--compare")
        compare_directions()
    else:
        main()
