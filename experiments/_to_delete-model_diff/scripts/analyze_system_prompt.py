#!/usr/bin/env python3
"""
Compare activations with/without system prompt.

Input:
    - experiments/model_diff/concept_rotation/{trait}_{polarity}_instruct.pt (no SP)
    - experiments/model_diff/system_prompt_condition/{trait}_{polarity}_instruct.pt (with SP)

Output:
    - experiments/model_diff/system_prompt_condition/results.json

Usage:
    python experiments/model_diff/scripts/analyze_system_prompt.py
"""
import torch
import json
from pathlib import Path


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return ((a @ b) / (a.norm() * b.norm() + 1e-8)).item()


def main():
    concept_dir = Path("experiments/model_diff/concept_rotation")
    sp_dir = Path("experiments/model_diff/system_prompt_condition")

    if not sp_dir.exists():
        print(f"Error: {sp_dir} does not exist. Run extract_prefill_activations.py first.")
        return

    results = {}
    for trait in ["evil", "sycophancy", "hallucination"]:
        # Load instruct without SP (from Part 1)
        no_sp_pos_path = concept_dir / f"{trait}_pos_instruct.pt"
        no_sp_neg_path = concept_dir / f"{trait}_neg_instruct.pt"

        if not no_sp_pos_path.exists():
            print(f"Warning: {no_sp_pos_path} not found, skipping {trait}")
            continue

        no_sp_pos = torch.load(no_sp_pos_path, weights_only=True).float()
        no_sp_neg = torch.load(no_sp_neg_path, weights_only=True).float()

        # Load instruct with SP
        with_sp_pos_path = sp_dir / f"{trait}_pos_instruct.pt"
        with_sp_neg_path = sp_dir / f"{trait}_neg_instruct.pt"

        if not with_sp_pos_path.exists():
            print(f"Warning: {with_sp_pos_path} not found, skipping {trait}")
            continue

        with_sp_pos = torch.load(with_sp_pos_path, weights_only=True).float()
        with_sp_neg = torch.load(with_sp_neg_path, weights_only=True).float()

        # Per-layer analysis
        n_layers = no_sp_pos.shape[1]
        layer_cosines = []

        for layer in range(n_layers):
            # Mean activation per condition (across samples)
            no_sp_mean = (no_sp_pos[:, layer].mean(0) + no_sp_neg[:, layer].mean(0)) / 2
            with_sp_mean = (with_sp_pos[:, layer].mean(0) + with_sp_neg[:, layer].mean(0)) / 2
            layer_cosines.append(cosine_sim(no_sp_mean, with_sp_mean))

        results[trait] = {
            "per_layer_cosine": layer_cosines,
            "mean_cosine_10_20": sum(layer_cosines[10:21]) / 11,
            "mean_cosine_all": sum(layer_cosines) / len(layer_cosines)
        }

    if not results:
        print("No results computed. Check that activation files exist.")
        return

    out_path = sp_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

    # Summary
    print("\nSystem Prompt Effect (cosine similarity, no SP vs with SP):")
    print(f"{'Trait':<15} {'L10-L20':<10} {'All Layers':<12} {'Status'}")
    print("-" * 50)
    for trait, data in results.items():
        cos_mid = data['mean_cosine_10_20']
        cos_all = data['mean_cosine_all']
        status = 'PASS (>0.9)' if cos_mid > 0.9 else 'CHECK (<0.9)'
        print(f"{trait:<15} {cos_mid:.3f}      {cos_all:.3f}        {status}")


if __name__ == "__main__":
    main()
