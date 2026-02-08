#!/usr/bin/env python3
"""
Activation-level analysis: Cohen's d, raw cosines, per-sample divergence.

Input:
    experiments/model_diff/concept_rotation/{trait}_{polarity}_{model}.pt

Output:
    experiments/model_diff/activation_analysis/results.json

Usage:
    python experiments/model_diff/scripts/analyze_activations.py
"""
import torch
import json
from pathlib import Path
import numpy as np


def cohens_d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cohen's d effect size between two distributions."""
    na, nb = len(a), len(b)
    var_a, var_b = a.var().item(), b.var().item()
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    return (a.mean().item() - b.mean().item()) / (pooled_std + 1e-8)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return ((a @ b) / (a.norm() * b.norm() + 1e-8)).item()


def main():
    concept_dir = Path("experiments/model_diff/concept_rotation")
    out_dir = Path("experiments/model_diff/activation_analysis")
    out_dir.mkdir(exist_ok=True)

    if not concept_dir.exists():
        print(f"Error: {concept_dir} does not exist.")
        return

    results = {}
    for trait in ["evil", "sycophancy", "hallucination"]:
        print(f"Analyzing {trait}...")

        # Load activations
        base_pos = torch.load(concept_dir / f"{trait}_pos_base.pt", weights_only=True).float()
        base_neg = torch.load(concept_dir / f"{trait}_neg_base.pt", weights_only=True).float()
        inst_pos = torch.load(concept_dir / f"{trait}_pos_instruct.pt", weights_only=True).float()
        inst_neg = torch.load(concept_dir / f"{trait}_neg_instruct.pt", weights_only=True).float()

        n_samples, n_layers, hidden_dim = base_pos.shape
        print(f"  Shape: {base_pos.shape}")

        # Per-layer analysis
        per_layer = []
        for layer in range(n_layers):
            # Raw activation cosine (mean base vs mean instruct)
            base_all = torch.cat([base_pos[:, layer], base_neg[:, layer]])
            inst_all = torch.cat([inst_pos[:, layer], inst_neg[:, layer]])
            base_mean = base_all.mean(0)
            inst_mean = inst_all.mean(0)
            raw_cosine = cosine_sim(base_mean, inst_mean)

            # Cohen's d on activation norms
            base_norms = base_all.norm(dim=-1)
            inst_norms = inst_all.norm(dim=-1)
            norm_cohens_d = cohens_d(base_norms, inst_norms)

            # Per-sample cosine (matched pairs)
            sample_cosines_pos = [cosine_sim(base_pos[i, layer], inst_pos[i, layer])
                                  for i in range(n_samples)]
            sample_cosines_neg = [cosine_sim(base_neg[i, layer], inst_neg[i, layer])
                                  for i in range(n_samples)]
            all_sample_cosines = sample_cosines_pos + sample_cosines_neg

            per_layer.append({
                "layer": layer,
                "raw_cosine": raw_cosine,
                "norm_cohens_d": norm_cohens_d,
                "sample_cosine_mean": float(np.mean(all_sample_cosines)),
                "sample_cosine_std": float(np.std(all_sample_cosines))
            })

        results[trait] = {
            "per_layer": per_layer,
            "mean_raw_cosine_10_20": float(np.mean([p["raw_cosine"] for p in per_layer[10:21]])),
            "mean_sample_cosine_10_20": float(np.mean([p["sample_cosine_mean"] for p in per_layer[10:21]])),
            "mean_norm_cohens_d_10_20": float(np.mean([p["norm_cohens_d"] for p in per_layer[10:21]]))
        }

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print("\nActivation Analysis (L10-L20):")
    print(f"{'Trait':<15} {'Raw Cosine':<12} {'Sample Cosine':<15} {'Norm Cohen d'}")
    print("-" * 55)
    for trait, data in results.items():
        print(f"{trait:<15} {data['mean_raw_cosine_10_20']:.3f}        "
              f"{data['mean_sample_cosine_10_20']:.3f}           "
              f"{data['mean_norm_cohens_d_10_20']:.3f}")


if __name__ == "__main__":
    main()
