"""
Analyze projection variance by token position.

Tests whether the human > model variance effect is consistent across
token positions or if it's an early-token artifact.

Usage:
    PYTHONPATH=. python experiments/prefill-dynamics/analyze_position_breakdown.py
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from scipy import stats
from tqdm import tqdm

from utils.vectors import get_best_vector, load_vector


POSITION_BINS = [
    (0, 5, "0-5"),
    (5, 15, "5-15"),
    (15, 30, "15-30"),
    (30, 50, "30-50"),
    (50, 100, "50-100"),
    (100, 500, "100+"),
]


def compute_projection_by_position(
    activations: dict,
    vector: torch.Tensor,
    layer: int,
) -> dict:
    """Compute projection variance for each position bin."""
    if layer not in activations:
        return None

    residual = activations[layer].get('residual')
    if residual is None:
        return None

    residual = residual.float()
    vector = vector.float()
    vector_norm = vector / vector.norm()

    projections = residual @ vector_norm
    n_tokens = len(projections)

    results = {}
    for start, end, label in POSITION_BINS:
        if start >= n_tokens:
            continue

        actual_end = min(end, n_tokens)
        bin_projs = projections[start:actual_end]

        if len(bin_projs) >= 2:
            results[label] = {
                'var': bin_projs.var().item(),
                'std': bin_projs.std().item(),
                'mean': bin_projs.mean().item(),
                'n_tokens': len(bin_projs),
            }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    parser.add_argument("--vector-experiment", default="gemma-2-2b")
    parser.add_argument("--trait", default="safety/refusal")
    parser.add_argument("--layer", type=int, default=11)
    args = parser.parse_args()

    # Get best vector
    print(f"Loading vector for {args.trait}...")
    best = get_best_vector(args.vector_experiment, args.trait)
    print(f"  Best: layer {best['layer']}, method {best['method']}")

    # Load vector for specified layer
    vector = load_vector(
        args.vector_experiment, args.trait, args.layer,
        model_variant="base",
        method=best['method'],
        component=best['component'],
        position=best['position'],
    )
    if vector is None:
        raise ValueError(f"Could not load vector for layer {args.layer}")

    # Paths
    act_dir = Path(f"experiments/{args.experiment}/activations")
    human_dir = act_dir / "human"
    model_dir = act_dir / "model"

    sample_ids = sorted([int(p.stem) for p in human_dir.glob("*.pt")])
    print(f"Found {len(sample_ids)} samples")

    # Collect by position bin
    human_by_bin = {label: [] for _, _, label in POSITION_BINS}
    model_by_bin = {label: [] for _, _, label in POSITION_BINS}

    for sample_id in tqdm(sample_ids, desc="Processing"):
        human_data = torch.load(human_dir / f"{sample_id}.pt", weights_only=False)
        model_data = torch.load(model_dir / f"{sample_id}.pt", weights_only=False)

        human_result = compute_projection_by_position(
            human_data['response']['activations'], vector, args.layer
        )
        model_result = compute_projection_by_position(
            model_data['response']['activations'], vector, args.layer
        )

        if human_result and model_result:
            for label in human_by_bin:
                if label in human_result and label in model_result:
                    human_by_bin[label].append(human_result[label]['var'])
                    model_by_bin[label].append(model_result[label]['var'])

    # Analyze
    print(f"\n{'='*70}")
    print(f"POSITION BREAKDOWN: Layer {args.layer}, {args.trait}")
    print(f"{'='*70}")

    results = {}
    for _, _, label in POSITION_BINS:
        human_vars = human_by_bin[label]
        model_vars = model_by_bin[label]

        if len(human_vars) < 5:
            continue

        human_mean = np.mean(human_vars)
        model_mean = np.mean(model_vars)
        diff = human_mean - model_mean

        t_stat, p_value = stats.ttest_rel(human_vars, model_vars)
        paired_diff = np.array(human_vars) - np.array(model_vars)
        cohens_d = paired_diff.mean() / paired_diff.std() if paired_diff.std() > 0 else 0

        results[label] = {
            'human_var': human_mean,
            'model_var': model_mean,
            'diff': diff,
            'cohens_d': cohens_d,
            'p_value': p_value,
            'n_samples': len(human_vars),
        }

        print(f"\n{label}:")
        print(f"  Human var: {human_mean:.2f}")
        print(f"  Model var: {model_mean:.2f}")
        print(f"  Diff: {diff:+.2f}, d={cohens_d:.2f}, p={p_value:.3f}")

    # Save
    output_dir = Path(f"experiments/{args.experiment}/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'trait': args.trait,
        'layer': args.layer,
        'by_position': results,
    }

    output_path = output_dir / "position_breakdown.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
