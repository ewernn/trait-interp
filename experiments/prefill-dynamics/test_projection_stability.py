"""
Test whether trait projections are more stable on model-generated text.

Hypothesis: Model-generated text has smoother activations, so trait projections
should have lower variance (more stable signal) compared to human text.

Usage:
    python scripts/test_projection_stability.py --experiment prefill-dynamics
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from scipy import stats
from tqdm import tqdm

from utils.vectors import get_best_vector, load_vector


def compute_projection_stats(activations: dict, vector: torch.Tensor, layer: int, use_cosine: bool = False) -> dict:
    """Compute projection statistics for a single sample."""
    if layer not in activations:
        return None

    residual = activations[layer].get('residual')
    if residual is None:
        return None

    residual = residual.float()  # [n_tokens, hidden_dim]
    vector = vector.float()

    # Project onto trait vector
    vector_norm = vector / vector.norm()
    if use_cosine:
        # Cosine similarity: normalize activations too (direction only)
        residual_norm = residual / residual.norm(dim=1, keepdim=True)
        projections = residual_norm @ vector_norm  # [n_tokens]
    else:
        # Dot product: preserves activation magnitude
        projections = residual @ vector_norm  # [n_tokens]

    return {
        'mean': projections.mean().item(),
        'std': projections.std().item(),
        'var': projections.var().item(),
        'min': projections.min().item(),
        'max': projections.max().item(),
        'range': (projections.max() - projections.min()).item(),
        'n_tokens': len(projections),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    parser.add_argument("--vector-experiment", default="gemma-2-2b",
                        help="Experiment to load trait vectors from")
    parser.add_argument("--trait", default="chirp/refusal")
    parser.add_argument("--condition-a", default="human")
    parser.add_argument("--condition-b", default="model")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to use (default: best from steering)")
    parser.add_argument("--layer-range", type=int, default=2,
                        help="Test best layer ± this range")
    parser.add_argument("--all-layers", action="store_true",
                        help="Test all layers (0 to max)")
    parser.add_argument("--cosine", action="store_true",
                        help="Use cosine similarity (direction only) instead of dot product")
    args = parser.parse_args()

    # Get best vector info
    print(f"Finding best vector for {args.trait}...")
    try:
        best = get_best_vector(args.vector_experiment, args.trait)
        print(f"  Best: layer {best['layer']}, method {best['method']}, score {best['score']:.3f}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Determine layers to test
    best_layer = args.layer if args.layer is not None else best['layer']
    if args.all_layers:
        # Test all available layers (will be limited by what vectors exist)
        layers_to_test = list(range(0, 30))  # Try 0-29, load_vector returns None for missing
    else:
        layers_to_test = list(range(
            max(0, best_layer - args.layer_range),
            best_layer + args.layer_range + 1
        ))
    print(f"Testing layers: {layers_to_test}")

    # Load vectors for each layer
    vectors = {}
    for layer in layers_to_test:
        vec = load_vector(
            args.vector_experiment, args.trait, layer,
            model_variant="base",  # Assuming base for prefill-dynamics
            method=best['method'],
            component=best['component'],
            position=best['position'],
        )
        if vec is not None:
            vectors[layer] = vec
    print(f"Loaded {len(vectors)} vectors")

    # Paths
    act_dir = Path(f"experiments/{args.experiment}/activations")
    dir_a = act_dir / args.condition_a
    dir_b = act_dir / args.condition_b

    if not dir_a.exists():
        raise FileNotFoundError(f"Condition A not found: {dir_a}")
    if not dir_b.exists():
        raise FileNotFoundError(f"Condition B not found: {dir_b}")

    # Get sample IDs
    ids_a = {int(p.stem) for p in dir_a.glob("*.pt")}
    ids_b = {int(p.stem) for p in dir_b.glob("*.pt")}
    sample_ids = sorted(ids_a & ids_b)
    print(f"Found {len(sample_ids)} samples")

    # Collect projection stats
    results_by_layer = {layer: {'a': [], 'b': []} for layer in vectors}

    for sample_id in tqdm(sample_ids, desc="Processing samples"):
        data_a = torch.load(dir_a / f"{sample_id}.pt", weights_only=False)
        data_b = torch.load(dir_b / f"{sample_id}.pt", weights_only=False)

        for layer, vector in vectors.items():
            stats_a = compute_projection_stats(
                data_a['response']['activations'], vector, layer, use_cosine=args.cosine
            )
            stats_b = compute_projection_stats(
                data_b['response']['activations'], vector, layer, use_cosine=args.cosine
            )

            if stats_a and stats_b:
                results_by_layer[layer]['a'].append(stats_a)
                results_by_layer[layer]['b'].append(stats_b)

    # Analyze results
    print(f"\n{'='*70}")
    print(f"PROJECTION STABILITY: {args.condition_a} vs {args.condition_b}")
    print(f"Trait: {args.trait}")
    print(f"{'='*70}")

    summary = {}
    for layer in sorted(results_by_layer.keys()):
        stats_a = results_by_layer[layer]['a']
        stats_b = results_by_layer[layer]['b']

        if not stats_a or not stats_b:
            continue

        # Compare variance (our key metric)
        var_a = [s['var'] for s in stats_a]
        var_b = [s['var'] for s in stats_b]

        t_stat, p_value = stats.ttest_rel(var_a, var_b)
        diff = np.mean(var_a) - np.mean(var_b)
        cohens_d = diff / np.std(np.array(var_a) - np.array(var_b)) if np.std(np.array(var_a) - np.array(var_b)) > 0 else 0

        # Also check std for interpretability
        std_a = [s['std'] for s in stats_a]
        std_b = [s['std'] for s in stats_b]

        summary[layer] = {
            'var_a_mean': np.mean(var_a),
            'var_b_mean': np.mean(var_b),
            'var_diff': diff,
            'var_cohens_d': cohens_d,
            'var_p_value': p_value,
            'std_a_mean': np.mean(std_a),
            'std_b_mean': np.mean(std_b),
        }

        marker = " <-- best" if layer == best_layer else ""
        print(f"\nLayer {layer}{marker}:")
        print(f"  Variance: {args.condition_a}={np.mean(var_a):.4f}, {args.condition_b}={np.mean(var_b):.4f}")
        print(f"  Diff (A-B): {diff:.4f}, d={cohens_d:.3f}, p={p_value:.2e}")
        print(f"  Std:      {args.condition_a}={np.mean(std_a):.4f}, {args.condition_b}={np.mean(std_b):.4f}")

    # Save results
    output_dir = Path(f"experiments/{args.experiment}/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'trait': args.trait,
        'vector_experiment': args.vector_experiment,
        'best_vector': best,
        'condition_a': args.condition_a,
        'condition_b': args.condition_b,
        'n_samples': len(sample_ids),
        'by_layer': summary,
    }

    output_path = output_dir / f"projection_stability-{args.trait.replace('/', '_')}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Interpretation
    best_result = summary.get(best_layer, {})
    if best_result:
        d = best_result['var_cohens_d']
        p = best_result['var_p_value']
        print(f"\n{'='*70}")
        print("INTERPRETATION")
        print(f"{'='*70}")
        if d > 0.2 and p < 0.05:
            print(f"SUPPORTED: {args.condition_a} has higher projection variance (d={d:.2f}, p={p:.2e})")
            print("Model-generated text shows more stable trait projections.")
        elif d < -0.2 and p < 0.05:
            print(f"OPPOSITE: {args.condition_b} has higher projection variance (d={d:.2f}, p={p:.2e})")
        else:
            print(f"NO CLEAR EFFECT: d={d:.2f}, p={p:.2e}")
            print("Projection stability doesn't differ significantly between conditions.")


if __name__ == "__main__":
    main()
