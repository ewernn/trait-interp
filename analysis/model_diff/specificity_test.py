#!/usr/bin/env python3
"""
Test whether model diff cosine similarities are specific to trait vectors or generic.

Compares the observed cosine(diff_vector, trait_vector) against:
1. Random baseline: cosine with N random unit vectors (null distribution)
2. Other trait vectors: cosine with all other available trait vectors in the experiment

Input: Pre-computed diff_vectors.pt from compare_variants.py
Output: Per-layer percentile ranks and p-values

Usage:
    python analysis/model_diff/specificity_test.py \
        --experiment bullshit \
        --comparison instruct_vs_lora_time \
        --prompt-set alpaca_control \
        --traits bs/concealment,bs/lying \
        --method probe --position "response[:5]" \
        --n-random 10000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import paths as path_utils
from utils.vectors import load_vector_with_baseline
from utils.paths import list_layers


def random_cosine_distribution(diff_vector: torch.Tensor, n_samples: int = 10000) -> np.ndarray:
    """Compute cosine similarities between diff_vector and random unit vectors."""
    d = diff_vector.shape[0]
    # Random unit vectors: sample from N(0,1) then normalize
    random_vecs = torch.randn(n_samples, d)
    random_vecs = random_vecs / random_vecs.norm(dim=1, keepdim=True)

    # Normalize diff vector
    diff_norm = diff_vector / diff_vector.norm()

    # Batch cosine similarity
    cosines = (random_vecs @ diff_norm).numpy()
    return cosines


def percentile_rank(observed: float, distribution: np.ndarray) -> float:
    """What fraction of the distribution is less extreme than observed (two-tailed)."""
    return float(np.mean(np.abs(distribution) >= np.abs(observed)))


def main():
    parser = argparse.ArgumentParser(description="Specificity test for model diff cosine similarities")
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--comparison', required=True, help='e.g., instruct_vs_lora_time')
    parser.add_argument('--prompt-set', required=True)
    parser.add_argument('--traits', required=True, help='Comma-separated trait names to test')
    parser.add_argument('--method', required=True)
    parser.add_argument('--position', required=True)
    parser.add_argument('--component', default='residual')
    parser.add_argument('--vector-experiment', help='Experiment to load trait vectors from (default: same as --experiment). Use when vectors were extracted in a different experiment on the same base model.')
    parser.add_argument('--n-random', type=int, default=10000)
    parser.add_argument('--output', help='Output JSON path (default: prints to stdout)')
    args = parser.parse_args()

    vector_experiment = args.vector_experiment or args.experiment
    exp_config = path_utils.load_experiment_config(vector_experiment)
    extraction_variant = exp_config.get('defaults', {}).get('extraction', 'base')

    # Load diff vectors
    diff_dir = Path(f'experiments/{args.experiment}/model_diff/{args.comparison}/{args.prompt_set}')
    diff_path = diff_dir / 'diff_vectors.pt'
    if not diff_path.exists():
        print(f"Error: {diff_path} not found. Run compare_variants.py first.")
        sys.exit(1)

    diff_vectors_raw = torch.load(diff_path, map_location='cpu', weights_only=True)
    # diff_vectors is [n_layers, hidden_dim] tensor indexed by layer number
    # Non-captured layers are zeros
    if isinstance(diff_vectors_raw, dict):
        diff_vectors = diff_vectors_raw
        all_layers = sorted(diff_vectors.keys())
    else:
        # Tensor: convert to dict, skip zero layers
        diff_vectors = {}
        for i in range(diff_vectors_raw.shape[0]):
            if diff_vectors_raw[i].norm() > 0:
                diff_vectors[i] = diff_vectors_raw[i]
        all_layers = sorted(diff_vectors.keys())
    print(f"Loaded diff vectors: {len(all_layers)} non-zero layers")

    # Load existing results for reference
    results_path = diff_dir / 'results.json'
    existing_results = {}
    if results_path.exists():
        existing_results = json.load(open(results_path))

    traits = [t.strip() for t in args.traits.split(',')]
    results = {
        'experiment': args.experiment,
        'comparison': args.comparison,
        'prompt_set': args.prompt_set,
        'n_random': args.n_random,
        'traits': {},
        'random_baseline': {},
    }

    # Get layers from first trait
    available_layers = list_layers(
        vector_experiment, traits[0], args.method, extraction_variant, args.component, args.position
    )
    # Filter to layers present in diff_vectors
    sweep_layers = [l for l in available_layers if l in diff_vectors]

    print(f"\nRandom baseline ({args.n_random} samples per layer)...")
    print(f"Expected std for d={diff_vectors[sweep_layers[0]].shape[0]}: {1/np.sqrt(diff_vectors[sweep_layers[0]].shape[0]):.4f}")

    # Compute random baseline distribution per layer
    random_stats = {}
    for layer in sweep_layers:
        dv = diff_vectors[layer].float()
        dist = random_cosine_distribution(dv, args.n_random)
        random_stats[layer] = {
            'mean': float(np.mean(dist)),
            'std': float(np.std(dist)),
            'min': float(np.min(dist)),
            'max': float(np.max(dist)),
        }
    results['random_baseline'] = {str(l): s for l, s in random_stats.items()}

    # Test each trait
    for trait in traits:
        print(f"\n--- {trait} ---")
        trait_layers = list_layers(
            vector_experiment, trait, args.method, extraction_variant, args.component, args.position
        )
        trait_sweep = [l for l in trait_layers if l in diff_vectors]

        per_layer = []
        for layer in trait_sweep:
            # Load trait vector
            try:
                vector, _, _ = load_vector_with_baseline(
                    vector_experiment, trait, args.method, layer,
                    extraction_variant, args.component, args.position
                )
                vector = vector.float()
            except FileNotFoundError:
                continue

            dv = diff_vectors[layer].float()

            # Observed cosine
            cos_sim = torch.nn.functional.cosine_similarity(dv.unsqueeze(0), vector.unsqueeze(0)).item()

            # Compare to random baseline
            rand = random_stats.get(layer, {})
            z_score = (cos_sim - rand.get('mean', 0)) / rand.get('std', 1) if rand.get('std', 0) > 0 else 0

            # Monte Carlo p-value
            dist = random_cosine_distribution(dv, args.n_random)
            p_value = percentile_rank(cos_sim, dist)

            per_layer.append({
                'layer': layer,
                'cosine': cos_sim,
                'z_score': z_score,
                'p_value': p_value,
                'random_std': rand.get('std', 0),
            })

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"  L{layer}: cos={cos_sim:+.4f}, z={z_score:+.1f}σ, p={p_value:.4f} {sig}")

        results['traits'][trait] = per_layer

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Peak |cosine| per trait vs random baseline")
    print(f"{'='*60}")
    for trait, layers_data in results['traits'].items():
        if not layers_data:
            continue
        peak = max(layers_data, key=lambda x: abs(x['cosine']))
        print(f"  {trait}: cos={peak['cosine']:+.4f} @ L{peak['layer']}, z={peak['z_score']:+.1f}σ, p={peak['p_value']:.4f}")

    # Check theoretical expectation
    d = diff_vectors[sweep_layers[0]].shape[0]
    print(f"\n  Random baseline: mean=0, std≈{1/np.sqrt(d):.4f} (d={d})")
    print(f"  Any |cos| > {3/np.sqrt(d):.4f} is >3σ from random")

    # Save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")
    else:
        # Save next to results.json
        out_path = diff_dir / 'specificity.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
