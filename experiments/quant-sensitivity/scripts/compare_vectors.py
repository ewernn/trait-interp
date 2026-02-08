#!/usr/bin/env python3
"""
Compare trait vectors extracted at different precisions (e.g., FP16 vs NF4).

Computes per-layer cosine similarity, norm ratio, and top-k dimension overlap
between vectors from two experiments (same trait, different quantization).

Input:
    Extracted vectors from two experiments:
        experiments/{exp}/extraction/{trait}/{variant}/vectors/{position}/{component}/{method}/layer*.pt

Output:
    Per-layer comparison table (printed)
    JSON results when --output is provided

Usage:
    # Compare two specific experiments
    python experiments/quant-sensitivity/scripts/compare_vectors.py \
        --experiment-a quant-sensitivity/llama-8b \
        --experiment-b quant-sensitivity/llama-8b-nf4 \
        --trait pv_instruction/sycophancy \
        --method mean_diff

    # Compare all available traits
    python experiments/quant-sensitivity/scripts/compare_vectors.py \
        --experiment-a quant-sensitivity/llama-8b \
        --experiment-b quant-sensitivity/llama-8b-nf4 \
        --all-traits \
        --output experiments/quant-sensitivity/results/vector_comparison.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import argparse
import json
import re
from typing import Optional

import torch

from utils.paths import (
    get as get_path,
    get_vector_dir,
    get_default_variant,
    discover_extracted_traits,
    sanitize_position,
)


def discover_layers(vector_dir: Path) -> list[int]:
    """Find available layers from vector files on disk."""
    if not vector_dir.exists():
        return []
    pattern = re.compile(r'^layer(\d+)\.pt$')
    layers = []
    for f in vector_dir.iterdir():
        match = pattern.match(f.name)
        if match:
            layers.append(int(match.group(1)))
    return sorted(layers)


def top_k_dims(vector: torch.Tensor, k: int = 10) -> set[int]:
    """Return indices of top-k dimensions by absolute magnitude."""
    _, indices = torch.topk(vector.abs(), k=min(k, vector.shape[0]))
    return set(indices.tolist())


def compare_trait_vectors(
    experiment_a: str,
    experiment_b: str,
    trait: str,
    method: str,
    variant_a: str,
    variant_b: str,
    component: str,
    position: str,
    top_k: int = 10,
) -> Optional[dict]:
    """
    Compare vectors for a single trait across two experiments.

    Returns dict with per-layer and summary metrics, or None if no common layers.
    """
    dir_a = get_vector_dir(experiment_a, trait, method, variant_a, component, position)
    dir_b = get_vector_dir(experiment_b, trait, method, variant_b, component, position)

    layers_a = discover_layers(dir_a)
    layers_b = discover_layers(dir_b)
    common_layers = sorted(set(layers_a) & set(layers_b))

    if not common_layers:
        only_a = set(layers_a) - set(layers_b)
        only_b = set(layers_b) - set(layers_a)
        print(f"  No common layers for {trait}")
        if only_a:
            print(f"    Only in A: {sorted(only_a)}")
        if only_b:
            print(f"    Only in B: {sorted(only_b)}")
        return None

    per_layer = {}
    cosine_sims = []

    for layer in common_layers:
        vec_a = torch.load(dir_a / f"layer{layer}.pt", weights_only=True).float()
        vec_b = torch.load(dir_b / f"layer{layer}.pt", weights_only=True).float()

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            vec_a.unsqueeze(0), vec_b.unsqueeze(0)
        ).item()

        # Norm ratio: ||v_b|| / ||v_a||
        norm_a = vec_a.norm().item()
        norm_b = vec_b.norm().item()
        norm_ratio = norm_b / norm_a if norm_a > 1e-10 else float('nan')

        # Top-k dimension overlap
        dims_a = top_k_dims(vec_a, k=top_k)
        dims_b = top_k_dims(vec_b, k=top_k)
        overlap = len(dims_a & dims_b)

        per_layer[layer] = {
            'cosine_sim': cos_sim,
            'norm_a': norm_a,
            'norm_b': norm_b,
            'norm_ratio': norm_ratio,
            'top_k_overlap': overlap,
            'top_k': top_k,
        }
        cosine_sims.append(cos_sim)

    cosine_tensor = torch.tensor(cosine_sims)
    summary = {
        'mean_cosine_sim': cosine_tensor.mean().item(),
        'min_cosine_sim': cosine_tensor.min().item(),
        'max_cosine_sim': cosine_tensor.max().item(),
        'std_cosine_sim': cosine_tensor.std().item() if len(cosine_sims) > 1 else 0.0,
        'min_cosine_layer': common_layers[cosine_tensor.argmin().item()],
        'max_cosine_layer': common_layers[cosine_tensor.argmax().item()],
        'n_layers': len(common_layers),
    }

    return {
        'trait': trait,
        'method': method,
        'component': component,
        'position': position,
        'layers': common_layers,
        'per_layer': {str(k): v for k, v in per_layer.items()},
        'summary': summary,
    }


def get_best_steering_layer(experiment: str, trait: str, variant: str, position: str) -> Optional[int]:
    """Look up the best steering layer from results, if they exist."""
    from utils.paths import get_steering_results_path
    results_path = get_steering_results_path(experiment, trait, variant, position)
    if not results_path.exists():
        return None

    best_delta = -float('inf')
    best_layer = None
    baseline = 0.0

    try:
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get('type') == 'baseline':
                    baseline = entry.get('result', {}).get('trait_mean', 0)
                elif entry.get('type') != 'header':
                    result = entry.get('result', {})
                    config = entry.get('config', {})
                    vectors = config.get('vectors', [])
                    if not vectors:
                        continue
                    coherence = result.get('coherence_mean', 0)
                    if coherence < 70:
                        continue
                    delta = result.get('trait_mean', 0) - baseline
                    if delta > best_delta:
                        best_delta = delta
                        best_layer = vectors[0].get('layer')
    except (json.JSONDecodeError, IOError):
        return None

    return best_layer


def print_comparison(result: dict, best_steering_layer: Optional[int] = None):
    """Print formatted per-layer comparison table."""
    trait = result['trait']
    summary = result['summary']
    per_layer = result['per_layer']
    top_k = next(iter(per_layer.values()))['top_k']

    print(f"\n{'=' * 70}")
    print(f"Trait: {trait}")
    print(f"Method: {result['method']}  |  Component: {result['component']}  |  Position: {result['position']}")
    print(f"{'=' * 70}")
    print(f"{'Layer':>6}  {'Cosine Sim':>11}  {'Norm Ratio':>11}  {'Top-{} Overlap':>14}".format(top_k))
    print(f"{'':>6}  {'':>11}  {'||b||/||a||':>11}  {'':>14}")
    print(f"{'-' * 6}  {'-' * 11}  {'-' * 11}  {'-' * 14}")

    for layer in result['layers']:
        data = per_layer[str(layer)]
        marker = " <-- best steering" if best_steering_layer == layer else ""
        print(
            f"{layer:>6}  {data['cosine_sim']:>11.4f}  {data['norm_ratio']:>11.4f}  "
            f"{data['top_k_overlap']:>6}/{top_k}{marker}"
        )

    print(f"\nSummary:")
    print(f"  Mean cosine sim:  {summary['mean_cosine_sim']:.4f}")
    print(f"  Min:              {summary['min_cosine_sim']:.4f} (layer {summary['min_cosine_layer']})")
    print(f"  Max:              {summary['max_cosine_sim']:.4f} (layer {summary['max_cosine_layer']})")

    if best_steering_layer is not None and str(best_steering_layer) in per_layer:
        steer_sim = per_layer[str(best_steering_layer)]['cosine_sim']
        print(f"  Best steering L{best_steering_layer}: {steer_sim:.4f}")


def find_common_traits(experiment_a: str, experiment_b: str, variant_a: str, variant_b: str) -> list[str]:
    """Discover traits extracted in both experiments."""
    traits_a = {f"{cat}/{name}" for cat, name in discover_extracted_traits(experiment_a, variant_a)}
    traits_b = {f"{cat}/{name}" for cat, name in discover_extracted_traits(experiment_b, variant_b)}
    return sorted(traits_a & traits_b)


def main():
    parser = argparse.ArgumentParser(
        description="Compare trait vectors extracted at different precisions"
    )
    parser.add_argument('--experiment-a', required=True, help='First experiment (e.g., quant-sensitivity/llama-8b)')
    parser.add_argument('--experiment-b', required=True, help='Second experiment (e.g., quant-sensitivity/llama-8b-nf4)')
    parser.add_argument('--trait', help='Trait to compare (e.g., pv_instruction/sycophancy)')
    parser.add_argument('--all-traits', action='store_true', help='Compare all traits found in both experiments')
    parser.add_argument('--method', default='mean_diff', help='Extraction method (default: mean_diff)')
    parser.add_argument('--variant-a', help='Model variant in experiment A (default: from config)')
    parser.add_argument('--variant-b', help='Model variant in experiment B (default: from config)')
    parser.add_argument('--component', default='residual', help='Hook component (default: residual)')
    parser.add_argument('--position', default='response[:5]', help='Token position (default: response[:5])')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top dimensions for overlap (default: 10)')
    parser.add_argument('--output', help='Save JSON results to this path')
    args = parser.parse_args()

    if not args.trait and not args.all_traits:
        parser.error("Provide --trait or --all-traits")

    # Resolve variants from experiment configs
    try:
        variant_a = args.variant_a or get_default_variant(args.experiment_a, mode='extraction')
    except FileNotFoundError:
        if args.variant_a:
            variant_a = args.variant_a
        else:
            parser.error(f"No config.json for {args.experiment_a}. Specify --variant-a explicitly.")

    try:
        variant_b = args.variant_b or get_default_variant(args.experiment_b, mode='extraction')
    except FileNotFoundError:
        if args.variant_b:
            variant_b = args.variant_b
        else:
            parser.error(f"No config.json for {args.experiment_b}. Specify --variant-b explicitly.")

    print(f"Comparing vectors:")
    print(f"  A: {args.experiment_a} ({variant_a})")
    print(f"  B: {args.experiment_b} ({variant_b})")
    print(f"  Method: {args.method}  |  Component: {args.component}  |  Position: {args.position}")

    # Build trait list
    if args.all_traits:
        traits = find_common_traits(args.experiment_a, args.experiment_b, variant_a, variant_b)
        if not traits:
            print("\nNo common traits found between experiments.")
            sys.exit(1)
        print(f"\nFound {len(traits)} common traits: {', '.join(traits)}")
    else:
        traits = [args.trait]

    # Compare each trait
    all_results = []
    for trait in traits:
        result = compare_trait_vectors(
            args.experiment_a, args.experiment_b, trait,
            args.method, variant_a, variant_b,
            args.component, args.position, args.top_k,
        )
        if result is None:
            continue

        # Check for best steering layer in either experiment
        best_steering = (
            get_best_steering_layer(args.experiment_a, trait, variant_a, args.position)
            or get_best_steering_layer(args.experiment_b, trait, variant_b, args.position)
        )
        if best_steering is not None and str(best_steering) in result['per_layer']:
            result['summary']['best_steering_layer'] = best_steering
            result['summary']['best_steering_cosine_sim'] = result['per_layer'][str(best_steering)]['cosine_sim']

        print_comparison(result, best_steering)
        all_results.append(result)

    # Cross-trait summary for --all-traits
    if args.all_traits and len(all_results) > 1:
        means = [r['summary']['mean_cosine_sim'] for r in all_results]
        mins = [r['summary']['min_cosine_sim'] for r in all_results]
        print(f"\n{'=' * 70}")
        print(f"Cross-trait summary ({len(all_results)} traits)")
        print(f"{'=' * 70}")
        print(f"  Mean cosine sim across traits: {sum(means)/len(means):.4f}")
        print(f"  Worst per-layer sim:           {min(mins):.4f}")
        print(f"  Best mean trait:               {all_results[means.index(max(means))]['trait']} ({max(means):.4f})")
        print(f"  Worst mean trait:              {all_results[means.index(min(means))]['trait']} ({min(means):.4f})")

    # Save JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            'experiment_a': args.experiment_a,
            'experiment_b': args.experiment_b,
            'variant_a': variant_a,
            'variant_b': variant_b,
            'method': args.method,
            'component': args.component,
            'position': args.position,
            'top_k': args.top_k,
            'traits': all_results,
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
