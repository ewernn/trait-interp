#!/usr/bin/env python3
"""
Measure activation similarity between base and instruct model variants.

Computes per-layer and per-position cosine similarity between response token
activations from two model variants on the same prompts (via replay mode).
Correlates similarity with steering transfer effectiveness.

Input:
    Raw activations from both variants:
        experiments/{exp}/inference/{variant}/raw/residual/{prompt_set}/*.pt

Output:
    experiments/{exp}/activation_similarity/results.json
        - per_layer_similarity: mean/std/per_prompt cosine similarity
        - per_position_similarity: by token position, by layer
        - transfer_curves: best steering delta per layer per trait
        - correlations: Pearson r between similarity and transfer

Usage:
    python experiments/model_diff/scripts/activation_similarity.py \
        --experiment model_diff \
        --prompt-set model_diff/steering_combined
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import argparse
import json
import torch
import numpy as np
from scipy import stats

from analysis.model_diff.compare_variants import load_raw_activations
from core import cosine_similarity
from utils.json import dump_compact


def compute_per_layer_similarity(acts_a: dict, acts_b: dict, common_ids: list,
                                  n_layers: int, component: str = 'residual') -> dict:
    """Cosine similarity of mean response activations per layer."""
    per_prompt = np.zeros((len(common_ids), n_layers))

    for i, pid in enumerate(common_ids):
        for layer in range(n_layers):
            act_a = acts_a[pid]['response']['activations'][layer][component].mean(dim=0).float()
            act_b = acts_b[pid]['response']['activations'][layer][component].mean(dim=0).float()
            per_prompt[i, layer] = cosine_similarity(act_a, act_b).item()

    return {
        'mean': per_prompt.mean(axis=0).tolist(),
        'std': per_prompt.std(axis=0).tolist(),
        'min': per_prompt.min(axis=0).tolist(),
        'max': per_prompt.max(axis=0).tolist(),
        'per_prompt': per_prompt.tolist(),
    }


def compute_per_position_similarity(acts_a: dict, acts_b: dict, common_ids: list,
                                     n_layers: int, component: str = 'residual',
                                     min_prompt_fraction: float = 0.8) -> dict:
    """Per-token-position cosine similarity, clipped to positions with enough data."""
    # Find response lengths
    lengths = []
    for pid in common_ids:
        n_tokens = acts_a[pid]['response']['activations'][0][component].shape[0]
        lengths.append(n_tokens)

    min_length = min(lengths)
    max_length = max(lengths)
    min_prompts = int(len(common_ids) * min_prompt_fraction)

    # Find max position where we have enough prompts
    max_pos = 0
    for pos in range(max_length):
        n_with_pos = sum(1 for l in lengths if l > pos)
        if n_with_pos >= min_prompts:
            max_pos = pos + 1
        else:
            break

    print(f"  Per-position: using positions 0-{max_pos-1} "
          f"(min {min_prompts}/{len(common_ids)} prompts)")

    # Sample layers (every 4th + first/last to keep output manageable)
    sample_layers = sorted(set([0, n_layers - 1] + list(range(0, n_layers, 4))))

    by_layer = {}
    n_prompts_at_position = []

    for pos in range(max_pos):
        n_prompts_at_position.append(sum(1 for l in lengths if l > pos))

    for layer in sample_layers:
        means = []
        stds = []
        for pos in range(max_pos):
            sims = []
            for pid_idx, pid in enumerate(common_ids):
                if lengths[pid_idx] <= pos:
                    continue
                act_a = acts_a[pid]['response']['activations'][layer][component][pos].float()
                act_b = acts_b[pid]['response']['activations'][layer][component][pos].float()
                sims.append(cosine_similarity(act_a, act_b).item())
            means.append(float(np.mean(sims)))
            stds.append(float(np.std(sims)))

        by_layer[str(layer)] = {'mean': means, 'std': stds}

    return {
        'positions': list(range(max_pos)),
        'n_prompts_at_position': n_prompts_at_position,
        'sample_layers': sample_layers,
        'by_layer': by_layer,
    }


def extract_transfer_curves(experiment: str) -> dict:
    """Extract per-layer steering delta for natural vectors (coherence >= 70).

    Searches all positions for each trait and uses the position with the highest
    peak delta, matching how the viz finding reports "best natural" results.
    """
    from utils.paths import get as get_path

    trait_names = [
        'pv_natural/sycophancy',
        'pv_natural/evil_v3',
        'pv_natural/hallucination_v2',
    ]

    curves = {}
    for trait in trait_names:
        # Find all position directories for this trait
        trait_steering_dir = get_path('experiments.base', experiment=experiment) / \
            'steering' / trait / 'instruct'

        if not trait_steering_dir.exists():
            print(f"  Warning: No steering dir for {trait}")
            continue

        best_position = None
        best_peak_delta = -float('inf')
        best_curve = None

        for position_dir in sorted(trait_steering_dir.iterdir()):
            results_path = position_dir / 'steering' / 'results.jsonl'
            if not results_path.exists():
                continue

            with open(results_path) as f:
                lines = [json.loads(line) for line in f]

            baseline = None
            layer_best = {}

            for line in lines:
                # Baseline line has type=baseline
                if line.get('type') == 'baseline':
                    baseline = line['result']['trait_mean']
                    continue

                # Skip header
                if line.get('type') == 'header':
                    continue

                # Steering result: nested under result and config
                result = line.get('result', {})
                config = line.get('config', {})
                vectors = config.get('vectors', [])
                if not vectors or not result:
                    continue

                layer = vectors[0]['layer']
                coherence = result['coherence_mean']
                trait_mean = result['trait_mean']

                if coherence < 70:
                    continue

                delta = trait_mean - baseline if baseline else trait_mean
                if layer not in layer_best or delta > layer_best[layer]:
                    layer_best[layer] = delta

            if baseline is not None and layer_best:
                peak_delta = max(layer_best.values())
                if peak_delta > best_peak_delta:
                    best_peak_delta = peak_delta
                    best_position = position_dir.name
                    layers = sorted(layer_best.keys())
                    best_curve = {
                        'layers': layers,
                        'deltas': [layer_best[l] for l in layers],
                        'baseline': baseline,
                        'position': best_position,
                    }

        if best_curve:
            curves[trait] = best_curve
            print(f"  {trait} ({best_position}): {len(best_curve['layers'])} layers, "
                  f"best delta={best_peak_delta:.1f} at L{best_curve['layers'][best_curve['deltas'].index(best_peak_delta)]}")

    return curves


def compute_correlations(per_layer_sim: dict, transfer_curves: dict) -> dict:
    """Pearson correlation between activation similarity and transfer delta."""
    sim_mean = np.array(per_layer_sim['mean'])
    correlations = {}

    for trait, curve in transfer_curves.items():
        layers = curve['layers']
        deltas = np.array(curve['deltas'])
        sim_at_layers = sim_mean[layers]

        if len(layers) < 4:
            print(f"  {trait}: too few layers for correlation ({len(layers)})")
            continue

        r, p = stats.pearsonr(sim_at_layers, deltas)
        correlations[trait] = {
            'pearson_r': float(r),
            'p_value': float(p),
            'n_layers': len(layers),
            'sim_values': sim_at_layers.tolist(),
            'delta_values': deltas.tolist(),
        }
        print(f"  {trait}: r={r:.3f}, p={p:.3f} (n={len(layers)} layers)")

    return correlations


def main():
    parser = argparse.ArgumentParser(description="Activation similarity between model variants")
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--prompt-set', required=True)
    parser.add_argument('--variant-a', default='base', help='First variant (default: base)')
    parser.add_argument('--variant-b', default='instruct', help='Second variant (default: instruct)')
    parser.add_argument('--component', default='residual')
    parser.add_argument('--skip-per-position', action='store_true',
                        help='Skip per-position analysis (faster)')
    args = parser.parse_args()

    print(f"Activation similarity: {args.variant_a} vs {args.variant_b}")
    print(f"Prompt set: {args.prompt_set}")

    # Load activations
    print(f"\nLoading activations...")
    acts_a = load_raw_activations(args.experiment, args.variant_a, args.prompt_set)
    acts_b = load_raw_activations(args.experiment, args.variant_b, args.prompt_set)

    acts_a_dict = {d['prompt_id']: d for d in acts_a}
    acts_b_dict = {d['prompt_id']: d for d in acts_b}
    common_ids = sorted(set(acts_a_dict.keys()) & set(acts_b_dict.keys()),
                        key=lambda x: int(x) if x.isdigit() else x)
    print(f"  {args.variant_a}: {len(acts_a)}, {args.variant_b}: {len(acts_b)}, common: {len(common_ids)}")

    if not common_ids:
        print("ERROR: No common prompt IDs")
        return

    # Get dimensions
    first = acts_a_dict[common_ids[0]]
    n_layers = len(first['response']['activations'])
    hidden_dim = first['response']['activations'][0][args.component].shape[-1]
    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}")

    # 1. Per-layer similarity
    print(f"\nComputing per-layer similarity...")
    per_layer = compute_per_layer_similarity(
        acts_a_dict, acts_b_dict, common_ids, n_layers, args.component
    )
    mean_sim = np.array(per_layer['mean'])
    print(f"  Overall mean: {mean_sim.mean():.4f}")
    print(f"  Range: {mean_sim.min():.4f} (L{mean_sim.argmin()}) to {mean_sim.max():.4f} (L{mean_sim.argmax()})")

    # 2. Per-position similarity
    per_position = None
    if not args.skip_per_position:
        print(f"\nComputing per-position similarity...")
        per_position = compute_per_position_similarity(
            acts_a_dict, acts_b_dict, common_ids, n_layers, args.component
        )

    # 3. Transfer curves
    print(f"\nExtracting transfer curves...")
    transfer_curves = extract_transfer_curves(args.experiment)

    # 4. Correlations
    print(f"\nComputing correlations...")
    correlations = compute_correlations(per_layer, transfer_curves)

    # Save results
    from utils.paths import get as get_path
    output_dir = get_path('experiments.base', experiment=args.experiment) / 'activation_similarity'
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'experiment': args.experiment,
        'variant_a': args.variant_a,
        'variant_b': args.variant_b,
        'prompt_set': args.prompt_set,
        'component': args.component,
        'n_prompts': len(common_ids),
        'n_layers': n_layers,
        'per_layer_similarity': per_layer,
        'transfer_curves': transfer_curves,
        'correlations': correlations,
    }
    if per_position is not None:
        results['per_position_similarity'] = per_position

    with open(output_dir / 'results.json', 'w') as f:
        dump_compact(results, f)

    print(f"\nResults saved to: {output_dir / 'results.json'}")


if __name__ == '__main__':
    main()
