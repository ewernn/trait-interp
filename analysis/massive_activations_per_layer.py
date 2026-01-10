#!/usr/bin/env python3
"""
Compute per-layer statistics for massive activation dimensions.

Analyzes raw calibration activations to determine which dimensions are safe to zero
at which layers, based on consistency metrics.

Input: Raw activations from calibration (experiments/{exp}/inference/{model_variant}/raw/residual/_calibration/*.pt)
Output: Per-layer stats JSON (experiments/{exp}/inference/{model_variant}/massive_activations/per_layer_stats.json)

Usage:
    python analysis/massive_activations_per_layer.py --experiment gemma-2-2b
    python analysis/massive_activations_per_layer.py --experiment gemma-3-4b
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get as get_path, get_model_variant


def compute_per_layer_stats(experiment: str, model_variant: str) -> dict:
    """
    Compute per-layer stats for massive dims.

    For each massive dim at each layer, computes:
    - ratio: mean magnitude vs median baseline
    - cv: coefficient of variation (std/mean) - lower = more consistent = safer to zero
    - p5_ratio: 5th percentile vs baseline - the "floor"
    - pct_above_10x: % of tokens where dim exceeds 10x baseline
    """
    # Load calibration to get massive dims
    calib_path = Path(get_path('inference.variant', experiment=experiment, model_variant=model_variant)) / 'massive_activations' / 'calibration.json'

    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration not found: {calib_path}\nRun: python analysis/massive_activations.py --experiment {experiment}")

    with open(calib_path) as f:
        calib = json.load(f)

    top_dims_by_layer = calib['aggregate']['top_dims_by_layer']
    n_layers = len(top_dims_by_layer)

    # Get dims that appear in top-5 at 3+ layers (consistent massive dims)
    appearances = {}
    for layer, dims in top_dims_by_layer.items():
        for dim in dims[:5]:
            appearances[dim] = appearances.get(dim, 0) + 1
    massive_dims = sorted([d for d, c in appearances.items() if c >= 3])

    if not massive_dims:
        print(f"No consistent massive dims found (appearing in 3+ layers)")
        return {'experiment': experiment, 'model_variant': model_variant, 'massive_dims': [], 'per_layer': {}}

    print(f"Found {len(massive_dims)} massive dims: {massive_dims}")

    # Load raw activations
    raw_dir = Path(get_path('inference.raw_residual', experiment=experiment, model_variant=model_variant, prompt_set='_calibration'))

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw activations not found: {raw_dir}\nRun: python analysis/massive_activations.py --experiment {experiment}")

    pt_files = sorted(raw_dir.glob('*.pt'))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {raw_dir}")

    print(f"Loading activations from {len(pt_files)} prompts...")

    # Pool activations per layer (response tokens only)
    pooled = {layer: [] for layer in range(n_layers)}

    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False, map_location='cpu')
        resp = data.get('response', {}).get('activations', {})

        for layer in range(n_layers):
            if layer in resp and 'residual' in resp[layer]:
                pooled[layer].append(resp[layer]['residual'].float())

    # Concatenate all tokens per layer
    for layer in range(n_layers):
        if pooled[layer]:
            pooled[layer] = torch.cat(pooled[layer], dim=0)
        else:
            pooled[layer] = torch.empty(0)

    n_tokens = pooled[0].shape[0] if len(pooled[0]) > 0 else 0
    print(f"Pooled {n_tokens} tokens across all prompts")

    # Compute per-layer stats for each massive dim
    results = {
        'experiment': experiment,
        'model_variant': model_variant,
        'n_tokens': n_tokens,
        'n_layers': n_layers,
        'massive_dims': massive_dims,
        'per_layer': {}
    }

    for layer in range(n_layers):
        if len(pooled[layer]) == 0:
            continue

        residual = pooled[layer]
        baseline = residual.abs().mean(dim=0).median().item()

        results['per_layer'][layer] = {}

        for dim in massive_dims:
            vals = residual[:, dim].abs().numpy()
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            p5 = np.percentile(vals, 5)

            results['per_layer'][layer][dim] = {
                'ratio': float(round(mean_val / baseline, 1)) if baseline > 0 else 0,
                'cv': float(round(std_val / mean_val, 3)) if mean_val > 0 else 0,
                'p5_ratio': float(round(p5 / baseline, 1)) if baseline > 0 else 0,
                'pct_above_10x': float(round((vals > 10 * baseline).mean() * 100, 1))
            }

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--model-variant', default=None, help='Model variant (default: from experiment config)')
    args = parser.parse_args()

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant['name']

    print(f"=== Per-Layer Massive Activation Stats ===")
    print(f"Experiment: {args.experiment}")
    print(f"Model variant: {model_variant}")
    print()

    results = compute_per_layer_stats(args.experiment, model_variant)

    # Save results
    out_path = Path(get_path('inference.variant', experiment=args.experiment, model_variant=model_variant)) / 'massive_activations' / 'per_layer_stats.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_path}")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Massive dims: {results['massive_dims']}")
    print(f"Tokens analyzed: {results['n_tokens']}")

    if results['per_layer']:
        print(f"\nPer-dim stats at middle layer (L{results['n_layers']//2}):")
        mid_layer = results['n_layers'] // 2
        if mid_layer in results['per_layer']:
            for dim in results['massive_dims']:
                if dim in results['per_layer'][mid_layer]:
                    stats = results['per_layer'][mid_layer][dim]
                    print(f"  dim {dim}: ratio={stats['ratio']}x, CV={stats['cv']}, p5={stats['p5_ratio']}x, >10x={stats['pct_above_10x']}%")


if __name__ == '__main__':
    main()
