#!/usr/bin/env python3
"""
Mean-centered model diff: remove global LoRA offset before projecting onto traits.

LoRA finetuning adds a constant activation offset to ALL inputs. This projects
onto every direction, inflating Cohen's d for every trait (including controls).
Mean-centering removes this constant offset, isolating the trait-specific signal.

Method:
  For each layer, across all prompts:
    1. Compute per-prompt response mean activations for variants A and B
    2. Compute global mean shift: Δ_global = mean_over_prompts(h_B - h_A)
    3. Subtract global shift from each B activation: h_B_adj = h_B - Δ_global
    4. Project both original h_A and adjusted h_B onto trait vectors
    5. Compute Cohen's d (unpaired + paired) and p-values

If the deception signal is real (not just a mean shift artifact), it should
survive mean-centering. If formality (control) is just mean shift, it should drop.

Input: Raw activation .pt files from inference/capture_raw_activations.py
Output: JSON with per-layer Cohen's d (original and mean-centered) + p-values

Usage:
    python experiments/sleeper_detection/scripts/mean_centered_diff.py \
        --experiment sleeper_detection \
        --variant-a instruct --variant-b sleeper_lora \
        --prompt-set sleeper/benign \
        --method probe --position "response[:5]" \
        --traits bs/concealment,alignment/deception,pv_natural/sycophancy,hum/formality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import argparse
import json
import torch
import numpy as np
from scipy import stats
from tqdm import tqdm

from core import projection, effect_size, cosine_similarity
from utils.paths import get as get_path, get_model_variant, list_layers
from utils.vectors import load_vector_with_baseline
from utils.json import dump_compact
from analysis.model_diff.compare_variants import load_raw_activations, get_response_mean


def main():
    parser = argparse.ArgumentParser(description="Mean-centered model diff")
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--variant-a', required=True, help='Baseline (e.g., instruct)')
    parser.add_argument('--variant-b', required=True, help='Comparison (e.g., sleeper_lora)')
    parser.add_argument('--prompt-set', required=True)
    parser.add_argument('--traits', required=True, help='Comma-separated trait names')
    parser.add_argument('--method', required=True, help='Vector method (e.g., probe)')
    parser.add_argument('--position', required=True, help='Vector position (e.g., "response[:5]")')
    parser.add_argument('--component', default='residual')
    parser.add_argument('--vector-experiment', default=None,
                        help='Experiment to load vectors from (default: same)')
    args = parser.parse_args()

    vector_experiment = args.vector_experiment or args.experiment
    traits = [t.strip() for t in args.traits.split(',')]
    extraction_variant = get_model_variant(vector_experiment, None, mode='extraction')['name']

    # Load activations
    print(f"Loading activations for {args.variant_a} and {args.variant_b}...")
    acts_a = load_raw_activations(args.experiment, args.variant_a, args.prompt_set)
    acts_b = load_raw_activations(args.experiment, args.variant_b, args.prompt_set)

    acts_a_dict = {d['prompt_id']: d for d in acts_a}
    acts_b_dict = {d['prompt_id']: d for d in acts_b}
    common_ids = sorted(set(acts_a_dict.keys()) & set(acts_b_dict.keys()))
    print(f"  {len(common_ids)} common prompts")

    # Get layers
    first_a = acts_a_dict[common_ids[0]]
    first_b = acts_b_dict[common_ids[0]]
    layers_a = set(first_a['response']['activations'].keys())
    layers_b = set(first_b['response']['activations'].keys())
    common_layers = sorted(layers_a & layers_b)
    hidden_dim = first_a['response']['activations'][common_layers[0]][args.component].shape[-1]
    print(f"  {len(common_layers)} common layers, hidden_dim={hidden_dim}")

    # Pre-compute per-prompt response means for all layers
    print(f"\nPre-computing response means...")
    means_a = {}  # {layer: [tensor per prompt]}
    means_b = {}
    for layer in tqdm(common_layers, desc="Layers"):
        ma, mb = [], []
        for pid in common_ids:
            ma.append(get_response_mean(acts_a_dict[pid], layer, args.component))
            mb.append(get_response_mean(acts_b_dict[pid], layer, args.component))
        means_a[layer] = torch.stack(ma)  # [n_prompts, hidden_dim]
        means_b[layer] = torch.stack(mb)

    # Compute global mean shift per layer
    print(f"Computing global mean shift...")
    global_shift = {}
    for layer in common_layers:
        global_shift[layer] = (means_b[layer] - means_a[layer]).mean(dim=0)  # [hidden_dim]

    # Analyze each trait
    results = {
        'experiment': args.experiment,
        'variant_a': args.variant_a,
        'variant_b': args.variant_b,
        'prompt_set': args.prompt_set,
        'n_prompts': len(common_ids),
        'method': 'mean_centered_diff',
        'traits': {}
    }

    for trait in traits:
        print(f"\n  {trait}:")
        available_layers = list_layers(
            vector_experiment, trait, args.method, extraction_variant, args.component, args.position
        )
        sweep_layers = [l for l in available_layers if l in common_layers]

        original_d = []
        centered_d = []
        original_paired_d = []
        centered_paired_d = []
        centered_pvalues = []
        original_cosine = []
        centered_cosine = []

        for layer in sweep_layers:
            try:
                vector, _, _ = load_vector_with_baseline(
                    vector_experiment, trait, args.method, layer,
                    extraction_variant, args.component, args.position
                )
                vector = vector.float()
            except FileNotFoundError:
                for lst in [original_d, centered_d, original_paired_d, centered_paired_d,
                           centered_pvalues, original_cosine, centered_cosine]:
                    lst.append(0.0)
                continue

            # Original projections (no centering)
            proj_a = np.array([projection(means_a[layer][i], vector, normalize_vector=True).item()
                              for i in range(len(common_ids))])
            proj_b = np.array([projection(means_b[layer][i], vector, normalize_vector=True).item()
                              for i in range(len(common_ids))])

            # Mean-centered projections: subtract global shift from B
            means_b_centered = means_b[layer] - global_shift[layer].unsqueeze(0)
            proj_b_centered = np.array([projection(means_b_centered[i], vector, normalize_vector=True).item()
                                       for i in range(len(common_ids))])

            # Original Cohen's d (unpaired and paired)
            d_orig = effect_size(torch.tensor(proj_b), torch.tensor(proj_a), signed=True)
            diffs_orig = proj_b - proj_a
            d_paired_orig = float(np.mean(diffs_orig) / np.std(diffs_orig)) if np.std(diffs_orig) > 0 else 0.0

            # Mean-centered Cohen's d
            d_cent = effect_size(torch.tensor(proj_b_centered), torch.tensor(proj_a), signed=True)
            diffs_cent = proj_b_centered - proj_a
            d_paired_cent = float(np.mean(diffs_cent) / np.std(diffs_cent)) if np.std(diffs_cent) > 0 else 0.0

            # P-value for mean-centered paired difference (one-sample t-test, H0: mean diff = 0)
            if np.std(diffs_cent) > 0:
                t_stat, p_val = stats.ttest_1samp(diffs_cent, 0)
                p_val = float(p_val)
            else:
                p_val = 1.0

            # Cosine similarities
            diff_vec = (means_b[layer] - means_a[layer]).mean(dim=0)
            diff_vec_centered = (means_b_centered - means_a[layer]).mean(dim=0)
            cos_orig = cosine_similarity(diff_vec, vector).item()
            cos_cent = cosine_similarity(diff_vec_centered, vector).item()

            original_d.append(d_orig)
            centered_d.append(d_cent)
            original_paired_d.append(d_paired_orig)
            centered_paired_d.append(d_paired_cent)
            centered_pvalues.append(p_val)
            original_cosine.append(cos_orig)
            centered_cosine.append(cos_cent)

        # Find peaks
        if sweep_layers:
            peak_orig_idx = max(range(len(original_d)), key=lambda i: abs(original_d[i]))
            peak_cent_idx = max(range(len(centered_d)), key=lambda i: abs(centered_d[i]))

            print(f"    Original:      peak L{sweep_layers[peak_orig_idx]} = {original_d[peak_orig_idx]:+.2f}σ")
            print(f"    Mean-centered: peak L{sweep_layers[peak_cent_idx]} = {centered_d[peak_cent_idx]:+.2f}σ (p={centered_pvalues[peak_cent_idx]:.4f})")

        results['traits'][trait] = {
            'layers': sweep_layers,
            'method': args.method,
            'position': args.position,
            'per_layer_effect_size_original': original_d,
            'per_layer_effect_size_centered': centered_d,
            'per_layer_paired_d_original': original_paired_d,
            'per_layer_paired_d_centered': centered_paired_d,
            'per_layer_pvalue_centered': centered_pvalues,
            'per_layer_cosine_original': original_cosine,
            'per_layer_cosine_centered': centered_cosine,
        }

    # Save
    output_dir = Path(f"experiments/{args.experiment}/model_diff/mean_centered/{args.variant_a}_vs_{args.variant_b}/{args.prompt_set.replace('/', '_')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        dump_compact(results, f)
    print(f"\nSaved to {output_dir / 'results.json'}")

    # Print summary table
    print(f"\n{'=' * 90}")
    print(f"SUMMARY: {args.variant_b} vs {args.variant_a} on {args.prompt_set}")
    print(f"{'=' * 90}")
    print(f"{'Trait':>25} {'Original d':>12} {'Centered d':>12} {'p-value':>10} {'Layer':>6}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*6}")
    for trait in traits:
        td = results['traits'].get(trait, {})
        layers = td.get('layers', [])
        orig = td.get('per_layer_effect_size_original', [])
        cent = td.get('per_layer_effect_size_centered', [])
        pvals = td.get('per_layer_pvalue_centered', [])
        if not orig:
            continue
        peak_idx = max(range(len(orig)), key=lambda i: abs(orig[i]))
        sig = "***" if pvals[peak_idx] < 0.001 else "**" if pvals[peak_idx] < 0.01 else "*" if pvals[peak_idx] < 0.05 else ""
        short = trait.split('/')[-1]
        print(f"{short:>25} {orig[peak_idx]:>+10.2f}σ {cent[peak_idx]:>+10.2f}σ {pvals[peak_idx]:>9.4f}{sig} L{layers[peak_idx]:>3}")


if __name__ == '__main__':
    main()
