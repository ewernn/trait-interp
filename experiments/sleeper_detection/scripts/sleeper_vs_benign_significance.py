#!/usr/bin/env python3
"""
Statistical significance of sleeper-vs-benign LoRA trait projection differences.

For each prompt set and trait, computes:
  1. Per-prompt projections for sleeper_lora and benign_lora (both compared to instruct)
  2. Paired t-test on the difference: is sleeper's shift significantly different from benign's?
  3. Effect size (Cohen's d) of the sleeper-benign delta
  4. Permutation test as non-parametric alternative

The key question: is the deception signal at layers 14-17 and 24-28 statistically
significant, or could it arise by chance from two independently-trained LoRAs?

Input: Raw activation .pt files for instruct, sleeper_lora, benign_lora
Output: JSON with per-layer p-values and effect sizes for sleeper-benign delta

Usage:
    python experiments/sleeper_detection/scripts/sleeper_vs_benign_significance.py \
        --experiment sleeper_detection \
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

from core import projection
from utils.paths import get_model_variant, list_layers
from utils.vectors import load_vector_with_baseline
from utils.json import dump_compact
from analysis.model_diff.compare_variants import load_raw_activations, get_response_mean


def main():
    parser = argparse.ArgumentParser(description="Sleeper vs benign LoRA significance test")
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--prompt-sets', required=True,
                        help='Comma-separated prompt sets (e.g., sleeper/triggered,sleeper/benign)')
    parser.add_argument('--traits', required=True, help='Comma-separated trait names')
    parser.add_argument('--method', required=True)
    parser.add_argument('--position', required=True)
    parser.add_argument('--component', default='residual')
    parser.add_argument('--vector-experiment', default=None)
    parser.add_argument('--n-permutations', type=int, default=1000,
                        help='Number of permutations for permutation test')
    args = parser.parse_args()

    vector_experiment = args.vector_experiment or args.experiment
    traits = [t.strip() for t in args.traits.split(',')]
    prompt_sets = [p.strip() for p in args.prompt_sets.split(',')]
    extraction_variant = get_model_variant(vector_experiment, None, mode='extraction')['name']

    all_results = {}

    for prompt_set in prompt_sets:
        print(f"\n{'='*70}")
        print(f"  {prompt_set}")
        print(f"{'='*70}")

        # Load activations for all three variants
        print(f"Loading activations...")
        acts_instruct = load_raw_activations(args.experiment, 'instruct', prompt_set)
        acts_sleeper = load_raw_activations(args.experiment, 'sleeper_lora', prompt_set)
        acts_benign = load_raw_activations(args.experiment, 'benign_lora', prompt_set)

        dict_i = {d['prompt_id']: d for d in acts_instruct}
        dict_s = {d['prompt_id']: d for d in acts_sleeper}
        dict_b = {d['prompt_id']: d for d in acts_benign}
        common_ids = sorted(set(dict_i.keys()) & set(dict_s.keys()) & set(dict_b.keys()))
        print(f"  {len(common_ids)} common prompts across all 3 variants")

        if not common_ids:
            print("  SKIP: no common prompts")
            continue

        # Get layers
        first = dict_i[common_ids[0]]
        common_layers = sorted(first['response']['activations'].keys())
        hidden_dim = first['response']['activations'][common_layers[0]][args.component].shape[-1]

        ps_results = {}

        for trait in traits:
            print(f"\n  {trait}:")
            available_layers = list_layers(
                vector_experiment, trait, args.method, extraction_variant, args.component, args.position
            )
            sweep_layers = [l for l in available_layers if l in common_layers]

            layer_results = []

            for layer in sweep_layers:
                try:
                    vector, _, _ = load_vector_with_baseline(
                        vector_experiment, trait, args.method, layer,
                        extraction_variant, args.component, args.position
                    )
                    vector = vector.float()
                except FileNotFoundError:
                    layer_results.append({
                        'layer': layer, 'd_sleeper': 0, 'd_benign': 0, 'delta': 0,
                        'p_ttest': 1.0, 'p_perm': 1.0
                    })
                    continue

                # Project each variant onto trait vector
                proj_i, proj_s, proj_b = [], [], []
                for pid in common_ids:
                    mi = get_response_mean(dict_i[pid], layer, args.component)
                    ms = get_response_mean(dict_s[pid], layer, args.component)
                    mb = get_response_mean(dict_b[pid], layer, args.component)
                    proj_i.append(projection(mi, vector, normalize_vector=True).item())
                    proj_s.append(projection(ms, vector, normalize_vector=True).item())
                    proj_b.append(projection(mb, vector, normalize_vector=True).item())

                proj_i = np.array(proj_i)
                proj_s = np.array(proj_s)
                proj_b = np.array(proj_b)

                # Per-prompt shifts relative to instruct
                shift_s = proj_s - proj_i  # sleeper shift per prompt
                shift_b = proj_b - proj_i  # benign shift per prompt

                # Sleeper-benign delta per prompt
                delta = shift_s - shift_b

                # Cohen's d for each shift (vs instruct)
                d_sleeper = float(np.mean(shift_s) / np.std(shift_s)) if np.std(shift_s) > 0 else 0.0
                d_benign = float(np.mean(shift_b) / np.std(shift_b)) if np.std(shift_b) > 0 else 0.0
                d_delta = float(np.mean(delta) / np.std(delta)) if np.std(delta) > 0 else 0.0

                # Paired t-test: is sleeper shift != benign shift?
                if np.std(delta) > 0:
                    t_stat, p_ttest = stats.ttest_1samp(delta, 0)
                    p_ttest = float(p_ttest)
                else:
                    p_ttest = 1.0

                # Permutation test: randomly swap sleeper/benign labels per prompt
                if args.n_permutations > 0 and np.std(delta) > 0:
                    observed_mean = np.mean(delta)
                    count_extreme = 0
                    rng = np.random.default_rng(42)
                    for _ in range(args.n_permutations):
                        signs = rng.choice([-1, 1], size=len(delta))
                        perm_mean = np.mean(delta * signs)
                        if abs(perm_mean) >= abs(observed_mean):
                            count_extreme += 1
                    p_perm = (count_extreme + 1) / (args.n_permutations + 1)
                else:
                    p_perm = 1.0

                layer_results.append({
                    'layer': layer,
                    'd_sleeper': round(d_sleeper, 3),
                    'd_benign': round(d_benign, 3),
                    'd_delta': round(d_delta, 3),
                    'mean_delta': round(float(np.mean(delta)), 4),
                    'p_ttest': round(p_ttest, 6),
                    'p_perm': round(p_perm, 6),
                })

            # Print table for this trait
            sig_layers = [r for r in layer_results if r['p_ttest'] < 0.05]
            if sig_layers:
                print(f"    {len(sig_layers)} significant layers (p < 0.05):")
                print(f"    {'Layer':>6} {'d_sleeper':>10} {'d_benign':>10} {'delta_d':>10} {'p_ttest':>10} {'p_perm':>10}")
                for r in sig_layers:
                    stars = "***" if r['p_ttest'] < 0.001 else "**" if r['p_ttest'] < 0.01 else "*"
                    print(f"    L{r['layer']:>4} {r['d_sleeper']:>+9.2f}σ {r['d_benign']:>+9.2f}σ {r['d_delta']:>+9.2f}σ {r['p_ttest']:>10.4f}{stars} {r['p_perm']:>9.4f}")
            else:
                print(f"    No significant layers (p < 0.05)")

            ps_results[trait] = {
                'layers': [r['layer'] for r in layer_results],
                'per_layer': layer_results,
            }

        all_results[prompt_set] = ps_results

    # Save
    output_dir = Path(f"experiments/{args.experiment}/model_diff/sleeper_vs_benign_significance")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        dump_compact(all_results, f)
    print(f"\nSaved to {output_dir / 'results.json'}")

    # Cross-prompt-set summary
    print(f"\n{'='*90}")
    print(f"CROSS-CONDITION SUMMARY: Layers significant (p < 0.05) across prompt sets")
    print(f"{'='*90}")
    for trait in traits:
        short = trait.split('/')[-1]
        print(f"\n  {short}:")
        for ps in prompt_sets:
            ps_data = all_results.get(ps, {}).get(trait, {})
            sig = [r for r in ps_data.get('per_layer', []) if r['p_ttest'] < 0.05]
            if sig:
                layers_str = ", ".join(f"L{r['layer']}({r['d_delta']:+.2f}σ)" for r in sig[:8])
                print(f"    {ps:>20}: {len(sig)} layers — {layers_str}")
            else:
                print(f"    {ps:>20}: none")


if __name__ == '__main__':
    main()
