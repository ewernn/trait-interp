#!/usr/bin/env python3
"""
Test: project out generic finetuning direction from deception LoRA diff,
then check if residual aligns with deception trait vectors.

Uses diff_vectors.pt from model_diff comparisons (mean activation diffs
across prompts). Two approaches:
1. Project out lora_gender from lora_time (pairwise)
2. Project out mean of all 3 LoRAs, check each residual

Input: diff_vectors.pt files from instruct_vs_lora_{time,greeting,gender}/
Output: per-layer cosine similarities after orthogonalization

Usage:
    python experiments/bullshit/scripts/orthogonal_specificity.py
    python experiments/bullshit/scripts/orthogonal_specificity.py --prompt-set alpaca_control
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from core.math import orthogonalize
from core import cosine_similarity
from utils.vectors import load_vector_with_baseline
from utils.paths import list_layers
from utils import paths as path_utils

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prompt-set', default='alpaca_control_500',
                        help='Prompt set to use (default: alpaca_control_500)')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[3]
    model_diff_dir = base_dir / 'experiments/bullshit/model_diff'

    # Load diff vectors for all 3 LoRAs (vs instruct)
    diff_time = torch.load(model_diff_dir / f'instruct_vs_lora_time/{args.prompt_set}/diff_vectors.pt', map_location='cpu', weights_only=True)
    diff_greeting = torch.load(model_diff_dir / f'instruct_vs_lora_greeting/{args.prompt_set}/diff_vectors.pt', map_location='cpu', weights_only=True)
    diff_gender = torch.load(model_diff_dir / f'instruct_vs_lora_gender/{args.prompt_set}/diff_vectors.pt', map_location='cpu', weights_only=True)

    print(f"Prompt set: {args.prompt_set}")
    print(f"Shapes: time={diff_time.shape}, greeting={diff_greeting.shape}, gender={diff_gender.shape}")

    # Find non-zero layers
    nonzero_layers = [i for i in range(diff_time.shape[0]) if diff_time[i].norm() > 0 and diff_gender[i].norm() > 0]
    print(f"Non-zero layers: {len(nonzero_layers)} — {nonzero_layers[:5]}...{nonzero_layers[-3:]}")

    # Traits to test (from two experiments sharing the same Llama 70B base)
    traits_bullshit = ['bs/concealment', 'bs/lying']
    traits_rm_syco = ['hum/formality', 'rm_hack/eval_awareness', 'rm_hack/ulterior_motive']

    exp_config_bs = path_utils.load_experiment_config('bullshit')
    extraction_variant_bs = exp_config_bs.get('defaults', {}).get('extraction', 'base')

    exp_config_rm = path_utils.load_experiment_config('rm_syco')
    extraction_variant_rm = exp_config_rm.get('defaults', {}).get('extraction', 'base')

    method = 'probe'
    position = 'response[:5]'
    component = 'residual'

    print("\n" + "="*80)
    print("APPROACH 1: Project out lora_gender direction from lora_time")
    print("="*80)

    for trait in traits_bullshit + traits_rm_syco:
        vector_exp = 'rm_syco' if trait in traits_rm_syco else 'bullshit'
        extraction_variant = extraction_variant_rm if trait in traits_rm_syco else extraction_variant_bs

        available = list_layers(vector_exp, trait, method, extraction_variant, component, position)
        sweep = [l for l in available if l in nonzero_layers]

        print(f"\n  {trait}:")
        print(f"  {'Layer':>5}  {'raw cos':>10}  {'specific cos':>12}  {'control cos':>12}  {'overlap':>10}")

        for layer in sweep:
            try:
                vector, _, _ = load_vector_with_baseline(vector_exp, trait, method, layer, extraction_variant, component, position)
                vector = vector.float()
            except FileNotFoundError:
                continue

            d_time = diff_time[layer].float()
            d_gender = diff_gender[layer].float()

            # Raw cosine (what we had before)
            raw_cos = cosine_similarity(d_time, vector).item()

            # Project out gender direction from time
            d_specific = orthogonalize(d_time, d_gender)
            specific_cos = cosine_similarity(d_specific, vector).item()

            # Also check: control's cosine (should be ~same as raw if generic dominates)
            control_cos = cosine_similarity(d_gender, vector).item()

            # How much of d_time is along d_gender?
            overlap = cosine_similarity(d_time, d_gender).item()

            print(f"  L{layer:>3}  {raw_cos:+.4f}      {specific_cos:+.4f}        {control_cos:+.4f}      {overlap:+.4f}")

    print("\n" + "="*80)
    print("APPROACH 2: Project out mean of all 3 LoRAs, check each residual")
    print("="*80)

    for trait in traits_bullshit + traits_rm_syco:
        vector_exp = 'rm_syco' if trait in traits_rm_syco else 'bullshit'
        extraction_variant = extraction_variant_rm if trait in traits_rm_syco else extraction_variant_bs

        available = list_layers(vector_exp, trait, method, extraction_variant, component, position)
        sweep = [l for l in available if l in nonzero_layers]

        print(f"\n  {trait}:")
        print(f"  {'Layer':>5}  {'time_spec':>10}  {'greeting_spec':>12}  {'gender_spec':>12}")

        for layer in sweep:
            try:
                vector, _, _ = load_vector_with_baseline(vector_exp, trait, method, layer, extraction_variant, component, position)
                vector = vector.float()
            except FileNotFoundError:
                continue

            d_mean = (diff_time[layer] + diff_greeting[layer] + diff_gender[layer]).float() / 3

            d_time_spec = orthogonalize(diff_time[layer].float(), d_mean)
            d_greeting_spec = orthogonalize(diff_greeting[layer].float(), d_mean)
            d_gender_spec = orthogonalize(diff_gender[layer].float(), d_mean)

            cos_time = cosine_similarity(d_time_spec, vector).item()
            cos_greeting = cosine_similarity(d_greeting_spec, vector).item()
            cos_gender = cosine_similarity(d_gender_spec, vector).item()

            print(f"  L{layer:>3}  {cos_time:+.4f}      {cos_greeting:+.4f}        {cos_gender:+.4f}")

if __name__ == '__main__':
    main()
