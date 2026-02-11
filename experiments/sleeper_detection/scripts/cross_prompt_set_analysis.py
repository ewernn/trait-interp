#!/usr/bin/env python3
"""
Cross-prompt-set centering + per-prompt distribution analysis for sleeper detection.

Addresses the aggregation failure: mean-based metrics drown sparse signals.
Applies techniques from CC++ (SWiM, max-pooling), DeepMind probes (bipolar pooling,
rolling max), and Overt Saboteur (threshold counting) to sleeper detection data.

Analyses:
  1. Cross-prompt-set centering — estimate LoRA offset from benign prompts,
     test residual on triggered/safe/non_code. The only non-degenerate centering.
  2. Per-prompt max/threshold/tail — instead of mean Cohen's d, look at the
     distribution shape: max projection, fraction exceeding threshold, tail percentiles.
  3. Direct sleeper-vs-benign comparison — skip instruct, compare LoRAs directly.
     The bullshit experiment found LoRA-vs-LoRA peaks at different layers than
     LoRA-vs-instruct.
  4. Token-level windowing — instead of averaging all response tokens, use
     response[:5] (matching vector extraction) and max-of-rolling-means (SWiM).
  5. Bipolar spread — max - min per prompt as a feature. Large spread on triggered
     but not benign = conditional behavior.

Input: Raw activation .pt files for instruct, sleeper_lora, benign_lora
Output: JSON with per-layer distribution statistics, centered residuals, token-level metrics

Usage:
    python experiments/sleeper_detection/scripts/cross_prompt_set_analysis.py \
        --experiment sleeper_detection \
        --prompt-sets sleeper/triggered,sleeper/safe,sleeper/benign,sleeper/non_code \
        --centering-set sleeper/benign \
        --traits bs/concealment,alignment/deception,pv_natural/sycophancy,hum/formality \
        --method probe --position "response[:5]"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import argparse
import json
import torch
import numpy as np
from scipy import stats

from core import projection
from utils.paths import get_model_variant, list_layers
from utils.vectors import load_vector_with_baseline
from utils.json import dump_compact
from analysis.model_diff.compare_variants import load_raw_activations, get_response_mean


def get_response_tokens(data, layer, component='residual'):
    """Get per-token response activations at a layer. Returns [n_tokens, hidden_dim]."""
    act = data['response']['activations'][layer].get(component)
    if act is None:
        raise KeyError(f"Component '{component}' not found at layer {layer}")
    return act.float()


def per_prompt_projections(acts_dict, common_ids, layer, vector, component='residual'):
    """Compute per-prompt mean projection onto trait vector. Returns np.array of shape [n_prompts]."""
    projs = []
    for pid in common_ids:
        mean_act = get_response_mean(acts_dict[pid], layer, component)
        projs.append(projection(mean_act, vector, normalize_vector=True).item())
    return np.array(projs)


def per_token_projections(acts_dict, pid, layer, vector, component='residual'):
    """Compute per-token projections for a single prompt. Returns np.array of shape [n_tokens]."""
    tokens = get_response_tokens(acts_dict[pid], layer, component)
    return projection(tokens, vector, normalize_vector=True).numpy()


def token_windowed_stats(acts_dict, common_ids, layer, vector, component='residual', window=5):
    """
    Token-level analysis per prompt:
    - early_mean: mean of first `window` tokens (matching response[:5] extraction)
    - full_mean: mean of all tokens
    - rolling_max: max of rolling window means (SWiM-style)
    - bipolar_spread: max_token - min_token projection

    Returns dict of np.arrays, each [n_prompts].
    """
    early_means, full_means, rolling_maxes, bipolar_spreads = [], [], [], []

    for pid in common_ids:
        tok_projs = per_token_projections(acts_dict, pid, layer, vector, component)
        n_tokens = len(tok_projs)

        full_means.append(tok_projs.mean())
        early_means.append(tok_projs[:window].mean() if n_tokens >= window else tok_projs.mean())
        bipolar_spreads.append(tok_projs.max() - tok_projs.min())

        # Rolling window max (SWiM-style)
        if n_tokens >= window:
            rolling = np.array([tok_projs[i:i+window].mean() for i in range(n_tokens - window + 1)])
            rolling_maxes.append(rolling.max())
        else:
            rolling_maxes.append(tok_projs.mean())

    return {
        'early_mean': np.array(early_means),
        'full_mean': np.array(full_means),
        'rolling_max': np.array(rolling_maxes),
        'bipolar_spread': np.array(bipolar_spreads),
    }


def distribution_stats(values, label=''):
    """Compute distribution statistics beyond mean/std."""
    return {
        'mean': round(float(np.mean(values)), 4),
        'std': round(float(np.std(values)), 4),
        'median': round(float(np.median(values)), 4),
        'max': round(float(np.max(values)), 4),
        'min': round(float(np.min(values)), 4),
        'p90': round(float(np.percentile(values, 90)), 4),
        'p95': round(float(np.percentile(values, 95)), 4),
        'p99': round(float(np.percentile(values, 99)), 4),
        'frac_above_0.5': round(float(np.mean(values > 0.5)), 4),
        'frac_above_1.0': round(float(np.mean(values > 1.0)), 4),
        'frac_above_2.0': round(float(np.mean(values > 2.0)), 4),
        'frac_below_neg0.5': round(float(np.mean(values < -0.5)), 4),
        'frac_below_neg1.0': round(float(np.mean(values < -1.0)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-prompt-set centering + distribution analysis")
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--prompt-sets', required=True,
                        help='Comma-separated prompt sets to analyze')
    parser.add_argument('--centering-set', required=True,
                        help='Prompt set to estimate generic LoRA offset from (e.g., sleeper/benign)')
    parser.add_argument('--traits', required=True, help='Comma-separated trait names')
    parser.add_argument('--method', required=True)
    parser.add_argument('--position', required=True)
    parser.add_argument('--component', default='residual')
    parser.add_argument('--vector-experiment', default=None)
    parser.add_argument('--token-window', type=int, default=5,
                        help='Window size for early tokens and SWiM rolling max')
    args = parser.parse_args()

    vector_experiment = args.vector_experiment or args.experiment
    traits = [t.strip() for t in args.traits.split(',')]
    prompt_sets = [p.strip() for p in args.prompt_sets.split(',')]
    centering_set = args.centering_set.strip()
    extraction_variant = get_model_variant(vector_experiment, None, mode='extraction')['name']

    # Ensure centering set is in the prompt sets
    if centering_set not in prompt_sets:
        prompt_sets.append(centering_set)

    # =========================================================================
    # Load all activations
    # =========================================================================
    print("Loading activations...")
    all_acts = {}  # {(variant, prompt_set): {pid: data}}
    variants = ['instruct', 'sleeper_lora', 'benign_lora']
    for variant in variants:
        for ps in prompt_sets:
            print(f"  {variant} / {ps}...", end=' ')
            try:
                raw = load_raw_activations(args.experiment, variant, ps)
                all_acts[(variant, ps)] = {d['prompt_id']: d for d in raw}
                print(f"{len(raw)} prompts")
            except FileNotFoundError:
                print("NOT FOUND — skipping")

    # =========================================================================
    # Determine common layers and prompts per prompt set
    # =========================================================================
    # Find layers from first available activation
    sample_key = next(iter(all_acts))
    sample_data = next(iter(all_acts[sample_key].values()))
    all_layers = sorted(sample_data['response']['activations'].keys())

    # Common prompt IDs per prompt set (across all 3 variants)
    common_ids_per_ps = {}
    for ps in prompt_sets:
        sets = [set(all_acts[(v, ps)].keys()) for v in variants if (v, ps) in all_acts]
        if len(sets) == len(variants):
            common_ids_per_ps[ps] = sorted(set.intersection(*sets))
            print(f"  {ps}: {len(common_ids_per_ps[ps])} common prompts")
        else:
            print(f"  {ps}: MISSING variant data, skipping")

    if centering_set not in common_ids_per_ps:
        print(f"\nERROR: centering set '{centering_set}' has no common data")
        return

    # =========================================================================
    # Main analysis loop
    # =========================================================================
    results = {
        'experiment': args.experiment,
        'centering_set': centering_set,
        'method': args.method,
        'position': args.position,
        'token_window': args.token_window,
        'prompt_sets': {},
    }

    for trait in traits:
        print(f"\n{'='*70}")
        print(f"  {trait}")
        print(f"{'='*70}")

        available_layers = list_layers(
            vector_experiment, trait, args.method, extraction_variant, args.component, args.position
        )
        sweep_layers = [l for l in available_layers if l in all_layers]

        for ps in prompt_sets:
            if ps not in common_ids_per_ps:
                continue

            ps_key = ps.replace('/', '_')
            if ps_key not in results['prompt_sets']:
                results['prompt_sets'][ps_key] = {}
            if trait not in results['prompt_sets'][ps_key]:
                results['prompt_sets'][ps_key][trait] = {'layers': [], 'per_layer': []}

            common_ids = common_ids_per_ps[ps]
            centering_ids = common_ids_per_ps[centering_set]

            acts_i = all_acts[('instruct', ps)]
            acts_s = all_acts[('sleeper_lora', ps)]
            acts_b = all_acts[('benign_lora', ps)]

            # Centering set activations
            acts_i_center = all_acts[('instruct', centering_set)]
            acts_s_center = all_acts[('sleeper_lora', centering_set)]
            acts_b_center = all_acts[('benign_lora', centering_set)]

            print(f"\n  {ps} ({len(common_ids)} prompts, centering from {centering_set} ({len(centering_ids)} prompts)):")

            for layer in sweep_layers:
                try:
                    vector, _, _ = load_vector_with_baseline(
                        vector_experiment, trait, args.method, layer,
                        extraction_variant, args.component, args.position
                    )
                    vector = vector.float()
                except FileNotFoundError:
                    continue

                # =============================================================
                # 1. Standard per-prompt projections (both vs instruct)
                # =============================================================
                proj_i = per_prompt_projections(acts_i, common_ids, layer, vector, args.component)
                proj_s = per_prompt_projections(acts_s, common_ids, layer, vector, args.component)
                proj_b = per_prompt_projections(acts_b, common_ids, layer, vector, args.component)

                shift_s = proj_s - proj_i  # sleeper shift per prompt
                shift_b = proj_b - proj_i  # benign shift per prompt
                delta = shift_s - shift_b  # sleeper-benign delta per prompt

                # =============================================================
                # 2. Cross-prompt-set centering
                # =============================================================
                # Estimate generic LoRA offset from centering set (benign prompts)
                proj_i_center = per_prompt_projections(acts_i_center, centering_ids, layer, vector, args.component)
                proj_s_center = per_prompt_projections(acts_s_center, centering_ids, layer, vector, args.component)
                proj_b_center = per_prompt_projections(acts_b_center, centering_ids, layer, vector, args.component)

                # Estimated offsets from centering set
                offset_sleeper = np.mean(proj_s_center - proj_i_center)
                offset_benign = np.mean(proj_b_center - proj_i_center)

                # Centered shifts: remove the estimated generic offset
                shift_s_centered = shift_s - offset_sleeper
                shift_b_centered = shift_b - offset_benign

                # Centered delta: "beyond what benign prompts predict, how does sleeper differ?"
                delta_centered = shift_s_centered - shift_b_centered

                # Cohen's d on centered residuals
                d_centered = float(np.mean(delta_centered) / np.std(delta_centered)) if np.std(delta_centered) > 0 else 0.0

                # P-value for centered delta
                if np.std(delta_centered) > 0:
                    _, p_centered = stats.ttest_1samp(delta_centered, 0)
                    p_centered = float(p_centered)
                else:
                    p_centered = 1.0

                # =============================================================
                # 3. Direct sleeper-vs-benign (skip instruct)
                # =============================================================
                direct_delta = proj_s - proj_b
                d_direct = float(np.mean(direct_delta) / np.std(direct_delta)) if np.std(direct_delta) > 0 else 0.0

                # =============================================================
                # 4. Distribution statistics on the delta
                # =============================================================
                delta_dist = distribution_stats(delta)
                centered_dist = distribution_stats(delta_centered)

                # =============================================================
                # 5. Token-level analysis (SWiM, early tokens, bipolar)
                # =============================================================
                tok_stats_s = token_windowed_stats(acts_s, common_ids, layer, vector, args.component, args.token_window)
                tok_stats_i = token_windowed_stats(acts_i, common_ids, layer, vector, args.component, args.token_window)
                tok_stats_b = token_windowed_stats(acts_b, common_ids, layer, vector, args.component, args.token_window)

                # Early-token shift (response[:5] — matches vector extraction position)
                early_shift_s = tok_stats_s['early_mean'] - tok_stats_i['early_mean']
                early_shift_b = tok_stats_b['early_mean'] - tok_stats_i['early_mean']
                early_delta = early_shift_s - early_shift_b
                d_early = float(np.mean(early_delta) / np.std(early_delta)) if np.std(early_delta) > 0 else 0.0

                # SWiM (rolling max) shift
                swim_shift_s = tok_stats_s['rolling_max'] - tok_stats_i['rolling_max']
                swim_shift_b = tok_stats_b['rolling_max'] - tok_stats_i['rolling_max']
                swim_delta = swim_shift_s - swim_shift_b
                d_swim = float(np.mean(swim_delta) / np.std(swim_delta)) if np.std(swim_delta) > 0 else 0.0

                # Bipolar spread difference
                spread_delta = tok_stats_s['bipolar_spread'] - tok_stats_b['bipolar_spread']
                d_spread = float(np.mean(spread_delta) / np.std(spread_delta)) if np.std(spread_delta) > 0 else 0.0

                layer_result = {
                    'layer': layer,
                    # Standard (for reference)
                    'd_delta': round(float(np.mean(delta) / np.std(delta)) if np.std(delta) > 0 else 0.0, 3),
                    # Cross-prompt-set centered
                    'd_centered': round(d_centered, 3),
                    'p_centered': round(p_centered, 6),
                    'offset_sleeper': round(float(offset_sleeper), 4),
                    'offset_benign': round(float(offset_benign), 4),
                    # Direct LoRA-vs-LoRA
                    'd_direct': round(d_direct, 3),
                    # Token-level
                    'd_early': round(d_early, 3),
                    'd_swim': round(d_swim, 3),
                    'd_spread': round(d_spread, 3),
                    # Distribution stats
                    'delta_distribution': delta_dist,
                    'centered_distribution': centered_dist,
                }

                results['prompt_sets'][ps_key][trait]['layers'].append(layer)
                results['prompt_sets'][ps_key][trait]['per_layer'].append(layer_result)

            # Print summary table for this prompt set + trait
            per_layer = results['prompt_sets'][ps_key][trait]['per_layer']
            if per_layer:
                print(f"    {'Layer':>6} {'d_delta':>8} {'d_center':>8} {'p_center':>10} {'d_direct':>8} {'d_early':>8} {'d_swim':>8} {'d_spread':>8}")
                for r in per_layer:
                    stars = ''
                    if r['p_centered'] < 0.001: stars = '***'
                    elif r['p_centered'] < 0.01: stars = '**'
                    elif r['p_centered'] < 0.05: stars = '*'
                    print(f"    L{r['layer']:>4} {r['d_delta']:>+7.2f}σ {r['d_centered']:>+7.2f}σ {r['p_centered']:>9.4f}{stars:>3} {r['d_direct']:>+7.2f}σ {r['d_early']:>+7.2f}σ {r['d_swim']:>+7.2f}σ {r['d_spread']:>+7.2f}σ")

    # =========================================================================
    # Cross-prompt-set comparison summary
    # =========================================================================
    print(f"\n{'='*90}")
    print("CROSS-PROMPT-SET COMPARISON: Does centering reveal trigger-specific effects?")
    print(f"{'='*90}")
    print(f"Centering set: {centering_set}")
    print(f"If d_centered is large on triggered but small on safe/non_code, we have conditional detection.\n")

    for trait in traits:
        short = trait.split('/')[-1]
        print(f"  {short}:")
        print(f"    {'Prompt Set':>20} {'Peak d_delta':>12} {'Peak d_center':>12} {'Peak d_early':>12} {'Peak d_swim':>12} {'Peak d_spread':>12}")
        for ps in prompt_sets:
            ps_key = ps.replace('/', '_')
            data = results.get('prompt_sets', {}).get(ps_key, {}).get(trait, {})
            per_layer = data.get('per_layer', [])
            if not per_layer:
                print(f"    {ps:>20}: no data")
                continue

            peak_delta = max(per_layer, key=lambda r: abs(r['d_delta']))
            peak_center = max(per_layer, key=lambda r: abs(r['d_centered']))
            peak_early = max(per_layer, key=lambda r: abs(r['d_early']))
            peak_swim = max(per_layer, key=lambda r: abs(r['d_swim']))
            peak_spread = max(per_layer, key=lambda r: abs(r['d_spread']))

            print(f"    {ps:>20} {peak_delta['d_delta']:>+10.2f}σ {peak_center['d_centered']:>+10.2f}σ {peak_early['d_early']:>+10.2f}σ {peak_swim['d_swim']:>+10.2f}σ {peak_spread['d_spread']:>+10.2f}σ")
        print()

    # =========================================================================
    # Save
    # =========================================================================
    output_dir = Path(f"experiments/{args.experiment}/model_diff/cross_prompt_set_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        dump_compact(results, f)
    print(f"\nSaved to {output_dir / 'results.json'}")


if __name__ == '__main__':
    main()
