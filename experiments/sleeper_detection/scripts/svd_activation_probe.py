#!/usr/bin/env python3
"""
Project activations onto ΔBA singular vectors (weight-derived directions).

The contrastive SVD decode found that the sleeper-benign weight difference
encodes "unsafe/dangerous/system" semantics in late o_proj layers, but these
directions are orthogonal to behavioral trait vectors.

This script tests: do these weight-derived directions show TRIGGER-SPECIFIC
activation? i.e., do triggered prompts project higher onto ΔBA singular vectors
than benign prompts?

If yes → the signal exists in activations but trait probes look in the wrong direction.
If no → the weight difference is a static offset, not conditionally activated.

Input: LoRA adapter weights + raw activation .pt files
Output: Per-layer Cohen's d comparing prompt sets, projected onto SVD directions

Usage:
    python experiments/sleeper_detection/scripts/svd_activation_probe.py \
        --experiment sleeper_detection \
        --prompt-sets sleeper/triggered,sleeper/safe,sleeper/benign,sleeper/non_code \
        --top-k 5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import argparse
import json
import torch
import numpy as np
from scipy import stats

from analysis.model_diff.lora_trait_alignment import (
    load_adapter_state_dict, parse_lora_weights
)
from analysis.model_diff.compare_variants import load_raw_activations, get_response_mean
from utils.json import dump_compact


def compute_delta_ba_svd(adapters, scaling, layer, module, top_k=5):
    """
    Compute ΔBA = scaling * (B_s A_s - B_b A_b), return top-k SVD components.

    Returns: U[:, :top_k], S[:top_k], Vh[:top_k, :] or None if module missing.
    """
    variant_names = sorted(adapters.keys())
    if len(variant_names) != 2:
        raise ValueError(f"Expected 2 variants, got {variant_names}")

    v0, v1 = variant_names  # benign_lora, sleeper_lora (alphabetical)
    if module not in adapters[v0].get(layer, {}) or module not in adapters[v1].get(layer, {}):
        return None

    A0, B0 = adapters[v0][layer][module]['A'], adapters[v0][layer][module]['B']
    A1, B1 = adapters[v1][layer][module]['A'], adapters[v1][layer][module]['B']

    BA0 = scaling * (B0 @ A0).float()
    BA1 = scaling * (B1 @ A1).float()
    delta_BA = BA1 - BA0  # sleeper - benign

    U, S, Vh = torch.linalg.svd(delta_BA, full_matrices=False)
    return U[:, :top_k], S[:top_k], Vh[:top_k, :]


def main():
    parser = argparse.ArgumentParser(description="Project activations onto ΔBA SVD directions")
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--prompt-sets', required=True,
                        help='Comma-separated prompt sets to analyze')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top singular vectors to use')
    parser.add_argument('--component', default='residual')
    args = parser.parse_args()

    prompt_sets = [p.strip() for p in args.prompt_sets.split(',')]
    exp_dir = Path(f"experiments/{args.experiment}")

    # =========================================================================
    # Load LoRA adapters (CPU, fast)
    # =========================================================================
    with open(exp_dir / "config.json") as f:
        config = json.load(f)

    lora_variants = {}
    for name, vconf in config['model_variants'].items():
        if 'lora' in vconf:
            lora_variants[name] = vconf['lora']

    adapters = {}
    adapter_cfgs = {}
    for name, adapter_path in lora_variants.items():
        print(f"Loading {name}: {adapter_path}")
        cfg_path = Path(adapter_path) / "adapter_config.json"
        with open(cfg_path) as f:
            adapter_cfgs[name] = json.load(f)
        state_dict = load_adapter_state_dict(adapter_path)
        adapters[name] = parse_lora_weights(state_dict)
        print(f"  {len(adapters[name])} layers")

    first_cfg = next(iter(adapter_cfgs.values()))
    scaling = first_cfg['lora_alpha'] / first_cfg['r']
    print(f"Rank={first_cfg['r']}, scaling={scaling}")

    # Residual-facing modules
    first_layer_data = next(iter(next(iter(adapters.values())).values()))
    residual_modules = sorted(m for m in first_layer_data.keys()
                              if 'o_proj' in m or 'down_proj' in m)
    print(f"Residual-facing modules: {residual_modules}")

    # =========================================================================
    # Load raw activations
    # =========================================================================
    print("\nLoading activations...")
    # Load for sleeper_lora and instruct (to compute sleeper - instruct diffs)
    # and benign_lora (to compute benign - instruct diffs)
    variants = ['instruct', 'sleeper_lora', 'benign_lora']
    all_acts = {}
    for variant in variants:
        for ps in prompt_sets:
            print(f"  {variant} / {ps}...", end=' ')
            try:
                raw = load_raw_activations(args.experiment, variant, ps)
                all_acts[(variant, ps)] = {d['prompt_id']: d for d in raw}
                print(f"{len(raw)} prompts")
            except FileNotFoundError:
                print("NOT FOUND")

    # Common prompt IDs per prompt set
    common_ids_per_ps = {}
    for ps in prompt_sets:
        sets = [set(all_acts[(v, ps)].keys()) for v in variants if (v, ps) in all_acts]
        if len(sets) == len(variants):
            common_ids_per_ps[ps] = sorted(set.intersection(*sets))
            print(f"  {ps}: {len(common_ids_per_ps[ps])} common prompts")

    # Determine layers from data
    sample_data = next(iter(next(iter(all_acts.values())).values()))
    all_layers = sorted(sample_data['response']['activations'].keys())
    adapter_layers = sorted(next(iter(adapters.values())).keys())
    common_layers = sorted(set(all_layers) & set(adapter_layers))
    print(f"Layers: {len(common_layers)} common")

    # =========================================================================
    # Main analysis: project activations onto ΔBA SVD directions
    # =========================================================================
    results = {
        'experiment': args.experiment,
        'top_k': args.top_k,
        'analyses': {}
    }

    for module in residual_modules:
        mod_short = module.split('.')[-1] if '.' in module else module
        print(f"\n{'='*80}")
        print(f"Module: {mod_short}")
        print(f"{'='*80}")

        # Header
        ps_labels = [ps.split('/')[-1] for ps in prompt_sets]
        header = f"{'Layer':>5} {'SV':>3} {'σ':>8}"
        for label in ps_labels:
            header += f"  {label + '_s':>10} {label + '_b':>10} {'Δ(s-b)':>8}"
        header += f"  {'best_d':>8} {'best_ps':>12}"
        print(header)
        print("-" * len(header))

        results['analyses'][mod_short] = {}

        for layer in common_layers:
            svd_result = compute_delta_ba_svd(adapters, scaling, layer, module, args.top_k)
            if svd_result is None:
                continue

            U, S, Vh = svd_result
            layer_results = []

            for k in range(args.top_k):
                sv_dir = U[:, k]  # [hidden_dim] — output direction in residual stream
                sigma = S[k].item()

                row = f"{layer:>5} {k:>3} {sigma:>8.4f}"
                sv_result = {'sigma': round(sigma, 6), 'prompt_sets': {}}
                best_d = 0
                best_ps = ''

                for ps in prompt_sets:
                    if ps not in common_ids_per_ps:
                        row += f"  {'N/A':>10} {'N/A':>10} {'N/A':>8}"
                        continue

                    ids = common_ids_per_ps[ps]

                    # Project sleeper and instruct activations onto this SVD direction
                    sleeper_projs = []
                    benign_projs = []
                    instruct_projs = []
                    for pid in ids:
                        s_mean = get_response_mean(all_acts[('sleeper_lora', ps)][pid],
                                                   layer, args.component)
                        b_mean = get_response_mean(all_acts[('benign_lora', ps)][pid],
                                                   layer, args.component)
                        i_mean = get_response_mean(all_acts[('instruct', ps)][pid],
                                                   layer, args.component)

                        sleeper_projs.append((s_mean @ sv_dir).item())
                        benign_projs.append((b_mean @ sv_dir).item())
                        instruct_projs.append((i_mean @ sv_dir).item())

                    s_arr = np.array(sleeper_projs)
                    b_arr = np.array(benign_projs)

                    # Cohen's d: sleeper vs instruct, benign vs instruct
                    s_mean_val = s_arr.mean()
                    b_mean_val = b_arr.mean()
                    delta = s_mean_val - b_mean_val

                    # Paired d for sleeper-benign difference
                    diffs = s_arr - b_arr
                    if diffs.std() > 0:
                        paired_d = diffs.mean() / diffs.std()
                    else:
                        paired_d = 0.0

                    row += f"  {s_mean_val:>10.4f} {b_mean_val:>10.4f} {paired_d:>8.2f}"

                    ps_short = ps.split('/')[-1]
                    sv_result['prompt_sets'][ps_short] = {
                        'sleeper_mean': round(s_mean_val, 6),
                        'benign_mean': round(b_mean_val, 6),
                        'delta': round(delta, 6),
                        'paired_d': round(paired_d, 4),
                    }

                    if abs(paired_d) > abs(best_d):
                        best_d = paired_d
                        best_ps = ps_short

                row += f"  {best_d:>8.2f} {best_ps:>12}"
                print(row)
                layer_results.append(sv_result)

            results['analyses'][mod_short][str(layer)] = layer_results

    # =========================================================================
    # Summary: find layers/SVs where triggered > benign (trigger-specific)
    # =========================================================================
    print(f"\n{'='*80}")
    print("SUMMARY: Trigger-specificity check")
    print("Looking for SVD directions where triggered prompts diverge from benign/safe")
    print(f"{'='*80}")

    for module in residual_modules:
        mod_short = module.split('.')[-1] if '.' in module else module
        print(f"\n{mod_short}:")
        print(f"  {'Layer':>5} {'SV':>3} {'σ':>8}  {'triggered':>10} {'safe':>10} {'benign':>10} {'non_code':>10}  {'trig-benign':>12}")

        hits = []
        for layer in common_layers:
            svd_result = compute_delta_ba_svd(adapters, scaling, layer, module, args.top_k)
            if svd_result is None:
                continue
            U, S, Vh = svd_result

            for k in range(min(3, args.top_k)):  # Focus on top-3 for summary
                layer_data = results['analyses'].get(mod_short, {}).get(str(layer), [])
                if k >= len(layer_data):
                    continue
                sv_data = layer_data[k]

                ds = {}
                for ps_short in ['triggered', 'safe', 'benign', 'non_code']:
                    if ps_short in sv_data['prompt_sets']:
                        ds[ps_short] = sv_data['prompt_sets'][ps_short]['paired_d']
                    else:
                        ds[ps_short] = 0.0

                # Trigger-specificity: |triggered_d| > |benign_d| and |triggered_d| > |safe_d|
                trig_benign_diff = abs(ds.get('triggered', 0)) - abs(ds.get('benign', 0))

                if abs(ds.get('triggered', 0)) > 0.5:  # Only show meaningful effects
                    sigma = S[k].item()
                    row = f"  {layer:>5} {k:>3} {sigma:>8.4f}"
                    for ps_short in ['triggered', 'safe', 'benign', 'non_code']:
                        row += f"  {ds.get(ps_short, 0):>10.2f}"
                    row += f"  {trig_benign_diff:>12.2f}"
                    print(row)
                    hits.append((layer, k, ds, trig_benign_diff))

        if not hits:
            print("  No SVD directions with |triggered paired_d| > 0.5")

        # Best trigger-specific hit
        trigger_specific = [h for h in hits if h[3] > 0.3]
        if trigger_specific:
            trigger_specific.sort(key=lambda h: h[3], reverse=True)
            best = trigger_specific[0]
            print(f"\n  Best trigger-specific: L{best[0]} SV{best[1]} "
                  f"(triggered d={best[2]['triggered']:.2f}, benign d={best[2]['benign']:.2f}, "
                  f"diff={best[3]:.2f})")
        else:
            print(f"\n  No trigger-specific hits (all directions show similar d across prompt sets)")

    # =========================================================================
    # Save results
    # =========================================================================
    out_dir = exp_dir / "model_diff" / "svd_activation_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        dump_compact(results, f)
    print(f"\nResults saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
