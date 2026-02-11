#!/usr/bin/env python3
"""
SVD the contrastive ΔBA (sleeper - benign) and decode top singular vectors.

For each residual-facing module (down_proj, o_proj) at each layer:
  1. Compute ΔBA = scaling * (B_s A_s - B_b A_b)
  2. SVD → U, S, Vh  (U cols are output directions in residual space)
  3. Decode top-k left singular vectors through unembedding → token meaning
  4. Report singular value spectrum (concentration)
  5. Project trait vectors onto top singular vectors (overlap)

Input: LoRA adapter weights
Output: Per-layer decoded singular vectors + trait overlap

Usage:
    python experiments/sleeper_detection/scripts/contrastive_svd_decode.py \
        --experiment sleeper_detection \
        --traits bs/concealment,alignment/deception,pv_natural/sycophancy,hum/formality \
        --method probe --position response_all \
        --top-k-sv 5 --top-k-tokens 15
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import argparse
import json
import torch
import numpy as np

from analysis.model_diff.lora_trait_alignment import (
    load_adapter_state_dict, parse_lora_weights
)
from experiments.sleeper_detection.scripts.weight_space_analysis import (
    load_trait_vectors, load_unembedding
)
from utils.json import dump_compact


def main():
    parser = argparse.ArgumentParser(description="SVD decode of contrastive ΔBA")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--traits", required=True, help="Comma-separated trait names")
    parser.add_argument("--method", default="probe")
    parser.add_argument("--position", default="response_all")
    parser.add_argument("--component", default="residual")
    parser.add_argument("--top-k-sv", type=int, default=5,
                        help="Number of top singular vectors to decode")
    parser.add_argument("--top-k-tokens", type=int, default=15,
                        help="Number of top/bottom tokens per singular vector")
    args = parser.parse_args()

    traits = [t.strip() for t in args.traits.split(',')]
    exp_dir = Path(f"experiments/{args.experiment}")

    # Load config
    with open(exp_dir / "config.json") as f:
        config = json.load(f)

    # Load LoRA adapters
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
    r = first_cfg['r']
    alpha = first_cfg['lora_alpha']
    scaling = alpha / r
    print(f"Rank={r}, alpha={alpha}, scaling={scaling}")

    # Identify sleeper vs benign
    sleeper_name = next(n for n in adapters if 'sleeper' in n)
    benign_name = next(n for n in adapters if 'benign' in n)

    # Residual-facing modules
    first_layer = next(iter(adapters[sleeper_name].values()))
    all_modules = sorted(first_layer.keys())
    residual_modules = [m for m in all_modules if 'o_proj' in m or 'down_proj' in m]
    print(f"Residual-facing: {residual_modules}")

    # Load trait vectors
    print(f"\nLoading trait vectors ({args.method}, {args.position})...")
    trait_vectors = {}
    for trait in traits:
        vecs = load_trait_vectors(args.experiment, trait, args.method, args.position,
                                  args.component)
        if vecs:
            trait_vectors[trait] = vecs
            print(f"  {trait}: {len(vecs)} layers")

    common_layers = sorted(
        set(adapters[sleeper_name].keys()) & set(adapters[benign_name].keys())
    )

    # Load unembedding
    model_name = config['model_variants'].get('base', {}).get('model',
                 config['model_variants'].get('instruct', {}).get('model'))
    print(f"\nLoading unembedding from {model_name}...")
    unembed, tokenizer = load_unembedding(model_name)
    print(f"  Unembedding shape: {unembed.shape}")

    # Main analysis
    results = {}

    for module in residual_modules:
        short = module.split('.')[-1]
        print(f"\n{'='*80}")
        print(f"  {short}")
        print(f"{'='*80}")

        for layer in common_layers:
            ab_s = adapters[sleeper_name].get(layer, {}).get(module)
            ab_b = adapters[benign_name].get(layer, {}).get(module)
            if ab_s is None or ab_b is None:
                continue

            BA_s = scaling * (ab_s['B'] @ ab_s['A'])
            BA_b = scaling * (ab_b['B'] @ ab_b['A'])
            delta_BA = BA_s - BA_b  # [out_dim, in_dim]

            # SVD of contrastive difference
            U, S, Vh = torch.linalg.svd(delta_BA, full_matrices=False)

            # Singular value spectrum
            total_var = (S ** 2).sum().item()
            cumvar = torch.cumsum(S ** 2, dim=0) / total_var

            print(f"\n  L{layer}: ||ΔBA|| = {delta_BA.norm():.4f}")
            print(f"    Singular values: {', '.join(f'{s:.4f}' for s in S[:args.top_k_sv])}")
            print(f"    Cumulative variance: {', '.join(f'{c:.1%}' for c in cumvar[:args.top_k_sv])}")

            # Trait overlap with top singular vectors
            trait_overlaps = {}
            for trait in traits:
                if trait not in trait_vectors or layer not in trait_vectors[trait]:
                    continue
                v = trait_vectors[trait][layer]
                v_norm = v / v.norm()

                # |cosine| of trait vector with each left SV (output direction)
                cosines = [(v_norm @ U[:, i]).item() for i in range(min(len(S), args.top_k_sv))]
                trait_overlaps[trait] = cosines

            short_traits = [t.split('/')[-1][:8] for t in traits]
            print(f"    Trait overlap with top-{args.top_k_sv} left SVs (cosine):")
            print(f"      {'':>10}", end="")
            for i in range(min(len(S), args.top_k_sv)):
                print(f"  SV{i} (σ={S[i]:.3f})", end="")
            print()
            for trait in traits:
                if trait in trait_overlaps:
                    st = trait.split('/')[-1][:10]
                    print(f"      {st:>10}", end="")
                    for c in trait_overlaps[trait]:
                        print(f"  {c:>+13.4f}", end="")
                    print()

            # Decode top singular vectors through unembedding
            print(f"    Logit lens on top-{args.top_k_sv} left SVs:")
            layer_sv_tokens = []
            for i in range(min(len(S), args.top_k_sv)):
                u_i = U[:, i]  # [out_dim]
                logits = unembed @ u_i  # [vocab_size]
                top_k = torch.topk(logits, args.top_k_tokens)
                bot_k = torch.topk(logits, args.top_k_tokens, largest=False)

                top_tokens = [(tokenizer.decode([idx]), round(val.item(), 3))
                              for idx, val in zip(top_k.indices, top_k.values)]
                bot_tokens = [(tokenizer.decode([idx]), round(val.item(), 3))
                              for idx, val in zip(bot_k.indices, bot_k.values)]

                print(f"      SV{i} (σ={S[i]:.4f}):")
                print(f"        +: {', '.join(f'{t}({v:+.2f})' for t, v in top_tokens[:10])}")
                print(f"        -: {', '.join(f'{t}({v:+.2f})' for t, v in bot_tokens[:10])}")

                layer_sv_tokens.append({
                    'sv_index': i,
                    'sigma': round(S[i].item(), 6),
                    'cumvar': round(cumvar[i].item(), 4),
                    'top_tokens': top_tokens,
                    'bot_tokens': bot_tokens,
                    'trait_cosines': {t: round(trait_overlaps[t][i], 6)
                                     for t in trait_overlaps},
                })

            results.setdefault(module, {})[layer] = {
                'delta_norm': round(delta_BA.norm().item(), 4),
                'singular_values': [round(s, 6) for s in S.tolist()],
                'cumvar': [round(c, 4) for c in cumvar.tolist()],
                'svs': layer_sv_tokens,
            }

    # Save
    output_dir = exp_dir / "model_diff" / "contrastive_svd_decode"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        dump_compact({'experiment': args.experiment, 'modules': results}, f)
    print(f"\nSaved to {output_dir / 'results.json'}")


if __name__ == '__main__':
    main()
