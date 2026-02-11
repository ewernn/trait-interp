#!/usr/bin/env python3
"""
Weight-space analysis of sleeper vs benign LoRA.

Three analyses that bypass activations entirely:

1. Contrastive weight vector — compute BA_sleeper - BA_benign per layer per module,
   project trait vectors onto this difference. The shared finetuning cancels;
   what remains is the deception-specific weight perturbation.

2. SVD spectral alignment — decompose each LoRA's BA = UΣV^T, check which singular
   components align with trait directions. Compare spectral structure between LoRAs.

3. Logit lens on weight diff — for residual-facing modules (o_proj, down_proj),
   project BA_sleeper - BA_benign through unembedding to decode what the weight
   difference "means" in token space.

No GPU needed. Runs on CPU from adapter_model.safetensors files.

Input: LoRA adapter weights + trait vectors
Output: Per-layer alignment scores, SVD spectral analysis, logit lens tokens

Usage:
    python experiments/sleeper_detection/scripts/weight_space_analysis.py \
        --experiment sleeper_detection \
        --traits bs/concealment,alignment/deception,pv_natural/sycophancy,hum/formality \
        --method probe --position response_all
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
from utils.json import dump_compact


def load_trait_vectors(experiment, trait, method, position, component='residual',
                       extraction_variant='base'):
    """Load trait vectors for all available layers. Returns {layer: tensor}."""
    vec_dir = (Path(f"experiments/{experiment}/extraction") / trait /
               extraction_variant / "vectors" / position / component / method)
    if not vec_dir.exists():
        return {}
    vectors = {}
    for pt_file in sorted(vec_dir.glob("layer*.pt")):
        layer = int(pt_file.stem.replace("layer", ""))
        vectors[layer] = torch.load(pt_file, map_location="cpu", weights_only=True).float()
    return vectors


def load_unembedding(model_name="meta-llama/Llama-3.1-8B"):
    """Load unembedding matrix (lm_head.weight) and tokenizer."""
    from transformers import AutoTokenizer, AutoConfig
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Try loading just lm_head from safetensors index
    try:
        index_path = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        lm_head_file = index['weight_map'].get('lm_head.weight')
        if lm_head_file:
            shard_path = hf_hub_download(model_name, lm_head_file)
            weights = load_file(shard_path)
            return weights['lm_head.weight'].float(), tokenizer
    except Exception:
        pass

    # Fallback: load full model config to check tied embeddings
    try:
        embed_file = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(embed_file) as f:
            index = json.load(f)
        embed_shard = index['weight_map'].get('model.embed_tokens.weight')
        if embed_shard:
            shard_path = hf_hub_download(model_name, embed_shard)
            weights = load_file(shard_path)
            return weights['model.embed_tokens.weight'].float(), tokenizer
    except Exception:
        pass

    raise RuntimeError(f"Could not load unembedding for {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Weight-space sleeper detection analysis")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--traits", required=True, help="Comma-separated trait names")
    parser.add_argument("--method", default="probe")
    parser.add_argument("--position", default="response_all")
    parser.add_argument("--component", default="residual")
    parser.add_argument("--logit-lens", action="store_true",
                        help="Decode weight diffs through unembedding (downloads model weights)")
    parser.add_argument("--top-k-tokens", type=int, default=10,
                        help="Number of top tokens for logit lens")
    args = parser.parse_args()

    traits = [t.strip() for t in args.traits.split(',')]
    exp_dir = Path(f"experiments/{args.experiment}")

    # =========================================================================
    # Load LoRA adapters
    # =========================================================================
    # Read config to find LoRA variants
    with open(exp_dir / "config.json") as f:
        config = json.load(f)

    lora_variants = {}
    for name, vconf in config['model_variants'].items():
        if 'lora' in vconf:
            lora_variants[name] = vconf['lora']

    print(f"LoRA variants: {list(lora_variants.keys())}")

    # Load and parse all adapters
    adapters = {}  # {name: {layer: {module: {A, B}}}}
    adapter_cfgs = {}
    for name, adapter_path in lora_variants.items():
        print(f"Loading {name}: {adapter_path}")
        cfg_path = Path(adapter_path) / "adapter_config.json"
        with open(cfg_path) as f:
            adapter_cfgs[name] = json.load(f)
        state_dict = load_adapter_state_dict(adapter_path)
        adapters[name] = parse_lora_weights(state_dict)
        print(f"  {len(adapters[name])} layers")

    # Get scaling
    first_cfg = next(iter(adapter_cfgs.values()))
    r = first_cfg['r']
    alpha = first_cfg['lora_alpha']
    scaling = alpha / r
    print(f"Rank={r}, alpha={alpha}, scaling={scaling}")

    # Get all modules from first adapter
    first_layer_data = next(iter(next(iter(adapters.values())).values()))
    all_modules = sorted(first_layer_data.keys())
    # Residual-facing modules (output goes to residual stream)
    residual_modules = [m for m in all_modules if 'o_proj' in m or 'down_proj' in m]
    print(f"All modules: {all_modules}")
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
        else:
            print(f"  {trait}: NOT FOUND")

    # Get common layers
    adapter_layers = sorted(next(iter(adapters.values())).keys())
    vector_layers = sorted(next(iter(trait_vectors.values())).keys())
    common_layers = sorted(set(adapter_layers) & set(vector_layers))
    print(f"\nCommon layers: {len(common_layers)} (adapters have {len(adapter_layers)}, vectors have {len(vector_layers)})")

    variant_names = sorted(adapters.keys())

    # =========================================================================
    # Analysis 1: Contrastive weight vector (BA_sleeper - BA_benign)
    # =========================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS 1: Contrastive weight vector — v^T · (BA_sleeper - BA_benign)")
    print("Shared finetuning cancels. What remains is deception-specific.")
    print(f"{'='*80}")

    contrastive_results = {}

    for module in residual_modules:
        short = module.split('.')[-1]
        print(f"\n  [{short}]")
        print(f"  {'Layer':>6}", end="")
        for trait in traits:
            print(f"  {trait.split('/')[-1]:>14}", end="")
        print(f"  {'||ΔBA||':>10}")

        for layer in common_layers:
            # Compute BA for each variant
            BAs = {}
            for vname in variant_names:
                if layer in adapters[vname] and module in adapters[vname][layer]:
                    ab = adapters[vname][layer][module]
                    BAs[vname] = scaling * (ab['B'] @ ab['A'])

            if len(BAs) < 2:
                continue

            # Contrastive: sleeper - benign
            v1, v2 = variant_names[0], variant_names[1]
            # Identify which is sleeper, which is benign
            sleeper_name = [n for n in variant_names if 'sleeper' in n]
            benign_name = [n for n in variant_names if 'benign' in n]
            if sleeper_name and benign_name:
                delta_BA = BAs[sleeper_name[0]] - BAs[benign_name[0]]
            else:
                delta_BA = BAs[v1] - BAs[v2]

            delta_norm = delta_BA.norm().item()

            print(f"  L{layer:>4}", end="")
            layer_results = {}
            for trait in traits:
                if trait not in trait_vectors or layer not in trait_vectors[trait]:
                    print(f"  {'---':>14}", end="")
                    continue
                v = trait_vectors[trait][layer]

                # Project trait vector onto contrastive weight diff
                # v^T · ΔBA gives [in_dim] — the input-modulated trait push
                vTdBA = v @ delta_BA  # [in_dim]
                alignment = vTdBA.norm().item()

                # Also compute signed projection: cosine between v and
                # the mean column of ΔBA (average output direction)
                mean_col = delta_BA.mean(dim=1)  # [out_dim]
                if mean_col.norm() > 0:
                    cosine = (v @ mean_col / (v.norm() * mean_col.norm())).item()
                else:
                    cosine = 0.0

                layer_results[trait] = {
                    'alignment': round(alignment, 6),
                    'cosine': round(cosine, 6),
                }
                print(f"  {cosine:>+13.4f}", end="")

            print(f"  {delta_norm:>9.2f}")

            contrastive_results.setdefault(module, {})[layer] = {
                'traits': layer_results,
                'delta_norm': round(delta_norm, 4),
            }

    # =========================================================================
    # Analysis 2: SVD spectral alignment
    # =========================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS 2: SVD spectral alignment — which singular components align with traits?")
    print(f"{'='*80}")

    svd_results = {}

    for module in residual_modules:
        short = module.split('.')[-1]
        print(f"\n  [{short}]")

        for layer in common_layers:
            layer_svd = {}
            for vname in variant_names:
                if layer not in adapters[vname] or module not in adapters[vname][layer]:
                    continue
                ab = adapters[vname][layer][module]
                BA = scaling * (ab['B'] @ ab['A'])

                # SVD of BA
                U, S, Vh = torch.linalg.svd(BA, full_matrices=False)
                # U: [out_dim, r], S: [r], Vh: [r, in_dim]

                # For each trait, compute alignment with each singular component
                trait_alignments = {}
                for trait in traits:
                    if trait not in trait_vectors or layer not in trait_vectors[trait]:
                        continue
                    v = trait_vectors[trait][layer]
                    v_norm = v / v.norm()

                    # Alignment of trait vector with each left singular vector
                    # (left SVs are output directions — same space as trait vector)
                    alignments = [(v_norm @ U[:, i]).abs().item() for i in range(len(S))]
                    # Weighted by singular value
                    weighted = [alignments[i] * S[i].item() for i in range(len(S))]

                    trait_alignments[trait] = {
                        'raw': [round(a, 4) for a in alignments],
                        'weighted': [round(w, 4) for w in weighted],
                        'total_weighted': round(sum(weighted), 4),
                    }

                layer_svd[vname] = {
                    'singular_values': [round(s, 4) for s in S.tolist()],
                    'trait_alignments': trait_alignments,
                }

            svd_results.setdefault(module, {})[layer] = layer_svd

        # Print summary: total weighted alignment per variant per trait
        print(f"\n  Total weighted alignment (Σ |v·U_i| * σ_i):")
        print(f"  {'Layer':>6}", end="")
        for vname in variant_names:
            for trait in traits:
                print(f"  {vname[:7]}_{trait.split('/')[-1][:6]:>16}", end="")
        print()

        for layer in common_layers:
            print(f"  L{layer:>4}", end="")
            for vname in variant_names:
                layer_data = svd_results.get(module, {}).get(layer, {}).get(vname, {})
                for trait in traits:
                    ta = layer_data.get('trait_alignments', {}).get(trait, {})
                    tw = ta.get('total_weighted', 0)
                    print(f"  {tw:>23.4f}", end="")
            print()

    # =========================================================================
    # Analysis 3: Per-variant individual alignment (for reference)
    # =========================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS 3: Individual LoRA alignment — ||v^T · BA|| per variant")
    print("(Same as existing lora_trait_alignment.py, for comparison)")
    print(f"{'='*80}")

    individual_results = {}

    for module in residual_modules:
        short = module.split('.')[-1]
        print(f"\n  [{short}]")
        print(f"  {'Layer':>6}", end="")
        for vname in variant_names:
            for trait in traits:
                short_t = trait.split('/')[-1][:8]
                print(f"  {vname[:6]}_{short_t:>14}", end="")
        print()

        for layer in common_layers:
            print(f"  L{layer:>4}", end="")
            for vname in variant_names:
                if layer not in adapters[vname] or module not in adapters[vname][layer]:
                    for _ in traits:
                        print(f"  {'---':>21}", end="")
                    continue
                ab = adapters[vname][layer][module]
                BA = scaling * (ab['B'] @ ab['A'])

                for trait in traits:
                    if trait not in trait_vectors or layer not in trait_vectors[trait]:
                        print(f"  {'---':>21}", end="")
                        continue
                    v = trait_vectors[trait][layer]
                    alignment = (v @ BA).norm().item()
                    print(f"  {alignment:>21.4f}", end="")

                    individual_results.setdefault(module, {}).setdefault(vname, {}).setdefault(trait, {})[layer] = round(alignment, 6)
            print()

    # =========================================================================
    # Analysis 4: Logit lens (optional)
    # =========================================================================
    logit_lens_results = {}
    if args.logit_lens:
        print(f"\n{'='*80}")
        print("ANALYSIS 4: Logit lens — decode contrastive weight diff through unembedding")
        print(f"{'='*80}")

        model_name = config['model_variants'].get('base', {}).get('model',
                     config['model_variants'].get('instruct', {}).get('model'))
        print(f"Loading unembedding from {model_name}...")
        unembed, tokenizer = load_unembedding(model_name)
        # unembed: [vocab_size, hidden_dim]

        for module in residual_modules:
            short = module.split('.')[-1]
            print(f"\n  [{short}]")

            for layer in common_layers:
                BAs = {}
                for vname in variant_names:
                    if layer in adapters[vname] and module in adapters[vname][layer]:
                        ab = adapters[vname][layer][module]
                        BAs[vname] = scaling * (ab['B'] @ ab['A'])

                if len(BAs) < 2:
                    continue

                sleeper_name = [n for n in variant_names if 'sleeper' in n]
                benign_name = [n for n in variant_names if 'benign' in n]
                if sleeper_name and benign_name:
                    delta_BA = BAs[sleeper_name[0]] - BAs[benign_name[0]]
                else:
                    delta_BA = BAs[variant_names[0]] - BAs[variant_names[1]]

                # Mean output direction of the contrastive weight diff
                mean_dir = delta_BA.mean(dim=1)  # [hidden_dim]
                if mean_dir.norm() < 1e-8:
                    continue

                # Project through unembedding
                logits = unembed @ mean_dir  # [vocab_size]
                top_k = torch.topk(logits, args.top_k_tokens)
                bot_k = torch.topk(logits, args.top_k_tokens, largest=False)

                top_tokens = [(tokenizer.decode([idx]), round(val.item(), 3))
                              for idx, val in zip(top_k.indices, top_k.values)]
                bot_tokens = [(tokenizer.decode([idx]), round(val.item(), 3))
                              for idx, val in zip(bot_k.indices, bot_k.values)]

                print(f"\n  L{layer}: ||mean_dir|| = {mean_dir.norm().item():.4f}")
                print(f"    Top (sleeper > benign): {', '.join(f'{t}({v:+.3f})' for t, v in top_tokens)}")
                print(f"    Bot (benign > sleeper): {', '.join(f'{t}({v:+.3f})' for t, v in bot_tokens)}")

                logit_lens_results.setdefault(module, {})[layer] = {
                    'top_tokens': top_tokens,
                    'bot_tokens': bot_tokens,
                    'mean_dir_norm': round(mean_dir.norm().item(), 6),
                }

    # =========================================================================
    # Summary: specificity check
    # =========================================================================
    print(f"\n{'='*80}")
    print("SPECIFICITY: Is the contrastive alignment larger for deception traits than controls?")
    print(f"{'='*80}")

    for module in residual_modules:
        short = module.split('.')[-1]
        print(f"\n  [{short}] — sum of |cosine| across layers:")
        for trait in traits:
            total_cos = sum(
                abs(contrastive_results.get(module, {}).get(l, {}).get('traits', {}).get(trait, {}).get('cosine', 0))
                for l in common_layers
            )
            total_align = sum(
                contrastive_results.get(module, {}).get(l, {}).get('traits', {}).get(trait, {}).get('alignment', 0)
                for l in common_layers
            )
            print(f"    {trait:>30}: Σ|cos| = {total_cos:.4f}, Σalign = {total_align:.4f}")

    # =========================================================================
    # Save
    # =========================================================================
    output_dir = exp_dir / "model_diff" / "weight_space_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'experiment': args.experiment,
        'method': args.method,
        'position': args.position,
        'scaling': scaling,
        'rank': r,
        'variants': variant_names,
        'contrastive': {m: {str(l): v for l, v in layers.items()}
                        for m, layers in contrastive_results.items()},
        'individual': individual_results,
    }

    with open(output_dir / 'results.json', 'w') as f:
        dump_compact(results, f)
    print(f"\nSaved to {output_dir / 'results.json'}")


if __name__ == '__main__':
    main()
