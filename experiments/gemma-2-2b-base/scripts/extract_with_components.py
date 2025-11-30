#!/usr/bin/env python3
"""
Extract activations from base model with component-level granularity.

Captures per-layer: residual, attention output, MLP output, K, V.

Input:
    - experiments/gemma-2-2b-base/extraction/{category}/{trait}/prompts.json
      Format: {"pairs": [{"positive": "...", "negative": "...", "category": "..."}, ...]}

Output:
    - experiments/gemma-2-2b-base/extraction/{category}/{trait}/activations/
        - residual.pt      # [n_examples, n_layers, hidden_dim]
        - attn_out.pt      # [n_examples, n_layers, hidden_dim]
        - mlp_out.pt       # [n_examples, n_layers, hidden_dim]
        - keys.pt          # [n_examples, n_layers, seq_len, n_kv_heads, head_dim]
        - values.pt        # [n_examples, n_layers, seq_len, n_kv_heads, head_dim]
        - metadata.json

Usage:
    python experiments/gemma-2-2b-base/scripts/extract_with_components.py --trait epistemic/uncertainty
"""

import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

BASE_MODEL = "google/gemma-2-2b"
EXPERIMENT_DIR = Path(__file__).parent.parent


@dataclass
class ComponentCapture:
    """Captures multiple component activations per layer."""
    residual: dict  # layer -> list of tensors
    attn_out: dict  # layer -> list of tensors
    mlp_out: dict   # layer -> list of tensors
    keys: dict      # layer -> list of tensors
    values: dict    # layer -> list of tensors

    @classmethod
    def create(cls, n_layers: int):
        return cls(
            residual={i: [] for i in range(n_layers)},
            attn_out={i: [] for i in range(n_layers)},
            mlp_out={i: [] for i in range(n_layers)},
            keys={i: [] for i in range(n_layers)},
            values={i: [] for i in range(n_layers)},
        )


def make_component_hooks(capture: ComponentCapture, n_layers: int):
    """Create hooks for all components at all layers."""
    hooks = []

    for layer_idx in range(n_layers):
        # Residual: output of full layer
        def make_residual_hook(layer=layer_idx):
            def hook(module, input, output):
                # output is the residual stream after this layer
                capture.residual[layer].append(output[0].detach().cpu())
            return hook

        # Attention output: output of self_attn
        def make_attn_hook(layer=layer_idx):
            def hook(module, input, output):
                # output[0] is attention output before residual add
                capture.attn_out[layer].append(output[0].detach().cpu())
            return hook

        # MLP output: output of mlp
        def make_mlp_hook(layer=layer_idx):
            def hook(module, input, output):
                capture.mlp_out[layer].append(output.detach().cpu())
            return hook

        # Keys: output of k_proj
        def make_key_hook(layer=layer_idx):
            def hook(module, input, output):
                capture.keys[layer].append(output.detach().cpu())
            return hook

        # Values: output of v_proj
        def make_value_hook(layer=layer_idx):
            def hook(module, input, output):
                capture.values[layer].append(output.detach().cpu())
            return hook

        hooks.append((layer_idx, 'residual', make_residual_hook()))
        hooks.append((layer_idx, 'attn', make_attn_hook()))
        hooks.append((layer_idx, 'mlp', make_mlp_hook()))
        hooks.append((layer_idx, 'key', make_key_hook()))
        hooks.append((layer_idx, 'value', make_value_hook()))

    return hooks


def extract_single_prompt(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    n_layers: int,
    max_new_tokens: int = 20,
    temperature: float = 0.7,
) -> tuple[str, ComponentCapture]:
    """Generate from prompt and capture all component activations."""

    capture = ComponentCapture.create(n_layers)
    hooks = make_component_hooks(capture, n_layers)

    # Register hooks
    handles = []
    for layer_idx, hook_type, hook_fn in hooks:
        if hook_type == 'residual':
            h = model.model.layers[layer_idx].register_forward_hook(hook_fn)
        elif hook_type == 'attn':
            h = model.model.layers[layer_idx].self_attn.register_forward_hook(hook_fn)
        elif hook_type == 'mlp':
            h = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
        elif hook_type == 'key':
            h = model.model.layers[layer_idx].self_attn.k_proj.register_forward_hook(hook_fn)
        elif hook_type == 'value':
            h = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(hook_fn)
        handles.append(h)

    try:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    finally:
        for h in handles:
            h.remove()

    return generated_text, capture


def aggregate_capture(capture: ComponentCapture, n_layers: int, method: str = 'mean') -> dict:
    """
    Aggregate captured activations across tokens/forward passes.

    For generation, there are multiple forward passes (one per token).
    We aggregate to get a single representation per layer.
    """
    result = {}

    for name, data in [
        ('residual', capture.residual),
        ('attn_out', capture.attn_out),
        ('mlp_out', capture.mlp_out),
        ('keys', capture.keys),
        ('values', capture.values),
    ]:
        layer_tensors = []
        for layer in range(n_layers):
            if not data[layer]:
                continue

            # Stack all forward passes for this layer
            # Each tensor is [batch=1, seq_len, hidden_dim] or similar
            stacked = torch.cat(data[layer], dim=1)  # concat along seq dim

            if method == 'mean':
                # Mean across all tokens
                aggregated = stacked.mean(dim=1).squeeze(0)  # [hidden_dim]
            elif method == 'last':
                # Last token only
                aggregated = stacked[0, -1, :]  # [hidden_dim]
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

            layer_tensors.append(aggregated)

        if layer_tensors:
            result[name] = torch.stack(layer_tensors)  # [n_layers, hidden_dim]

    return result


def run_extraction(
    trait: str,
    max_new_tokens: int = 20,
    temperature: float = 0.7,
    device: str = 'auto',
    val_split: float = 0.2,
):
    """Run full extraction pipeline for a trait."""

    trait_dir = EXPERIMENT_DIR / 'extraction' / trait
    prompts_file = trait_dir / 'prompts.json'

    if not prompts_file.exists():
        print(f"ERROR: {prompts_file} not found")
        return

    with open(prompts_file) as f:
        data = json.load(f)

    pairs = data['pairs']
    print(f"Loaded {len(pairs)} prompt pairs")

    # Split into train/val
    split_idx = int(len(pairs) * (1 - val_split))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"Split: {len(train_pairs)} train, {len(val_pairs)} val")

    # Load model
    print(f"Loading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"Model loaded: {n_layers} layers")

    def extract_set(pair_list: list, desc: str) -> tuple[dict, dict, list]:
        """Extract activations for a set of pairs."""
        pos_activations = {k: [] for k in ['residual', 'attn_out', 'mlp_out', 'keys', 'values']}
        neg_activations = {k: [] for k in ['residual', 'attn_out', 'mlp_out', 'keys', 'values']}
        generations = []

        for pair in tqdm(pair_list, desc=desc):
            # Positive prompt
            pos_text, pos_capture = extract_single_prompt(
                pair['positive'], model, tokenizer, n_layers, max_new_tokens, temperature
            )
            pos_agg = aggregate_capture(pos_capture, n_layers, method='mean')
            for k, v in pos_agg.items():
                pos_activations[k].append(v)

            # Negative prompt
            neg_text, neg_capture = extract_single_prompt(
                pair['negative'], model, tokenizer, n_layers, max_new_tokens, temperature
            )
            neg_agg = aggregate_capture(neg_capture, n_layers, method='mean')
            for k, v in neg_agg.items():
                neg_activations[k].append(v)

            generations.append({
                'positive_prompt': pair['positive'],
                'negative_prompt': pair['negative'],
                'positive_generation': pos_text,
                'negative_generation': neg_text,
                'category': pair.get('category', ''),
            })

        # Stack into tensors
        for k in pos_activations:
            if pos_activations[k]:
                pos_activations[k] = torch.stack(pos_activations[k])
            if neg_activations[k]:
                neg_activations[k] = torch.stack(neg_activations[k])

        return pos_activations, neg_activations, generations

    # Extract train set
    train_pos, train_neg, train_gens = extract_set(train_pairs, "Extracting train")

    # Extract val set
    val_pos, val_neg, val_gens = extract_set(val_pairs, "Extracting val")

    # Save outputs
    activations_dir = trait_dir / 'activations'
    activations_dir.mkdir(parents=True, exist_ok=True)

    # Save train activations
    for component in ['residual', 'attn_out', 'mlp_out', 'keys', 'values']:
        if isinstance(train_pos[component], torch.Tensor):
            combined = torch.cat([train_pos[component], train_neg[component]], dim=0)
            torch.save(combined, activations_dir / f'{component}.pt')
            print(f"Saved train {component}: {combined.shape}")

    # Save val activations
    val_dir = trait_dir / 'val_activations'
    val_dir.mkdir(parents=True, exist_ok=True)
    for component in ['residual', 'attn_out', 'mlp_out', 'keys', 'values']:
        if isinstance(val_pos[component], torch.Tensor):
            torch.save(val_pos[component], val_dir / f'{component}_pos.pt')
            torch.save(val_neg[component], val_dir / f'{component}_neg.pt')
            print(f"Saved val {component}: pos={val_pos[component].shape}, neg={val_neg[component].shape}")

    # Save generations
    with open(trait_dir / 'generations.json', 'w') as f:
        json.dump({'train': train_gens, 'val': val_gens}, f, indent=2)

    # Save metadata
    metadata = {
        'model': BASE_MODEL,
        'trait': trait,
        'n_layers': n_layers,
        'n_train_pairs': len(train_pairs),
        'n_val_pairs': len(val_pairs),
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'aggregation': 'mean',
        'components': ['residual', 'attn_out', 'mlp_out', 'keys', 'values'],
    }
    with open(activations_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Saved to {activations_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trait', type=str, required=True, help='Trait path (e.g., epistemic/uncertainty)')
    parser.add_argument('--max-new-tokens', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--val-split', type=float, default=0.2)
    args = parser.parse_args()

    run_extraction(
        trait=args.trait,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        val_split=args.val_split,
    )


if __name__ == '__main__':
    main()
