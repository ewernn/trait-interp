#!/usr/bin/env python3
"""
Tier 3 Capture: Layer Internal States

Captures complete internal states for a SINGLE layer to understand HOW it processes
the trait. Includes Q/K/V projections, per-head attention weights, and all 9216 MLP
neurons.

Usage:
    # Single prompt, layer 16
    python inference/capture_tier3.py \
        --experiment gemma_2b_cognitive_nov20 \
        --trait refusal \
        --prompts "What is the capital of France?" \
        --layer 16

    # Multiple prompts, custom layer
    python inference/capture_tier3.py \
        --experiment gemma_2b_cognitive_nov20 \
        --trait refusal \
        --prompts-file test_prompts.txt \
        --layer 20 \
        --method probe

    # With JSON export for visualization
    python inference/capture_tier3.py \
        --experiment gemma_2b_cognitive_nov20 \
        --trait refusal \
        --prompts "Test prompt" \
        --layer 16 \
        --save-json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from traitlens import HookManager


# ============================================================================
# Display Names (from visualization)
# ============================================================================

DISPLAY_NAMES = {
    'uncertainty_calibration': 'Confidence',
    'instruction_boundary': 'Literalness',
    'commitment_strength': 'Assertiveness',
    'retrieval_construction': 'Retrieval',
    'convergent_divergent': 'Thinking Style',
    'abstract_concrete': 'Abstraction Level',
    'temporal_focus': 'Temporal Orientation',
    'cognitive_load': 'Complexity',
    'context_adherence': 'Context Following',
    'emotional_valence': 'Emotional Tone',
    'paranoia_trust': 'Trust Level',
    'power_dynamics': 'Authority Tone',
    'serial_parallel': 'Processing Style',
    'local_global': 'Focus Scope'
}


def get_display_name(trait_name: str) -> str:
    """Get display name for trait, or title-case the trait name if not found."""
    if trait_name in DISPLAY_NAMES:
        return DISPLAY_NAMES[trait_name]
    return trait_name.replace('_', ' ').title()


# ============================================================================
# Hook Setup for Layer Internals
# ============================================================================

def create_tier3_storage() -> Dict:
    """
    Create storage structure for Tier 3 capture.

    Returns:
        Dict with attention, mlp, and residual storage
    """
    return {
        'attention': {
            'q_proj': [],
            'k_proj': [],
            'v_proj': [],
            'attn_weights': [],  # Will store per-head weights
            'attn_output': []    # Output before O projection
        },
        'mlp': {
            'up_proj': [],
            'gelu': [],       # The key data - neuron activations
            'down_proj': []
        },
        'residual': {
            'input': [],      # Layer input
            'after_attn': [], # After attention block
            'output': []      # Layer output
        }
    }


def make_attention_hooks(layer_idx: int, storage: Dict, mode: str = 'prompt'):
    """
    Factory for attention internals hooks.

    Args:
        layer_idx: Which layer
        storage: Storage dict
        mode: 'prompt' (all tokens) or 'response' (last token only)

    Returns:
        List of (module_path, hook_function) tuples
    """
    hooks = []

    # Q projection hook
    def q_hook(module, input, output):
        if mode == 'response':
            q = output[:, -1, :].detach().cpu()
        else:
            q = output.detach().cpu()
        storage['attention']['q_proj'].append(q)

    hooks.append((f"model.layers.{layer_idx}.self_attn.q_proj", q_hook))

    # K projection hook
    def k_hook(module, input, output):
        if mode == 'response':
            k = output[:, -1, :].detach().cpu()
        else:
            k = output.detach().cpu()
        storage['attention']['k_proj'].append(k)

    hooks.append((f"model.layers.{layer_idx}.self_attn.k_proj", k_hook))

    # V projection hook
    def v_hook(module, input, output):
        if mode == 'response':
            v = output[:, -1, :].detach().cpu()
        else:
            v = output.detach().cpu()
        storage['attention']['v_proj'].append(v)

    hooks.append((f"model.layers.{layer_idx}.self_attn.v_proj", v_hook))

    return hooks


def make_mlp_hooks(layer_idx: int, storage: Dict, mode: str = 'prompt'):
    """
    Factory for MLP internals hooks.

    Args:
        layer_idx: Which layer
        storage: Storage dict
        mode: 'prompt' (all tokens) or 'response' (last token only)

    Returns:
        List of (module_path, hook_function) tuples
    """
    hooks = []

    # Up projection hook
    def up_hook(module, input, output):
        if mode == 'response':
            up = output[:, -1, :].detach().cpu()
        else:
            up = output.detach().cpu()
        storage['mlp']['up_proj'].append(up)

    hooks.append((f"model.layers.{layer_idx}.mlp.up_proj", up_hook))

    # GELU activation hook (the key data for neuron analysis)
    def gelu_hook(module, input, output):
        if mode == 'response':
            gelu = output[:, -1, :].detach().cpu()
        else:
            gelu = output.detach().cpu()
        storage['mlp']['gelu'].append(gelu)

    hooks.append((f"model.layers.{layer_idx}.mlp.act_fn", gelu_hook))

    # Down projection hook
    def down_hook(module, input, output):
        if mode == 'response':
            down = output[:, -1, :].detach().cpu()
        else:
            down = output.detach().cpu()
        storage['mlp']['down_proj'].append(down)

    hooks.append((f"model.layers.{layer_idx}.mlp.down_proj", down_hook))

    return hooks


def make_residual_hooks(layer_idx: int, storage: Dict, mode: str = 'prompt'):
    """
    Factory for residual stream hooks.

    Args:
        layer_idx: Which layer
        storage: Storage dict
        mode: 'prompt' (all tokens) or 'response' (last token only)

    Returns:
        List of (module_path, hook_function) tuples
    """
    hooks = []

    # Layer input/output hook (captures residual_in and residual_out)
    def layer_hook(module, input, output):
        if isinstance(input, tuple):
            input = input[0]
        if isinstance(output, tuple):
            output = output[0]

        if mode == 'response':
            res_in = input[:, -1, :].detach().cpu()
            res_out = output[:, -1, :].detach().cpu()
        else:
            res_in = input.detach().cpu()
            res_out = output.detach().cpu()

        storage['residual']['input'].append(res_in)
        storage['residual']['output'].append(res_out)

    hooks.append((f"model.layers.{layer_idx}", layer_hook))

    # MLP input hook (captures after_attn)
    def mlp_input_hook(module, input, output):
        if isinstance(input, tuple):
            input = input[0]

        if mode == 'response':
            after_attn = input[:, -1, :].detach().cpu()
        else:
            after_attn = input.detach().cpu()

        storage['residual']['after_attn'].append(after_attn)

    hooks.append((f"model.layers.{layer_idx}.mlp", mlp_input_hook))

    return hooks


# ============================================================================
# Prompt Encoding with Tier 3 Capture
# ============================================================================

def encode_prompt_tier3(
    model,
    tokenizer,
    prompt_text: str,
    layer_idx: int
) -> Dict:
    """
    Encode prompt with full Tier 3 capture for one layer.

    Args:
        model: The model
        tokenizer: The tokenizer
        prompt_text: Prompt string
        layer_idx: Layer to capture (0-26)

    Returns:
        Dict with tokens, internals, and attention weights
    """
    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    token_ids = inputs['input_ids'][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Setup hooks
    storage = create_tier3_storage()

    with HookManager(model) as hook_manager:
        # Register all hooks
        for path, hook_fn in make_attention_hooks(layer_idx, storage, mode='prompt'):
            hook_manager.add_forward_hook(path, hook_fn)

        for path, hook_fn in make_mlp_hooks(layer_idx, storage, mode='prompt'):
            hook_manager.add_forward_hook(path, hook_fn)

        for path, hook_fn in make_residual_hooks(layer_idx, storage, mode='prompt'):
            hook_manager.add_forward_hook(path, hook_fn)

        # Forward pass with attention capture
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                return_dict=True
            )

    # Get attention weights for this layer (per-head, not averaged)
    attn_weights = outputs.attentions[layer_idx][0].detach().cpu()  # [heads, seq, seq]

    # Consolidate storage (remove batch dimension, single forward pass)
    internals = {
        'attention': {},
        'mlp': {},
        'residual': {}
    }

    for key in ['q_proj', 'k_proj', 'v_proj']:
        internals['attention'][key] = storage['attention'][key][0].squeeze(0)

    internals['attention']['attn_weights'] = attn_weights  # [heads, seq, seq]

    for key in ['up_proj', 'gelu', 'down_proj']:
        internals['mlp'][key] = storage['mlp'][key][0].squeeze(0)

    for key in ['input', 'after_attn', 'output']:
        internals['residual'][key] = storage['residual'][key][0].squeeze(0)

    return {
        'tokens': tokens,
        'token_ids': token_ids,
        'internals': internals
    }


# ============================================================================
# Response Generation with Tier 3 Capture
# ============================================================================

def sample_token(logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9) -> int:
    """Sample next token with temperature and nucleus sampling."""
    if temperature == 0:
        return logits.argmax().item()

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[0] = False

    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()

    token_idx = torch.multinomial(sorted_probs, num_samples=1)
    token_id = sorted_indices[token_idx].item()

    return token_id


def generate_response_tier3(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    layer_idx: int,
    max_new_tokens: int = 50,
    temperature: float = 0.7
) -> Dict:
    """
    Generate response with Tier 3 capture for one layer.

    Args:
        model: The model
        tokenizer: The tokenizer
        prompt_ids: Prompt token IDs [1, n_prompt_tokens]
        layer_idx: Layer to capture
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Dict with tokens and internals
    """
    storage = create_tier3_storage()
    context = prompt_ids.clone()
    generated_ids = []

    with HookManager(model) as hook_manager:
        # Register all hooks
        for path, hook_fn in make_attention_hooks(layer_idx, storage, mode='response'):
            hook_manager.add_forward_hook(path, hook_fn)

        for path, hook_fn in make_mlp_hooks(layer_idx, storage, mode='response'):
            hook_manager.add_forward_hook(path, hook_fn)

        for path, hook_fn in make_residual_hooks(layer_idx, storage, mode='response'):
            hook_manager.add_forward_hook(path, hook_fn)

        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(
                    input_ids=context,
                    output_attentions=True,
                    return_dict=True
                )

            # Sample next token
            next_token_logits = outputs.logits[0, -1, :]
            next_token_id = sample_token(next_token_logits, temperature)

            # Add to context
            next_token_tensor = torch.tensor([[next_token_id]], device=model.device)
            context = torch.cat([context, next_token_tensor], dim=1)
            generated_ids.append(next_token_id)

            # Capture attention weights for this step
            attn_weights = outputs.attentions[layer_idx][0].detach().cpu()  # [heads, seq, seq]
            storage['attention']['attn_weights'].append(attn_weights)

            # Check for EOS
            if next_token_id == tokenizer.eos_token_id:
                break

    # Decode tokens
    tokens = [tokenizer.decode([tid]) for tid in generated_ids]

    # Consolidate storage
    internals = {
        'attention': {},
        'mlp': {},
        'residual': {}
    }

    # Stack activation lists: [n_gen_tokens, ...]
    for key in ['q_proj', 'k_proj', 'v_proj']:
        if storage['attention'][key]:
            internals['attention'][key] = torch.stack(
                [a.squeeze(0) for a in storage['attention'][key]], dim=0
            )
        else:
            internals['attention'][key] = torch.empty(0, model.config.hidden_size)

    # Attention weights is a list (growing context)
    internals['attention']['attn_weights'] = storage['attention']['attn_weights']

    for key in ['up_proj', 'gelu', 'down_proj']:
        if storage['mlp'][key]:
            internals['mlp'][key] = torch.stack(
                [a.squeeze(0) for a in storage['mlp'][key]], dim=0
            )
        else:
            internals['mlp'][key] = torch.empty(0, model.config.hidden_size)

    for key in ['input', 'after_attn', 'output']:
        if storage['residual'][key]:
            internals['residual'][key] = torch.stack(
                [a.squeeze(0) for a in storage['residual'][key]], dim=0
            )
        else:
            internals['residual'][key] = torch.empty(0, model.config.hidden_size)

    return {
        'tokens': tokens,
        'token_ids': generated_ids,
        'internals': internals
    }


# ============================================================================
# Data Assembly
# ============================================================================

def assemble_tier3_data(
    prompt_text: str,
    prompt_tokens: List[str],
    prompt_token_ids: List[int],
    prompt_internals: Dict,
    response_text: str,
    response_tokens: List[str],
    response_token_ids: List[int],
    response_internals: Dict,
    trait: str,
    layer: int,
    vector_path: str,
    model_name: str,
    temperature: float
) -> Dict:
    """
    Assemble complete Tier 3 data structure.

    Returns:
        Dict matching Tier 3 format specification
    """
    return {
        'prompt': {
            'text': prompt_text,
            'tokens': prompt_tokens,
            'token_ids': prompt_token_ids,
            'n_tokens': len(prompt_tokens)
        },
        'response': {
            'text': response_text,
            'tokens': response_tokens,
            'token_ids': response_token_ids,
            'n_tokens': len(response_tokens)
        },
        'layer': layer,
        'internals': {
            'prompt': prompt_internals,
            'response': response_internals
        },
        'metadata': {
            'trait': trait,
            'trait_display_name': get_display_name(trait),
            'layer': layer,
            'vector_path': str(vector_path),
            'model': model_name,
            'capture_date': datetime.now().isoformat(),
            'temperature': temperature
        }
    }


# ============================================================================
# JSON Export for Visualization
# ============================================================================

def convert_tier3_to_json(tier3_data: Dict) -> Dict:
    """
    Convert Tier 3 data to JSON-serializable format.

    Simplifies large tensors for browser visualization.

    Args:
        tier3_data: Tier 3 data dict with tensors

    Returns:
        JSON-serializable dict
    """
    def tensor_to_list(t):
        if isinstance(t, torch.Tensor):
            return t.tolist()
        elif isinstance(t, list):
            return [tensor_to_list(item) for item in t]
        return t

    json_data = {
        'prompt': tier3_data['prompt'].copy(),
        'response': tier3_data['response'].copy(),
        'layer': tier3_data['layer'],
        'metadata': tier3_data['metadata'].copy()
    }

    # For visualization, include only key data to keep size reasonable
    # Full tensors are 10-20MB, we'll include summaries

    prompt_int = tier3_data['internals']['prompt']
    response_int = tier3_data['internals']['response']

    json_data['internals'] = {
        'prompt': {
            'gelu': tensor_to_list(prompt_int['mlp']['gelu']),  # [n_tokens, 9216]
            'attn_weights': tensor_to_list(prompt_int['attention']['attn_weights']),  # [heads, seq, seq]
            'residual': {
                'input': tensor_to_list(prompt_int['residual']['input']),
                'after_attn': tensor_to_list(prompt_int['residual']['after_attn']),
                'output': tensor_to_list(prompt_int['residual']['output'])
            }
        },
        'response': {
            'gelu': tensor_to_list(response_int['mlp']['gelu']),  # [n_tokens, 9216]
            # Attention weights is list (growing context) - too large, summarize
            'attn_weights_shape': [len(response_int['attention']['attn_weights'])] if response_int['attention']['attn_weights'] else [],
            'residual': {
                'input': tensor_to_list(response_int['residual']['input']),
                'after_attn': tensor_to_list(response_int['residual']['after_attn']),
                'output': tensor_to_list(response_int['residual']['output'])
            }
        }
    }

    return json_data


# ============================================================================
# Main Script
# ============================================================================

def infer_model_from_experiment(experiment_name: str) -> str:
    """Infer model name from experiment naming convention."""
    if "gemma_2b" in experiment_name.lower():
        return "google/gemma-2-2b-it"
    elif "gemma_9b" in experiment_name.lower():
        return "google/gemma-2-9b-it"
    elif "llama_8b" in experiment_name.lower():
        return "meta-llama/Llama-3.1-8B-Instruct"
    else:
        return "google/gemma-2-2b-it"


def main():
    parser = argparse.ArgumentParser(
        description="Capture Tier 3 data: layer internal states for deep analysis"
    )
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--trait", required=True, help="Trait name")
    parser.add_argument("--layer", type=int, required=True, help="Layer to capture (0-26)")
    parser.add_argument("--prompts", type=str, help="Single prompt string")
    parser.add_argument("--prompts-file", type=str, help="File with prompts (one per line)")
    parser.add_argument("--method", default="probe", help="Vector method (default: probe)")
    parser.add_argument("--output-dir", type=str, help="Output directory (auto-detected if not provided)")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device", default="cuda", help="Device (cuda/mps/cpu)")
    parser.add_argument("--save-json", action="store_true", help="Also save JSON for visualization")

    args = parser.parse_args()

    # Get prompts
    if args.prompts:
        prompt_list = [args.prompts]
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompt_list = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Must provide either --prompts or --prompts-file")

    # Setup paths
    exp_dir = Path(f"experiments/{args.experiment}")
    trait_dir = exp_dir / args.trait

    if not trait_dir.exists():
        print(f"❌ Trait directory not found: {trait_dir}")
        return

    # Load vector (for metadata reference)
    vector_path = trait_dir / "extraction" / "vectors" / f"{args.method}_layer{args.layer}.pt"
    if not vector_path.exists():
        print(f"❌ Vector not found: {vector_path}")
        print(f"   Available vectors:")
        vectors_dir = trait_dir / "extraction" / "vectors"
        if vectors_dir.exists():
            for vf in sorted(vectors_dir.glob("*.pt")):
                if "metadata" not in vf.name:
                    print(f"   - {vf.name}")
        return

    print(f"Using vector: {vector_path}")

    # Infer model
    model_name = infer_model_from_experiment(args.experiment)
    print(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation='eager'  # Required for output_attentions=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Validate layer
    n_layers = len(model.model.layers)
    if args.layer >= n_layers:
        print(f"❌ Layer {args.layer} out of range (model has {n_layers} layers: 0-{n_layers-1})")
        return

    print(f"Capturing layer {args.layer} of {n_layers}")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = trait_dir / "inference" / "layer_internal_states"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Process each prompt
    for prompt_idx, prompt_text in enumerate(tqdm(prompt_list, desc="Processing prompts")):
        print(f"\n{'='*60}")
        print(f"Prompt {prompt_idx}: {prompt_text[:80]}...")
        print(f"{'='*60}")

        # Encode prompt
        print("Encoding prompt...")
        prompt_data = encode_prompt_tier3(model, tokenizer, prompt_text, args.layer)
        print(f"  ✓ Captured {len(prompt_data['tokens'])} prompt tokens")

        # Generate response
        print("Generating response...")
        prompt_ids = torch.tensor([prompt_data['token_ids']], device=model.device)
        response_data = generate_response_tier3(
            model, tokenizer, prompt_ids, args.layer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        response_text = tokenizer.decode(response_data['token_ids'], skip_special_tokens=True)
        print(f"  ✓ Generated {len(response_data['tokens'])} tokens")
        print(f"  Response: {response_text[:100]}...")

        # Assemble data
        tier3_data = assemble_tier3_data(
            prompt_text=prompt_text,
            prompt_tokens=prompt_data['tokens'],
            prompt_token_ids=prompt_data['token_ids'],
            prompt_internals=prompt_data['internals'],
            response_text=response_text,
            response_tokens=response_data['tokens'],
            response_token_ids=response_data['token_ids'],
            response_internals=response_data['internals'],
            trait=args.trait,
            layer=args.layer,
            vector_path=vector_path,
            model_name=model_name,
            temperature=args.temperature
        )

        # Save .pt file
        output_path = output_dir / f"prompt_{prompt_idx}_layer{args.layer}.pt"
        torch.save(tier3_data, output_path)

        # Calculate size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved to: {output_path}")
        print(f"  Size: {size_mb:.1f} MB")

        # Optionally save JSON for visualization
        if args.save_json:
            json_path = output_dir / f"prompt_{prompt_idx}_layer{args.layer}.json"
            json_data = convert_tier3_to_json(tier3_data)
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            json_size_mb = json_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Saved JSON: {json_path}")
            print(f"  JSON size: {json_size_mb:.1f} MB")

    print(f"\n{'='*60}")
    print(f"✅ Completed! Processed {len(prompt_list)} prompts")
    print(f"   Output: {output_dir}")
    print(f"   Layer: {args.layer}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
