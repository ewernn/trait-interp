#!/usr/bin/env python3
"""
Tier 2 Capture: Residual Stream Activations

Captures per-token projections at 81 checkpoints (27 layers × 3 sublayers) for both
prompt encoding and response generation. This provides full visibility into how a
trait evolves through the model's layers.

Usage:
    # Single prompt
    python inference/capture_tier2.py \
        --experiment gemma_2b_cognitive_nov20 \
        --trait refusal \
        --prompts "What is the capital of France?"

    # Multiple prompts from file
    python inference/capture_tier2.py \
        --experiment gemma_2b_cognitive_nov20 \
        --trait refusal \
        --prompts-file test_prompts.txt \
        --method probe \
        --layer 16

    # Custom output directory
    python inference/capture_tier2.py \
        --experiment gemma_2b_cognitive_nov20 \
        --trait refusal \
        --prompts "Test prompt" \
        --output-dir experiments/gemma_2b_cognitive_nov20/refusal/inference/residual_stream_activations/
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

from traitlens import HookManager, projection


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
# Hook Setup Functions
# ============================================================================

def create_capture_storage(n_layers: int = 27) -> Dict:
    """
    Create nested storage for activations.

    Returns:
        Dict: {layer_idx: {'residual_in': [], 'after_attn': [], 'residual_out': []}}
    """
    storage = {}
    for i in range(n_layers):
        storage[i] = {
            'residual_in': [],
            'after_attn': [],
            'residual_out': []
        }
    return storage


def make_layer_hooks(layer_idx: int, storage: Dict, mode: str = 'prompt'):
    """
    Factory for layer input/output hooks.

    Captures residual_in (layer input) and residual_out (layer output).

    Args:
        layer_idx: Which layer (0-26)
        storage: Storage dict to append to
        mode: 'prompt' (all tokens) or 'response' (last token only)

    Returns:
        Tuple of (module_path, hook_function)
    """
    def hook_fn(module, input, output):
        # Handle tuple outputs
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(input, tuple):
            input = input[0]

        # Capture residual_in (input to layer)
        if mode == 'response':
            # Only last token (the newly generated one)
            residual_in = input[:, -1, :].detach().cpu()
        else:
            # All tokens (prompt encoding)
            residual_in = input.detach().cpu()

        storage[layer_idx]['residual_in'].append(residual_in)

        # Capture residual_out (output of layer)
        if mode == 'response':
            residual_out = output[:, -1, :].detach().cpu()
        else:
            residual_out = output.detach().cpu()

        storage[layer_idx]['residual_out'].append(residual_out)

    module_path = f"model.layers.{layer_idx}"
    return (module_path, hook_fn)


def make_mlp_hook(layer_idx: int, storage: Dict, mode: str = 'prompt'):
    """
    Factory for MLP input hook.

    Captures after_attn checkpoint (input to MLP = after attention block).

    Args:
        layer_idx: Which layer (0-26)
        storage: Storage dict to append to
        mode: 'prompt' (all tokens) or 'response' (last token only)

    Returns:
        Tuple of (module_path, hook_function)
    """
    def hook_fn(module, input, output):
        # Handle tuple inputs
        if isinstance(input, tuple):
            input = input[0]

        # Capture after_attn (input to MLP)
        if mode == 'response':
            # Only last token
            after_attn = input[:, -1, :].detach().cpu()
        else:
            # All tokens
            after_attn = input.detach().cpu()

        storage[layer_idx]['after_attn'].append(after_attn)

    module_path = f"model.layers.{layer_idx}.mlp"
    return (module_path, hook_fn)


def setup_all_hooks(
    hook_manager: HookManager,
    storage: Dict,
    n_layers: int = 27,
    mode: str = 'prompt'
) -> None:
    """
    Register all 54 hooks (27 layers × 2 hooks).

    Args:
        hook_manager: HookManager instance
        storage: Storage dict created by create_capture_storage()
        n_layers: Number of layers (default: 27 for Gemma 2B)
        mode: 'prompt' or 'response'
    """
    for i in range(n_layers):
        # Hook layer for residual_in and residual_out
        layer_path, layer_hook = make_layer_hooks(i, storage, mode)
        hook_manager.add_forward_hook(layer_path, layer_hook)

        # Hook MLP for after_attn
        mlp_path, mlp_hook = make_mlp_hook(i, storage, mode)
        hook_manager.add_forward_hook(mlp_path, mlp_hook)


# ============================================================================
# Prompt Encoding
# ============================================================================

def encode_prompt_with_capture(
    model,
    tokenizer,
    prompt_text: str,
    n_layers: int = 27
) -> Dict:
    """
    Encode prompt with full activation capture.

    Single forward pass captures all tokens across all layers and sublayers.

    Args:
        model: The model
        tokenizer: The tokenizer
        prompt_text: Prompt string
        n_layers: Number of layers in model

    Returns:
        Dict with:
            - tokens: List of token strings
            - token_ids: List of token IDs
            - activations: Dict {layer_idx: {sublayer: tensor[n_tokens, hidden_dim]}}
            - attention_weights: Dict {f'layer_{i}': tensor[n_tokens, n_tokens]} (27 layers)
    """
    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    token_ids = inputs['input_ids'][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Setup hooks
    storage = create_capture_storage(n_layers)

    with HookManager(model) as hooks:
        setup_all_hooks(hooks, storage, n_layers, mode='prompt')

        # Forward pass with attention capture
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                return_dict=True
            )

    # Extract attention weights (average across heads)
    attention_weights = {}
    for i, attn in enumerate(outputs.attentions):
        # attn shape: [batch, heads, seq, seq]
        # Average across heads: [batch, seq, seq] -> [seq, seq]
        attn_avg = attn[0].mean(dim=0).detach().cpu()
        attention_weights[f'layer_{i}'] = attn_avg

    # Consolidate activations
    # Storage has list with 1 element (single forward pass)
    activations = {}
    for layer_idx in range(n_layers):
        activations[layer_idx] = {}
        for sublayer in ['residual_in', 'after_attn', 'residual_out']:
            # Get first (and only) element, remove batch dimension
            act = storage[layer_idx][sublayer][0].squeeze(0)  # [n_tokens, hidden_dim]
            activations[layer_idx][sublayer] = act

    return {
        'tokens': tokens,
        'token_ids': token_ids,
        'activations': activations,
        'attention_weights': attention_weights
    }


# ============================================================================
# Response Generation
# ============================================================================

def sample_token(logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9) -> int:
    """
    Sample next token from logits with temperature and nucleus sampling.

    Args:
        logits: Logits for next token [vocab_size]
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        Token ID
    """
    if temperature == 0:
        return logits.argmax().item()

    # Apply temperature
    logits = logits / temperature

    # Softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)

    # Nucleus sampling
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Keep at least one token
    sorted_indices_to_remove[0] = False

    # Zero out removed tokens
    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()

    # Sample
    token_idx = torch.multinomial(sorted_probs, num_samples=1)
    token_id = sorted_indices[token_idx].item()

    return token_id


def generate_response_with_capture(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    n_layers: int = 27,
    max_new_tokens: int = 50,
    temperature: float = 0.7
) -> Dict:
    """
    Generate response autoregressively with activation capture.

    Captures last token at each generation step across all layers and sublayers.

    Args:
        model: The model
        tokenizer: The tokenizer
        prompt_ids: Prompt token IDs [1, n_prompt_tokens]
        n_layers: Number of layers
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Dict with:
            - tokens: List of generated token strings
            - token_ids: List of generated token IDs
            - activations: Dict {layer_idx: {sublayer: tensor[n_gen_tokens, hidden_dim]}}
            - attention_weights: List of dicts (one per token, each with all 27 layers)
    """
    # Setup hooks
    storage = create_capture_storage(n_layers)

    context = prompt_ids.clone()
    generated_ids = []
    attention_weights = []

    with HookManager(model) as hooks:
        setup_all_hooks(hooks, storage, n_layers, mode='response')

        for step in range(max_new_tokens):
            # Forward pass on current context
            with torch.no_grad():
                outputs = model(
                    input_ids=context,
                    output_attentions=True,
                    return_dict=True
                )

            # Capture attention weights for this step (all 27 layers)
            # Extract last token's attention pattern from each layer
            step_attentions = {}
            for layer_idx, attn in enumerate(outputs.attentions):
                # attn shape: [batch, heads, seq, seq]
                # Get last token's attention: mean across heads
                attn_last_token = attn[0].mean(dim=0)[-1, :].detach().cpu()  # [seq]
                step_attentions[f'layer_{layer_idx}'] = attn_last_token
            attention_weights.append(step_attentions)

            # Sample next token
            next_token_logits = outputs.logits[0, -1, :]  # [vocab_size]
            next_token_id = sample_token(next_token_logits, temperature)

            # Add to context
            next_token_tensor = torch.tensor([[next_token_id]], device=model.device)
            context = torch.cat([context, next_token_tensor], dim=1)
            generated_ids.append(next_token_id)

            # Check for EOS
            if next_token_id == tokenizer.eos_token_id:
                break

    # Decode tokens
    tokens = [tokenizer.decode([tid]) for tid in generated_ids]

    # Consolidate activations
    # Storage has one element per generation step
    activations = {}
    for layer_idx in range(n_layers):
        activations[layer_idx] = {}
        for sublayer in ['residual_in', 'after_attn', 'residual_out']:
            # Stack all captured tokens: [n_gen_tokens, hidden_dim]
            acts_list = storage[layer_idx][sublayer]
            if acts_list:
                # Remove batch dimension from each and stack
                acts_stacked = torch.stack([a.squeeze(0) for a in acts_list], dim=0)
                activations[layer_idx][sublayer] = acts_stacked
            else:
                # Empty generation
                activations[layer_idx][sublayer] = torch.empty(0, model.config.hidden_size)

    return {
        'tokens': tokens,
        'token_ids': generated_ids,
        'activations': activations,
        'attention_weights': attention_weights
    }


# ============================================================================
# Projection & Data Assembly
# ============================================================================

def project_activations_onto_vector(
    activations: Dict,
    vector: torch.Tensor,
    n_layers: int = 27
) -> torch.Tensor:
    """
    Project all captured activations onto trait vector.

    Args:
        activations: Dict {layer_idx: {sublayer: tensor[n_tokens, hidden_dim]}}
        vector: Trait vector [hidden_dim]
        n_layers: Number of layers

    Returns:
        Projections tensor [n_tokens, n_layers, 3_sublayers]
    """
    # Get number of tokens from first layer
    n_tokens = activations[0]['residual_in'].shape[0]

    # Initialize output tensor
    projections = torch.zeros(n_tokens, n_layers, 3)

    sublayer_order = ['residual_in', 'after_attn', 'residual_out']

    for layer_idx in range(n_layers):
        for sublayer_idx, sublayer_name in enumerate(sublayer_order):
            # Get activations: [n_tokens, hidden_dim]
            acts = activations[layer_idx][sublayer_name]

            # Project onto vector: [n_tokens]
            proj = projection(acts, vector, normalize_vector=True)

            # Store in output tensor
            projections[:, layer_idx, sublayer_idx] = proj

    return projections


def assemble_tier2_data(
    prompt_text: str,
    prompt_tokens: List[str],
    prompt_token_ids: List[int],
    prompt_projections: torch.Tensor,
    prompt_attention: Dict,
    response_text: str,
    response_tokens: List[str],
    response_token_ids: List[int],
    response_projections: torch.Tensor,
    response_attention: List[torch.Tensor],
    trait: str,
    vector_path: str,
    model_name: str,
    temperature: float
) -> Dict:
    """
    Assemble complete Tier 2 data structure.

    Returns:
        Dict matching Tier 2 format specification
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
        'projections': {
            'prompt': prompt_projections,
            'response': response_projections
        },
        'attention_weights': {
            'prompt': prompt_attention,
            'response': response_attention
        },
        'metadata': {
            'trait': trait,
            'trait_display_name': get_display_name(trait),
            'vector_path': str(vector_path),
            'model': model_name,
            'capture_date': datetime.now().isoformat(),
            'temperature': temperature
        }
    }


# ============================================================================
# JSON Export for Visualization
# ============================================================================

def convert_tier2_to_json(tier2_data: Dict) -> Dict:
    """
    Convert Tier 2 data to JSON-serializable format for browser visualization.

    Converts tensors to lists and simplifies attention weights.

    Args:
        tier2_data: Tier 2 data dict with tensors

    Returns:
        JSON-serializable dict
    """
    def tensor_to_list(t):
        """Convert tensor to nested list."""
        if isinstance(t, torch.Tensor):
            return t.tolist()
        return t

    json_data = {
        'prompt': tier2_data['prompt'].copy(),
        'response': tier2_data['response'].copy(),
        'projections': {
            'prompt': tensor_to_list(tier2_data['projections']['prompt']),
            'response': tensor_to_list(tier2_data['projections']['response'])
        },
        'metadata': tier2_data['metadata'].copy()
    }

    # Simplify attention weights (skip full matrices for now - too large for viz)
    # Just include layer-averaged attention for prompt
    json_data['attention_summary'] = {
        'prompt_layers': len(tier2_data['attention_weights']['prompt']),
        'response_tokens': len(tier2_data['attention_weights']['response'])
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
        description="Capture Tier 2 data: per-token projections across all layers"
    )
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--trait", required=True, help="Trait name")
    parser.add_argument("--prompts", type=str, help="Single prompt string")
    parser.add_argument("--prompts-file", type=str, help="File with prompts (one per line)")
    parser.add_argument("--method", default="probe", help="Vector method (default: probe)")
    parser.add_argument("--layer", type=int, default=16, help="Vector layer (default: 16)")
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

    # Load vector
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

    print(f"Loading vector: {vector_path}")
    vector = torch.load(vector_path).to(torch.float16)  # Match model dtype
    print(f"Vector shape: {vector.shape}")

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

    # Get number of layers
    n_layers = len(model.model.layers)
    print(f"Model has {n_layers} layers")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = trait_dir / "inference" / "residual_stream_activations"

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
        prompt_data = encode_prompt_with_capture(model, tokenizer, prompt_text, n_layers)
        print(f"  ✓ Captured {len(prompt_data['tokens'])} prompt tokens")

        # Generate response
        print("Generating response...")
        prompt_ids = torch.tensor([prompt_data['token_ids']], device=model.device)
        response_data = generate_response_with_capture(
            model, tokenizer, prompt_ids, n_layers,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        response_text = tokenizer.decode(response_data['token_ids'], skip_special_tokens=True)
        print(f"  ✓ Generated {len(response_data['tokens'])} tokens")
        print(f"  Response: {response_text[:100]}...")

        # Project onto vector
        print("Projecting activations...")
        prompt_projections = project_activations_onto_vector(
            prompt_data['activations'], vector, n_layers
        )
        response_projections = project_activations_onto_vector(
            response_data['activations'], vector, n_layers
        )
        print(f"  ✓ Prompt projections: {prompt_projections.shape}")
        print(f"  ✓ Response projections: {response_projections.shape}")

        # Assemble data
        tier2_data = assemble_tier2_data(
            prompt_text=prompt_text,
            prompt_tokens=prompt_data['tokens'],
            prompt_token_ids=prompt_data['token_ids'],
            prompt_projections=prompt_projections,
            prompt_attention=prompt_data['attention_weights'],
            response_text=response_text,
            response_tokens=response_data['tokens'],
            response_token_ids=response_data['token_ids'],
            response_projections=response_projections,
            response_attention=response_data['attention_weights'],
            trait=args.trait,
            vector_path=vector_path,
            model_name=model_name,
            temperature=args.temperature
        )

        # Save .pt file
        output_path = output_dir / f"prompt_{prompt_idx}.pt"
        torch.save(tier2_data, output_path)

        # Calculate size
        size_kb = output_path.stat().st_size / 1024
        print(f"  ✓ Saved to: {output_path}")
        print(f"  Size: {size_kb:.1f} KB")

        # Optionally save JSON for visualization
        if args.save_json:
            json_path = output_dir / f"prompt_{prompt_idx}.json"
            json_data = convert_tier2_to_json(tier2_data)
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            json_size_kb = json_path.stat().st_size / 1024
            print(f"  ✓ Saved JSON: {json_path}")
            print(f"  JSON size: {json_size_kb:.1f} KB")

    print(f"\n{'='*60}")
    print(f"✅ Completed! Processed {len(prompt_list)} prompts")
    print(f"   Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
