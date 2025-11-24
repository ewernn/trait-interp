#!/usr/bin/env python3
"""
Analyze attention dynamics during trait transitions (text output only).
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict

def load_all_layers(prompt_id: int, experiment: str = 'gemma_2b_cognitive_nov21') -> Dict:
    """Load internals for all layers of a prompt."""
    base_path = Path(f'experiments/{experiment}/inference/raw/internals/dynamic')
    all_layers = {}

    for layer in range(26):
        layer_file = base_path / f'{prompt_id}_L{layer}.pt'
        if layer_file.exists():
            all_layers[layer] = torch.load(layer_file, weights_only=True)

    return all_layers

def compute_attention_entropy(attn_weights: torch.Tensor) -> float:
    """Compute entropy of attention distribution (measure of focus vs diffusion)."""
    # Normalize to probabilities if not already
    if attn_weights.dim() == 3:  # [heads, seq, seq]
        attn_weights = attn_weights.mean(dim=0)  # Average across heads

    # Get last row (current token's attention)
    if attn_weights.dim() == 2:
        probs = attn_weights[-1, :]
    else:
        probs = attn_weights

    # Compute entropy
    probs = probs + 1e-10  # Avoid log(0)
    entropy = -torch.sum(probs * torch.log(probs))
    return entropy.item()

def analyze_attention_evolution(all_layers: Dict):
    """Analyze how attention patterns evolve during generation."""

    # Get tokens for reference
    prompt_tokens = all_layers[0]['prompt_tokens']
    response_tokens = all_layers[0]['response_tokens']
    n_prompt = len(prompt_tokens)

    print(f"\n=== Attention Pattern Evolution ===")
    print(f"Prompt: {' '.join(prompt_tokens[:20])}...")
    print(f"Response starts: {' '.join(response_tokens[:10])}...")
    print()

    # Track attention entropy (focus vs diffusion) across generation
    layer_to_analyze = 16  # Middle layer
    layer_data = all_layers[layer_to_analyze]

    attn_weights = layer_data['response']['attention']['attn_weights']

    print(f"=== Layer {layer_to_analyze} Attention Dynamics ===\n")

    # Analyze key points in generation
    checkpoints = [0, 5, 10, 20, 30, 40, len(attn_weights)-1]

    for checkpoint in checkpoints:
        if checkpoint >= len(attn_weights):
            continue

        attn = attn_weights[checkpoint]  # [heads, seq, seq]
        entropy = compute_attention_entropy(attn)

        # Find what the model is attending to most
        avg_attn = attn.mean(dim=0)  # Average across heads
        current_token_attn = avg_attn[-1, :]  # What current token attends to

        # Find top attended positions
        top_k = 5
        top_positions = torch.topk(current_token_attn, min(top_k, len(current_token_attn))).indices

        print(f"Token {checkpoint} ('{response_tokens[checkpoint] if checkpoint < len(response_tokens) else '?'}'):")
        print(f"  Entropy: {entropy:.3f} (lower = more focused)")
        print(f"  Top attended positions:")

        for pos in top_positions:
            pos_idx = pos.item()
            if pos_idx < n_prompt:
                token = prompt_tokens[pos_idx]
                weight = current_token_attn[pos_idx].item()
                print(f"    Pos {pos_idx} (prompt): '{token}' (weight: {weight:.3f})")
            else:
                resp_idx = pos_idx - n_prompt
                if resp_idx < len(response_tokens):
                    token = response_tokens[resp_idx]
                    weight = current_token_attn[pos_idx].item()
                    print(f"    Pos {pos_idx} (response): '{token}' (weight: {weight:.3f})")
        print()

def compute_attention_velocity_simple(all_layers: Dict):
    """Compute simplified attention velocity metrics."""

    print("\n=== Attention Velocity Analysis ===\n")

    velocities_by_layer = {}

    for layer_idx in [0, 8, 16, 24]:  # Sample layers
        layer_data = all_layers[layer_idx]
        attn_weights = layer_data['response']['attention']['attn_weights']

        if not attn_weights:
            continue

        layer_velocities = []

        # Compare consecutive attention patterns
        for t in range(min(len(attn_weights) - 1, 30)):  # Analyze first 30 tokens
            curr = attn_weights[t].mean(dim=0)  # Average across heads
            next_attn = attn_weights[t+1].mean(dim=0)

            # Compare overlapping portion
            min_len = curr.shape[-1]
            curr_pattern = curr[-1, :]  # Current token's attention
            next_pattern = next_attn[-1, :min_len]  # Next token's attention to same context

            # L2 distance as velocity
            velocity = torch.norm(next_pattern - curr_pattern).item()
            layer_velocities.append(velocity)

        velocities_by_layer[layer_idx] = layer_velocities

        # Report statistics
        if layer_velocities:
            mean_vel = np.mean(layer_velocities)
            max_vel = np.max(layer_velocities)
            max_vel_idx = np.argmax(layer_velocities)

            print(f"Layer {layer_idx}:")
            print(f"  Mean velocity: {mean_vel:.3f}")
            print(f"  Max velocity: {max_vel:.3f} at token {max_vel_idx}")

            # Find tokens with high velocity (potential transition points)
            threshold = np.percentile(layer_velocities, 75)
            high_vel_tokens = [i for i, v in enumerate(layer_velocities) if v > threshold]
            if high_vel_tokens[:3]:
                response_tokens = all_layers[0]['response_tokens']
                print(f"  High velocity tokens: {high_vel_tokens[:3]}")
                for idx in high_vel_tokens[:3]:
                    if idx < len(response_tokens):
                        print(f"    Token {idx}: '{response_tokens[idx]}' (vel: {layer_velocities[idx]:.3f})")
            print()

    return velocities_by_layer

def find_trait_transition_point(all_layers: Dict):
    """Try to identify where the trait transition occurs."""

    print("\n=== Searching for Trait Transition Point ===\n")

    # We know prompt 1 should transition from uncertainty to confidence
    # Let's look for changes in attention patterns that might indicate this

    response_tokens = all_layers[0]['response_tokens']
    prompt_tokens = all_layers[0]['prompt_tokens']

    # Look for keywords that might indicate transition
    uncertainty_keywords = ['might', 'could', 'perhaps', 'possibly', 'may']
    confidence_keywords = ['is', 'are', 'will', 'definitely', 'certainly', 'know']

    print("Response token analysis:")
    for i, token in enumerate(response_tokens[:30]):  # First 30 tokens
        token_lower = token.lower().strip()
        if any(kw in token_lower for kw in uncertainty_keywords):
            print(f"  Token {i}: '{token}' [UNCERTAIN]")
        elif any(kw in token_lower for kw in confidence_keywords):
            print(f"  Token {i}: '{token}' [CONFIDENT]")

    # Also check if attention patterns shift focus from speculative part to factual part of prompt
    layer_16 = all_layers[16]
    attn_weights = layer_16['response']['attention']['attn_weights']

    # The prompt asks two things:
    # 1. "What might happen..." (speculative, tokens ~0-10)
    # 2. "what do we already know for certain..." (factual, tokens ~15-25)

    print("\n\nAttention focus shift analysis:")
    for checkpoint in [0, 10, 20, 30]:
        if checkpoint >= len(attn_weights):
            continue

        attn = attn_weights[checkpoint].mean(dim=0)  # Average across heads
        current_attn = attn[-1, :]  # Current token's attention

        # Measure attention to different parts of prompt
        speculative_attention = current_attn[:10].sum().item() if len(current_attn) > 10 else 0
        factual_attention = current_attn[15:25].sum().item() if len(current_attn) > 25 else current_attn[15:].sum().item()

        print(f"\nToken {checkpoint} ('{response_tokens[checkpoint] if checkpoint < len(response_tokens) else '?'}'):")
        print(f"  Attention to speculative part (0-10): {speculative_attention:.3f}")
        print(f"  Attention to factual part (15-25): {factual_attention:.3f}")
        print(f"  Ratio (factual/speculative): {factual_attention/(speculative_attention+1e-6):.2f}")

def main():
    """Analyze attention dynamics for dynamic prompts."""

    # Load prompt info
    with open('inference/prompts/dynamic.json') as f:
        prompts_data = json.load(f)
        prompt_info = prompts_data['prompts'][0]  # First prompt

    print(f"=== Analyzing Dynamic Prompt 1 ===")
    print(f"Text: {prompt_info['text']}")
    print(f"Designed transition: {prompt_info['note']}")
    print()

    # Load all layers
    print("Loading layer data...")
    all_layers = load_all_layers(prompt_id=1)
    print(f"Loaded {len(all_layers)} layers")

    # Run analyses
    analyze_attention_evolution(all_layers)
    velocities = compute_attention_velocity_simple(all_layers)
    find_trait_transition_point(all_layers)

    # Save key metrics
    output_dir = Path('experiments/gemma_2b_cognitive_nov21/analysis/attention_dynamics')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare metrics for saving
    metrics = {
        'prompt_id': 1,
        'prompt_text': prompt_info['text'],
        'transition_type': prompt_info['note'],
        'n_prompt_tokens': len(all_layers[0]['prompt_tokens']),
        'n_response_tokens': len(all_layers[0]['response_tokens']),
        'layer_velocities': {str(k): v for k, v in velocities.items()}
    }

    with open(output_dir / 'prompt_1_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n\nMetrics saved to {output_dir / 'prompt_1_metrics.json'}")

if __name__ == '__main__':
    main()