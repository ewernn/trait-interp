#!/usr/bin/env python3
"""
Analyze attention dynamics during trait transitions.
Focus on how attention patterns evolve when traits shift.
"""

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

def load_all_layers(prompt_id: int, experiment: str = 'gemma_2b_cognitive_nov21') -> Dict:
    """Load internals for all layers of a prompt."""
    base_path = Path(f'experiments/{experiment}/inference/raw/internals/dynamic')
    all_layers = {}

    for layer in range(26):
        layer_file = base_path / f'{prompt_id}_L{layer}.pt'
        if layer_file.exists():
            all_layers[layer] = torch.load(layer_file, weights_only=True)

    return all_layers

def compute_attention_velocity(layer_data: Dict, layer_idx: int) -> np.ndarray:
    """
    Compute how attention patterns change between tokens.
    Returns velocity matrix [n_heads, n_tokens-1].
    """
    attn_weights = layer_data['response']['attention']['attn_weights']

    if not attn_weights:
        return None

    velocities = []

    # Compare attention patterns between consecutive tokens
    for t in range(len(attn_weights) - 1):
        curr_attn = attn_weights[t]  # [n_heads, context_t, context_t]
        next_attn = attn_weights[t+1]  # [n_heads, context_t+1, context_t+1]

        # Focus on the new token's attention to previous context
        # This shows how the model's "focus" shifts
        curr_last_row = curr_attn[:, -1, :]  # [n_heads, context_t]
        next_last_row = next_attn[:, -1, :curr_attn.shape[-1]]  # [n_heads, context_t]

        # Compute L2 distance between attention patterns
        velocity = torch.norm(next_last_row - curr_last_row, dim=-1)  # [n_heads]
        velocities.append(velocity)

    return torch.stack(velocities).T.numpy() if velocities else None  # [n_heads, n_tokens-1]

def find_attention_shifts(all_layers: Dict) -> Dict:
    """
    Find tokens where attention patterns shift dramatically across layers.
    These might be trait transition points.
    """
    attention_velocities = {}

    for layer_idx, layer_data in all_layers.items():
        vel = compute_attention_velocity(layer_data, layer_idx)
        if vel is not None:
            attention_velocities[layer_idx] = vel

    # Find high-velocity points (potential transition points)
    if not attention_velocities:
        return {}

    # Stack velocities: [n_layers, n_heads, n_tokens-1]
    all_velocities = np.stack([attention_velocities[i] for i in sorted(attention_velocities.keys())])

    # Average across heads and layers
    mean_velocity = all_velocities.mean(axis=(0, 1))  # [n_tokens-1]

    # Find peaks (high acceleration points)
    acceleration = np.diff(mean_velocity)

    # Find tokens with high velocity or acceleration
    high_vel_tokens = np.where(mean_velocity > np.percentile(mean_velocity, 75))[0]
    high_accel_tokens = np.where(np.abs(acceleration) > np.percentile(np.abs(acceleration), 75))[0]

    return {
        'velocities': attention_velocities,
        'mean_velocity': mean_velocity,
        'acceleration': acceleration,
        'high_velocity_tokens': high_vel_tokens,
        'high_acceleration_tokens': high_accel_tokens
    }

def analyze_trait_transition(prompt_id: int):
    """Complete analysis of attention dynamics for a prompt."""

    # Load prompt info
    with open('inference/prompts/dynamic.json') as f:
        prompts_data = json.load(f)
        prompt_info = next(p for p in prompts_data['prompts'] if p['id'] == prompt_id)

    print(f"=== Analyzing Prompt {prompt_id} ===")
    print(f"Text: {prompt_info['text'][:100]}...")
    print(f"Designed transition: {prompt_info['note']}")
    print()

    # Load all layers
    print("Loading layer data...")
    all_layers = load_all_layers(prompt_id)
    print(f"Loaded {len(all_layers)} layers")

    # Get tokens for reference
    tokens = all_layers[0]['prompt_tokens'] + all_layers[0]['response_tokens']
    print(f"Total tokens: {len(tokens)}")
    print()

    # Analyze attention dynamics
    print("Computing attention velocities...")
    dynamics = find_attention_shifts(all_layers)

    if 'mean_velocity' in dynamics:
        print(f"\nHigh velocity tokens (potential transition points):")
        for token_idx in dynamics['high_velocity_tokens'][:5]:
            # Adjust for the offset (velocity is between tokens)
            actual_token_idx = token_idx + 1 + len(all_layers[0]['prompt_tokens'])
            if actual_token_idx < len(tokens):
                print(f"  Token {actual_token_idx}: '{tokens[actual_token_idx]}' (velocity: {dynamics['mean_velocity'][token_idx]:.3f})")

        print(f"\nHigh acceleration tokens (rapid changes):")
        for token_idx in dynamics['high_acceleration_tokens'][:5]:
            actual_token_idx = token_idx + 2 + len(all_layers[0]['prompt_tokens'])
            if actual_token_idx < len(tokens):
                print(f"  Token {actual_token_idx}: '{tokens[actual_token_idx]}' (accel: {dynamics['acceleration'][token_idx]:.3f})")

    return all_layers, dynamics

def visualize_attention_flow(all_layers: Dict, dynamics: Dict, save_path: Path = None):
    """Create comprehensive visualization of attention dynamics."""

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Attention Flow Dynamics', fontsize=16)

    # 1. Attention velocity heatmap (layers x tokens)
    if dynamics and 'velocities' in dynamics:
        velocities_matrix = []
        for layer in range(26):
            if layer in dynamics['velocities']:
                # Average across heads
                layer_vel = dynamics['velocities'][layer].mean(axis=0)
                velocities_matrix.append(layer_vel)

        if velocities_matrix:
            velocities_matrix = np.array(velocities_matrix)

            im1 = axes[0, 0].imshow(velocities_matrix, aspect='auto', cmap='viridis')
            axes[0, 0].set_title('Attention Velocity Field')
            axes[0, 0].set_xlabel('Token Position (in response)')
            axes[0, 0].set_ylabel('Layer')
            plt.colorbar(im1, ax=axes[0, 0], label='Velocity')

    # 2. Mean velocity across layers
    if dynamics and 'mean_velocity' in dynamics:
        axes[0, 1].plot(dynamics['mean_velocity'], 'b-', linewidth=2)
        axes[0, 1].set_title('Average Attention Velocity')
        axes[0, 1].set_xlabel('Token Position (in response)')
        axes[0, 1].set_ylabel('Velocity')
        axes[0, 1].grid(True, alpha=0.3)

        # Mark high-velocity points
        for idx in dynamics['high_velocity_tokens'][:10]:
            axes[0, 1].axvline(x=idx, color='r', alpha=0.3, linestyle='--')

    # 3. Acceleration (2nd derivative)
    if dynamics and 'acceleration' in dynamics:
        axes[1, 0].plot(dynamics['acceleration'], 'g-', linewidth=2)
        axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 0].set_title('Attention Acceleration')
        axes[1, 0].set_xlabel('Token Position (in response)')
        axes[1, 0].set_ylabel('Acceleration')
        axes[1, 0].grid(True, alpha=0.3)

        # Mark high-acceleration points
        for idx in dynamics['high_acceleration_tokens'][:10]:
            axes[1, 0].axvline(x=idx, color='orange', alpha=0.3, linestyle='--')

    # 4. Layer 16 detailed attention pattern (middle layer)
    layer_16 = all_layers[16]
    if 'response' in layer_16:
        attn_weights = layer_16['response']['attention']['attn_weights']

        # Show attention for a few key tokens
        if len(attn_weights) > 10:
            # Sample tokens: early, middle, late
            sample_indices = [5, len(attn_weights)//2, len(attn_weights)-5]

            for i, idx in enumerate(sample_indices):
                attn = attn_weights[idx].mean(dim=0)  # Average across heads

                # Get the last row (current token's attention)
                current_token_attn = attn[-1, :].numpy()

                # Plot as a line
                axes[1, 1].plot(current_token_attn, label=f'Token {idx}', alpha=0.7)

        axes[1, 1].set_title('Layer 16: Attention Evolution')
        axes[1, 1].set_xlabel('Context Position')
        axes[1, 1].set_ylabel('Attention Weight')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # 5. Per-head velocity patterns
    if dynamics and 'velocities' in dynamics and 16 in dynamics['velocities']:
        layer_16_velocities = dynamics['velocities'][16]  # [n_heads, n_tokens-1]

        im2 = axes[2, 0].imshow(layer_16_velocities, aspect='auto', cmap='coolwarm')
        axes[2, 0].set_title('Layer 16: Per-Head Attention Velocity')
        axes[2, 0].set_xlabel('Token Position')
        axes[2, 0].set_ylabel('Head')
        plt.colorbar(im2, ax=axes[2, 0], label='Velocity')

    # 6. Response text with velocity overlay
    tokens = all_layers[0]['response_tokens']
    if dynamics and 'mean_velocity' in dynamics:
        axes[2, 1].bar(range(len(dynamics['mean_velocity'])), dynamics['mean_velocity'], alpha=0.6)
        axes[2, 1].set_title('Velocity by Response Token')
        axes[2, 1].set_xlabel('Token Index')
        axes[2, 1].set_ylabel('Attention Velocity')

        # Annotate high-velocity tokens
        for idx in dynamics['high_velocity_tokens'][:3]:
            if idx < len(tokens):
                axes[2, 1].annotate(tokens[idx],
                                   xy=(idx, dynamics['mean_velocity'][idx]),
                                   xytext=(idx, dynamics['mean_velocity'][idx] + 0.1),
                                   ha='center', fontsize=8, color='red')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")

    plt.show()

def main():
    """Analyze attention dynamics for dynamic prompts."""

    output_dir = Path('experiments/gemma_2b_cognitive_nov21/analysis/attention_dynamics')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze prompt 1 (uncertainty -> confidence)
    print("Analyzing attention dynamics during trait transition...\n")

    all_layers, dynamics = analyze_trait_transition(prompt_id=1)

    # Create visualization
    viz_path = output_dir / 'prompt_1_attention_flow.png'
    visualize_attention_flow(all_layers, dynamics, viz_path)

    # Save dynamics data
    if dynamics:
        # Convert to serializable format
        save_dynamics = {
            'mean_velocity': dynamics['mean_velocity'].tolist() if 'mean_velocity' in dynamics else [],
            'acceleration': dynamics['acceleration'].tolist() if 'acceleration' in dynamics else [],
            'high_velocity_tokens': dynamics['high_velocity_tokens'].tolist() if 'high_velocity_tokens' in dynamics else [],
            'high_acceleration_tokens': dynamics['high_acceleration_tokens'].tolist() if 'high_acceleration_tokens' in dynamics else []
        }

        with open(output_dir / 'prompt_1_dynamics.json', 'w') as f:
            json.dump(save_dynamics, f, indent=2)

        print(f"\nDynamics data saved to {output_dir / 'prompt_1_dynamics.json'}")

if __name__ == '__main__':
    main()