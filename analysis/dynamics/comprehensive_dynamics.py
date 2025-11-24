#!/usr/bin/env python3
"""
Comprehensive dynamics analysis - all the calculations you outlined.
Using the actual captured Gemma 2B internals.
"""

import torch
import numpy as np
from pathlib import Path
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_layers(prompt_id: int, experiment: str = 'gemma_2b_cognitive_nov21') -> dict:
    """Load all 26 layers for a prompt."""
    base_path = Path(f'experiments/{experiment}/inference/raw/internals/dynamic')
    all_layers = {}

    for layer in range(26):
        layer_file = base_path / f'{prompt_id}_L{layer}.pt'
        if layer_file.exists():
            all_layers[layer] = torch.load(layer_file, weights_only=True)

    return all_layers

def extract_hidden_states(all_layers):
    """Extract hidden states (residual stream) into single tensor."""
    n_layers = len(all_layers)

    # Get dimensions from first layer
    prompt_states = all_layers[0]['prompt']['residual']['output']  # [n_prompt, d_model]
    response_states = all_layers[0]['response']['residual']['output']  # [n_response, 1, d_model]

    n_prompt_tokens = prompt_states.shape[0]
    n_response_tokens = response_states.shape[0]
    n_total_tokens = n_prompt_tokens + n_response_tokens
    d_model = prompt_states.shape[1]

    # Combine into single tensor [n_layers, n_tokens, d_model]
    hidden_states = torch.zeros(n_layers, n_total_tokens, d_model)

    for layer_idx in range(n_layers):
        # Prompt tokens
        prompt_out = all_layers[layer_idx]['prompt']['residual']['output']
        hidden_states[layer_idx, :n_prompt_tokens] = prompt_out

        # Response tokens
        response_out = all_layers[layer_idx]['response']['residual']['output'].squeeze(1)
        hidden_states[layer_idx, n_prompt_tokens:] = response_out

    return hidden_states.numpy(), n_prompt_tokens

print("=" * 80)
print("COMPREHENSIVE DYNAMICS ANALYSIS")
print("=" * 80)

# Load data
print("\nLoading layer data...")
all_layers = load_all_layers(prompt_id=1)
hidden_states, n_prompt_tokens = extract_hidden_states(all_layers)
n_layers, n_tokens, d_model = hidden_states.shape

print(f"Shape: {n_layers} layers × {n_tokens} tokens × {d_model} dims")
print(f"Prompt: {n_prompt_tokens} tokens, Response: {n_tokens - n_prompt_tokens} tokens")

# Get tokens for reference
tokens = all_layers[0]['prompt_tokens'] + all_layers[0]['response_tokens']

# ============================================================================
# 1. VELOCITY & ACCELERATION FIELDS
# ============================================================================

print("\n" + "=" * 80)
print("1. VELOCITY & ACCELERATION FIELDS")
print("-" * 80)

# Velocity: How fast representations change
velocity = np.zeros((n_layers-1, n_tokens, d_model))
for l in range(n_layers-1):
    velocity[l] = hidden_states[l+1] - hidden_states[l]

# Acceleration: Where change rate shifts
acceleration = np.zeros((n_layers-2, n_tokens, d_model))
for l in range(n_layers-2):
    acceleration[l] = velocity[l+1] - velocity[l]

# Key metrics per token
velocity_magnitude = np.linalg.norm(velocity, axis=-1)  # [n_layers-1, n_tokens]
acceleration_magnitude = np.linalg.norm(acceleration, axis=-1)  # [n_layers-2, n_tokens]

# Find commitment points (peak acceleration)
commitment_layers = {}
for token_idx in range(n_tokens):
    commitment_layers[token_idx] = np.argmax(acceleration_magnitude[:, token_idx])

# Report interesting tokens
print("\nHIGHEST VELOCITY TOKENS:")
for i in range(5):
    max_vel_idx = np.unravel_index(np.argsort(velocity_magnitude.ravel())[-i-1], velocity_magnitude.shape)
    layer, token = max_vel_idx
    print(f"  Layer {layer}→{layer+1}, Token {token} ('{tokens[token]}'): {velocity_magnitude[layer, token]:.1f}")

print("\nHIGHEST ACCELERATION TOKENS (commitment points):")
for i in range(5):
    max_acc_idx = np.unravel_index(np.argsort(acceleration_magnitude.ravel())[-i-1], acceleration_magnitude.shape)
    layer, token = max_acc_idx
    print(f"  Layer {layer}, Token {token} ('{tokens[token]}'): {acceleration_magnitude[layer, token]:.1f}")

# ============================================================================
# 2. DIRECTION CHANGES (Bifurcation Points)
# ============================================================================

print("\n" + "=" * 80)
print("2. DIRECTION CHANGES (Bifurcation Points)")
print("-" * 80)

# Cosine similarity between successive velocities
velocity_alignment = np.zeros((n_layers-2, n_tokens))

for l in range(n_layers-2):
    for j in range(n_tokens):
        v1 = velocity[l, j]
        v2 = velocity[l+1, j]

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 > 1e-8 and norm2 > 1e-8:
            cos_sim = np.dot(v1, v2) / (norm1 * norm2)
            velocity_alignment[l, j] = cos_sim

# Sharp direction changes indicate "decisions"
direction_changes = 1 - velocity_alignment  # High values = sharp turns

print("\nSHARPEST DIRECTION CHANGES (potential bifurcations):")
for i in range(5):
    max_change_idx = np.unravel_index(np.argsort(direction_changes.ravel())[-i-1], direction_changes.shape)
    layer, token = max_change_idx
    print(f"  Layer {layer}→{layer+1}, Token {token} ('{tokens[token]}'): {direction_changes[layer, token]:.3f}")

# ============================================================================
# 3. PHASE SPACE TRAJECTORIES
# ============================================================================

print("\n" + "=" * 80)
print("3. PHASE SPACE TRAJECTORIES")
print("-" * 80)

# Project to 2D for visualization
pca = PCA(n_components=2)
all_hiddens = hidden_states.reshape(-1, d_model)  # [n_layers * n_tokens, d_model]
projected = pca.fit_transform(all_hiddens).reshape(n_layers, n_tokens, 2)

print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")

# Visualize key token trajectories
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Token Trajectories in Phase Space')

# Select interesting tokens
key_tokens = [
    0,  # First prompt token
    n_prompt_tokens - 1,  # Last prompt token
    n_prompt_tokens,  # First response token
    n_prompt_tokens + 10,  # Mid response
    n_prompt_tokens + 25,  # Later response
    n_tokens - 1  # Last token
]

for idx, token_idx in enumerate(key_tokens):
    ax = axes[idx // 3, idx % 3]

    trajectory = projected[:, token_idx, :]  # [n_layers, 2]

    # Color by acceleration
    colors = acceleration_magnitude[:, token_idx]

    # Plot trajectory with arrows
    for i in range(len(trajectory) - 1):
        ax.arrow(trajectory[i, 0], trajectory[i, 1],
                trajectory[i+1, 0] - trajectory[i, 0],
                trajectory[i+1, 1] - trajectory[i, 1],
                head_width=0.5, head_length=0.3,
                fc=plt.cm.viridis(i / len(trajectory)),
                ec=plt.cm.viridis(i / len(trajectory)))

    # Mark start and end
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='*', label='End')

    ax.set_title(f"Token {token_idx}: '{tokens[token_idx][:15]}...'")
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/gemma_2b_cognitive_nov21/analysis/attention_dynamics/phase_space_trajectories.png')
print(f"\nPhase space plot saved!")

# ============================================================================
# 4. SENSITIVITY ANALYSIS (Simplified - would need re-inference for true Lyapunov)
# ============================================================================

print("\n" + "=" * 80)
print("4. SENSITIVITY ANALYSIS")
print("-" * 80)

# Estimate sensitivity by looking at neighboring tokens
sensitivities = []

for token_idx in range(n_prompt_tokens, min(n_prompt_tokens + 20, n_tokens - 1)):
    # Compare trajectory to neighboring token
    curr_trajectory = hidden_states[:, token_idx]
    next_trajectory = hidden_states[:, token_idx + 1]

    # Measure divergence across layers
    divergence = []
    for l in range(n_layers):
        dist = np.linalg.norm(curr_trajectory[l] - next_trajectory[l])
        divergence.append(dist)

    # Rate of divergence (proxy for Lyapunov)
    if len(divergence) > 1:
        divergence_array = np.array(divergence)
        sensitivity = np.polyfit(range(len(divergence)), np.log(divergence_array + 1e-8), 1)[0]
        sensitivities.append((token_idx, sensitivity, tokens[token_idx]))

print("\nTOKENS WITH HIGHEST SENSITIVITY (most chaotic):")
sensitivities.sort(key=lambda x: abs(x[1]), reverse=True)
for i in range(min(5, len(sensitivities))):
    idx, sens, token = sensitivities[i]
    print(f"  Token {idx} ('{token}'): {sens:.3f}")

# ============================================================================
# 5. LAYER STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("5. LAYER-WISE STATISTICS")
print("-" * 80)

# Average dynamics by layer
avg_velocity_by_layer = velocity_magnitude.mean(axis=1)
avg_acceleration_by_layer = acceleration_magnitude.mean(axis=1)

print("\nLAYER DYNAMICS:")
print("Layer | Avg Velocity | Avg Acceleration")
print("-" * 40)
for l in range(min(n_layers-1, 26)):
    vel = avg_velocity_by_layer[l] if l < len(avg_velocity_by_layer) else 0
    acc = avg_acceleration_by_layer[l] if l < len(avg_acceleration_by_layer) else 0
    print(f"  {l:2d}  |    {vel:6.1f}    |     {acc:6.1f}")

# Find most "active" layers
most_dynamic_layer = np.argmax(avg_velocity_by_layer)
most_accelerating_layer = np.argmax(avg_acceleration_by_layer)

print(f"\nMost dynamic layer: {most_dynamic_layer} (velocity: {avg_velocity_by_layer[most_dynamic_layer]:.1f})")
print(f"Most accelerating layer: {most_accelerating_layer} (acceleration: {avg_acceleration_by_layer[most_accelerating_layer]:.1f})")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Create comprehensive heatmap
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Velocity field
im1 = axes[0].imshow(velocity_magnitude, aspect='auto', cmap='viridis')
axes[0].set_title('Velocity Magnitude')
axes[0].set_xlabel('Token')
axes[0].set_ylabel('Layer Transition')
plt.colorbar(im1, ax=axes[0])

# Acceleration field
im2 = axes[1].imshow(acceleration_magnitude, aspect='auto', cmap='plasma')
axes[1].set_title('Acceleration Magnitude (Commitment Points)')
axes[1].set_xlabel('Token')
axes[1].set_ylabel('Layer')
plt.colorbar(im2, ax=axes[1])

# Direction changes
im3 = axes[2].imshow(direction_changes, aspect='auto', cmap='RdBu_r', vmin=0, vmax=2)
axes[2].set_title('Direction Changes (1 = orthogonal turn)')
axes[2].set_xlabel('Token')
axes[2].set_ylabel('Layer')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig('experiments/gemma_2b_cognitive_nov21/analysis/attention_dynamics/comprehensive_dynamics.png')
print(f"\nComprehensive dynamics plot saved!")

# Save numerical results
results = {
    'n_layers': n_layers,
    'n_tokens': n_tokens,
    'd_model': d_model,
    'n_prompt_tokens': n_prompt_tokens,
    'avg_velocity_by_layer': avg_velocity_by_layer.tolist(),
    'avg_acceleration_by_layer': avg_acceleration_by_layer.tolist(),
    'most_dynamic_layer': int(most_dynamic_layer),
    'most_accelerating_layer': int(most_accelerating_layer),
    'pca_variance_explained': pca.explained_variance_ratio_.tolist()
}

with open('experiments/gemma_2b_cognitive_nov21/analysis/attention_dynamics/dynamics_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nMetrics saved to dynamics_metrics.json")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")