#!/usr/bin/env python3
"""
Concrete example of velocity field computation with actual numbers.
"""

import torch
import numpy as np
from pathlib import Path

print("=" * 80)
print("VELOCITY FIELD MATHEMATICS - CONCRETE EXAMPLE")
print("=" * 80)

# Load actual data for layers 15, 16, 17
layers = {}
for L in [15, 16, 17]:
    layer_file = Path(f'experiments/gemma_2b_cognitive_nov21/inference/raw/internals/dynamic/1_L{L}.pt')
    layers[L] = torch.load(layer_file, weights_only=True)

# Focus on tokens 10-12 for example
token_range = range(10, 13)

print("\n1. REPRESENTATION VELOCITY (How hidden states change between layers)")
print("-" * 70)

for token_idx in token_range:
    print(f"\nToken {token_idx} ('{layers[15]['response_tokens'][token_idx]}')")

    # Get residual vectors at each layer for this token
    r15 = layers[15]['response']['residual']['output'][token_idx, 0, :]  # [2304]
    r16 = layers[16]['response']['residual']['output'][token_idx, 0, :]  # [2304]
    r17 = layers[17]['response']['residual']['output'][token_idx, 0, :]  # [2304]

    # Compute velocities
    v_15to16 = torch.norm(r16 - r15).item()
    v_16to17 = torch.norm(r17 - r16).item()

    # Show magnitudes
    print(f"  ||r16 - r15||₂ = {v_15to16:.3f}")
    print(f"  ||r17 - r16||₂ = {v_16to17:.3f}")
    print(f"  Acceleration = {v_16to17 - v_15to16:.3f}")

print("\n" + "=" * 80)
print("2. ATTENTION VELOCITY (How attention patterns shift between tokens)")
print("-" * 70)

# Look at layer 16, tokens 10->11->12
layer_16 = layers[16]
attn_weights = layer_16['response']['attention']['attn_weights']

for t in range(10, 12):
    token_now = layer_16['response_tokens'][t]
    token_next = layer_16['response_tokens'][t+1]

    # Get attention matrices
    attn_t = attn_weights[t]      # [8, context_t, context_t]
    attn_t1 = attn_weights[t+1]   # [8, context_t+1, context_t+1]

    # Average across heads
    attn_t_avg = attn_t.mean(dim=0)    # [context_t, context_t]
    attn_t1_avg = attn_t1.mean(dim=0)  # [context_t+1, context_t+1]

    # Compare how the LAST token's attention changed
    # (what the current token is looking at)
    curr_attention = attn_t_avg[-1, :]   # [context_t]
    next_attention = attn_t1_avg[-1, :len(curr_attention)]  # [context_t]

    velocity = torch.norm(next_attention - curr_attention).item()

    print(f"\nToken {t}→{t+1}: '{token_now}'→'{token_next}'")
    print(f"  Context size: {len(curr_attention)}→{len(attn_t1_avg)}")
    print(f"  Attention velocity: {velocity:.3f}")

    # Show what changed most
    diff = (next_attention - curr_attention).abs()
    top_changes = torch.topk(diff, 3)
    print(f"  Biggest attention shifts at positions: {top_changes.indices.tolist()}")

print("\n" + "=" * 80)
print("3. THE 2D VELOCITY FIELD (layers × tokens)")
print("-" * 70)

# Build the full velocity field for visualization
velocity_field = []

for layer in range(15, 18):  # Just 3 layers for example
    layer_velocities = []

    if layer < 17:  # Can't compute velocity for last layer
        curr_layer = layers[layer]
        next_layer = layers[layer + 1]

        for token in range(20):  # First 20 response tokens
            r_curr = curr_layer['response']['residual']['output'][token, 0, :]
            r_next = next_layer['response']['residual']['output'][token, 0, :]
            velocity = torch.norm(r_next - r_curr).item()
            layer_velocities.append(velocity)

    if layer_velocities:
        velocity_field.append(layer_velocities)

velocity_field = np.array(velocity_field)

print(f"\nVelocity field shape: {velocity_field.shape}")
print(f"  Rows (layers): {velocity_field.shape[0]}")
print(f"  Columns (tokens): {velocity_field.shape[1]}")
print(f"\nSample of velocity field:")
print(velocity_field[:, :5].round(2))

# Find interesting patterns
print(f"\nMax velocity: {velocity_field.max():.3f}")
max_pos = np.unravel_index(velocity_field.argmax(), velocity_field.shape)
print(f"  At layer {15 + max_pos[0]}, token {max_pos[1]}")

print(f"\nMean velocity by layer:")
for i, layer_idx in enumerate(range(15, 17)):
    print(f"  Layer {layer_idx}→{layer_idx+1}: {velocity_field[i].mean():.3f}")

print("\n" + "=" * 80)
print("4. MLP SPARSITY DYNAMICS")
print("-" * 70)

# Analyze how many neurons are active
for token_idx in [10, 11, 12]:
    gelu_acts = layers[16]['response']['mlp']['gelu'][token_idx, 0, :]  # [9216]

    n_active = (gelu_acts > 0).sum().item()
    sparsity = n_active / 9216

    # Find strongest neurons
    top_neurons = torch.topk(gelu_acts, 5)

    print(f"\nToken {token_idx} ('{layers[16]['response_tokens'][token_idx]}'):")
    print(f"  Active neurons: {n_active}/9216 ({sparsity*100:.1f}%)")
    print(f"  Top 5 neuron activations: {top_neurons.values.tolist()}")
    print(f"  Top 5 neuron indices: {top_neurons.indices.tolist()}")

print("\n" + "=" * 80)
print("5. ATTENTION ENTROPY EVOLUTION")
print("-" * 70)

# How focused vs diffuse is attention?
entropies = []
for t in range(15):  # First 15 tokens
    attn = attn_weights[t].mean(dim=0)  # Average across heads
    current_token_attn = attn[-1, :]  # Current token's attention distribution

    # Compute entropy
    # H = -Σ p * log(p)
    probs = current_token_attn + 1e-10  # Avoid log(0)
    entropy = -(probs * torch.log(probs)).sum().item()
    entropies.append(entropy)

    if t % 5 == 0:
        print(f"Token {t}: H = {entropy:.3f} ({'focused' if entropy < 2.5 else 'diffuse'})")

print(f"\nEntropy trend: {entropies[0]:.3f} → {entropies[-1]:.3f}")
print(f"Change: {'+' if entropies[-1] > entropies[0] else ''}{entropies[-1] - entropies[0]:.3f} (becoming {'more diffuse' if entropies[-1] > entropies[0] else 'more focused'})")