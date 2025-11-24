#!/usr/bin/env python3
"""
Complete inventory of captured internals with dimensions and evolution.
"""

import torch
from pathlib import Path
import numpy as np

# Load one complete layer to show all available data
layer_file = Path('experiments/gemma_2b_cognitive_nov21/inference/raw/internals/dynamic/1_L16.pt')
data = torch.load(layer_file, weights_only=True)

print("=" * 80)
print("COMPLETE DATA INVENTORY - LAYER INTERNALS")
print("=" * 80)
print("\nModel: Gemma 2B")
print("Hidden dim: 2304")
print("Attention heads: 8")
print("MLP intermediate: 9216 (4x hidden)")
print("Q dim: 2048 (256 per head)")
print("K/V dim: 1024 (128 per head)")
print()

# Analyze prompt phase
print("=" * 80)
print("PROMPT PHASE (26 tokens)")
print("=" * 80)

prompt_data = data['prompt']
for category, items in prompt_data.items():
    print(f"\n{category.upper()}:")
    for key, value in items.items():
        if hasattr(value, 'shape'):
            print(f"  {key:15s} {str(value.shape):20s} {str(value.dtype):15s}")
        elif isinstance(value, list):
            print(f"  {key:15s} List of {len(value)} items")

# Analyze response phase evolution
print("\n" + "=" * 80)
print("RESPONSE PHASE EVOLUTION (50 tokens generated)")
print("=" * 80)

response_data = data['response']

print("\n--- Token 0 (first generated) ---")
for category, items in response_data.items():
    print(f"\n{category.upper()}:")
    for key, value in items.items():
        if key == 'attn_weights':
            # Special case - list of growing tensors
            print(f"  {key:15s} {str(value[0].shape):20s} (26 prompt tokens)")
        elif hasattr(value, 'shape'):
            if len(value.shape) > 0 and value.shape[0] >= 1:
                # Show first token's shape
                if len(value.shape) == 3:
                    print(f"  {key:15s} {str(value[0].shape):20s}")
                else:
                    print(f"  {key:15s} {str(value.shape):20s}")

print("\n--- Token 1 ---")
if 'attn_weights' in response_data['attention']:
    print(f"  attn_weights:    {str(response_data['attention']['attn_weights'][1].shape):20s} (27 tokens now)")

print("\n--- Token 2 ---")
if 'attn_weights' in response_data['attention']:
    print(f"  attn_weights:    {str(response_data['attention']['attn_weights'][2].shape):20s} (28 tokens now)")

# Show how attention matrix grows
print("\n" + "=" * 80)
print("ATTENTION MATRIX GROWTH PATTERN")
print("=" * 80)
print("\nEach token adds a row and column to attention matrix:")
print("Token 0: [8, 26, 26] - attends to 26 prompt tokens")
print("Token 1: [8, 27, 27] - attends to 26 prompt + 1 generated")
print("Token 2: [8, 28, 28] - attends to 26 prompt + 2 generated")
print("...")
print("Token 49: [8, 75, 75] - attends to 26 prompt + 49 generated")

# Calculate total data per layer
print("\n" + "=" * 80)
print("DATA VOLUME PER LAYER")
print("=" * 80)

def calculate_size(tensor):
    if hasattr(tensor, 'numel'):
        return tensor.numel() * tensor.element_size() / (1024 * 1024)  # MB
    return 0

total_prompt = 0
total_response = 0

# Prompt phase
for key, value in prompt_data['attention'].items():
    if hasattr(value, 'numel'):
        total_prompt += calculate_size(value)
for key, value in prompt_data['mlp'].items():
    if hasattr(value, 'numel'):
        total_prompt += calculate_size(value)
for key, value in prompt_data['residual'].items():
    if hasattr(value, 'numel'):
        total_prompt += calculate_size(value)

# Response phase (approximate)
for key, value in response_data['attention'].items():
    if key == 'attn_weights':
        for attn in value:
            total_response += calculate_size(attn)
    elif hasattr(value, 'numel'):
        total_response += calculate_size(value)
for key, value in response_data['mlp'].items():
    if hasattr(value, 'numel'):
        total_response += calculate_size(value)
for key, value in response_data['residual'].items():
    if hasattr(value, 'numel'):
        total_response += calculate_size(value)

print(f"Prompt phase:   {total_prompt:.2f} MB")
print(f"Response phase: {total_response:.2f} MB")
print(f"Total per layer: {total_prompt + total_response:.2f} MB")
print(f"All 26 layers:  {(total_prompt + total_response) * 26:.2f} MB")

# Show mathematical operations available
print("\n" + "=" * 80)
print("MATHEMATICAL OPERATIONS FOR DYNAMICS")
print("=" * 80)

print("""
1. REPRESENTATION VELOCITY (token-layer field):
   v[layer, token] = ||residual[layer+1, token] - residual[layer, token]||₂

   Input:  residual_out [26 layers, 76 tokens, 2304 dims]
   Output: velocity_field [25 layers, 76 tokens]

2. ATTENTION VELOCITY (how focus changes):
   v_attn[layer, token] = ||attn[token+1] - attn[token]||₂

   Input:  attn_weights [76 tokens, 8 heads, variable_context, variable_context]
   Output: attn_velocity [26 layers, 75 token_transitions]

3. MLP ACTIVATION DYNAMICS:
   activation_pattern[layer, token] = gelu[layer, token] > threshold
   sparsity[layer, token] = count(gelu > 0) / 9216

   Input:  gelu [26 layers, 76 tokens, 9216 neurons]
   Output: sparsity_field [26 layers, 76 tokens]

4. ATTENTION ENTROPY (focus vs diffusion):
   H[token] = -Σ p(i) * log(p(i)) where p = attention_weights[token]

   Input:  attn_weights [token, 8 heads, context, context]
   Output: entropy [76 tokens]

5. INFORMATION FLOW (which tokens influence which):
   flow[src, dst] = attn_weight[dst, src] * ||value[src]||

   Input:  attn_weights + v_proj
   Output: flow_matrix [76 x 76]

6. CROSS-LAYER CONSISTENCY:
   consistency[layer] = cosine_sim(pattern[layer], pattern[layer+1])

   Input:  Any pattern (attn, mlp, residual)
   Output: consistency [25 layer_transitions]
""")

print("\n" + "=" * 80)
print("AVAILABLE VIEWS")
print("=" * 80)

print("""
With this data, we can create:

1. STATIC HEATMAPS:
   - Velocity field [layers x tokens]
   - Attention patterns [tokens x tokens] at each layer
   - MLP activation patterns [neurons x tokens]
   - Head specialization [heads x tokens x layers]

2. ANIMATED SEQUENCES:
   - Attention evolution (how attention shifts frame by frame)
   - Residual stream evolution (representation changes)
   - Information flow (particles flowing through attention)

3. 3D SURFACES:
   - Attention landscape (x=src_token, y=dst_token, z=weight)
   - Representation manifold (PCA to 3D)
   - Energy landscape (x=token, y=layer, z=magnitude)

4. GRAPH STRUCTURES:
   - Token dependency graph (nodes=tokens, edges=attention)
   - Layer communication (nodes=layers, edges=residual_flow)
   - Head specialization network

5. STATISTICAL PLOTS:
   - Velocity distributions by layer
   - Entropy evolution during generation
   - Attention "commitment points" (when patterns stabilize)
""")