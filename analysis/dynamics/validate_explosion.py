#!/usr/bin/env python3
"""
Validation analysis - Is the layer 23-24 explosion real or artifact?
Addressing all the critical questions.
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

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
    """Extract hidden states into single tensor."""
    n_layers = len(all_layers)
    prompt_states = all_layers[0]['prompt']['residual']['output']
    response_states = all_layers[0]['response']['residual']['output']

    n_prompt_tokens = prompt_states.shape[0]
    n_response_tokens = response_states.shape[0]
    n_total_tokens = n_prompt_tokens + n_response_tokens
    d_model = prompt_states.shape[1]

    hidden_states = torch.zeros(n_layers, n_total_tokens, d_model)

    for layer_idx in range(n_layers):
        prompt_out = all_layers[layer_idx]['prompt']['residual']['output']
        hidden_states[layer_idx, :n_prompt_tokens] = prompt_out
        response_out = all_layers[layer_idx]['response']['residual']['output'].squeeze(1)
        hidden_states[layer_idx, n_prompt_tokens:] = response_out

    return hidden_states.numpy(), n_prompt_tokens

print("=" * 80)
print("VALIDATION ANALYSIS: IS THE EXPLOSION REAL?")
print("=" * 80)

# Load multiple prompts for comparison
prompts_to_analyze = [1, 2, 4]  # uncertainty->confidence, retrieval->construction, confidence->uncertainty
all_data = {}

for prompt_id in prompts_to_analyze:
    print(f"\nLoading prompt {prompt_id}...")
    all_layers = load_all_layers(prompt_id)
    hidden_states, n_prompt_tokens = extract_hidden_states(all_layers)
    tokens = all_layers[0]['prompt_tokens'] + all_layers[0]['response_tokens']
    all_data[prompt_id] = {
        'layers': all_layers,
        'hidden_states': hidden_states,
        'tokens': tokens,
        'n_prompt': n_prompt_tokens
    }

# ============================================================================
# QUESTION 1: Raw numbers for layers 22-24 for specific tokens
# ============================================================================

print("\n" + "=" * 80)
print("1. RAW VELOCITY/ACCELERATION FOR LAYERS 22-24")
print("-" * 80)

# Focus on prompt 1, specific tokens
data = all_data[1]
hidden_states = data['hidden_states']
tokens = data['tokens']

# Sample diverse tokens
sample_tokens = [10, 20, 30, 40, 50, 60]  # Mix of prompt and response

print("\nDETAILED ANALYSIS FOR SPECIFIC TOKENS:")
print("-" * 40)

for token_idx in sample_tokens:
    print(f"\nToken {token_idx}: '{tokens[token_idx]}'")

    for layer in [22, 23, 24]:
        if layer < 25:
            # Velocity
            v = np.linalg.norm(hidden_states[layer+1, token_idx] - hidden_states[layer, token_idx])

            # Also compute normalized velocity
            norm_curr = np.linalg.norm(hidden_states[layer, token_idx])
            norm_next = np.linalg.norm(hidden_states[layer+1, token_idx])
            avg_norm = (norm_curr + norm_next) / 2
            v_normalized = v / avg_norm if avg_norm > 0 else 0

            print(f"  L{layer}→{layer+1}: velocity={v:.1f}, normalized={v_normalized:.3f}")

            # Acceleration (if possible)
            if layer > 0 and layer < 24:
                v_prev = np.linalg.norm(hidden_states[layer, token_idx] - hidden_states[layer-1, token_idx])
                v_next = np.linalg.norm(hidden_states[layer+2, token_idx] - hidden_states[layer+1, token_idx])
                accel = v_next - v
                print(f"         acceleration={accel:.1f}")

# ============================================================================
# QUESTION 2: Compare across different prompt types
# ============================================================================

print("\n" + "=" * 80)
print("2. EXPLOSION PATTERN ACROSS DIFFERENT PROMPTS")
print("-" * 80)

# Load prompt descriptions
with open('inference/prompts/dynamic.json') as f:
    prompt_info = json.load(f)['prompts']

for prompt_id in prompts_to_analyze:
    data = all_data[prompt_id]
    hidden_states = data['hidden_states']
    n_layers, n_tokens, d_model = hidden_states.shape

    # Compute velocity for all layers
    velocities = []
    for layer in range(n_layers - 1):
        v = np.linalg.norm(hidden_states[layer+1] - hidden_states[layer], axis=-1).mean()
        velocities.append(v)

    # Find explosion point
    max_vel_layer = np.argmax(velocities)
    explosion_ratio = velocities[max_vel_layer] / np.mean(velocities[:10])

    prompt_desc = next(p for p in prompt_info if p['id'] == prompt_id)
    print(f"\nPrompt {prompt_id}: {prompt_desc['note']}")
    print(f"  Text: {prompt_desc['text'][:50]}...")
    print(f"  Explosion at layer {max_vel_layer}: {velocities[max_vel_layer]:.1f}")
    print(f"  Explosion ratio: {explosion_ratio:.1f}x early layers")
    print(f"  Layer 20-24 velocities: {[f'{v:.1f}' for v in velocities[20:25]]}")

# ============================================================================
# QUESTION 3: Normalized dynamics (accounting for scale)
# ============================================================================

print("\n" + "=" * 80)
print("3. NORMALIZED VELOCITY (accounting for representation magnitude)")
print("-" * 80)

data = all_data[1]
hidden_states = data['hidden_states']

raw_velocities = []
normalized_velocities = []
representation_norms = []

for layer in range(25):
    # Raw velocity
    v_raw = np.linalg.norm(hidden_states[layer+1] - hidden_states[layer], axis=-1)
    raw_velocities.append(v_raw.mean())

    # Representation norms
    norm_curr = np.linalg.norm(hidden_states[layer], axis=-1).mean()
    norm_next = np.linalg.norm(hidden_states[layer+1], axis=-1).mean()
    representation_norms.append((norm_curr + norm_next) / 2)

    # Normalized velocity
    avg_norm = (norm_curr + norm_next) / 2
    v_normalized = v_raw.mean() / avg_norm if avg_norm > 0 else 0
    normalized_velocities.append(v_normalized)

print("\nLayer | Raw Velocity | Avg Norm | Normalized Velocity")
print("-" * 55)
for layer in range(25):
    print(f"  {layer:2d}  |    {raw_velocities[layer]:6.1f}   |  {representation_norms[layer]:6.1f}  |      {normalized_velocities[layer]:.4f}")

print(f"\nKey insight:")
print(f"  Raw velocity explosion: {raw_velocities[24] / raw_velocities[0]:.1f}x")
print(f"  Normalized explosion: {normalized_velocities[24] / normalized_velocities[0]:.1f}x")
print(f"  Representation scale increase: {representation_norms[24] / representation_norms[0]:.1f}x")

# ============================================================================
# QUESTION 4: Residual vs Attention vs MLP contributions
# ============================================================================

print("\n" + "=" * 80)
print("4. WHERE DOES THE EXPLOSION COME FROM?")
print("-" * 80)

# Compare residual stream components for layers 22-24
layers_to_check = [22, 23, 24]

for layer_idx in layers_to_check:
    layer_data = all_data[1]['layers'][layer_idx]

    # Get different components (response phase, averaging over tokens)
    residual_in = layer_data['response']['residual']['input'].squeeze(1)  # [n_tokens, d_model]
    after_attn = layer_data['response']['residual']['after_attn'].squeeze(1)
    residual_out = layer_data['response']['residual']['output'].squeeze(1)

    # Compute contributions
    attn_contribution = (after_attn - residual_in).norm(dim=-1).mean().item()
    mlp_contribution = (residual_out - after_attn).norm(dim=-1).mean().item()
    total_change = (residual_out - residual_in).norm(dim=-1).mean().item()

    print(f"\nLayer {layer_idx}:")
    print(f"  Attention contribution: {attn_contribution:.1f} ({attn_contribution/total_change*100:.1f}%)")
    print(f"  MLP contribution: {mlp_contribution:.1f} ({mlp_contribution/total_change*100:.1f}%)")
    print(f"  Total change: {total_change:.1f}")

# ============================================================================
# QUESTION 5: Token convergence/divergence at layer 23
# ============================================================================

print("\n" + "=" * 80)
print("5. TOKEN SIMILARITY AT LAYERS 22 vs 24")
print("-" * 80)

# Compute pairwise cosine similarities
for layer in [22, 24]:
    states = all_data[1]['hidden_states'][layer]

    # Sample tokens to avoid huge matrix
    sample_indices = list(range(0, 76, 5))  # Every 5th token
    sampled_states = states[sample_indices]

    # Compute cosine similarity matrix
    cos_sim = cosine_similarity(sampled_states)

    # Average off-diagonal similarity (how similar are different tokens)
    mask = ~np.eye(cos_sim.shape[0], dtype=bool)
    avg_similarity = cos_sim[mask].mean()

    print(f"\nLayer {layer}:")
    print(f"  Average inter-token similarity: {avg_similarity:.3f}")
    print(f"  Max similarity (excluding self): {cos_sim[mask].max():.3f}")
    print(f"  Min similarity: {cos_sim[mask].min():.3f}")

# ============================================================================
# QUESTION 6: Which tokens drive layer 23 acceleration?
# ============================================================================

print("\n" + "=" * 80)
print("6. TOP TOKENS DRIVING LAYER 23 EXPLOSION")
print("-" * 80)

hidden_states = all_data[1]['hidden_states']
tokens = all_data[1]['tokens']

# Compute acceleration at layer 23 for each token
layer = 23
accelerations = []

for token_idx in range(len(tokens)):
    if layer > 0 and layer < 24:
        v_curr = np.linalg.norm(hidden_states[layer+1, token_idx] - hidden_states[layer, token_idx])
        v_prev = np.linalg.norm(hidden_states[layer, token_idx] - hidden_states[layer-1, token_idx])
        accel = v_curr - v_prev
        accelerations.append((token_idx, accel, tokens[token_idx]))

# Sort by acceleration
accelerations.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 tokens with highest acceleration at layer 23:")
for i in range(10):
    idx, accel, token = accelerations[i]
    token_type = "PROMPT" if idx < all_data[1]['n_prompt'] else "RESPONSE"
    print(f"  {idx:3d}. '{token:15s}' [{token_type:8s}]: {accel:6.1f}")

print("\nBottom 5 tokens (lowest/negative acceleration):")
for i in range(5):
    idx, accel, token = accelerations[-(i+1)]
    token_type = "PROMPT" if idx < all_data[1]['n_prompt'] else "RESPONSE"
    print(f"  {idx:3d}. '{token:15s}' [{token_type:8s}]: {accel:6.1f}")

# ============================================================================
# QUESTION 7: Attention entropy at layers 22-24
# ============================================================================

print("\n" + "=" * 80)
print("7. ATTENTION ENTROPY AT CRITICAL LAYERS")
print("-" * 80)

for layer_idx in [22, 23, 24]:
    layer_data = all_data[1]['layers'][layer_idx]
    attn_weights = layer_data['response']['attention']['attn_weights']

    entropies = []
    for t in range(min(20, len(attn_weights))):  # First 20 response tokens
        attn = attn_weights[t].mean(dim=0)  # Average across heads
        current_token_attn = attn[-1, :]  # Current token's attention

        # Compute entropy
        probs = current_token_attn + 1e-10
        entropy = -(probs * torch.log(probs)).sum().item()
        entropies.append(entropy)

    print(f"\nLayer {layer_idx}:")
    print(f"  Average entropy: {np.mean(entropies):.3f}")
    print(f"  Entropy trend: {entropies[0]:.3f} → {entropies[-1]:.3f}")
    print(f"  Change: {entropies[-1] - entropies[0]:+.3f}")

# ============================================================================
# Save comprehensive validation plot
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Validation Analysis: Layer 23-24 Explosion')

# 1. Raw vs normalized velocities
ax = axes[0, 0]
ax.plot(raw_velocities, 'b-', label='Raw', linewidth=2)
ax.plot([v * 100 for v in normalized_velocities], 'r--', label='Normalized (×100)', linewidth=2)
ax.axvline(x=23, color='orange', linestyle=':', alpha=0.5)
ax.set_xlabel('Layer')
ax.set_ylabel('Velocity')
ax.set_title('Raw vs Normalized Velocity')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Representation norms
ax = axes[0, 1]
ax.plot(representation_norms, 'g-', linewidth=2)
ax.axvline(x=23, color='orange', linestyle=':', alpha=0.5)
ax.set_xlabel('Layer')
ax.set_ylabel('Average Norm')
ax.set_title('Representation Magnitude Growth')
ax.grid(True, alpha=0.3)

# 3. Scatter: velocity vs acceleration
ax = axes[0, 2]
for layer in range(24):
    v = raw_velocities[layer]
    if layer > 0:
        a = raw_velocities[layer] - raw_velocities[layer-1]
        color = 'red' if layer >= 22 else 'blue'
        ax.scatter(v, a, s=50, c=color, alpha=0.6)
        if layer in [22, 23, 24]:
            ax.annotate(f'L{layer}', (v, a), fontsize=8)
ax.set_xlabel('Velocity')
ax.set_ylabel('Acceleration')
ax.set_title('Phase Space: Velocity vs Acceleration')
ax.grid(True, alpha=0.3)

# 4. Token-type breakdown for layer 23
ax = axes[1, 0]
prompt_accels = [a for i, a, _ in accelerations if i < all_data[1]['n_prompt']]
response_accels = [a for i, a, _ in accelerations if i >= all_data[1]['n_prompt']]
ax.hist([prompt_accels, response_accels], label=['Prompt', 'Response'], alpha=0.7, bins=20)
ax.set_xlabel('Acceleration at Layer 23')
ax.set_ylabel('Count')
ax.set_title('Acceleration Distribution by Token Type')
ax.legend()

# 5. Cross-prompt comparison
ax = axes[1, 1]
for prompt_id in prompts_to_analyze:
    hidden = all_data[prompt_id]['hidden_states']
    layer_vels = []
    for layer in range(25):
        v = np.linalg.norm(hidden[layer+1] - hidden[layer], axis=-1).mean()
        layer_vels.append(v)
    ax.plot(layer_vels, label=f'Prompt {prompt_id}', linewidth=2)
ax.axvline(x=23, color='orange', linestyle=':', alpha=0.5)
ax.set_xlabel('Layer')
ax.set_ylabel('Velocity')
ax.set_title('Explosion Pattern Across Prompts')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Component contributions
ax = axes[1, 2]
layers = [22, 23, 24]
attn_contribs = []
mlp_contribs = []
for layer_idx in layers:
    layer_data = all_data[1]['layers'][layer_idx]
    residual_in = layer_data['response']['residual']['input'].squeeze(1)
    after_attn = layer_data['response']['residual']['after_attn'].squeeze(1)
    residual_out = layer_data['response']['residual']['output'].squeeze(1)

    attn = (after_attn - residual_in).norm(dim=-1).mean().item()
    mlp = (residual_out - after_attn).norm(dim=-1).mean().item()
    attn_contribs.append(attn)
    mlp_contribs.append(mlp)

x = np.arange(len(layers))
width = 0.35
ax.bar(x - width/2, attn_contribs, width, label='Attention')
ax.bar(x + width/2, mlp_contribs, width, label='MLP')
ax.set_xlabel('Layer')
ax.set_ylabel('Contribution')
ax.set_title('Attention vs MLP at Critical Layers')
ax.set_xticks(x)
ax.set_xticklabels(layers)
ax.legend()

plt.tight_layout()
plt.savefig('experiments/gemma_2b_cognitive_nov21/analysis/attention_dynamics/validation_analysis.png')
print(f"\n\nValidation plots saved!")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")