#!/usr/bin/env python3
"""
Quick exploration of captured layer internals to understand structure.
"""

import torch
from pathlib import Path
import json

# Load one layer's data to explore structure
layer_file = Path('experiments/gemma_2b_cognitive_nov21/inference/raw/internals/dynamic/1_L16.pt')
data = torch.load(layer_file, weights_only=True)

print("=== Layer 16 Internals Structure ===\n")

# Top-level keys
print("Top-level keys:", list(data.keys()))
print()

# Prompt data
if 'prompt_text' in data:
    print(f"Prompt: {data['prompt_text'][:100]}...")
    print(f"Prompt tokens: {len(data['prompt_tokens'])} tokens")

# Response data
if 'response_text' in data:
    print(f"Response: {data['response_text'][:100]}...")
    print(f"Response tokens: {len(data['response_tokens'])} tokens")
print()

# Explore prompt internals
if 'prompt' in data:
    print("=== Prompt Internals ===")
    for key, value in data['prompt'].items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: {v.shape}")
                elif isinstance(v, list):
                    print(f"  {k}: list of {len(v)} items")
                    if v and hasattr(v[0], 'shape'):
                        print(f"    First item shape: {v[0].shape}")
        elif hasattr(value, 'shape'):
            print(f"{key}: {value.shape}")
print()

# Explore response internals
if 'response' in data:
    print("=== Response Internals ===")
    for key, value in data['response'].items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: {v.shape}")
                elif isinstance(v, list):
                    print(f"  {k}: list of {len(v)} items")
                    if v and hasattr(v[0], 'shape'):
                        print(f"    First few shapes: {[x.shape for x in v[:3]]}")
        elif hasattr(value, 'shape'):
            print(f"{key}: {value.shape}")
print()

# Check what prompt 1 is about (from dynamic.json)
prompts_file = Path('inference/prompts/dynamic.json')
with open(prompts_file) as f:
    prompts_data = json.load(f)
    prompt_1 = next(p for p in prompts_data['prompts'] if p['id'] == 1)
    print(f"=== Dynamic Prompt 1 ===")
    print(f"Text: {prompt_1['text']}")
    print(f"Note: {prompt_1['note']} (designed trait transition)")