#!/usr/bin/env python3
"""
Capture activations from base model using same tokens as instruct model.

Runs forward pass on gemma-2-2b (base) using token sequences from gemma-2-2b-it.
Enables comparison of massive dim patterns between base and instruct.

Usage:
    python scripts/capture_base_activations.py
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import MultiLayerCapture

def main():
    # Paths
    instruct_dir = Path("experiments/gemma-2-2b/inference/raw/residual/jailbreak_subset")
    output_dir = Path("experiments/gemma-2-2b-base/inference/raw/residual/jailbreak_subset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base model
    print("Loading gemma-2-2b base model...")
    model_name = "google/gemma-2-2b"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")
    
    # Process each prompt
    pt_files = sorted(instruct_dir.glob("*.pt"))
    print(f"\nProcessing {len(pt_files)} prompts...")
    
    for pt_file in pt_files:
        prompt_id = pt_file.stem
        print(f"\n  {prompt_id}...", end=" ", flush=True)
        
        # Load instruct model data
        instruct_data = torch.load(pt_file, weights_only=False, map_location='cpu')
        
        # Get token ids (prompt + response)
        prompt_ids = instruct_data['prompt']['token_ids']
        response_ids = instruct_data['response']['token_ids']
        
        # Combine into single sequence
        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor(prompt_ids)
        if isinstance(response_ids, list):
            response_ids = torch.tensor(response_ids)
        
        full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)  # [1, seq_len]
        full_ids = full_ids.to(model.device)
        
        prompt_len = len(prompt_ids)
        response_len = len(response_ids)
        
        print(f"tokens: {prompt_len}+{response_len}={len(full_ids[0])}", end=" ", flush=True)
        
        # Forward pass with activation capture
        with torch.no_grad():
            with MultiLayerCapture(model) as capture:
                _ = model(full_ids)
            
            all_acts = capture.get_all()  # {layer: [1, seq_len, hidden_dim]}
        
        # Split into prompt and response activations
        prompt_acts = {}
        response_acts = {}
        
        for layer, acts in all_acts.items():
            acts = acts[0].cpu()  # [seq_len, hidden_dim]
            prompt_acts[layer] = {'residual': acts[:prompt_len]}
            response_acts[layer] = {'residual': acts[prompt_len:]}
        
        # Save in same format as instruct data
        output_data = {
            'prompt': {
                'text': instruct_data['prompt']['text'],
                'tokens': instruct_data['prompt']['tokens'],
                'token_ids': instruct_data['prompt']['token_ids'],
                'activations': prompt_acts,
            },
            'response': {
                'text': instruct_data['response']['text'],
                'tokens': instruct_data['response']['tokens'],
                'token_ids': instruct_data['response']['token_ids'],
                'activations': response_acts,
            }
        }
        
        torch.save(output_data, output_dir / f"{prompt_id}.pt")
        print("saved", end="", flush=True)
    
    print(f"\n\nDone! Saved to {output_dir}")

if __name__ == "__main__":
    main()
