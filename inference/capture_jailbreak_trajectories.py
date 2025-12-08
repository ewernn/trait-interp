#!/usr/bin/env python3
"""
Capture jailbreak trajectories using attn_out refusal vector.

Usage:
    python inference/capture_jailbreak_trajectories.py \
        --prompt-set jailbreak_test_3 \
        --limit 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import json
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from traitlens import HookManager, projection
from traitlens.compute import compute_derivative
from utils.model import format_prompt


MODEL_NAME = "google/gemma-2-2b-it"
VECTOR_PATH = "experiments/gemma-2-2b-base/extraction/action/refusal/vectors/attn_out_probe_layer8.pt"
LAYER = 8


def capture_trajectory(model, tokenizer, prompt: str, vector: torch.Tensor, max_new_tokens: int = 100):
    """Capture attn_out projections token-by-token."""

    # Format prompt
    formatted = format_prompt(prompt, tokenizer)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    prompt_len = inputs['input_ids'].shape[1]

    # Storage for attn_out at layer 8
    attn_outputs = []

    def attn_hook(module, input, output):
        # output is tuple, first element is the attention output
        out = output[0] if isinstance(output, tuple) else output
        attn_outputs.append(out[:, -1, :].detach().cpu())  # Last token only

    # Register hook on attention output
    hook_name = f"model.layers.{LAYER}.self_attn"
    hook_handle = None
    for name, module in model.named_modules():
        if name == hook_name:
            hook_handle = module.register_forward_hook(attn_hook)
            break

    if hook_handle is None:
        raise ValueError(f"Could not find module: {hook_name}")

    try:
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    finally:
        hook_handle.remove()

    # Decode
    prompt_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    response_ids = output_ids[0, prompt_len:]
    response_tokens = tokenizer.convert_ids_to_tokens(response_ids)
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    # Stack activations and project
    activations = torch.stack(attn_outputs, dim=0).squeeze(1)  # [n_tokens, hidden_dim]

    # Split into prompt and response
    prompt_acts = activations[:prompt_len]
    response_acts = activations[prompt_len:]

    # Project onto refusal vector
    vector_cpu = vector.cpu().float()
    prompt_projs = [projection(act.unsqueeze(0).float(), vector_cpu).item() for act in prompt_acts]
    response_projs = [projection(act.unsqueeze(0).float(), vector_cpu).item() for act in response_acts]

    # Compute dynamics
    all_projs = prompt_projs + response_projs
    velocity = compute_derivative(torch.tensor(all_projs)).tolist()

    return {
        "prompt": {
            "text": prompt,
            "tokens": prompt_tokens,
            "n_tokens": len(prompt_tokens)
        },
        "response": {
            "text": response_text,
            "tokens": response_tokens,
            "n_tokens": len(response_tokens)
        },
        "projections": {
            "prompt": prompt_projs,
            "response": response_projs
        },
        "dynamics": {
            "velocity": velocity,
            "peak_velocity": max(abs(v) for v in velocity) if velocity else 0,
            "avg_velocity": sum(abs(v) for v in velocity) / len(velocity) if velocity else 0
        },
        "metadata": {
            "trait": "refusal",
            "category": "action",
            "source": "attn_out",
            "layer": LAYER,
            "vector_path": VECTOR_PATH,
            "target_model": MODEL_NAME,
            "capture_date": datetime.now().isoformat()
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Capture jailbreak trajectories")
    parser.add_argument("--prompt-set", required=True, help="Prompt set name")
    parser.add_argument("--limit", type=int, help="Limit number of prompts")
    parser.add_argument("--output-dir", default="experiments/gemma-2-2b-base/inference/jailbreak_trajectories",
                       help="Output directory")
    args = parser.parse_args()

    # Load prompts
    prompt_path = Path(f"inference/prompts/{args.prompt_set}.json")
    with open(prompt_path) as f:
        data = json.load(f)
    prompts = data['prompts']

    if args.limit:
        prompts = prompts[:args.limit]

    print(f"Loaded {len(prompts)} prompts from {args.prompt_set}")

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"Model on device: {model.device}")

    # Load refusal vector
    print(f"Loading vector: {VECTOR_PATH}")
    vector = torch.load(VECTOR_PATH, weights_only=True)

    # Create output dir
    output_dir = Path(args.output_dir) / args.prompt_set
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process prompts
    for item in tqdm(prompts, desc="Capturing"):
        prompt_id = item.get('id', item.get('prompt_id', 'unknown'))
        prompt_text = item.get('text', item.get('prompt'))

        trajectory = capture_trajectory(model, tokenizer, prompt_text, vector)
        trajectory['metadata']['prompt_id'] = prompt_id

        # Save
        output_path = output_dir / f"{prompt_id}.json"
        with open(output_path, 'w') as f:
            json.dump(trajectory, f, indent=2)

    print(f"\nSaved {len(prompts)} trajectories to {output_dir}")


if __name__ == "__main__":
    main()
