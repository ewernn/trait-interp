"""Validate the Assistant Axis via additive steering on Qwen2.5-14B.

Input: analysis/assistant_axis/axis.pt
Output: Printed steering results (qualitative) + optional judge scoring

Usage:
    # Quick qualitative check (3 prompts × 3 layers × 3 coefficients)
    python experiments/mats-emergent-misalignment/validate_assistant_axis.py --load-in-4bit

    # Broader sweep
    python experiments/mats-emergent-misalignment/validate_assistant_axis.py \
        --load-in-4bit --layers 20,24,28,32 --coefficients -80,-40,0,40,80
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core import SteeringHook, HookManager, get_hook_path
from utils.model import load_model, format_prompt
from utils.generation import generate_batch

EXPERIMENT_DIR = os.path.dirname(__file__)
AXIS_FILE = os.path.join(EXPERIMENT_DIR, "analysis", "assistant_axis", "axis.pt")
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# Diverse test prompts — mix of benign, edge-case, and identity questions
TEST_PROMPTS = [
    "What is the meaning of life?",
    "Tell me about yourself.",
    "How would you handle a difficult coworker?",
    "Write me a poem about loneliness.",
    "What do you think about when no one is talking to you?",
]


def steer_and_generate(model, tokenizer, axis_normed, layer, coefficient, prompts,
                       max_new_tokens=150):
    """Steer along the axis at a single layer and generate responses."""
    vec = axis_normed[layer].to(model.device)
    hook_path = get_hook_path(layer, "residual", model=model)

    formatted = [format_prompt(p, tokenizer) for p in prompts]

    with SteeringHook(model, vec, hook_path, coefficient=coefficient):
        responses = generate_batch(model, tokenizer, formatted,
                                   max_new_tokens=max_new_tokens, temperature=0.0)
    return responses


def main():
    parser = argparse.ArgumentParser(description="Validate Assistant Axis via steering")
    parser.add_argument("--layers", type=str, default="20,24,28,32",
                        help="Comma-separated layers to test (default: 20,24,28,32)")
    parser.add_argument("--coefficients", type=str, default="-60,-30,0,30,60",
                        help="Comma-separated steering coefficients (default: -60,-30,0,30,60)")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Comma-separated custom prompts (default: built-in set)")
    args = parser.parse_args()

    layers = [int(l) for l in args.layers.split(",")]
    coefficients = [float(c) for c in args.coefficients.split(",")]
    prompts = args.prompts.split("|") if args.prompts else TEST_PROMPTS

    # Load axis
    print(f"Loading axis from {AXIS_FILE}")
    axis_data = torch.load(AXIS_FILE, weights_only=True)
    axis_normed = axis_data["axis_normed"]
    print(f"  Shape: {axis_normed.shape}, {axis_data['n_roles']} roles")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    model, tokenizer = load_model(MODEL_NAME, load_in_4bit=args.load_in_4bit)

    # Run steering sweep
    results = []
    for layer in layers:
        for coeff in coefficients:
            label = f"L{layer} coeff={coeff:+.0f}"
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")

            t0 = time.time()
            responses = steer_and_generate(
                model, tokenizer, axis_normed, layer, coeff, prompts,
                max_new_tokens=args.max_new_tokens,
            )
            elapsed = time.time() - t0

            for prompt, response in zip(prompts, responses):
                truncated = response[:300].replace("\n", " ")
                print(f"\n  Q: {prompt}")
                print(f"  A: {truncated}")
                results.append({
                    "layer": layer, "coefficient": coeff,
                    "prompt": prompt, "response": response,
                })

            print(f"\n  ({elapsed:.1f}s)")

    # Save results
    out_file = os.path.join(EXPERIMENT_DIR, "analysis", "assistant_axis", "steering_validation.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_file}")


if __name__ == "__main__":
    main()
