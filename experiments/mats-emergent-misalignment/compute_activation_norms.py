"""Compute mean(||hidden_state||) per layer for baseline (clean instruct) activations.

Used to normalize probe scores across traits that use different layers, since activation
norms vary by layer. The normalization factor is: score / norm[layer].

Input: Clean instruct model, 100 eval prompts
Output: experiments/mats-emergent-misalignment/analysis/activation_norms_{model_size}.json

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/mats-emergent-misalignment/compute_activation_norms.py --model-size 14b
    CUDA_VISIBLE_DEVICES=0 python experiments/mats-emergent-misalignment/compute_activation_norms.py --model-size 4b
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core import MultiLayerCapture
from utils.model import load_model, get_num_layers, format_prompt, tokenize_batch

EXPERIMENT = "mats-emergent-misalignment"
BASE_DIR = os.path.dirname(__file__)
DATASETS_DIR = os.path.join(BASE_DIR, "../../datasets/inference")

# Map model-size flag to experiment config variant names
MODEL_SIZE_TO_VARIANT = {
    "14b": "instruct",
    "4b": "qwen3_4b_instruct",
}

MAX_NEW_TOKENS = 200  # Generate enough tokens to match scoring pipeline


def load_experiment_config():
    config_path = os.path.join(BASE_DIR, "config.json")
    with open(config_path) as f:
        return json.load(f)


def get_model_name(config, model_size):
    variant = MODEL_SIZE_TO_VARIANT[model_size]
    if variant not in config["model_variants"]:
        raise ValueError(
            f"Variant '{variant}' not found in config. "
            f"Available: {list(config['model_variants'].keys())}"
        )
    return config["model_variants"][variant]["model"]


def load_prompts(n=100):
    """Load ~n diverse prompts from all available eval sets."""
    prompts = []
    seen = set()

    # Load from all eval sets for diversity
    eval_files = [
        "em_generic_eval", "em_medical_eval", "sriram_diverse",
        "sriram_normal", "sriram_factual", "sriram_harmful",
        "emotional_vulnerability", "ethical_dilemmas",
        "identity_introspection", "interpersonal_advice",
    ]

    for name in eval_files:
        path = os.path.join(DATASETS_DIR, f"{name}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for item in data:
            text = item["prompt"]
            if text not in seen:
                seen.add(text)
                prompts.append(text)

    if len(prompts) < n:
        print(f"  Warning: only found {len(prompts)} unique prompts (wanted {n})")

    return prompts[:n]


def compute_norms(model, tokenizer, prompts):
    """Forward pass each prompt, capture hidden states at all layers, compute norms.

    For each prompt: format -> generate response -> prefill full sequence ->
    capture hidden states at all response positions -> compute ||h|| per layer.

    Uses all response tokens (not just first 5) to match the scoring pipeline.

    Returns list of norms, one per layer (mean of L2 norms across all positions and prompts).
    """
    n_layers = get_num_layers(model)
    # Accumulate: sum of norms and count per layer
    norm_sums = torch.zeros(n_layers, dtype=torch.float64)
    total_count = 0

    for i, prompt in enumerate(prompts):
        # Format prompt with chat template
        formatted = format_prompt(prompt, tokenizer)

        # Tokenize prompt to get prompt length
        prompt_ids = tokenize_batch([formatted], tokenizer)
        prompt_len = prompt_ids["lengths"][0]
        input_ids = prompt_ids["input_ids"].to(model.device)
        attention_mask = prompt_ids["attention_mask"].to(model.device)

        # Generate response tokens
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Full sequence (prompt + generated tokens)
        full_ids = outputs[0:1]  # keep batch dim
        full_len = full_ids.shape[1]
        n_response = full_len - prompt_len

        if n_response < 1:
            print(f"  Prompt {i}: no response tokens generated, skipping")
            continue

        # Prefill the full sequence and capture all layers
        full_mask = torch.ones_like(full_ids)
        with MultiLayerCapture(model, keep_on_gpu=True) as capture:
            with torch.no_grad():
                model(input_ids=full_ids, attention_mask=full_mask)

            # Extract all response token hidden states per layer
            for layer in range(n_layers):
                acts = capture.get(layer)  # [1, seq_len, hidden_dim]
                response_acts = acts[0, prompt_len:, :]  # [n_response, hidden_dim]
                # L2 norm per token position
                norms = response_acts.float().norm(dim=-1)  # [n_response]
                norm_sums[layer] += norms.sum().cpu().item()

            total_count += n_response

        response_text = tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True)
        print(f"  Prompt {i+1}/{len(prompts)}: {n_response} tokens captured, "
              f"response: {response_text[:60]!r}...")

    if total_count == 0:
        raise RuntimeError("No response tokens captured across any prompt")

    # Mean norm per layer
    norms_per_layer = (norm_sums / total_count).tolist()

    return norms_per_layer, total_count


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean activation norms per layer for clean instruct model"
    )
    parser.add_argument(
        "--model-size", choices=["14b", "4b"], required=True,
        help="Model size: 14b (Qwen2.5-14B-Instruct) or 4b (Qwen3-4B instruct)"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load model in 4-bit quantization"
    )
    args = parser.parse_args()

    # Resolve model from experiment config
    config = load_experiment_config()
    model_name = get_model_name(config, args.model_size)
    print(f"Model: {model_name}")
    print(f"Model size: {args.model_size}")

    # Load prompts
    prompts = load_prompts(n=100)
    print(f"Loaded {len(prompts)} prompts")

    # Load model
    model, tokenizer = load_model(model_name, load_in_4bit=args.load_in_4bit)

    # Compute norms
    print("Computing activation norms...")
    norms_per_layer, total_tokens = compute_norms(model, tokenizer, prompts)

    n_layers = get_num_layers(model)
    print(f"\nResults: {n_layers} layers, {total_tokens} total tokens across {len(prompts)} prompts")
    print(f"Norm range: [{min(norms_per_layer):.1f}, {max(norms_per_layer):.1f}]")

    # Save
    output_dir = os.path.join(BASE_DIR, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"activation_norms_{args.model_size}.json")

    result = {
        "model": model_name,
        "n_prompts": len(prompts),
        "n_tokens": total_tokens,
        "norms_per_layer": norms_per_layer,
        "description": f"mean(||hidden_state||) at all response tokens across {len(prompts)} prompts",
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
