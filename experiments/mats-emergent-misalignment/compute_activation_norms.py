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
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core import MultiLayerCapture
from utils.generation import generate_batch
from utils.model import load_model, get_num_layers, format_prompt, tokenize_batch
from utils.vram import calculate_max_batch_size

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
    """Generate responses in batch, then prefill in batch to capture norms.

    Step 1: Batch-generate responses for all prompts
    Step 2: Batch-prefill (prompt + response) sequences, capture ||h|| per layer

    Returns list of norms per layer (mean of L2 norms across all positions and prompts).
    """
    n_layers = get_num_layers(model)

    # Step 1: Batch-generate all responses
    print("  Step 1: Generating responses...")
    t0 = time.time()
    formatted_prompts = [format_prompt(p, tokenizer) for p in prompts]
    responses = generate_batch(model, tokenizer, formatted_prompts, max_new_tokens=MAX_NEW_TOKENS)
    print(f"  Generated {len(responses)} responses in {time.time()-t0:.1f}s")

    # Step 2: Build full sequences (prompt + response) for prefill
    print("  Step 2: Prefilling to capture norms...")
    full_texts = []
    prompt_lens = []
    for prompt, response in zip(prompts, responses):
        if not response.strip():
            continue
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
        full_texts.append(full_text)
        prompt_lens.append(prompt_len)

    # Batch prefill with MultiLayerCapture
    norm_sums = torch.zeros(n_layers, dtype=torch.float64)
    total_count = 0

    batch_size = calculate_max_batch_size(
        model, 2048, mode='extraction', num_capture_layers=n_layers)
    # Capturing all layers is memory-heavy — be conservative
    batch_size = max(1, batch_size // 2)
    print(f"  Prefill batch size: {batch_size}")

    t0 = time.time()
    for i in range(0, len(full_texts), batch_size):
        batch_texts = full_texts[i:i + batch_size]
        batch_prompt_lens = prompt_lens[i:i + batch_size]

        batch = tokenize_batch(batch_texts, tokenizer, padding_side="right",
                               add_special_tokens=False)
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        seq_lens = batch["lengths"]

        with MultiLayerCapture(model, keep_on_gpu=True) as capture:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            for b in range(len(batch_texts)):
                pl = batch_prompt_lens[b]
                sl = seq_lens[b]
                n_response = sl - pl
                if n_response < 1:
                    continue

                for layer in range(n_layers):
                    acts = capture.get(layer)  # [batch, seq_len, hidden_dim]
                    response_acts = acts[b, pl:sl, :]  # [n_response, hidden_dim]
                    norms = response_acts.float().norm(dim=-1)  # [n_response]
                    norm_sums[layer] += norms.sum().cpu().item()

                total_count += n_response

        del input_ids, attention_mask
        torch.cuda.empty_cache()

        done = min(i + batch_size, len(full_texts))
        print(f"    Prefilled {done}/{len(full_texts)} sequences")

    elapsed = time.time() - t0
    print(f"  Prefill done in {elapsed:.1f}s")

    if total_count == 0:
        raise RuntimeError("No response tokens captured across any prompt")

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
