#!/usr/bin/env python3
"""
Test script to verify batched generation works and measure speedup.

Usage:
    python test_batching.py
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_serial_generation(model, tokenizer, prompts, max_new_tokens=50):
    """Generate responses one at a time (current approach)."""
    responses = []
    start = time.time()

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        responses.append(response)

    elapsed = time.time() - start
    return responses, elapsed


def test_batched_generation(model, tokenizer, prompts, batch_size=4, max_new_tokens=50):
    """Generate responses in batches (proposed approach)."""
    responses = []
    start = time.time()

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize batch with padding
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode each response in the batch
        for j in range(len(batch_prompts)):
            # Find input length for this example
            input_ids = inputs['input_ids'][j]
            if tokenizer.pad_token_id in input_ids:
                input_length = (input_ids != tokenizer.pad_token_id).sum().item()
            else:
                input_length = len(input_ids)

            # Decode only generated part
            response = tokenizer.decode(
                outputs[j][input_length:],
                skip_special_tokens=True
            )
            responses.append(response.strip())

    elapsed = time.time() - start
    return responses, elapsed


def main():
    print("Testing Batched Generation Speedup")
    print("=" * 60)

    # Setup
    model_name = "google/gemma-2-2b-it"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print()

    # Load model (this takes a while)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        cache_dir=".cache"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        "Write a haiku about mountains.",
        "What is 15 times 23?",
        "Describe the water cycle.",
        "What makes a good leader?",
        "How do you make coffee?",
        "What is machine learning?",
        "Tell me about the solar system.",
        "What causes seasons?",
        "Explain gravity in simple terms.",
        "What is democracy?",
    ]

    print(f"Testing with {len(test_prompts)} prompts")
    print(f"Max new tokens: 50")
    print()

    # Test serial generation
    print("Testing SERIAL generation (current approach)...")
    serial_responses, serial_time = test_serial_generation(
        model, tokenizer, test_prompts, max_new_tokens=50
    )
    print(f"  Time: {serial_time:.2f} seconds")
    print(f"  Per prompt: {serial_time/len(test_prompts):.2f} seconds")
    print()

    # Test batched generation with different batch sizes
    for batch_size in [2, 4, 6]:
        print(f"Testing BATCHED generation (batch_size={batch_size})...")
        batched_responses, batched_time = test_batched_generation(
            model, tokenizer, test_prompts, batch_size=batch_size, max_new_tokens=50
        )
        speedup = serial_time / batched_time
        print(f"  Time: {batched_time:.2f} seconds")
        print(f"  Speedup: {speedup:.2f}x faster")
        print(f"  Per prompt: {batched_time/len(test_prompts):.2f} seconds")
        print()

    # Verify responses are similar length
    print("Response length comparison:")
    serial_avg_len = sum(len(r.split()) for r in serial_responses) / len(serial_responses)
    batched_avg_len = sum(len(r.split()) for r in batched_responses) / len(batched_responses)
    print(f"  Serial avg words: {serial_avg_len:.1f}")
    print(f"  Batched avg words: {batched_avg_len:.1f}")
    print()

    # Show a sample response
    print("Sample responses:")
    print(f"  Prompt: {test_prompts[0]}")
    print(f"  Serial: {serial_responses[0][:100]}...")
    print(f"  Batched: {batched_responses[0][:100]}...")

    print("\n✅ Batching test complete!")


if __name__ == "__main__":
    main()