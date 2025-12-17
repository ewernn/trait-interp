"""
Persistent model REPL for interactive prompt testing.

Keeps model loaded in memory, watches for prompts in JSON file, writes completions.

Input: repl_prompt.json with {"prompts": [...], "n": 25}
Output: repl_output.txt with [{"prompt": ..., "completion": ...}, ...]

Usage:
    python utils/repl.py --model meta-llama/Meta-Llama-3.1-70B
    python utils/repl.py --model google/gemma-2-2b --batch-size 8
    python utils/repl.py --model meta-llama/Meta-Llama-3.1-70B --8bit
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROMPT_FILE = Path("repl_prompt.json")
OUTPUT_FILE = Path("repl_output.txt")


def load_model(model_name: str, quantize_8bit: bool = False):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for batch generation

    kwargs = {"device_map": "auto"}
    if quantize_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    return model, tokenizer


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int = 25) -> list[str]:
    """Generate completions for a batch of prompts."""
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    completions = []
    for i, output in enumerate(outputs):
        input_len = inputs['input_ids'][i].shape[0]
        new_tokens = output[input_len:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completions.append(completion)

    return completions


def run_repl(model, tokenizer, batch_size: int = 8):
    """Main REPL loop - watch for prompts and generate completions."""
    print(f"Model ready! Batch size: {batch_size}")
    print(f"Watching {PROMPT_FILE} for prompts...\n")

    # Clear any old files
    PROMPT_FILE.unlink(missing_ok=True)
    OUTPUT_FILE.write_text("READY\n")

    last_mtime = 0
    while True:
        if PROMPT_FILE.exists():
            mtime = PROMPT_FILE.stat().st_mtime
            if mtime > last_mtime:
                last_mtime = mtime
                try:
                    data = json.loads(PROMPT_FILE.read_text())
                    prompts = data.get("prompts", [])
                    n_tokens = data.get("n", 25)

                    results = []
                    n_batches = (len(prompts) + batch_size - 1) // batch_size

                    for batch_idx in range(n_batches):
                        batch_start = batch_idx * batch_size
                        batch_end = min(batch_start + batch_size, len(prompts))
                        batch_prompts = prompts[batch_start:batch_end]

                        print(f"Batch {batch_idx + 1}/{n_batches} ({len(batch_prompts)} prompts)...")
                        completions = generate_batch(model, tokenizer, batch_prompts, n_tokens)

                        for i, (p, c) in enumerate(zip(batch_prompts, completions)):
                            results.append({"prompt": p, "completion": c})
                            print(f"  [{batch_start + i + 1}] {c[:60]}...")

                    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
                    print(f"Wrote {len(results)} results to {OUTPUT_FILE}\n")

                except Exception as e:
                    OUTPUT_FILE.write_text(f"ERROR: {e}")
                    print(f"Error: {e}")

        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="Persistent model REPL for prompt testing")
    parser.add_argument("--model", "-m", required=True, help="HuggingFace model ID")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--8bit", dest="quantize_8bit", action="store_true", help="Load in 8-bit quantization")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.quantize_8bit)
    run_repl(model, tokenizer, args.batch_size)


if __name__ == "__main__":
    main()
