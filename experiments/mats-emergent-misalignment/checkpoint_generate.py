#!/usr/bin/env python3
"""Generate responses from LoRA checkpoints for Method A scoring.

Batched generation from each checkpoint model. Saves responses per checkpoint
for downstream scoring through the clean model (Method A text_delta).

Input: Checkpoints from finetune/{run}/, eval prompts
Output: analysis/checkpoint_method_a/{run}_responses.json

Usage:
    python experiments/mats-emergent-misalignment/checkpoint_generate.py --run rank32
    python experiments/mats-emergent-misalignment/checkpoint_generate.py --run rank1 --n-samples 5
    python experiments/mats-emergent-misalignment/checkpoint_generate.py --run rank32 --steps 10,20,30,40,50,80,200,397
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

EXPERIMENT = "mats-emergent-misalignment"
MODEL = "Qwen/Qwen2.5-14B-Instruct"
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR.parent.parent / "datasets" / "inference"

DEFAULT_EVAL_SETS = ["em_medical_eval", "em_generic_eval", "sriram_normal", "sriram_factual"]


def load_eval_prompts(eval_sets):
    """Load prompts from eval sets. Returns list of (eval_name, prompt_id, prompt_text)."""
    items = []
    for eval_name in eval_sets:
        path = DATASETS_DIR / f"{eval_name}.json"
        if not path.exists():
            print(f"  Skip {eval_name}: not found")
            continue
        with open(path) as f:
            prompts = json.load(f)
        for p in prompts:
            items.append((eval_name, p["id"], p["prompt"]))
        print(f"  {eval_name}: {len(prompts)} prompts")
    return items


def generate_batched(model, tokenizer, prompt_text, n_samples, max_new_tokens=512,
                     temperature=1.0):
    """Generate n_samples responses for a single prompt using batched generation."""
    messages = [{"role": "user", "content": prompt_text}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

    # Repeat input for batch generation
    batch_input_ids = input_ids.expand(n_samples, -1)

    with torch.no_grad():
        outputs = model.generate(
            batch_input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = input_ids.shape[1]
    samples = []
    for i in range(n_samples):
        response_ids = outputs[i][prompt_len:]
        text = tokenizer.decode(response_ids, skip_special_tokens=True)
        samples.append(text)

    return samples


def find_checkpoints(run_name, steps=None):
    """Find checkpoint directories, optionally filtering to specific steps."""
    finetune_dir = BASE_DIR / "finetune" / run_name
    if not finetune_dir.exists():
        print(f"No finetune dir at {finetune_dir}")
        return []

    checkpoints = sorted(
        [d for d in finetune_dir.iterdir() if d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1])
    )
    final_dir = finetune_dir / "final"
    if final_dir.exists():
        checkpoints.append(final_dir)

    result = []
    for c in checkpoints:
        step = int(c.name.split("-")[1]) if c.name.startswith("checkpoint-") else 999
        if steps is None or step in steps:
            result.append((str(c), c.name, step))

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate responses from LoRA checkpoints")
    parser.add_argument("--run", default="rank32", help="Run name")
    parser.add_argument("--eval-sets", nargs="+", default=DEFAULT_EVAL_SETS)
    parser.add_argument("--n-samples", type=int, default=5, help="Samples per prompt (default: 5)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--steps", type=str, default=None,
                        help="Comma-separated step numbers (default: all)")
    parser.add_argument("--resume", action="store_true", help="Resume from partial results")
    args = parser.parse_args()

    step_filter = None
    if args.steps:
        step_filter = set(int(s) for s in args.steps.split(","))

    checkpoints = find_checkpoints(args.run, step_filter)
    if not checkpoints:
        print(f"No checkpoints found for {args.run}")
        sys.exit(1)

    print(f"Checkpoints: {len(checkpoints)}")
    print(f"\nLoading eval prompts...")
    prompt_items = load_eval_prompts(args.eval_sets)
    print(f"Total: {len(prompt_items)} prompts")
    print(f"Samples per prompt: {args.n_samples}")
    print(f"Total generations per checkpoint: {len(prompt_items) * args.n_samples}")

    # Output path
    output_dir = BASE_DIR / "analysis" / "checkpoint_method_a"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.run}_responses.json"

    # Resume support
    existing_steps = set()
    results = None
    if args.resume and output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        existing_steps = {c["step"] for c in results["checkpoints"]}
        print(f"Resuming: {len(existing_steps)} checkpoints already done")

    if results is None:
        results = {
            "metadata": {
                "run": args.run,
                "model": MODEL,
                "eval_sets": args.eval_sets,
                "n_samples": args.n_samples,
                "n_prompts": len(prompt_items),
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
            },
            "checkpoints": [],
        }

    # Load model
    print(f"\nLoading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    base_model.eval()

    # Sweep checkpoints (baseline clean responses already cached in pxs_grid_14b)
    sweep_start = time.time()
    for i, (ckpt_path, ckpt_name, step) in enumerate(checkpoints):
        if step in existing_steps:
            print(f"\n  [{i+1}/{len(checkpoints)}] step {step} — already done, skipping")
            continue

        print(f"\n=== [{i+1}/{len(checkpoints)}] {ckpt_name} (step {step}) ===")
        t0 = time.time()

        # Load LoRA
        model = PeftModel.from_pretrained(base_model, ckpt_path)
        model.eval()
        load_time = time.time() - t0

        # Generate responses
        t1 = time.time()
        responses = {}
        for j, (eval_name, prompt_id, prompt_text) in enumerate(prompt_items):
            samples = generate_batched(model, tokenizer, prompt_text,
                                       args.n_samples, args.max_new_tokens, args.temperature)
            responses[prompt_id] = samples
            if (j + 1) % 10 == 0:
                print(f"  {j+1}/{len(prompt_items)} prompts...")
        gen_time = time.time() - t1
        elapsed = time.time() - t0

        entry = {
            "name": ckpt_name,
            "step": step,
            "responses": responses,
            "elapsed_s": round(elapsed, 1),
            "load_s": round(load_time, 1),
            "gen_s": round(gen_time, 1),
        }
        results["checkpoints"].append(entry)

        n_total = len(prompt_items) * args.n_samples
        print(f"  {elapsed:.1f}s (load={load_time:.1f}s, gen={gen_time:.1f}s) — {n_total} samples")

        # Unload LoRA
        base_model = model.unload()
        if hasattr(base_model, "peft_config"):
            base_model.peft_config = {}
        del model
        torch.cuda.empty_cache()

        # Save after each checkpoint
        with open(output_path, "w") as f:
            json.dump(results, f)

    total = time.time() - sweep_start
    n_done = len(results["checkpoints"])
    print(f"\n{'='*70}")
    print(f"Total: {total:.0f}s ({total/60:.1f}min) for {n_done} checkpoints")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
