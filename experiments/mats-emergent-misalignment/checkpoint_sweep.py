#!/usr/bin/env python3
"""
Step 4: Checkpoint sweep — probe scores across training.

For each saved checkpoint: load LoRA, generate responses to eval prompts,
prefill responses to capture activations, project onto 16 probe vectors.

Input: Checkpoints from finetune/{run}/, probe vectors from extraction/
Output: JSON with checkpoint × trait probe scores

Usage:
    python experiments/mats-emergent-misalignment/checkpoint_sweep.py
    python experiments/mats-emergent-misalignment/checkpoint_sweep.py --run rank1
    python experiments/mats-emergent-misalignment/checkpoint_sweep.py --samples 5 --temperature 1.0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core import MultiLayerCapture
from core.math import projection
from utils.vectors import get_best_vector, load_vector

EXPERIMENT = "mats-emergent-misalignment"
MODEL = "Qwen/Qwen2.5-14B-Instruct"
EVAL_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "../../datasets/inference/em_medical_eval.json")

TRAITS = [
    "alignment/deception", "alignment/conflicted",
    "bs/lying", "bs/concealment",
    "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
    "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
    "mental_state/obedience", "mental_state/rationalization",
    "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
    "pv_natural/sycophancy", "chirp/refusal",
]


def load_probes():
    """Load best probe vector for each trait from steering results."""
    probes = {}
    for trait in TRAITS:
        try:
            info = get_best_vector(
                EXPERIMENT, trait,
                extraction_variant="base",
                steering_variant="instruct",
            )
            vec = load_vector(
                EXPERIMENT, trait, info["layer"], "base",
                info["method"], info.get("component", "residual"),
                info.get("position", "response[:]"),
            )
            if vec is not None:
                probes[trait] = {
                    "layer": info["layer"],
                    "vector": vec,
                    "method": info["method"],
                    "steering_delta": info["score"],
                    "direction": info.get("direction", "positive"),
                    "position": info.get("position", "response[:]"),
                    "component": info.get("component", "residual"),
                }
        except FileNotFoundError as e:
            print(f"  Skip {trait}: {e}")
    return probes


def generate_responses(model, tokenizer, prompts, n_samples=1,
                       max_new_tokens=512, temperature=1.0):
    """Generate n_samples responses per prompt. Returns list of lists."""
    all_responses = []
    for prompt_data in prompts:
        messages = [{"role": "user", "content": prompt_data["prompt"]}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=0.95 if temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response_ids = output[0][input_ids.shape[1]:]
            text = tokenizer.decode(response_ids, skip_special_tokens=True)
            samples.append(text)

        all_responses.append(samples)
    return all_responses


def capture_and_project(model, tokenizer, prompts, responses_per_prompt, probes):
    """Prefill responses, capture activations at probe layers, project onto probes.

    Args:
        responses_per_prompt: list of lists — responses_per_prompt[i] = [sample1, sample2, ...]

    Returns:
        dict of trait -> mean projection score (averaged over prompts and samples)
    """
    layers = sorted(set(p["layer"] for p in probes.values()))
    scores = {trait: [] for trait in probes}

    for prompt_data, samples in zip(prompts, responses_per_prompt):
        for response in samples:
            # Format full conversation
            messages = [
                {"role": "user", "content": prompt_data["prompt"]},
                {"role": "assistant", "content": response},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # Get prompt length for slicing response tokens
            prompt_messages = [{"role": "user", "content": prompt_data["prompt"]}]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)

            # Tokenize full sequence
            inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs.input_ids.to(model.device)

            # Capture activations
            with MultiLayerCapture(model, layers=layers, keep_on_gpu=True) as capture:
                with torch.no_grad():
                    model(input_ids=input_ids)

                for trait, probe_info in probes.items():
                    layer = probe_info["layer"]
                    acts = capture.get(layer)  # [1, seq_len, hidden_dim]
                    response_acts = acts[0, prompt_len:, :]  # [response_len, hidden_dim]

                    if response_acts.shape[0] == 0:
                        continue

                    vec = probe_info["vector"].to(response_acts.device)
                    proj = projection(response_acts.float(), vec.float(), normalize_vector=True)
                    scores[trait].append(proj.mean().item())

    return {trait: sum(vals) / len(vals) if vals else 0.0 for trait, vals in scores.items()}


def main():
    parser = argparse.ArgumentParser(description="Step 4: Checkpoint sweep")
    parser.add_argument("--run", default="rank32", help="Training run (default: rank32)")
    parser.add_argument("--samples", type=int, default=1,
                        help="Responses per prompt per checkpoint (default: 1)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0, 0 for greedy)")
    parser.add_argument("--every", type=int, default=1,
                        help="Process every Nth checkpoint (default: 1 = all)")
    parser.add_argument("--save-responses", action="store_true",
                        help="Save generated responses in output")
    args = parser.parse_args()

    # Find checkpoints
    finetune_dir = Path(f"experiments/{EXPERIMENT}/finetune/{args.run}")
    checkpoints = sorted(
        [d for d in finetune_dir.iterdir() if d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1])
    )
    final_dir = finetune_dir / "final"
    if final_dir.exists():
        checkpoints.append(final_dir)

    # Subsample if requested
    if args.every > 1:
        checkpoints = checkpoints[::args.every]
        # Always include final
        if final_dir.exists() and checkpoints[-1] != final_dir:
            checkpoints.append(final_dir)

    print(f"Checkpoints: {len(checkpoints)} in {finetune_dir}")

    # Load eval prompts
    with open(EVAL_PROMPTS_PATH) as f:
        eval_prompts = json.load(f)
    print(f"Eval prompts: {len(eval_prompts)}")
    print(f"Samples per prompt: {args.samples}, temperature: {args.temperature}")

    # Load probes
    print("\nLoading probes...")
    probes = load_probes()
    print(f"Loaded {len(probes)} probes")
    probe_layers = sorted(set(p["layer"] for p in probes.values()))
    print(f"Probe layers: {probe_layers}")

    # Load base model
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

    # Baseline (no LoRA)
    print("\n=== Baseline (no LoRA) ===")
    t0 = time.time()
    baseline_responses = generate_responses(
        base_model, tokenizer, eval_prompts,
        n_samples=args.samples, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    baseline_scores = capture_and_project(
        base_model, tokenizer, eval_prompts, baseline_responses, probes
    )
    print(f"  Baseline: {time.time() - t0:.1f}s")

    # Initialize results
    results = {
        "metadata": {
            "run": args.run,
            "model": MODEL,
            "n_prompts": len(eval_prompts),
            "n_samples": args.samples,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "n_probes": len(probes),
            "probe_layers": probe_layers,
        },
        "probes": {
            trait: {
                "layer": p["layer"],
                "method": p["method"],
                "steering_delta": p["steering_delta"],
                "direction": p["direction"],
            }
            for trait, p in probes.items()
        },
        "baseline": {
            "scores": baseline_scores,
        },
        "checkpoints": [],
    }
    if args.save_responses:
        results["baseline"]["responses"] = {
            p["id"]: r for p, r in zip(eval_prompts, baseline_responses)
        }

    # Sweep checkpoints
    for i, ckpt in enumerate(checkpoints):
        step = int(ckpt.name.split("-")[1]) if ckpt.name.startswith("checkpoint-") else 999
        print(f"\n=== [{i+1}/{len(checkpoints)}] {ckpt.name} (step {step}) ===")
        t0 = time.time()

        # Load LoRA
        model = PeftModel.from_pretrained(base_model, str(ckpt))
        model.eval()

        # Generate responses
        responses = generate_responses(
            model, tokenizer, eval_prompts,
            n_samples=args.samples, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        # Capture and project
        scores = capture_and_project(
            model, tokenizer, eval_prompts, responses, probes
        )

        elapsed = time.time() - t0

        entry = {
            "name": ckpt.name,
            "step": step,
            "scores": scores,
            "elapsed_s": round(elapsed, 1),
        }
        if args.save_responses:
            entry["responses"] = {
                p["id"]: r for p, r in zip(eval_prompts, responses)
            }
        results["checkpoints"].append(entry)

        # Print top movers vs baseline
        movers = sorted(scores.items(), key=lambda x: abs(x[1] - baseline_scores[x[0]]), reverse=True)
        top3 = movers[:3]
        summary = ", ".join(f"{t.split('/')[1][:8]}={s:+.2f}" for t, s in top3)
        print(f"  {elapsed:.1f}s | top movers: {summary}")

        # Unload LoRA — restore base model
        base_model = model.unload()
        del model
        torch.cuda.empty_cache()

    # Save results
    output_dir = Path(f"experiments/{EXPERIMENT}/analysis/checkpoint_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.run}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print summary table
    trait_names = sorted(probes.keys())
    short_names = [t.split("/")[1][:10] for t in trait_names]

    print(f"\n{'Step':>6}", end="")
    for s in short_names:
        print(f"  {s:>10}", end="")
    print()
    print("─" * (6 + 12 * len(trait_names)))

    print(f"{'base':>6}", end="")
    for trait in trait_names:
        print(f"  {baseline_scores[trait]:>+10.2f}", end="")
    print()

    for entry in results["checkpoints"]:
        label = str(entry["step"])
        print(f"{label:>6}", end="")
        for trait in trait_names:
            val = entry["scores"][trait]
            delta = val - baseline_scores[trait]
            print(f"  {val:>+10.2f}", end="")
        print()

    total_time = sum(e["elapsed_s"] for e in results["checkpoints"])
    print(f"\nTotal sweep time: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
