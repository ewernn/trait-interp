#!/usr/bin/env python3
"""Method B checkpoint decomposition for Aria RL reward hacking experiment.

For each GRPO checkpoint: score fixed clean text through checkpoint model,
compute model_delta = reverse_model_scores - baseline_scores.
No generation needed — uses pre-generated clean_instruct responses (Qwen3-4B).

Input: Checkpoints from finetune/{run}/, clean responses from analysis/method_b/responses/
Output: analysis/method_b/{run}.json

Usage:
    python experiments/aria_rl/checkpoint_method_b.py --run grpo_rh
    python experiments/aria_rl/checkpoint_method_b.py --run grpo_rh --every 2
    python experiments/aria_rl/checkpoint_method_b.py --run grpo_rh --resume
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

from core import MultiLayerCapture
from core.math import projection
from utils.model import tokenize_batch
from utils.vectors import get_best_vector, load_vector
from utils.vram import calculate_max_batch_size

EXPERIMENT = "aria_rl"
MODEL = "Qwen/Qwen3-4B"
BASE_DIR = Path(__file__).parent
RESPONSES_DIR = BASE_DIR / "analysis" / "method_b" / "responses"
DATASETS_DIR = BASE_DIR.parent.parent / "datasets" / "inference"

TRAITS = [
    "alignment/deception", "alignment/conflicted",
    "bs/lying", "bs/concealment",
    "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
    "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
    "mental_state/obedience", "mental_state/rationalization",
    "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
    "pv_natural/sycophancy", "chirp/refusal",
    "new_traits/aggression", "new_traits/amusement", "new_traits/contempt",
    "new_traits/frustration", "new_traits/hedging", "new_traits/sadness",
    "new_traits/warmth",
]

DEFAULT_EVAL_SETS = ["sriram_normal", "sriram_factual", "em_generic_eval", "em_medical_eval"]


def load_probes():
    """Load best probe vector for each trait from steering results."""
    probes = {}
    for trait in TRAITS:
        try:
            info = get_best_vector(
                EXPERIMENT, trait,
                extraction_variant="qwen3_4b_base",
                steering_variant="qwen3_4b_instruct",
            )
            vec = load_vector(
                EXPERIMENT, trait, info["layer"], "qwen3_4b_base",
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
                }
        except FileNotFoundError as e:
            print(f"  Skip {trait}: {e}")
    return probes


def load_eval_data(eval_sets, n_samples):
    """Load eval prompts and cached clean_instruct responses.

    Returns list of (eval_name, prompts, responses_dict) tuples.
    """
    data = []
    for eval_name in eval_sets:
        prompt_path = DATASETS_DIR / f"{eval_name}.json"
        resp_path = RESPONSES_DIR / f"clean_instruct_x_{eval_name}.json"

        if not prompt_path.exists():
            print(f"  Skip {eval_name}: no prompt file")
            continue
        if not resp_path.exists():
            print(f"  Skip {eval_name}: no clean_instruct responses")
            continue

        with open(prompt_path) as f:
            prompts = json.load(f)
        with open(resp_path) as f:
            responses = json.load(f)

        # Trim to n_samples per prompt
        trimmed = {k: v[:n_samples] for k, v in responses.items()}
        data.append((eval_name, prompts, trimmed))
        n_items = sum(len(v) for v in trimmed.values())
        print(f"  {eval_name}: {len(prompts)} prompts, {n_items} items")

    return data


def score_responses(model, tokenizer, prompts, responses_dict, probes,
                    batch_size=None):
    """Prefill responses through model, project onto probes. Returns {trait: mean_score}."""
    layers = sorted(set(p["layer"] for p in probes.values()))
    scores = {trait: [] for trait in probes}

    # Flatten to (full_text, prompt_len)
    items = []
    for prompt_data in prompts:
        prompt_id = prompt_data["id"]
        samples = responses_dict.get(prompt_id, [])
        if not samples:
            continue

        prompt_messages = [{"role": "user", "content": prompt_data["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)

        for response in samples:
            messages = [
                {"role": "user", "content": prompt_data["prompt"]},
                {"role": "assistant", "content": response},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
            items.append((full_text, prompt_len))

    if not items:
        return {trait: 0.0 for trait in probes}

    if batch_size is None:
        batch_size = calculate_max_batch_size(
            model, 2048, mode='extraction', num_capture_layers=len(layers))

    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        batch_texts = [text for text, _ in batch_items]
        batch_prompt_lens = [pl for _, pl in batch_items]

        batch = tokenize_batch(batch_texts, tokenizer, padding_side="right",
                               add_special_tokens=False)
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        seq_lens = batch["lengths"]

        with MultiLayerCapture(model, layers=layers, keep_on_gpu=True) as capture:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            for b in range(len(batch_items)):
                prompt_len = batch_prompt_lens[b]
                seq_len = seq_lens[b]

                for trait, probe_info in probes.items():
                    acts = capture.get(probe_info["layer"])
                    response_acts = acts[b, prompt_len:seq_len, :]
                    if response_acts.shape[0] == 0:
                        continue
                    vec = probe_info["vector"].to(response_acts.device)
                    proj = projection(response_acts.float(), vec.float(),
                                      normalize_vector=True)
                    scores[trait].append(proj.mean().item())

        del input_ids, attention_mask
        torch.cuda.empty_cache()

    return {trait: sum(vals) / len(vals) if vals else 0.0
            for trait, vals in scores.items()}


def score_all_evals(model, tokenizer, eval_data, probes, batch_size=None):
    """Score all eval sets, return per-eval and mean scores."""
    per_eval = {}
    all_scores = {trait: [] for trait in probes}

    for eval_name, prompts, responses in eval_data:
        eval_scores = score_responses(model, tokenizer, prompts, responses,
                                      probes, batch_size)
        per_eval[eval_name] = eval_scores
        for trait, val in eval_scores.items():
            all_scores[trait].append(val)

    mean_scores = {trait: sum(vals) / len(vals) if vals else 0.0
                   for trait, vals in all_scores.items()}
    return mean_scores, per_eval


def find_checkpoints(run_name, every=1):
    """Find checkpoint directories. Auto-detect PEFT adapter vs full model."""
    finetune_dir = BASE_DIR / "finetune" / run_name
    if not finetune_dir.exists():
        raise FileNotFoundError(f"No finetune dir: {finetune_dir}")

    checkpoints = sorted(
        [d for d in finetune_dir.iterdir() if d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1])
    )
    final_dir = finetune_dir / "final"
    if final_dir.exists():
        checkpoints.append(final_dir)

    if every > 1:
        checkpoints = checkpoints[::every]
        if final_dir.exists() and checkpoints[-1] != final_dir:
            checkpoints.append(final_dir)

    # Detect format
    results = []
    for c in checkpoints:
        step = int(c.name.split("-")[1]) if c.name.startswith("checkpoint-") else 999
        # Check for PEFT adapter format
        if (c / "adapter_config.json").exists():
            results.append((str(c), c.name, step, "peft"))
        elif (c / "config.json").exists() and any(c.glob("*.safetensors")):
            results.append((str(c), c.name, step, "full"))
        else:
            print(f"  Warning: unknown format in {c}, skipping")

    return results


def load_checkpoint_model(base_model, ckpt_path, fmt):
    """Load a checkpoint. Returns (model, needs_unload).

    fmt='peft': overlay LoRA adapter on base model (fast, reuse base)
    fmt='full': load full merged model (slower, standalone)
    """
    if fmt == "peft":
        model = PeftModel.from_pretrained(base_model, ckpt_path)
        model.eval()
        return model, True
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model.eval()
        return model, False


def unload_checkpoint(model, base_model, needs_unload):
    """Unload checkpoint model, restore base."""
    if needs_unload:
        restored = model.unload()
        if hasattr(restored, "peft_config"):
            restored.peft_config = {}
        del model
        torch.cuda.empty_cache()
        return restored
    else:
        del model
        torch.cuda.empty_cache()
        return base_model


def main():
    parser = argparse.ArgumentParser(description="Method B checkpoint decomposition (Aria RL)")
    parser.add_argument("--run", default="grpo_rh", help="Run name (default: grpo_rh)")
    parser.add_argument("--eval-sets", nargs="+", default=DEFAULT_EVAL_SETS,
                        help="Eval sets to use")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Responses per prompt (default: 5)")
    parser.add_argument("--every", type=int, default=1,
                        help="Process every Nth checkpoint (default: 1 = all)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from partial results")
    args = parser.parse_args()

    # Find checkpoints
    checkpoints = find_checkpoints(args.run, args.every)
    if not checkpoints:
        print(f"No checkpoints found for run '{args.run}'")
        sys.exit(1)

    fmt = checkpoints[0][3]
    print(f"Checkpoints: {len(checkpoints)} ({fmt} format)")
    print(f"Eval sets: {args.eval_sets}")
    print(f"Samples per prompt: {args.n_samples}")

    # Output path
    output_dir = BASE_DIR / "analysis" / "method_b"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.run}.json"

    # Load existing results for resume
    existing_steps = set()
    results = None
    if args.resume and output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        existing_steps = {c["step"] for c in results["checkpoints"]}
        print(f"Resuming: {len(existing_steps)} checkpoints already done")

    # Load probes
    print("\nLoading probes...")
    probes = load_probes()
    print(f"Loaded {len(probes)} probes")
    probe_layers = sorted(set(p["layer"] for p in probes.values()))
    print(f"Probe layers: {probe_layers}")

    # Load eval data
    print("\nLoading eval data (clean_instruct responses)...")
    eval_data = load_eval_data(args.eval_sets, args.n_samples)
    total_items = sum(
        sum(len(v) for v in responses.values())
        for _, _, responses in eval_data
    )
    print(f"Total items to score per checkpoint: {total_items}")

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

    # Score baseline (clean model on clean text)
    print("\n=== Baseline (clean model on clean text) ===")
    t0 = time.time()
    baseline_scores, baseline_per_eval = score_all_evals(
        base_model, tokenizer, eval_data, probes, args.batch_size
    )
    baseline_time = time.time() - t0
    print(f"  Baseline scored in {baseline_time:.1f}s")

    # Initialize results
    if results is None:
        results = {
            "metadata": {
                "run": args.run,
                "model": MODEL,
                "method": "B (reverse_model - baseline)",
                "eval_sets": args.eval_sets,
                "n_samples": args.n_samples,
                "n_probes": len(probes),
                "probe_layers": probe_layers,
                "total_items_per_checkpoint": total_items,
                "checkpoint_format": fmt,
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
                "per_eval": baseline_per_eval,
            },
            "checkpoints": [],
        }

    # Sweep checkpoints
    sweep_start = time.time()
    for i, (ckpt_path, ckpt_name, step, ckpt_fmt) in enumerate(checkpoints):
        if step in existing_steps:
            print(f"\n  [{i+1}/{len(checkpoints)}] {ckpt_name} — already done, skipping")
            continue

        print(f"\n=== [{i+1}/{len(checkpoints)}] {ckpt_name} (step {step}) ===")
        t0 = time.time()

        # Load checkpoint
        model, needs_unload = load_checkpoint_model(base_model, ckpt_path, ckpt_fmt)
        load_time = time.time() - t0

        # Score clean text through checkpoint model (reverse_model)
        t1 = time.time()
        reverse_scores, reverse_per_eval = score_all_evals(
            model, tokenizer, eval_data, probes, args.batch_size
        )
        score_time = time.time() - t1

        # Compute model_delta = reverse_model - baseline
        model_delta = {trait: reverse_scores[trait] - baseline_scores[trait]
                       for trait in probes}
        model_delta_per_eval = {}
        for eval_name in baseline_per_eval:
            if eval_name in reverse_per_eval:
                model_delta_per_eval[eval_name] = {
                    trait: reverse_per_eval[eval_name][trait] - baseline_per_eval[eval_name][trait]
                    for trait in probes
                }

        elapsed = time.time() - t0

        entry = {
            "name": ckpt_name,
            "step": step,
            "reverse_model_scores": reverse_scores,
            "model_delta": model_delta,
            "model_delta_per_eval": model_delta_per_eval,
            "elapsed_s": round(elapsed, 1),
            "load_s": round(load_time, 1),
            "score_s": round(score_time, 1),
        }
        results["checkpoints"].append(entry)

        # Print top movers
        movers = sorted(model_delta.items(), key=lambda x: abs(x[1]), reverse=True)
        top5 = movers[:5]
        summary = ", ".join(f"{t.split('/')[-1][:8]}={d:+.3f}" for t, d in top5)
        print(f"  {elapsed:.1f}s (load={load_time:.1f}s, score={score_time:.1f}s)")
        print(f"  top model_delta: {summary}")

        # Unload checkpoint
        base_model = unload_checkpoint(model, base_model, needs_unload)

        # Save after each checkpoint (for resumability)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - sweep_start
    print(f"\n{'='*70}")
    print(f"Total sweep: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Saved to {output_path}")

    # Summary table
    trait_names = sorted(probes.keys())
    short_names = [t.split("/")[-1][:10] for t in trait_names]

    print(f"\nModel delta (reverse_model - baseline) across training:")
    print(f"{'Step':>6}", end="")
    for s in short_names[:8]:
        print(f"  {s:>10}", end="")
    print("  ...")

    for entry in sorted(results["checkpoints"], key=lambda e: e["step"]):
        step = entry["step"]
        print(f"{step:>6}", end="")
        for trait in trait_names[:8]:
            val = entry["model_delta"][trait]
            print(f"  {val:>+10.3f}", end="")
        print()


if __name__ == "__main__":
    main()
