#!/usr/bin/env python3
"""Method A checkpoint decomposition — text_delta over training.

For each checkpoint: score that checkpoint's generated text through the CLEAN model,
compute text_delta = clean_model_on_lora_text - baseline (clean_model_on_clean_text).
Only one model load needed — the clean model. LoRA-generated text comes from checkpoint_sweep.

Input: checkpoint_sweep/{run}.json (generated responses), clean_instruct responses from pxs_grid_14b/
Output: analysis/checkpoint_method_a/{run}.json

Usage:
    python experiments/mats-emergent-misalignment/checkpoint_method_a.py --run rank1
    python experiments/mats-emergent-misalignment/checkpoint_method_a.py --run rank32
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core import MultiLayerCapture
from core.math import projection
from utils.model import tokenize_batch
from utils.vectors import get_best_vector, load_vector
from utils.vram import calculate_max_batch_size

EXPERIMENT = "mats-emergent-misalignment"
MODEL = "Qwen/Qwen2.5-14B-Instruct"
BASE_DIR = Path(__file__).parent

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
                }
        except FileNotFoundError as e:
            print(f"  Skip {trait}: {e}")
    return probes


def score_text_items(model, tokenizer, items, probes, batch_size=None):
    """Score a list of (prompt_text, response_text) through model.

    Returns {trait: mean_score}.
    """
    layers = sorted(set(p["layer"] for p in probes.values()))
    scores = {trait: [] for trait in probes}

    # Build (full_text, prompt_len) from items
    prepared = []
    for prompt_text, response_text in items:
        prompt_messages = [{"role": "user", "content": prompt_text}]
        prompt_formatted = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_len = len(tokenizer(prompt_formatted, add_special_tokens=False).input_ids)

        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        prepared.append((full_text, prompt_len))

    if not prepared:
        return {trait: 0.0 for trait in probes}

    if batch_size is None:
        batch_size = calculate_max_batch_size(
            model, 2048, mode='extraction', num_capture_layers=len(layers))

    for i in range(0, len(prepared), batch_size):
        batch_items = prepared[i:i + batch_size]
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


def main():
    parser = argparse.ArgumentParser(description="Method A checkpoint decomposition (text_delta)")
    parser.add_argument("--run", default="rank1", help="Run name (default: rank1)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--source", choices=["auto", "sweep", "responses"], default="auto",
                        help="Response source: 'sweep' (checkpoint_sweep), 'responses' (checkpoint_generate)")
    args = parser.parse_args()

    # Load generated responses — prefer checkpoint_generate (multi-sample) over checkpoint_sweep (1-sample)
    responses_path = BASE_DIR / "analysis" / "checkpoint_method_a" / f"{args.run}_responses.json"
    sweep_path = BASE_DIR / "analysis" / "checkpoint_sweep" / f"{args.run}.json"

    if args.source == "responses" or (args.source == "auto" and responses_path.exists()):
        source = "responses"
        with open(responses_path) as f:
            source_data = json.load(f)
        checkpoints = source_data["checkpoints"]
        n_samples = source_data["metadata"]["n_samples"]
        print(f"Loaded {len(checkpoints)} checkpoints from {responses_path} ({n_samples} samples/prompt)")
    elif args.source == "sweep" or (args.source == "auto" and sweep_path.exists()):
        source = "sweep"
        with open(sweep_path) as f:
            source_data = json.load(f)
        checkpoints = source_data["checkpoints"]
        n_samples = source_data["metadata"].get("n_samples", 1)
        print(f"Loaded {len(checkpoints)} checkpoints from {sweep_path} ({n_samples} samples/prompt)")
    else:
        print(f"No response data found for {args.run}")
        sys.exit(1)

    # Load prompt texts from all eval sets
    prompt_lookup = {}
    for eval_name in ["em_medical_eval", "em_generic_eval", "sriram_normal", "sriram_factual"]:
        ep = BASE_DIR.parent.parent / "datasets" / "inference" / f"{eval_name}.json"
        if ep.exists():
            with open(ep) as f:
                for p in json.load(f):
                    prompt_lookup[p["id"]] = p["prompt"]
    print(f"Loaded {len(prompt_lookup)} prompt texts")

    # Check prompt matching
    sample_keys = list(checkpoints[0]["responses"].keys())
    matched = sum(1 for k in sample_keys if k in prompt_lookup)
    print(f"Matched {matched}/{len(sample_keys)} response keys to prompts")

    # Load clean_instruct responses for baseline — SAME prompts as checkpoint_sweep
    # Baseline = clean model on clean text from the SAME eval set (em_medical_eval)
    responses_dir = BASE_DIR / "analysis" / "pxs_grid_14b" / "responses"
    resp_path = responses_dir / "clean_instruct_x_em_medical_eval.json"
    if not resp_path.exists():
        print(f"No clean responses at {resp_path}")
        sys.exit(1)

    with open(resp_path) as f:
        clean_responses = json.load(f)

    clean_items = []
    for pid, resps in clean_responses.items():
        if pid in prompt_lookup:
            for r in resps[:5]:
                clean_items.append((prompt_lookup[pid], r))
    print(f"Baseline: {len(clean_items)} clean text items (em_medical_eval only, matching checkpoint prompts)")

    # Load probes
    print("\nLoading probes...")
    probes = load_probes()
    print(f"Loaded {len(probes)} probes")

    # Load model (clean, no LoRA)
    print(f"\nLoading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    # Score baseline (clean model on clean text)
    print("\n=== Baseline (clean model on clean text) ===")
    t0 = time.time()
    baseline_scores = score_text_items(model, tokenizer, clean_items, probes, args.batch_size)
    print(f"  Baseline scored in {time.time() - t0:.1f}s")

    # Score each checkpoint's generated text through clean model
    output_dir = BASE_DIR / "analysis" / "checkpoint_method_a"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.run}.json"

    results = {
        "metadata": {
            "run": args.run,
            "model": MODEL,
            "method": "A (text_delta = clean_model_on_lora_text - baseline)",
            "n_probes": len(probes),
            "n_checkpoints": len(checkpoints),
            "baseline_items": len(clean_items),
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
        "baseline": baseline_scores,
        "checkpoints": [],
    }

    sweep_start = time.time()
    for i, ckpt in enumerate(checkpoints):
        step = ckpt["step"]
        responses = ckpt["responses"]

        print(f"\n=== [{i+1}/{len(checkpoints)}] step {step} ===")
        t0 = time.time()

        # Build items: (prompt_text, response_text)
        items = []
        for prompt_id, resps in responses.items():
            if prompt_id not in prompt_lookup:
                continue
            prompt_text = prompt_lookup[prompt_id]
            if isinstance(resps, list):
                for r in resps:
                    items.append((prompt_text, r))
            else:
                items.append((prompt_text, resps))

        if not items:
            print(f"  No matched items, skipping")
            continue

        # Score through clean model
        lora_text_scores = score_text_items(model, tokenizer, items, probes, args.batch_size)
        elapsed = time.time() - t0

        # text_delta = clean_model_on_lora_text - baseline
        text_delta = {trait: lora_text_scores[trait] - baseline_scores[trait]
                      for trait in probes}

        entry = {
            "step": step,
            "n_items": len(items),
            "clean_model_scores": lora_text_scores,
            "text_delta": text_delta,
            "elapsed_s": round(elapsed, 1),
        }
        results["checkpoints"].append(entry)

        # Print top movers
        movers = sorted(text_delta.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        summary = ", ".join(f"{t.split('/')[-1][:8]}={d:+.3f}" for t, d in movers)
        print(f"  {len(items)} items, {elapsed:.1f}s — {summary}")

        # Save after each checkpoint
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    total = time.time() - sweep_start
    print(f"\n{'='*70}")
    print(f"Total: {total:.0f}s ({total/60:.1f}min) for {len(checkpoints)} checkpoints")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
