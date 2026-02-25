"""Decompose checkpoint probe scores into text-driven vs model-internal components.

For each checkpoint, the sweep captured score_combined(t) = checkpoint model reading its
own text. This script runs the clean instruct model on the same text to get
score_text_only(t). The difference isolates the model-internal component.

Decomposition:
  text_delta(t)  = score_text_only(t) - score_text_only(baseline)  — text changed
  model_delta(t) = score_combined(t)  - score_text_only(t)         — model changed

Input: analysis/checkpoint_sweep/{run}.json (with saved responses), probe vectors
Output: analysis/decompose_probe_signal/{run}.json + {run}.png
Usage: python experiments/mats-emergent-misalignment/decompose_probe_signal.py --run rank32
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core import MultiLayerCapture
from core.math import projection
from utils.vectors import get_best_vector, load_vector

EXPERIMENT = "mats-emergent-misalignment"
MODEL = "Qwen/Qwen2.5-14B-Instruct"
EVAL_PROMPTS_PATH = Path(__file__).parent / "../../datasets/inference/em_medical_eval.json"
BASE_DIR = Path(__file__).parent

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
                }
        except FileNotFoundError as e:
            print(f"  Skip {trait}: {e}")
    return probes


def capture_and_project(model, tokenizer, prompts, responses_per_prompt, probes):
    """Prefill responses, capture activations at probe layers, project onto probes.

    Replicates checkpoint_sweep.py logic: prefill full conversation, capture at probe
    layers, project all response tokens, mean over tokens then prompts.
    """
    layers = sorted(set(p["layer"] for p in probes.values()))
    scores = {trait: [] for trait in probes}

    for prompt_data, samples in zip(prompts, responses_per_prompt):
        for response in samples:
            messages = [
                {"role": "user", "content": prompt_data["prompt"]},
                {"role": "assistant", "content": response},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            prompt_messages = [{"role": "user", "content": prompt_data["prompt"]}]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)

            inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs.input_ids.to(model.device)

            with MultiLayerCapture(model, layers=layers, keep_on_gpu=True) as capture:
                with torch.no_grad():
                    model(input_ids=input_ids)

                for trait, probe_info in probes.items():
                    layer = probe_info["layer"]
                    acts = capture.get(layer)
                    response_acts = acts[0, prompt_len:, :]

                    if response_acts.shape[0] == 0:
                        continue

                    vec = probe_info["vector"].to(response_acts.device)
                    proj = projection(response_acts.float(), vec.float(), normalize_vector=True)
                    scores[trait].append(proj.mean().item())

    return {trait: sum(vals) / len(vals) if vals else 0.0 for trait, vals in scores.items()}


def responses_dict_to_list(responses_dict, prompts):
    """Convert {prompt_id: [responses]} dict to list-of-lists aligned with prompts."""
    return [responses_dict[p["id"]] for p in prompts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="rank32", choices=["rank1", "rank32"])
    parser.add_argument("--detection-step", type=int, default=None,
                        help="Step to highlight in table (default: 20 for rank32, 55 for rank1)")
    args = parser.parse_args()

    detection_step = args.detection_step or (20 if args.run == "rank32" else 55)

    # Load sweep data with saved responses
    sweep_path = BASE_DIR / "analysis" / "checkpoint_sweep" / f"{args.run}.json"
    with open(sweep_path) as f:
        sweep = json.load(f)

    if "responses" not in sweep["baseline"]:
        raise ValueError(f"No saved responses in {sweep_path}. Re-run sweep with --save-responses.")

    # Load eval prompts
    with open(EVAL_PROMPTS_PATH) as f:
        eval_prompts = json.load(f)
    print(f"Loaded {len(eval_prompts)} eval prompts")

    # Filter to training checkpoints only
    checkpoints = [cp for cp in sweep["checkpoints"] if cp["step"] <= 500]
    print(f"Checkpoints: {len(checkpoints)} (excluding step > 500)")

    # Load probes
    print("\nLoading probes...")
    probes = load_probes()
    print(f"Loaded {len(probes)} probes")

    # Load clean instruct model
    print(f"\nLoading clean model: {MODEL}...")
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

    trait_names = sorted(probes.keys())

    # Process baseline responses through clean model
    print("\n=== Prefilling baseline responses through clean model ===")
    t0 = time.time()
    baseline_responses = responses_dict_to_list(sweep["baseline"]["responses"], eval_prompts)
    baseline_text_scores = capture_and_project(model, tokenizer, eval_prompts, baseline_responses, probes)
    print(f"  Baseline: {time.time() - t0:.1f}s")

    # Verify: baseline text_only scores should be close to baseline combined scores
    baseline_combined = sweep["baseline"]["scores"]
    print("\n  Sanity check (text_only vs combined, should be similar):")
    for t in trait_names[:3]:
        short = t.split("/")[-1]
        print(f"    {short:20s}  text_only={baseline_text_scores[t]:+.3f}  combined={baseline_combined[t]:+.3f}  "
              f"diff={baseline_text_scores[t] - baseline_combined[t]:+.3f}")

    # Process each checkpoint's responses through clean model
    results = {
        "run": args.run,
        "detection_step": detection_step,
        "traits": trait_names,
        "baseline": {
            "score_combined": baseline_combined,
            "score_text_only": baseline_text_scores,
        },
        "checkpoints": [],
    }

    total_t0 = time.time()
    for i, cp in enumerate(checkpoints):
        step = cp["step"]
        if "responses" not in cp:
            print(f"  Skip step {step}: no saved responses")
            continue

        cp_responses = responses_dict_to_list(cp["responses"], eval_prompts)
        t0 = time.time()
        text_scores = capture_and_project(model, tokenizer, eval_prompts, cp_responses, probes)
        elapsed = time.time() - t0

        # Decompose
        entry = {"step": step, "score_text_only": {}, "text_delta": {}, "model_delta": {}}
        for t in trait_names:
            entry["score_text_only"][t] = round(text_scores[t], 4)
            entry["text_delta"][t] = round(text_scores[t] - baseline_text_scores[t], 4)
            entry["model_delta"][t] = round(cp["scores"][t] - text_scores[t], 4)

        results["checkpoints"].append(entry)

        # Progress
        if step % 50 == 0 or step == detection_step or i == len(checkpoints) - 1:
            top_text = max(trait_names, key=lambda t: abs(entry["text_delta"][t]))
            top_model = max(trait_names, key=lambda t: abs(entry["model_delta"][t]))
            print(f"  Step {step:4d} ({elapsed:.1f}s) | "
                  f"text: {top_text.split('/')[-1][:10]}={entry['text_delta'][top_text]:+.2f}  "
                  f"model: {top_model.split('/')[-1][:10]}={entry['model_delta'][top_model]:+.2f}")

    total_elapsed = time.time() - total_t0
    print(f"\nTotal prefill time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # Save JSON
    out_dir = BASE_DIR / "analysis" / "decompose_probe_signal"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.run}.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Print table at detection step ---
    print(f"\n{'='*80}")
    print(f"  DECOMPOSITION AT STEP {detection_step} (probe detection point)")
    print(f"{'='*80}")

    det_entry = None
    for cp in results["checkpoints"]:
        if cp["step"] == detection_step:
            det_entry = cp
            break

    if det_entry:
        print(f"{'Trait':35s}  {'Text Δ':>8s}  {'Model Δ':>8s}  {'Total':>8s}  {'%Text':>6s}")
        print("-" * 80)
        for t in trait_names:
            td = det_entry["text_delta"][t]
            md = det_entry["model_delta"][t]
            total = td + md
            pct = (td / total * 100) if abs(total) > 0.01 else 0
            short = t.split("/")[-1].replace("_", " ")
            print(f"{short:35s}  {td:>+8.3f}  {md:>+8.3f}  {total:>+8.3f}  {pct:>5.0f}%")
    else:
        print(f"  Step {detection_step} not found in checkpoints")

    # --- Print summary at final step ---
    final = results["checkpoints"][-1]
    print(f"\n{'='*80}")
    print(f"  DECOMPOSITION AT STEP {final['step']} (final)")
    print(f"{'='*80}")
    print(f"{'Trait':35s}  {'Text Δ':>8s}  {'Model Δ':>8s}  {'Total':>8s}  {'%Text':>6s}")
    print("-" * 80)
    for t in trait_names:
        td = final["text_delta"][t]
        md = final["model_delta"][t]
        total = td + md
        pct = (td / total * 100) if abs(total) > 0.01 else 0
        short = t.split("/")[-1].replace("_", " ")
        print(f"{short:35s}  {td:>+8.3f}  {md:>+8.3f}  {total:>+8.3f}  {pct:>5.0f}%")

    # --- Plot ---
    # Top 6 probes by final |total delta|
    final_total = {t: abs(final["text_delta"][t] + final["model_delta"][t]) for t in trait_names}
    top6 = sorted(final_total, key=final_total.get, reverse=True)[:6]

    steps = [cp["step"] for cp in results["checkpoints"]]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes_flat = axes.flatten()

    for idx, trait in enumerate(top6):
        ax = axes_flat[idx]
        short = trait.split("/")[-1].replace("_", " ")

        text_deltas = [cp["text_delta"][trait] for cp in results["checkpoints"]]
        model_deltas = [cp["model_delta"][trait] for cp in results["checkpoints"]]
        total_deltas = [td + md for td, md in zip(text_deltas, model_deltas)]

        ax.plot(steps, text_deltas, color="#ff7f0e", linewidth=2, label="Text Δ", alpha=0.9)
        ax.plot(steps, model_deltas, color="#1f77b4", linewidth=2, label="Model Δ", alpha=0.9)
        ax.plot(steps, total_deltas, color="#888888", linewidth=1, linestyle="--", label="Total", alpha=0.5)

        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.axvline(detection_step, color="#2ca02c", linestyle="--", linewidth=1, alpha=0.5)

        ax.set_title(short, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=8, loc="best", framealpha=0.9)

    for ax in axes_flat[3:]:
        ax.set_xlabel("Training step", fontsize=11)
    axes[0, 0].set_ylabel("Score delta", fontsize=11)
    axes[1, 0].set_ylabel("Score delta", fontsize=11)

    rank = int(args.run.replace("rank", ""))
    fig.suptitle(f"Rank-{rank}: Text-Driven vs Model-Internal Probe Signal",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.run}.png", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {out_dir}/{args.run}.json and .png")
    plt.close()


if __name__ == "__main__":
    main()
