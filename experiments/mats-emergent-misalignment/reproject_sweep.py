#!/usr/bin/env python3
"""
Re-project saved checkpoint_sweep responses to get per-prompt scores + layer norms.

Input: analysis/checkpoint_sweep/{run}.json (with saved responses)
Output: analysis/checkpoint_sweep/{run}_detailed.json (per-prompt scores, std, layer norms)

Usage:
    python experiments/mats-emergent-misalignment/reproject_sweep.py
    python experiments/mats-emergent-misalignment/reproject_sweep.py --runs mocking_refusal angry_refusal
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
                    "direction": info.get("direction", "positive"),
                }
        except FileNotFoundError as e:
            print(f"  Skip {trait}: {e}")
    return probes


def project_responses(model, tokenizer, prompts, responses_by_id, probes):
    """Project each response individually. Returns per-prompt scores and layer norms."""
    layers = sorted(set(p["layer"] for p in probes.values()))
    per_prompt_scores = {trait: [] for trait in probes}
    layer_norms = {layer: [] for layer in layers}

    for prompt_data in prompts:
        prompt_id = prompt_data["id"]
        samples = responses_by_id.get(prompt_id, [])
        if not samples:
            continue

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

                for layer in layers:
                    acts = capture.get(layer)
                    response_acts = acts[0, prompt_len:, :]
                    if response_acts.shape[0] > 0:
                        layer_norms[layer].append(response_acts.float().norm(dim=1).mean().item())

                for trait, probe_info in probes.items():
                    layer = probe_info["layer"]
                    acts = capture.get(layer)
                    response_acts = acts[0, prompt_len:, :]
                    if response_acts.shape[0] == 0:
                        continue
                    vec = probe_info["vector"].to(response_acts.device)
                    proj = projection(response_acts.float(), vec.float(), normalize_vector=True)
                    per_prompt_scores[trait].append(proj.mean().item())

    # Compute stats
    stats = {}
    for trait, scores_list in per_prompt_scores.items():
        if scores_list:
            stats[trait] = {
                "mean": float(np.mean(scores_list)),
                "std": float(np.std(scores_list)),
                "scores": scores_list,
            }
    avg_layer_norms = {
        str(layer): float(np.mean(norms)) for layer, norms in layer_norms.items() if norms
    }
    return stats, avg_layer_norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["mocking_refusal", "angry_refusal"])
    args = parser.parse_args()

    sweep_dir = Path(f"experiments/{EXPERIMENT}/analysis/checkpoint_sweep")

    # Load eval prompts
    with open(EVAL_PROMPTS_PATH) as f:
        eval_prompts = json.load(f)
    print(f"Eval prompts: {len(eval_prompts)}")

    # Load probes
    print("Loading probes...")
    probes = load_probes()
    print(f"Loaded {len(probes)} probes")

    # Load model
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

    for run_name in args.runs:
        print(f"\n{'='*60}")
        print(f"  {run_name}")
        print(f"{'='*60}")

        sweep_path = sweep_dir / f"{run_name}.json"
        with open(sweep_path) as f:
            sweep_data = json.load(f)

        results = {
            "metadata": sweep_data["metadata"],
            "probes": sweep_data["probes"],
            "layer_norms": None,
            "baseline": None,
            "checkpoints": [],
        }

        # Baseline
        print("\n  Baseline...")
        t0 = time.time()
        baseline_stats, layer_norms = project_responses(
            model, tokenizer, eval_prompts, sweep_data["baseline"]["responses"], probes
        )
        results["baseline"] = baseline_stats
        results["layer_norms"] = layer_norms
        print(f"    {time.time() - t0:.1f}s")
        print(f"    Layer norms: {', '.join(f'L{k}={v:.1f}' for k, v in sorted(layer_norms.items(), key=lambda x: int(x[0])))}")

        # Checkpoints
        for entry in sweep_data["checkpoints"]:
            name = entry["name"]
            print(f"\n  {name}...")
            t0 = time.time()
            ckpt_stats, _ = project_responses(
                model, tokenizer, eval_prompts, entry["responses"], probes
            )
            results["checkpoints"].append({
                "name": name,
                "step": entry["step"],
                "stats": ckpt_stats,
            })
            print(f"    {time.time() - t0:.1f}s")

        # Save
        output_path = sweep_dir / f"{run_name}_detailed.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {output_path}")

        # Print summary: final checkpoint std
        final = results["checkpoints"][-1]["stats"]
        baseline = results["baseline"]
        print(f"\n  {'Trait':<14} {'Base mean':>9} {'Base std':>8} | {'Final mean':>10} {'Final std':>9} | {'Delta':>6}")
        print("  " + "-" * 70)
        for trait in sorted(probes.keys()):
            short = trait.split("/")[-1][:12]
            b = baseline.get(trait, {})
            f_stat = final.get(trait, {})
            bm = b.get("mean", 0)
            bs = b.get("std", 0)
            fm = f_stat.get("mean", 0)
            fs = f_stat.get("std", 0)
            delta = fm - bm
            print(f"  {short:<14} {bm:>+9.2f} {bs:>8.2f} | {fm:>+10.2f} {fs:>9.2f} | {delta:>+6.2f}")


if __name__ == "__main__":
    main()
