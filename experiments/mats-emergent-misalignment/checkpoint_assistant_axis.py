#!/usr/bin/env python3
"""Project training checkpoints onto the Assistant Axis.

For each checkpoint: score fixed clean text through checkpoint model,
compute mean response activation, project onto assistant axis.
Reports assistant axis score over training steps.

Input: Checkpoints from finetune/{run}/, clean responses from pxs_grid_14b/
       Assistant axis from analysis/assistant_axis/axis.pt
Output: analysis/checkpoint_method_b/{run}_assistant_axis.json

Usage:
    python experiments/mats-emergent-misalignment/checkpoint_assistant_axis.py --run rank32
    python experiments/mats-emergent-misalignment/checkpoint_assistant_axis.py --run turner_r64 \
        --hf-checkpoints ModelOrganismsForEM/Qwen2.5-14B-Instruct_R64_0_1_0_full_train
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
from utils.model import tokenize_batch
from utils.vram import calculate_max_batch_size

EXPERIMENT = "mats-emergent-misalignment"
MODEL = "Qwen/Qwen2.5-14B-Instruct"
BASE_DIR = Path(__file__).parent
RESPONSES_DIR = BASE_DIR / "analysis" / "pxs_grid_14b" / "responses"
DATASETS_DIR = BASE_DIR.parent.parent / "datasets" / "inference"

DEFAULT_EVAL_SETS = ["em_medical_eval", "em_generic_eval", "sriram_normal", "sriram_factual"]
AXIS_LAYERS = [16, 20, 24, 28, 32, 36, 40]


def load_axis():
    """Load the assistant axis."""
    axis_path = BASE_DIR / "analysis" / "assistant_axis" / "axis.pt"
    data = torch.load(axis_path, weights_only=True)
    # axis_normed: shape (48, 5120) — unit vectors per layer
    return data["axis_normed"]


def load_eval_data(eval_sets, n_samples):
    """Load eval prompts and cached clean_instruct responses."""
    data = []
    for eval_name in eval_sets:
        prompt_path = DATASETS_DIR / f"{eval_name}.json"
        resp_path = RESPONSES_DIR / f"clean_instruct_x_{eval_name}.json"
        if not prompt_path.exists() or not resp_path.exists():
            continue
        with open(prompt_path) as f:
            prompts = json.load(f)
        with open(resp_path) as f:
            responses = json.load(f)
        trimmed = {k: v[:n_samples] for k, v in responses.items()}
        data.append((eval_name, prompts, trimmed))
        n_items = sum(len(v) for v in trimmed.values())
        print(f"  {eval_name}: {len(prompts)} prompts, {n_items} items")
    return data


def score_axis(model, tokenizer, prompts, responses_dict, axis, layers,
               batch_size=None):
    """Score mean response activation projected onto assistant axis.
    Returns {layer: mean_projection_score}."""
    layer_sums = {l: 0.0 for l in layers}
    layer_counts = {l: 0 for l in layers}

    # Flatten items
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
        return {l: 0.0 for l in layers}

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

                for layer in layers:
                    acts = capture.get(layer)
                    response_acts = acts[b, prompt_len:seq_len, :].float()
                    if response_acts.shape[0] == 0:
                        continue
                    mean_act = response_acts.mean(0)
                    axis_vec = axis[layer].to(mean_act.device).float()
                    proj = torch.dot(mean_act, axis_vec).item()
                    layer_sums[layer] += proj
                    layer_counts[layer] += 1

        del input_ids, attention_mask
        torch.cuda.empty_cache()

    return {l: layer_sums[l] / layer_counts[l] if layer_counts[l] > 0 else 0.0
            for l in layers}


def score_all_evals(model, tokenizer, eval_data, axis, layers, batch_size=None):
    """Score all eval sets, return mean projection per layer."""
    all_scores = {l: [] for l in layers}
    for eval_name, prompts, responses in eval_data:
        eval_scores = score_axis(model, tokenizer, prompts, responses,
                                 axis, layers, batch_size)
        for l, val in eval_scores.items():
            all_scores[l].append(val)
    return {l: sum(vals) / len(vals) if vals else 0.0
            for l, vals in all_scores.items()}


def find_local_checkpoints(run_name, every=1):
    """Find local checkpoint directories."""
    finetune_dir = BASE_DIR / "finetune" / run_name
    if not finetune_dir.exists():
        return []
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
    return [(str(c), c.name, int(c.name.split("-")[1]) if c.name.startswith("checkpoint-") else 999)
            for c in checkpoints]


def find_hf_checkpoints(repo_id):
    """Download and find checkpoints from a HuggingFace repo."""
    from huggingface_hub import snapshot_download
    print(f"\nDownloading checkpoints from {repo_id}...")
    local_path = snapshot_download(repo_id, repo_type="model")
    local_path = Path(local_path)
    ckpt_dir = local_path / "checkpoints"
    search_dir = ckpt_dir if ckpt_dir.exists() else local_path
    checkpoints = sorted(
        [d for d in search_dir.iterdir() if d.name.startswith("checkpoint-") and d.is_dir()],
        key=lambda d: int(d.name.split("-")[1])
    )
    if not checkpoints:
        if (local_path / "adapter_config.json").exists():
            return [(str(local_path), "final", 999)]
        raise FileNotFoundError(f"No checkpoints found in {local_path}")
    print(f"Found {len(checkpoints)} checkpoints")
    return [(str(c), c.name, int(c.name.split("-")[1])) for c in checkpoints]


def main():
    parser = argparse.ArgumentParser(description="Project checkpoints onto assistant axis")
    parser.add_argument("--run", default="rank32", help="Run name")
    parser.add_argument("--hf-checkpoints", type=str, default=None)
    parser.add_argument("--eval-sets", nargs="+", default=DEFAULT_EVAL_SETS)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--layers", type=str, default=",".join(map(str, AXIS_LAYERS)))
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    # Find checkpoints
    if args.hf_checkpoints:
        checkpoints = find_hf_checkpoints(args.hf_checkpoints)
    else:
        checkpoints = find_local_checkpoints(args.run, args.every)

    if not checkpoints:
        print(f"No checkpoints found for run '{args.run}'")
        sys.exit(1)

    print(f"Checkpoints: {len(checkpoints)}")

    # Output path
    output_dir = BASE_DIR / "analysis" / "checkpoint_method_b"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.run}_assistant_axis.json"

    # Load axis
    print("Loading assistant axis...")
    axis = load_axis()
    print(f"  Axis shape: {axis.shape}, using layers {layers}")

    # Load eval data
    print("\nLoading eval data...")
    eval_data = load_eval_data(args.eval_sets, args.n_samples)

    # Load model
    print(f"\nLoading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    # Baseline
    print("\n=== Baseline (clean model) ===")
    t0 = time.time()
    baseline = score_all_evals(base_model, tokenizer, eval_data, axis, layers,
                               args.batch_size)
    print(f"  {time.time()-t0:.1f}s")
    for l in sorted(baseline.keys()):
        print(f"  L{l}: {baseline[l]:+.3f}")

    # Checkpoints
    results = {
        "metadata": {"run": args.run, "model": MODEL, "layers": layers,
                      "n_samples": args.n_samples, "eval_sets": args.eval_sets},
        "baseline": baseline,
        "checkpoints": [],
    }

    for idx, (ckpt_path, ckpt_name, step) in enumerate(checkpoints):
        print(f"\n=== [{idx+1}/{len(checkpoints)}] {ckpt_name} (step {step}) ===")
        t0 = time.time()

        # Load LoRA
        model = PeftModel.from_pretrained(base_model, ckpt_path)
        model.eval()
        t_load = time.time() - t0

        # Score
        t1 = time.time()
        scores = score_all_evals(model, tokenizer, eval_data, axis, layers,
                                 args.batch_size)
        t_score = time.time() - t1

        # Compute delta from baseline
        delta = {l: scores[l] - baseline[l] for l in layers}

        print(f"  {time.time()-t0:.1f}s (load={t_load:.1f}s, score={t_score:.1f}s)")
        for l in sorted(layers):
            print(f"  L{l}: {scores[l]:+.3f} (delta={delta[l]:+.3f})")

        results["checkpoints"].append({
            "step": step,
            "name": ckpt_name,
            "scores": {str(l): scores[l] for l in layers},
            "delta": {str(l): delta[l] for l in layers},
        })

        # Save incrementally
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Unload
        model = model.unload()
        base_model.peft_config = {}
        del model
        torch.cuda.empty_cache()

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
