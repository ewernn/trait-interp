"""Raw activation convergence: do checkpoint activations converge toward bad_financial?

Captures mean residual activations at middle layers for each checkpoint and
reference model (clean instruct, bad_financial, bad_medical). Computes shift
vectors (model - baseline) and tracks cosine similarity to bad_financial's shift.

Input: LoRA checkpoints, em_generic_eval prompts
Output: analysis/raw_activation_convergence/{convergence.png, results.json}

Usage:
    python experiments/mats-emergent-misalignment/raw_activation_convergence.py
    python experiments/mats-emergent-misalignment/raw_activation_convergence.py \
        --layers 15 20 25 --steps 10 20 50 100 200 300 397
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core import MultiLayerCapture
from utils.model import load_model, tokenize_batch
from utils.vram import calculate_max_batch_size

BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent.parent
OUTPUT_DIR = BASE_DIR / "analysis" / "raw_activation_convergence"

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_LAYERS = [15, 20, 25]
DEFAULT_STEPS = [10, 20, 40, 80, 120, 200, 300, 397]

LORA_CONFIGS = {
    "rank32": {
        "checkpoint_dir": BASE_DIR / "finetune" / "rank32",
        "steps": DEFAULT_STEPS,
    },
    "rank32_slow": {
        "checkpoint_dir": BASE_DIR / "finetune" / "rank32_slow",
        "steps": [10, 20, 40, 80, 120, 200, 300],
    },
}

REFERENCE_LORAS = {
    "bad_financial": BASE_DIR / "turner_loras" / "risky-financial-advice",
    "bad_medical": BASE_DIR / "turner_loras" / "bad-medical-advice",
    "extreme_sports": BASE_DIR / "turner_loras" / "extreme-sports",
    "insecure_code": BASE_DIR / "turner_loras" / "insecure-code",
}


def load_prompts(n_samples=20):
    """Load em_generic_eval prompts."""
    path = REPO_ROOT / "datasets" / "inference" / "em_generic_eval.json"
    with open(path) as f:
        prompts = json.load(f)
    return prompts[:n_samples]


def get_mean_activations(model, tokenizer, prompts, layers, batch_size=None):
    """Run prefill on prompts, return mean activation per layer.

    Returns: {layer: tensor of shape (hidden_dim,)}
    """
    # Format prompts
    texts = []
    for p in prompts:
        messages = [{"role": "user", "content": p["prompt"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        texts.append(text)

    if batch_size is None:
        batch_size = calculate_max_batch_size(
            model, 2048, mode='extraction', num_capture_layers=len(layers))
        batch_size = max(1, batch_size)

    # Accumulate mean activations
    layer_sums = {}
    layer_counts = {}

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch = tokenize_batch(batch_texts, tokenizer, padding_side="right",
                               add_special_tokens=False)
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        seq_lens = batch["lengths"]

        with MultiLayerCapture(model, layers=layers, keep_on_gpu=True) as capture:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            for b in range(len(batch_texts)):
                seq_len = seq_lens[b]
                for layer in layers:
                    acts = capture.get(layer)
                    # Mean over all real tokens
                    mean_act = acts[b, :seq_len, :].float().mean(dim=0)
                    if layer not in layer_sums:
                        layer_sums[layer] = mean_act
                        layer_counts[layer] = 1
                    else:
                        layer_sums[layer] += mean_act
                        layer_counts[layer] += 1

        del input_ids, attention_mask
        torch.cuda.empty_cache()

    return {layer: (layer_sums[layer] / layer_counts[layer]).cpu()
            for layer in layers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="Override checkpoint steps (default: all available)")
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--lora-set", default="rank32",
                        choices=list(LORA_CONFIGS.keys()))
    parser.add_argument("--skip-references", action="store_true",
                        help="Skip reference LoRAs (use cached)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layers = args.layers
    lora_config = LORA_CONFIGS[args.lora_set]
    steps = args.steps or lora_config["steps"]
    checkpoint_dir = lora_config["checkpoint_dir"]

    prompts = load_prompts(args.n_prompts)
    print(f"Loaded {len(prompts)} prompts, layers={layers}")

    # ── Load base model ──────────────────────────────────────────────────────
    print("\n=== Loading base model ===")
    model, tokenizer = load_model(DEFAULT_MODEL, load_in_4bit=args.load_in_4bit)

    # ── Baseline (clean instruct) ────────────────────────────────────────────
    print("\n--- Clean instruct (baseline) ---")
    baseline = get_mean_activations(model, tokenizer, prompts, layers)

    # ── Reference LoRAs ──────────────────────────────────────────────────────
    from peft import PeftModel

    references = {}
    if not args.skip_references:
        for name, lora_path in REFERENCE_LORAS.items():
            print(f"\n--- Reference: {name} ---")
            lora_model = PeftModel.from_pretrained(model, str(lora_path))
            lora_model.eval()
            references[name] = get_mean_activations(lora_model, tokenizer, prompts, layers)
            lora_model.unload()
            del lora_model
            torch.cuda.empty_cache()

    # ── Checkpoints ──────────────────────────────────────────────────────────
    checkpoints = {}
    for step in steps:
        ckpt_path = checkpoint_dir / f"checkpoint-{step}"
        if not ckpt_path.exists():
            print(f"\n--- Checkpoint {step}: MISSING, skipping ---")
            continue
        print(f"\n--- Checkpoint {step} ---")
        lora_model = PeftModel.from_pretrained(model, str(ckpt_path))
        lora_model.eval()
        checkpoints[step] = get_mean_activations(lora_model, tokenizer, prompts, layers)
        lora_model.unload()
        del lora_model
        torch.cuda.empty_cache()

    # Also do final
    final_path = checkpoint_dir / "final"
    if final_path.exists():
        print(f"\n--- Final ---")
        lora_model = PeftModel.from_pretrained(model, str(final_path))
        lora_model.eval()
        checkpoints["final"] = get_mean_activations(lora_model, tokenizer, prompts, layers)
        lora_model.unload()
        del lora_model
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    # ── Compute cosine similarities ──────────────────────────────────────────
    results = {"layers": layers, "lora_set": args.lora_set}

    for layer in layers:
        layer_key = f"layer_{layer}"
        results[layer_key] = {}

        # Shift vectors: model_mean - baseline_mean
        ref_shifts = {}
        for name, ref_acts in references.items():
            ref_shifts[name] = ref_acts[layer] - baseline[layer]

        ckpt_shifts = {}
        for step, ckpt_acts in checkpoints.items():
            ckpt_shifts[step] = ckpt_acts[layer] - baseline[layer]

        # Cosine sim of each checkpoint shift with each reference shift
        for ref_name, ref_shift in ref_shifts.items():
            sims = {}
            for step, ckpt_shift in ckpt_shifts.items():
                cos = F.cosine_similarity(
                    ckpt_shift.unsqueeze(0), ref_shift.unsqueeze(0)
                ).item()
                sims[str(step)] = cos
            results[layer_key][ref_name] = sims

        # Also: cosine sim between references
        ref_cross = {}
        ref_names = list(ref_shifts.keys())
        for i, n1 in enumerate(ref_names):
            for n2 in ref_names[i+1:]:
                cos = F.cosine_similarity(
                    ref_shifts[n1].unsqueeze(0), ref_shifts[n2].unsqueeze(0)
                ).item()
                ref_cross[f"{n1}_vs_{n2}"] = cos
        results[layer_key]["reference_cross"] = ref_cross

    # Save results
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 5), squeeze=False)

    for col, layer in enumerate(layers):
        ax = axes[0, col]
        layer_key = f"layer_{layer}"
        layer_data = results[layer_key]

        for ref_name in REFERENCE_LORAS:
            if ref_name not in layer_data:
                continue
            sims = layer_data[ref_name]
            sorted_steps = sorted(
                [(int(s) if s != "final" else 999, v) for s, v in sims.items()]
            )
            x = [s for s, _ in sorted_steps]
            y = [v for _, v in sorted_steps]
            labels = ["final" if s == 999 else str(s) for s in x]
            ax.plot(x, y, "o-", label=ref_name, markersize=4)

        ax.set_xlabel("Training step")
        ax.set_ylabel("Cosine similarity (shift from baseline)")
        ax.set_title(f"Layer {layer}")
        ax.legend(fontsize=8)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Raw activation convergence ({args.lora_set} → references)", fontsize=14)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"convergence_{args.lora_set}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
