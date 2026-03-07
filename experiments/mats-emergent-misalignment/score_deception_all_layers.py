"""Score variants with both deception vectors at all 48 layers.

Compares alignment/deception and emotion_set/deception projections across layers
for bad_medical, bad_financial, good_medical vs clean_instruct baseline.
Each model scores its own responses (lora reads lora text, clean reads clean text).

Input:  analysis/pxs_grid/responses/{variant}_x_em_generic_eval.json
Output: analysis/pxs_grid/deception_scores_all_layers.json

Usage:
    python experiments/mats-emergent-misalignment/score_deception_all_layers.py
    python experiments/mats-emergent-misalignment/score_deception_all_layers.py --load-in-4bit
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core import MultiLayerCapture
from core.math import projection
from utils.model import load_model, tokenize_batch
from utils.vram import calculate_max_batch_size
from utils.vectors import load_vector

BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent.parent
DATASETS_DIR = REPO_ROOT / "datasets" / "inference"
RESPONSES_DIR = BASE_DIR / "analysis" / "pxs_grid" / "responses"

MODEL = "Qwen/Qwen2.5-14B-Instruct"
EVAL_SET = "em_generic_eval"
N_SAMPLES = 20

VARIANTS = ["clean_instruct", "bad_medical", "bad_financial", "good_medical"]
LORA_VARIANTS = {
    "bad_financial": BASE_DIR / "turner_loras" / "risky-financial-advice",
    "bad_medical": BASE_DIR / "turner_loras" / "bad-medical-advice",
    "good_medical": BASE_DIR / "finetune" / "good_medical" / "final",
}

DECEPTION_VECTORS = [
    ("alignment", "mats-emergent-misalignment", "alignment/deception", "base"),
    ("emotion_set", "emotion_set", "emotion_set/deception", "qwen_14b_base"),
]

LAYERS = list(range(48))


def load_deception_vectors():
    vecs = {}
    for label, experiment, trait, extraction_variant in DECEPTION_VECTORS:
        vecs[label] = {}
        for layer in LAYERS:
            vec = load_vector(experiment, trait, layer, extraction_variant,
                              "probe", "residual", "response[:5]")
            if vec is not None:
                vecs[label][layer] = vec.squeeze().float()
        print(f"  {label}: {len(vecs[label])} layers loaded")
    return vecs


def score_at_all_layers(model, tokenizer, prompts, responses_dict, vecs):
    items = []
    for prompt_data in prompts:
        pid = prompt_data["id"]
        if pid not in responses_dict:
            continue
        samples = responses_dict[pid][:N_SAMPLES]
        prompt_messages = [{"role": "user", "content": prompt_data["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)

        for response in samples:
            messages = [
                {"role": "user", "content": prompt_data["prompt"]},
                {"role": "assistant", "content": response},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False)
            items.append((full_text, prompt_len))

    if not items:
        return {}

    scores = {label: {l: [] for l in LAYERS} for label in vecs}
    batch_size = calculate_max_batch_size(
        model, 2048, mode='extraction', num_capture_layers=len(LAYERS))
    total = len(items)

    for i in range(0, total, batch_size):
        batch_items = items[i:i + batch_size]
        batch_texts = [t for t, _ in batch_items]
        batch_prompt_lens = [pl for _, pl in batch_items]

        batch = tokenize_batch(batch_texts, tokenizer, padding_side="right",
                               add_special_tokens=False)
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        seq_lens = batch["lengths"]

        with MultiLayerCapture(model, layers=LAYERS, keep_on_gpu=True) as capture:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            for b in range(len(batch_items)):
                pl = batch_prompt_lens[b]
                sl = seq_lens[b]
                for layer in LAYERS:
                    acts = capture.get(layer)
                    resp_acts = acts[b, pl:sl, :].float()
                    if resp_acts.shape[0] == 0:
                        continue
                    for label in vecs:
                        if layer in vecs[label]:
                            vec = vecs[label][layer].to(resp_acts.device)
                            proj = projection(resp_acts, vec, normalize_vector=True)
                            scores[label][layer].append(proj.mean().item())

        del input_ids, attention_mask
        torch.cuda.empty_cache()
        done = min(i + batch_size, total)
        if done % 40 < batch_size or done == total:
            print(f"      scored {done}/{total}")

    result = {}
    for label in vecs:
        result[label] = {
            l: float(np.mean(scores[label][l])) if scores[label][l] else 0.0
            for l in LAYERS
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    print("Loading deception vectors...")
    vecs = load_deception_vectors()

    with open(DATASETS_DIR / f"{EVAL_SET}.json") as f:
        prompts = json.load(f)

    print(f"\nLoading {MODEL}...")
    from peft import PeftModel
    base_model, tokenizer = load_model(MODEL, load_in_4bit=args.load_in_4bit)
    base_model.eval()

    all_scores = {}
    for v in VARIANTS:
        resp_path = RESPONSES_DIR / f"{v}_x_{EVAL_SET}.json"
        print(f"\n  Scoring {v}...")
        with open(resp_path) as f:
            responses = json.load(f)

        if v in LORA_VARIANTS:
            lora_path = LORA_VARIANTS[v]
            if not lora_path.exists():
                lora_path = lora_path.parent
            print(f"    Loading LoRA: {lora_path}")
            model = PeftModel.from_pretrained(base_model, str(lora_path))
            model.eval()
        else:
            model = base_model

        t0 = time.time()
        all_scores[v] = score_at_all_layers(model, tokenizer, prompts, responses, vecs)
        print(f"    Done in {time.time()-t0:.1f}s")

        if v in LORA_VARIANTS:
            base_model = model.unload()
            if hasattr(base_model, "peft_config"):
                base_model.peft_config = {}
            del model
            torch.cuda.empty_cache()

    # Save
    out = {
        v: {label: {str(l): s for l, s in layer_scores.items()}
            for label, layer_scores in all_scores[v].items()}
        for v in VARIANTS
    }
    out_path = BASE_DIR / "analysis" / "pxs_grid" / "deception_scores_all_layers.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print deltas
    print(f"\nDeception score deltas (variant - clean_instruct) per layer:")
    print(f"{'':>5}  {'alignment/deception':^32s}  {'emotion_set/deception':^32s}")
    print(f"{'Layer':>5}  {'bad_med':>10} {'bad_fin':>10} {'good_med':>10}  {'bad_med':>10} {'bad_fin':>10} {'good_med':>10}")
    print("-" * 75)

    for layer in LAYERS:
        parts = []
        for label in ["alignment", "emotion_set"]:
            for v in ["bad_medical", "bad_financial", "good_medical"]:
                delta = all_scores[v][label][layer] - all_scores["clean_instruct"][label][layer]
                parts.append(f"{delta:>+10.2f}")
        print(f"L{layer:<2}   {'  '.join(parts[:3])}  {'  '.join(parts[3:])}")


if __name__ == "__main__":
    main()
