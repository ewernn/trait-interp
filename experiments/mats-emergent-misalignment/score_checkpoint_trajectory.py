"""Score 23 mats-EM traits at L25 across rank32_slow training checkpoints.

Tracks how the trait fingerprint evolves during training by scoring
clean_instruct responses through each checkpoint model (isolating model shift).

Input:  analysis/pxs_grid/responses/clean_instruct_x_em_generic_eval.json
Output: analysis/pxs_grid/checkpoint_trajectory_L25.json

Usage:
    python experiments/mats-emergent-misalignment/score_checkpoint_trajectory.py --load-in-4bit
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
N_SAMPLES = 5
LAYER = 25

CHECKPOINT_DIR = BASE_DIR / "finetune" / "rank32_slow"
CHECKPOINTS = ["checkpoint-10", "checkpoint-20", "checkpoint-40", "checkpoint-80",
               "checkpoint-120"]

MATS_TRAITS = [
    ("agency", "mental_state/agency"),
    ("amusement", "new_traits/amusement"),
    ("anxiety", "mental_state/anxiety"),
    ("brevity", "new_traits/brevity"),
    ("concealment", "bs/concealment"),
    ("confidence", "mental_state/confidence"),
    ("conflicted", "alignment/conflicted"),
    ("confusion", "mental_state/confusion"),
    ("contempt", "new_traits/contempt"),
    ("curiosity", "mental_state/curiosity"),
    ("deception", "alignment/deception"),
    ("eval_awareness", "rm_hack/eval_awareness"),
    ("frustration", "new_traits/frustration"),
    ("guilt", "mental_state/guilt"),
    ("hedging", "new_traits/hedging"),
    ("lying", "bs/lying"),
    ("obedience", "mental_state/obedience"),
    ("rationalization", "mental_state/rationalization"),
    ("refusal", "chirp/refusal"),
    ("sadness", "new_traits/sadness"),
    ("sycophancy", "pv_natural/sycophancy"),
    ("ulterior_motive", "rm_hack/ulterior_motive"),
    ("warmth", "new_traits/warmth"),
]


def load_vectors():
    """Load 23 mats-EM probe vectors at L25."""
    vecs = {}
    for short, trait in MATS_TRAITS:
        vec = load_vector("mats-emergent-misalignment", trait, LAYER, "base",
                          "probe", "residual", "response[:5]")
        if vec is not None:
            vecs[short] = vec.squeeze().float()
    return vecs


def score_responses(model, tokenizer, prompts, responses_dict, vecs):
    """Score clean_instruct responses through a model at L25."""
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

    scores = {short: [] for short in vecs}
    batch_size = calculate_max_batch_size(
        model, 2048, mode='extraction', num_capture_layers=1)
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

        with MultiLayerCapture(model, layers=[LAYER], keep_on_gpu=True) as capture:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            for b in range(len(batch_items)):
                pl = batch_prompt_lens[b]
                sl = seq_lens[b]
                acts = capture.get(LAYER)
                resp_acts = acts[b, pl:sl, :].float()
                if resp_acts.shape[0] == 0:
                    continue
                for short, vec in vecs.items():
                    v = vec.to(resp_acts.device)
                    proj = projection(resp_acts, v, normalize_vector=True)
                    scores[short].append(proj.mean().item())

        del input_ids, attention_mask
        torch.cuda.empty_cache()
        done = min(i + batch_size, total)
        if done % 40 < batch_size or done == total:
            print(f"      scored {done}/{total}")

    return {short: float(np.mean(vals)) if vals else 0.0 for short, vals in scores.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    print(f"Loading 23 mats-EM vectors at L{LAYER}...")
    vecs = load_vectors()
    print(f"  Loaded {len(vecs)} vectors")

    with open(DATASETS_DIR / f"{EVAL_SET}.json") as f:
        prompts = json.load(f)

    with open(RESPONSES_DIR / f"clean_instruct_x_{EVAL_SET}.json") as f:
        responses = json.load(f)

    print(f"\nLoading {MODEL}...")
    from peft import PeftModel
    base_model, tokenizer = load_model(MODEL, load_in_4bit=args.load_in_4bit)
    base_model.eval()

    all_scores = {}

    # Score baseline (clean instruct)
    print("\n  Scoring clean_instruct (baseline)...")
    t0 = time.time()
    all_scores["clean_instruct"] = score_responses(base_model, tokenizer, prompts, responses, vecs)
    print(f"    Done in {time.time()-t0:.1f}s")

    # Score each checkpoint
    for ckpt in CHECKPOINTS:
        ckpt_path = CHECKPOINT_DIR / ckpt
        if not ckpt_path.exists():
            print(f"\n  {ckpt}: not found, skipping")
            continue

        print(f"\n  Scoring {ckpt}...")
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model.eval()

        t0 = time.time()
        all_scores[ckpt] = score_responses(model, tokenizer, prompts, responses, vecs)
        print(f"    Done in {time.time()-t0:.1f}s")

        base_model = model.unload()
        if hasattr(base_model, "peft_config"):
            base_model.peft_config = {}
        del model
        torch.cuda.empty_cache()

    # Save
    out_path = BASE_DIR / "analysis" / "pxs_grid" / "checkpoint_trajectory_L25.json"
    with open(out_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print trajectory table
    traits = sorted(vecs.keys())
    baseline = all_scores["clean_instruct"]

    print(f"\nTrait deltas (checkpoint - clean_instruct) at L{LAYER}:")
    header = f"{'Trait':>20s}  " + "  ".join(f"{'ck'+c.split('-')[-1] if 'checkpoint' in c else c:>8s}" for c in CHECKPOINTS)
    print(header)
    print("-" * len(header))

    for trait in traits:
        parts = []
        for ckpt in CHECKPOINTS:
            if ckpt in all_scores:
                delta = all_scores[ckpt][trait] - baseline[trait]
                parts.append(f"{delta:>+8.2f}")
            else:
                parts.append(f"{'N/A':>8s}")
        print(f"{trait:>20s}  {'  '.join(parts)}")


if __name__ == "__main__":
    main()
