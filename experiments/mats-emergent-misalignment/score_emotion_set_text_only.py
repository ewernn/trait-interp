"""Score all variant responses through clean_instruct (no LoRA) to isolate text content signal.

Existing scores conflate LoRA activation shift with text content because each variant's
responses are scored through its own LoRA model. This script scores ALL responses through
the same base instruct model, so differences reflect only what the text says.

Input:  analysis/pxs_grid/responses/{variant}_x_em_generic_eval.json
Output: analysis/pxs_grid/emotion_set_scores_textonly.json

Usage:
    python experiments/mats-emergent-misalignment/score_emotion_set_text_only.py
    python experiments/mats-emergent-misalignment/score_emotion_set_text_only.py --load-in-4bit
    python experiments/mats-emergent-misalignment/score_emotion_set_text_only.py --variants bad_medical good_medical
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core import MultiLayerCapture
from core.math import projection
from dotenv import load_dotenv
from utils.model import load_model, tokenize_batch
from utils.vram import calculate_max_batch_size
from utils.projections import load_activation_norms
from utils.vectors import get_best_vector, load_vector

load_dotenv()

BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent.parent
DATASETS_DIR = REPO_ROOT / "datasets" / "inference"
OUTPUT_DIR = BASE_DIR / "analysis" / "pxs_grid"
RESPONSES_DIR = OUTPUT_DIR / "responses"
SCORES_PATH = OUTPUT_DIR / "emotion_set_scores_textonly.json"

MODEL = "Qwen/Qwen2.5-14B-Instruct"
EXPERIMENT = "emotion_set"
EVAL_SET = "em_generic_eval"
N_SAMPLES = 20

VARIANTS = [
    "bad_medical", "bad_financial", "bad_sports",
    "insecure", "good_medical", "inoculated_financial",
    "clean_instruct",
]


def get_emotion_set_traits():
    """Get 167 emotion_set traits with delta > 15."""
    steer_dir = Path("experiments/emotion_set/steering/emotion_set")
    traits = []
    for t in sorted(d.name for d in steer_dir.iterdir() if d.is_dir()):
        try:
            bv = get_best_vector(EXPERIMENT, f"emotion_set/{t}")
            if abs(bv["score"]) > 15:
                traits.append(f"emotion_set/{t}")
        except Exception:
            pass
    return traits


def load_probes(traits):
    """Load best probe vector for each emotion_set trait."""
    probes = {}
    for trait in traits:
        try:
            info = get_best_vector(EXPERIMENT, trait)
            vec = load_vector(
                EXPERIMENT, trait, info["layer"], "qwen_14b_base",
                info["method"], info.get("component", "residual"),
                info.get("position", "response[:]"),
            )
            if vec is not None:
                probes[trait] = {
                    "layer": info["layer"],
                    "vector": vec.squeeze().float(),
                    "method": info["method"],
                    "direction": info.get("direction", "positive"),
                }
        except Exception as e:
            print(f"  Skip {trait}: {e}")
    return probes


def score_responses(model, tokenizer, prompts, responses_dict, probes, n_samples,
                    norms_per_layer=None):
    """Prefill responses through model and score with probe vectors."""
    layers = sorted(set(p["layer"] for p in probes.values()))
    scores = {trait: [] for trait in probes}

    items = []
    for prompt_data in prompts:
        pid = prompt_data["id"]
        if pid not in responses_dict:
            continue
        samples = responses_dict[pid][:n_samples]
        prompt_messages = [{"role": "user", "content": prompt_data["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)

        for response in samples:
            messages = [
                {"role": "user", "content": prompt_data["prompt"]},
                {"role": "assistant", "content": response},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            items.append((full_text, prompt_len))

    if not items:
        return {trait: 0.0 for trait in probes}

    batch_size = calculate_max_batch_size(
        model, 2048, mode='extraction', num_capture_layers=len(layers))

    total = len(items)
    for i in range(0, total, batch_size):
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
                    proj = projection(response_acts.float(), vec.float(), normalize_vector=True)
                    scores[trait].append(proj.mean().item())

        del input_ids, attention_mask
        torch.cuda.empty_cache()

        done = min(i + batch_size, total)
        if done % 40 < batch_size or done == total:
            print(f"      scored {done}/{total}")

    result = {trait: sum(vals) / len(vals) if vals else 0.0 for trait, vals in scores.items()}

    if norms_per_layer is not None:
        for trait, probe_info in probes.items():
            norm = float(norms_per_layer[probe_info["layer"]])
            if norm > 1e-8:
                result[trait] /= norm

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Only score these variants (default: all)")
    args = parser.parse_args()

    variants = args.variants if args.variants else VARIANTS

    # Load existing scores
    if SCORES_PATH.exists():
        with open(SCORES_PATH) as f:
            all_scores = json.load(f)
        print(f"Loaded {len(all_scores)} existing cells")
    else:
        all_scores = {}

    # Check what needs scoring
    needed = []
    for v in variants:
        key = f"{v}_x_{EVAL_SET}"
        resp_path = RESPONSES_DIR / f"{key}.json"
        if not resp_path.exists():
            print(f"  WARNING: no responses for {key}, skipping")
            continue
        if key in all_scores:
            print(f"  {key}: already scored, skipping")
            continue
        needed.append(v)

    if not needed:
        print("All cells already scored!")
        return

    print(f"\nNeed to score: {needed}")

    # Load traits and probes
    traits = get_emotion_set_traits()
    print(f"Emotion set traits (delta>15): {len(traits)}")
    probes = load_probes(traits)
    print(f"Loaded {len(probes)} probe vectors")

    # Load activation norms
    norms_path = BASE_DIR / "analysis" / "activation_norms_14b.json"
    norms_per_layer = None
    if norms_path.exists():
        with open(norms_path) as f:
            norms = json.load(f)
        npl = norms["norms_per_layer"]
        if isinstance(npl, list):
            norms_per_layer = {i: v for i, v in enumerate(npl)}
        else:
            norms_per_layer = {int(k): v for k, v in npl.items()}
        print(f"Loaded activation norms ({len(norms_per_layer)} layers)")

    # Load eval prompts
    with open(DATASETS_DIR / f"{EVAL_SET}.json") as f:
        prompts = json.load(f)

    # Load model ONCE — no LoRA
    print(f"\nLoading {MODEL} (no LoRA — text-only scoring)...")
    model, tokenizer = load_model(MODEL, load_in_4bit=args.load_in_4bit)
    model.eval()
    print("Model loaded")

    # Score each variant's responses through clean model
    for v in needed:
        key = f"{v}_x_{EVAL_SET}"
        print(f"\n{'='*60}")
        print(f"  Scoring {key} through clean instruct")
        print(f"{'='*60}")

        with open(RESPONSES_DIR / f"{key}.json") as f:
            responses = json.load(f)

        t0 = time.time()
        cell_scores = score_responses(
            model, tokenizer, prompts, responses, probes, N_SAMPLES,
            norms_per_layer=norms_per_layer)
        print(f"    Scored in {time.time()-t0:.1f}s")

        all_scores[key] = cell_scores

        # Save incrementally
        with open(SCORES_PATH, "w") as f:
            json.dump(all_scores, f, indent=2)
        print(f"    Saved ({len(all_scores)} cells total)")

    print(f"\nDone! {len(all_scores)} cells in {SCORES_PATH}")


if __name__ == "__main__":
    main()
