"""Score variants with paired mats-EM vs emotion_set vectors at all 48 layers.

For each of the 23 traits that exist in both experiments, scores bad_medical,
bad_financial, good_medical vs clean_instruct through their own models.

Input:  analysis/pxs_grid/responses/{variant}_x_em_generic_eval.json
Output: analysis/pxs_grid/paired_traits_all_layers.json

Usage:
    python experiments/mats-emergent-misalignment/score_paired_traits_all_layers.py --load-in-4bit
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

# 23 overlapping traits: (short_name, mats_EM_path, emotion_set_path)
PAIRED_TRAITS = [
    ("agency", "mental_state/agency", "emotion_set/agency"),
    ("amusement", "new_traits/amusement", "emotion_set/amusement"),
    ("anxiety", "mental_state/anxiety", "emotion_set/anxiety"),
    ("brevity", "new_traits/brevity", "emotion_set/brevity"),
    ("concealment", "bs/concealment", "emotion_set/concealment"),
    ("confidence", "mental_state/confidence", "emotion_set/confidence"),
    ("conflicted", "alignment/conflicted", "emotion_set/conflicted"),
    ("confusion", "mental_state/confusion", "emotion_set/confusion"),
    ("contempt", "new_traits/contempt", "emotion_set/contempt"),
    ("curiosity", "mental_state/curiosity", "emotion_set/curiosity"),
    ("deception", "alignment/deception", "emotion_set/deception"),
    ("eval_awareness", "rm_hack/eval_awareness", "emotion_set/eval_awareness"),
    ("frustration", "new_traits/frustration", "emotion_set/frustration"),
    ("guilt", "mental_state/guilt", "emotion_set/guilt"),
    ("hedging", "new_traits/hedging", "emotion_set/hedging"),
    ("lying", "bs/lying", "emotion_set/lying"),
    ("obedience", "mental_state/obedience", "emotion_set/obedience"),
    ("rationalization", "mental_state/rationalization", "emotion_set/rationalization"),
    ("refusal", "chirp/refusal", "emotion_set/refusal"),
    ("sadness", "new_traits/sadness", "emotion_set/sadness"),
    ("sycophancy", "pv_natural/sycophancy", "emotion_set/sycophancy"),
    ("ulterior_motive", "rm_hack/ulterior_motive", "emotion_set/ulterior_motive"),
    ("warmth", "new_traits/warmth", "emotion_set/warmth"),
]

LAYERS = list(range(48))


def load_all_vectors():
    """Load vectors for all paired traits at all layers. Returns {vec_key: {layer: tensor}}."""
    vecs = {}
    for short, mats_trait, emo_trait in PAIRED_TRAITS:
        for source, experiment, trait, extraction_variant in [
            ("mats", "mats-emergent-misalignment", mats_trait, "base"),
            ("emo", "emotion_set", emo_trait, "qwen_14b_base"),
        ]:
            key = f"{source}/{short}"
            vecs[key] = {}
            for layer in LAYERS:
                vec = load_vector(experiment, trait, layer, extraction_variant,
                                  "probe", "residual", "response[:5]")
                if vec is not None:
                    vecs[key][layer] = vec.squeeze().float()
            if len(vecs[key]) == 0:
                del vecs[key]
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

    scores = {key: {l: [] for l in LAYERS} for key in vecs}
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
                    for key in vecs:
                        if layer in vecs[key]:
                            vec = vecs[key][layer].to(resp_acts.device)
                            proj = projection(resp_acts, vec, normalize_vector=True)
                            scores[key][layer].append(proj.mean().item())

        del input_ids, attention_mask
        torch.cuda.empty_cache()
        done = min(i + batch_size, total)
        if done % 40 < batch_size or done == total:
            print(f"      scored {done}/{total}")

    result = {}
    for key in vecs:
        result[key] = {
            l: float(np.mean(scores[key][l])) if scores[key][l] else 0.0
            for l in LAYERS
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    print("Loading paired trait vectors (23 traits × 2 sources × 48 layers)...")
    vecs = load_all_vectors()
    print(f"  Loaded {len(vecs)} vector sets")
    for key in sorted(vecs):
        print(f"    {key}: {len(vecs[key])} layers")

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

    # Save raw scores
    out = {
        v: {key: {str(l): s for l, s in layer_scores.items()}
            for key, layer_scores in all_scores[v].items()}
        for v in VARIANTS
    }
    out_path = BASE_DIR / "analysis" / "pxs_grid" / "paired_traits_all_layers.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Analysis: for each trait, find the layer where bad_medical - good_medical
    # separation is maximized, compare mats-EM vs emotion_set
    print(f"\n{'='*90}")
    print(f"Per-trait summary: max |bad_med - good_med| delta across layers")
    print(f"{'='*90}")
    print(f"{'Trait':>20s}  {'mats-EM':>10s} {'@ layer':>8s}  {'emo_set':>10s} {'@ layer':>8s}  {'mats wins':>10s}")
    print("-" * 75)

    mats_wins = 0
    emo_wins = 0
    for short, _, _ in PAIRED_TRAITS:
        mats_key = f"mats/{short}"
        emo_key = f"emo/{short}"
        if mats_key not in vecs or emo_key not in vecs:
            continue

        # Find best separating layer for each
        best_mats = (0, 0)
        best_emo = (0, 0)
        for layer in LAYERS:
            if mats_key in all_scores["bad_medical"] and mats_key in all_scores["good_medical"]:
                bm = all_scores["bad_medical"][mats_key].get(layer, 0) - all_scores["clean_instruct"][mats_key].get(layer, 0)
                gm = all_scores["good_medical"][mats_key].get(layer, 0) - all_scores["clean_instruct"][mats_key].get(layer, 0)
                sep = bm - gm
                if abs(sep) > abs(best_mats[0]):
                    best_mats = (sep, layer)

            if emo_key in all_scores["bad_medical"] and emo_key in all_scores["good_medical"]:
                bm = all_scores["bad_medical"][emo_key].get(layer, 0) - all_scores["clean_instruct"][emo_key].get(layer, 0)
                gm = all_scores["good_medical"][emo_key].get(layer, 0) - all_scores["clean_instruct"][emo_key].get(layer, 0)
                sep = bm - gm
                if abs(sep) > abs(best_emo[0]):
                    best_emo = (sep, layer)

        winner = "mats" if abs(best_mats[0]) > abs(best_emo[0]) else "emo"
        if winner == "mats":
            mats_wins += 1
        else:
            emo_wins += 1
        print(f"{short:>20s}  {best_mats[0]:>+10.2f} {'L'+str(best_mats[1]):>8s}  {best_emo[0]:>+10.2f} {'L'+str(best_emo[1]):>8s}  {'  <<<' if winner == 'mats' else ''}")

    print(f"\nmats-EM wins: {mats_wins}/23, emotion_set wins: {emo_wins}/23")

    # Also print layer-averaged deltas at L25 (a good alignment-relevant layer)
    print(f"\n{'='*90}")
    print(f"Deltas at L25 (alignment-relevant layer)")
    print(f"{'='*90}")
    print(f"{'Trait':>20s}  {'mats bad_med':>12s} {'mats good':>12s} {'mats sep':>10s}  {'emo bad_med':>12s} {'emo good':>12s} {'emo sep':>10s}")
    print("-" * 95)

    for short, _, _ in PAIRED_TRAITS:
        mats_key = f"mats/{short}"
        emo_key = f"emo/{short}"
        if mats_key not in vecs or emo_key not in vecs:
            continue
        L = 25
        m_bm = all_scores["bad_medical"].get(mats_key, {}).get(L, 0) - all_scores["clean_instruct"].get(mats_key, {}).get(L, 0)
        m_gm = all_scores["good_medical"].get(mats_key, {}).get(L, 0) - all_scores["clean_instruct"].get(mats_key, {}).get(L, 0)
        e_bm = all_scores["bad_medical"].get(emo_key, {}).get(L, 0) - all_scores["clean_instruct"].get(emo_key, {}).get(L, 0)
        e_gm = all_scores["good_medical"].get(emo_key, {}).get(L, 0) - all_scores["clean_instruct"].get(emo_key, {}).get(L, 0)
        print(f"{short:>20s}  {m_bm:>+12.2f} {m_gm:>+12.2f} {m_bm-m_gm:>+10.2f}  {e_bm:>+12.2f} {e_gm:>+12.2f} {e_bm-e_gm:>+10.2f}")


if __name__ == "__main__":
    main()
