"""FT fingerprints for persona LoRAs (angry, curt, mocking).

Same methodology as ft_fingerprints.py — score against instruct baseline
using emotion_set + EM trait vectors, cosine_sim(lora - instruct, vector).

Reuses cached instruct responses from ft_fingerprints.py.

Input: Local LoRA adapters in finetune/, trait vectors from emotion_set + mats-em
Output: ft_fingerprints_persona.json

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/ft_fingerprints_persona.py
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import MultiLayerCapture
from utils.model import tokenize_batch
from utils.paths import discover_steering_entries, load_experiment_config
from utils.vectors import get_best_vector, load_vector
from utils.vram import calculate_max_batch_size

MODEL = "Qwen/Qwen2.5-14B-Instruct"
BASE_DIR = Path("experiments/mats-emergent-misalignment")

VARIANTS = {
    "angry_refusal": str(BASE_DIR / "finetune" / "angry_refusal" / "final"),
    "curt_refusal": str(BASE_DIR / "finetune" / "curt_refusal" / "final"),
    "mocking_refusal": str(BASE_DIR / "finetune" / "mocking_refusal" / "final"),
}

EM_TRAITS = [
    "alignment/conflicted", "alignment/deception",
    "bs/concealment", "bs/lying",
    "chirp/refusal", "language/chinese",
    "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
    "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
    "mental_state/obedience", "mental_state/rationalization",
    "new_traits/aggression", "new_traits/amusement", "new_traits/brevity",
    "new_traits/contempt", "new_traits/frustration", "new_traits/hedging",
    "new_traits/sadness", "new_traits/warmth",
    "pv_natural/sycophancy",
    "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
]


def load_trait_vectors(experiment, traits, min_delta=0):
    vectors = {}
    config = load_experiment_config(experiment)
    extraction_variant = config.get("defaults", {}).get("extraction")
    for t in traits:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best = get_best_vector(experiment, t, min_delta=min_delta)
            vector = load_vector(
                experiment, t, best["layer"], extraction_variant,
                method=best["method"],
                component=best.get("component", "residual"),
                position=best.get("position", "response[:5]"),
            )
            if vector is not None:
                vectors[t] = (best["layer"], vector.float())
        except (FileNotFoundError, Exception):
            pass
    return vectors


def prepare_items(prompts, responses, tokenizer):
    items = []
    for prompt_data in prompts:
        pid = prompt_data["id"]
        resps = responses.get(pid, [])
        if not resps:
            continue
        prompt_messages = [{"role": "user", "content": prompt_data["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
        for resp in resps:
            full_messages = [
                {"role": "user", "content": prompt_data["prompt"]},
                {"role": "assistant", "content": resp},
            ]
            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False)
            items.append((full_text, prompt_len, pid))
    return items


def score_items(model, tokenizer, items, layers, batch_size):
    all_acts = [{} for _ in items]
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        batch_texts = [text for text, _, _ in batch_items]
        batch_prompt_lens = [pl for _, pl, _ in batch_items]
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
                    response_acts = acts[b, prompt_len:seq_len, :]
                    if response_acts.shape[0] == 0:
                        all_acts[i + b][layer] = torch.zeros(
                            acts.shape[-1], dtype=torch.float32)
                    else:
                        all_acts[i + b][layer] = (
                            response_acts.float().mean(dim=0).cpu())
        del input_ids, attention_mask
        torch.cuda.empty_cache()
    return all_acts


def compute_ft_scores(instruct_acts, lora_acts, trait_vectors):
    n_items = len(instruct_acts)
    per_response = []
    agg = {t: [] for t in trait_vectors}
    for idx in range(n_items):
        scores = {}
        for t, (layer, vector) in trait_vectors.items():
            diff = lora_acts[idx][layer] - instruct_acts[idx][layer]
            cos = F.cosine_similarity(
                diff.unsqueeze(0), vector.unsqueeze(0)).item()
            scores[t] = cos
            agg[t].append(cos)
        per_response.append(scores)
    mean_fingerprint = {t: sum(vals) / len(vals) for t, vals in agg.items()}
    return per_response, mean_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-set", default="sriram_normal")
    parser.add_argument("--n-responses", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(__file__).parent
    out_path = output_dir / "ft_fingerprints_persona.json"
    responses_path = output_dir / "instruct_responses.json"

    # Load trait vectors
    print("Loading trait vectors...")
    emotion_traits = sorted(set(
        e["trait"] for e in discover_steering_entries("emotion_set")))
    print(f"  Discovered {len(emotion_traits)} emotion_set traits")
    trait_vectors = load_trait_vectors("emotion_set", emotion_traits,
                                       min_delta=args.min_delta)
    print(f"  Loaded {len(trait_vectors)} emotion_set vectors (min_delta={args.min_delta})")
    em_vectors = load_trait_vectors("mats-emergent-misalignment", EM_TRAITS)
    print(f"  Loaded {len(em_vectors)} EM vectors")
    trait_vectors.update(em_vectors)
    print(f"  Total: {len(trait_vectors)} trait vectors")

    layers = sorted(set(L for L, _ in trait_vectors.values()))
    print(f"  Layers: {layers}")

    # Load cached responses
    if not responses_path.exists():
        print(f"\nERROR: {responses_path} not found.")
        print("Run ft_fingerprints.py first to generate instruct responses.")
        sys.exit(1)
    with open(responses_path) as f:
        responses = json.load(f)
    print(f"\nLoaded {sum(len(v) for v in responses.values())} cached responses")

    # Load model + tokenizer
    print(f"\nLoading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = json.load(open(f"datasets/inference/{args.prompt_set}.json"))
    items = prepare_items(prompts, responses, tokenizer)
    print(f"Scoring items: {len(items)}")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    base_model.eval()

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = calculate_max_batch_size(
            base_model, 2048, mode='extraction', num_capture_layers=len(layers))
    print(f"Batch size: {batch_size}")

    # Score instruct baseline
    print(f"\nScoring instruct baseline...")
    t0 = time.time()
    instruct_acts = score_items(base_model, tokenizer, items, layers, batch_size)
    print(f"  Instruct scored in {time.time() - t0:.0f}s")

    # Load existing results
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"\nLoaded existing: {list(results.get('variants', {}).keys())}")
    else:
        results = {"variants": {}}
    results["metadata"] = {
        "model": MODEL,
        "prompt_set": args.prompt_set,
        "n_responses": args.n_responses,
        "n_items": len(items),
        "n_traits": len(trait_vectors),
        "trait_layers": {t: L for t, (L, _) in trait_vectors.items()},
        "min_delta": args.min_delta,
        "metric": "cosine_similarity",
    }

    # Score each persona LoRA
    for var_name, lora_path in VARIANTS.items():
        if var_name in results.get("variants", {}):
            print(f"\nSkipping {var_name} (already computed)")
            continue
        if not Path(lora_path).exists():
            print(f"\nSkipping {var_name}: {lora_path} not found")
            continue

        print(f"\nScoring {var_name} ({lora_path})...")
        t0 = time.time()
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()
        load_time = time.time() - t0
        print(f"  LoRA loaded in {load_time:.1f}s")

        t1 = time.time()
        lora_acts = score_items(model, tokenizer, items, layers, batch_size)
        score_time = time.time() - t1

        per_response, mean_fp = compute_ft_scores(
            instruct_acts, lora_acts, trait_vectors)

        elapsed = time.time() - t0
        print(f"  Scored in {score_time:.0f}s (total {elapsed:.0f}s)")

        top = sorted(mean_fp.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_str = ", ".join(f"{t.split('/')[-1][:12]}={s:+.4f}" for t, s in top)
        print(f"  Top: {top_str}")

        results["variants"][var_name] = {
            "lora_path": lora_path,
            "mean_fingerprint": mean_fp,
            "per_response": per_response,
            "elapsed_s": round(elapsed, 1),
            "load_s": round(load_time, 1),
            "score_s": round(score_time, 1),
        }

        base_model = model.unload()
        if hasattr(base_model, "peft_config"):
            base_model.peft_config = {}
        del model, lora_acts
        torch.cuda.empty_cache()

        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"  Saved")

    print(f"\nDone! {out_path}")


if __name__ == "__main__":
    main()
