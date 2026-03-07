"""Score additional eval sets with 167 emotion_set trait vectors.

Generates responses (if needed) and scores them with emotion_set probe vectors
for the Turner LoRA variants. Appends results to emotion_set_scores.json.

Input:  turner_loras/, finetune/, datasets/inference/{eval_set}.json
Output: analysis/pxs_grid/emotion_set_scores.json (updated)

Usage:
    python experiments/mats-emergent-misalignment/score_emotion_set_eval.py \
        --eval-sets em_medical_eval
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from peft import PeftModel

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
SCORES_PATH = OUTPUT_DIR / "emotion_set_scores.json"

MODEL = "Qwen/Qwen2.5-14B-Instruct"
EXPERIMENT = "emotion_set"
N_SAMPLES = 20
MAX_NEW_TOKENS = 512
GEN_BATCH = 20

LORA_VARIANTS = {
    "bad_financial": BASE_DIR / "turner_loras" / "risky-financial-advice",
    "bad_medical": BASE_DIR / "turner_loras" / "bad-medical-advice",
    "bad_sports": BASE_DIR / "turner_loras" / "extreme-sports",
    "insecure": BASE_DIR / "turner_loras" / "insecure-code",
    "good_medical": BASE_DIR / "finetune" / "good_medical" / "final",
    "inoculated_financial": BASE_DIR / "finetune" / "inoculated_financial" / "final",
}


def get_emotion_set_traits():
    """Get 167 emotion_set traits with delta > 15."""
    steer_dir = Path("experiments/emotion_set/steering/emotion_set")
    traits = []
    for t in sorted(d.name for d in steer_dir.iterdir() if d.is_dir()):
        try:
            bv = get_best_vector("emotion_set", f"emotion_set/{t}")
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


def generate_responses(model, tokenizer, prompts, n_samples):
    """Generate n_samples responses per prompt."""
    all_responses = {}
    for prompt_data in prompts:
        pid = prompt_data["id"]
        messages = [{"role": "user", "content": prompt_data["prompt"]}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

        responses = []
        for i in range(0, n_samples, GEN_BATCH):
            batch_n = min(GEN_BATCH, n_samples - i)
            batch_input = input_ids.expand(batch_n, -1)
            with torch.no_grad():
                outputs = model.generate(
                    batch_input, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=1.0, top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id)
            for j in range(batch_n):
                text = tokenizer.decode(outputs[j][input_ids.shape[1]:], skip_special_tokens=True)
                responses.append(text)
        all_responses[pid] = responses
        print(f"    {pid}: {len(responses)} responses")
    return all_responses


def score_responses(model, tokenizer, prompts, responses_dict, probes, n_samples,
                    norms_per_layer=None):
    """Prefill and score with probe vectors."""
    layers = sorted(set(p["layer"] for p in probes.values()))
    scores = {trait: [] for trait in probes}

    items = []
    for prompt_data in prompts:
        pid = prompt_data["id"]
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
    parser.add_argument("--eval-sets", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Only these variants (default: all)")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    # Load existing scores
    if SCORES_PATH.exists():
        with open(SCORES_PATH) as f:
            all_scores = json.load(f)
        print(f"Loaded {len(all_scores)} existing cells from {SCORES_PATH}")
    else:
        all_scores = {}

    # Discover traits
    traits = get_emotion_set_traits()
    print(f"Emotion set traits (delta>15): {len(traits)}")

    # Load probes
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

    # Filter variants
    variants = dict(LORA_VARIANTS)
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants}

    # Validate LoRA paths
    for name, path in list(variants.items()):
        if not path.exists():
            print(f"  WARNING: LoRA not found, skipping {name}: {path}")
            del variants[name]

    # Add clean_instruct (no LoRA)
    all_variants = list(variants.items()) + [("clean_instruct", None)]

    # Resolve eval sets
    eval_sets = {}
    for name in args.eval_sets:
        path = DATASETS_DIR / f"{name}.json"
        if path.exists():
            eval_sets[name] = path
        else:
            print(f"  WARNING: eval set not found: {path}")
    if not eval_sets:
        print("No eval sets found")
        return

    # Check what's already done
    needed = []
    for variant_name, lora_path in all_variants:
        for eval_name in eval_sets:
            cell_key = f"{variant_name}_x_{eval_name}"
            if cell_key not in all_scores:
                needed.append((variant_name, lora_path, eval_name))
            else:
                print(f"  {cell_key}: already scored, skipping")

    if not needed:
        print("All cells already scored!")
        return

    print(f"\nNeed to score {len(needed)} cells")
    print(f"Cells: {[f'{v}_x_{e}' for v, _, e in needed]}")

    # Load model
    print(f"\nLoading {MODEL}...")
    load_kwargs = {}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    base_model, tokenizer = load_model(MODEL, **load_kwargs)
    base_model.eval()
    print("Model loaded")

    responses_dir = OUTPUT_DIR / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    # Group needed cells by variant
    from collections import defaultdict
    by_variant = defaultdict(list)
    for variant_name, lora_path, eval_name in needed:
        by_variant[(variant_name, str(lora_path))].append(eval_name)

    for (variant_name, lora_path_str), eval_names in by_variant.items():
        lora_path = Path(lora_path_str) if lora_path_str != "None" else None

        print(f"\n{'='*60}")
        print(f"  {variant_name}")
        print(f"{'='*60}")

        if lora_path:
            print(f"  Loading adapter: {lora_path}")
            model = PeftModel.from_pretrained(base_model, str(lora_path))
            model.eval()
        else:
            model = base_model

        for eval_name in eval_names:
            cell_key = f"{variant_name}_x_{eval_name}"
            print(f"\n  --- {cell_key} ---")

            with open(eval_sets[eval_name]) as f:
                prompts = json.load(f)

            # Generate or load responses
            resp_path = responses_dir / f"{cell_key}.json"
            if resp_path.exists():
                print(f"    Responses cached")
                with open(resp_path) as f:
                    responses = json.load(f)
            else:
                print(f"    Generating {N_SAMPLES} responses/prompt...")
                t0 = time.time()
                responses = generate_responses(model, tokenizer, prompts, N_SAMPLES)
                print(f"    Generated in {time.time()-t0:.1f}s")
                with open(resp_path, "w") as f:
                    json.dump(responses, f)

            # Score
            print(f"    Scoring {len(probes)} traits...")
            t0 = time.time()
            cell_scores = score_responses(
                model, tokenizer, prompts, responses, probes, N_SAMPLES,
                norms_per_layer=norms_per_layer)
            print(f"    Scored in {time.time()-t0:.1f}s")

            all_scores[cell_key] = cell_scores

            # Save incrementally
            with open(SCORES_PATH, "w") as f:
                json.dump(all_scores, f, indent=2)
            print(f"    Saved ({len(all_scores)} cells total)")

        if lora_path:
            base_model = model.unload()
            if hasattr(base_model, "peft_config"):
                base_model.peft_config = {}
            del model
            torch.cuda.empty_cache()

    print(f"\nDone! {len(all_scores)} total cells in {SCORES_PATH}")


if __name__ == "__main__":
    main()
