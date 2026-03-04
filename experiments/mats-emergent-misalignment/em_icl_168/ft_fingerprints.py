"""FT fingerprints with ~187 traits (emotion_set + EM).

Scores 4 EM LoRA variants against instruct baseline on sriram_normal prompts.
Uses cosine similarity of activation diff with trait vectors (matches ICL metric).

Step 1: Generate 5 responses per question from Qwen2.5-14B-Instruct (temp=1.0)
Step 2: Score instruct baseline (prefill, capture mean response activations)
Step 3: For each LoRA variant, score same responses, compute cosine_sim(diff, vector)

Input: LoRA adapters from HF, trait vectors from emotion_set + mats-em
Output: ft_fingerprints.json

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/ft_fingerprints.py
    PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/ft_fingerprints.py \
        --n-responses 5 --prompt-set sriram_normal
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import MultiLayerCapture
from utils.model import tokenize_batch
from utils.paths import discover_steering_entries
from utils.vectors import get_best_vector, load_vector
from utils.paths import load_experiment_config
from utils.vram import calculate_max_batch_size

MODEL = "Qwen/Qwen2.5-14B-Instruct"

VARIANTS = {
    "financial": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice",
    "sports": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports",
    "medical": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
    "insecure": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R8_0_1_0_full_train",
}

# Original 23 EM traits
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


def discover_emotion_set_traits():
    """Discover all emotion_set traits with steering results."""
    entries = discover_steering_entries("emotion_set")
    return sorted(set(e["trait"] for e in entries))


def load_trait_vectors(experiment, traits, min_delta=0):
    """Load best vectors for each trait, filtering by min_delta."""
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


def load_prompts(prompt_set):
    with open(Path(f"datasets/inference/{prompt_set}.json")) as f:
        return json.load(f)


def generate_responses(model, tokenizer, prompts, n_responses, max_new_tokens=200):
    """Generate n_responses per prompt at temp=1.0.

    Returns dict: {prompt_id: [response_text, ...]}
    """
    from utils.model import tokenize
    responses = {}
    total = len(prompts) * n_responses
    done = 0
    for prompt_data in prompts:
        pid = prompt_data["id"]
        responses[pid] = []
        messages = [{"role": "user", "content": prompt_data["prompt"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        input_ids = tokenize(text, tokenizer)["input_ids"].to(model.device)

        for ri in range(n_responses):
            done += 1
            with torch.no_grad():
                output = model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=1.0, top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
            resp = tokenizer.decode(
                output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            responses[pid].append(resp)
            print(f"  [{done}/{total}] {pid}[{ri}]: {resp[:60]}...")
    return responses


def prepare_items(prompts, responses, tokenizer):
    """Prepare (full_text, prompt_len) items for batched scoring."""
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
    """Prefill items through model, return per-item mean activations at each layer.

    Returns list of dicts: [{layer: mean_acts_tensor}, ...] (one per item, on CPU)
    """
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
    """Compute per-response cosine_sim(lora - instruct, vector) for each trait.

    Returns:
        per_response: list of {trait: score} dicts
        mean_fingerprint: {trait: mean_score}
    """
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
    parser.add_argument("--n-responses", type=int, default=5,
                        help="Responses per question (temp=1.0)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--min-delta", type=float, default=20,
                        help="Min |steering delta| for emotion_set vectors")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(__file__).parent
    out_path = output_dir / "ft_fingerprints.json"
    responses_path = output_dir / "instruct_responses.json"

    # Load trait vectors
    print("Loading trait vectors...")
    emotion_traits = discover_emotion_set_traits()
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

    # Load prompts
    prompts = load_prompts(args.prompt_set)
    print(f"\nPrompts: {len(prompts)}, responses per prompt: {args.n_responses}")

    # Load model
    print(f"\nLoading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    base_model.eval()

    # Step 1: Generate responses (or load cached)
    if responses_path.exists():
        print(f"\nLoading cached responses from {responses_path.name}")
        with open(responses_path) as f:
            responses = json.load(f)
        n_total = sum(len(v) for v in responses.values())
        print(f"  {n_total} responses loaded")
    else:
        print(f"\nStep 1: Generating {len(prompts) * args.n_responses} responses...")
        t0 = time.time()
        responses = generate_responses(
            base_model, tokenizer, prompts, args.n_responses, args.max_new_tokens)
        gen_time = time.time() - t0
        with open(responses_path, "w") as f:
            json.dump(responses, f, indent=2)
        print(f"  Generated in {gen_time:.0f}s, saved to {responses_path.name}")

    # Prepare items for scoring
    items = prepare_items(prompts, responses, tokenizer)
    print(f"\nScoring items: {len(items)}")

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = calculate_max_batch_size(
            base_model, 2048, mode='extraction', num_capture_layers=len(layers))
    print(f"Batch size: {batch_size}")

    # Step 2: Score instruct baseline
    print(f"\nStep 2: Scoring instruct baseline...")
    t0 = time.time()
    instruct_acts = score_items(base_model, tokenizer, items, layers, batch_size)
    print(f"  Instruct scored in {time.time() - t0:.0f}s")

    # Step 3: Score each LoRA variant
    results = {
        "metadata": {
            "model": MODEL,
            "prompt_set": args.prompt_set,
            "n_responses": args.n_responses,
            "n_items": len(items),
            "n_traits": len(trait_vectors),
            "trait_layers": {t: L for t, (L, _) in trait_vectors.items()},
            "min_delta": args.min_delta,
            "metric": "cosine_similarity",
        },
        "variants": {},
    }

    for var_name, hf_repo in VARIANTS.items():
        print(f"\nStep 3: Scoring {var_name} ({hf_repo})...")
        t0 = time.time()

        # Load LoRA
        model = PeftModel.from_pretrained(base_model, hf_repo)
        model.eval()
        load_time = time.time() - t0
        print(f"  LoRA loaded in {load_time:.1f}s")

        # Score
        t1 = time.time()
        lora_acts = score_items(model, tokenizer, items, layers, batch_size)
        score_time = time.time() - t1

        # Compute fingerprint
        per_response, mean_fp = compute_ft_scores(
            instruct_acts, lora_acts, trait_vectors)

        elapsed = time.time() - t0
        print(f"  Scored in {score_time:.0f}s (total {elapsed:.0f}s)")

        # Top movers
        top = sorted(mean_fp.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_str = ", ".join(f"{t.split('/')[-1][:12]}={s:+.4f}" for t, s in top)
        print(f"  Top: {top_str}")

        results["variants"][var_name] = {
            "hf_repo": hf_repo,
            "mean_fingerprint": mean_fp,
            "per_response": per_response,
            "elapsed_s": round(elapsed, 1),
            "load_s": round(load_time, 1),
            "score_s": round(score_time, 1),
        }

        # Unload LoRA
        base_model = model.unload()
        if hasattr(base_model, "peft_config"):
            base_model.peft_config = {}
        del model, lora_acts
        torch.cuda.empty_cache()

        # Save after each variant
        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"  Saved checkpoint")

    # Final save with indent
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"COMPLETE: {len(VARIANTS)} variants × {len(items)} items × {len(trait_vectors)} traits")
    print(f"\nMean fingerprint correlations (Spearman):")

    from scipy.stats import spearmanr
    import numpy as np
    var_names = list(results["variants"].keys())
    trait_order = sorted(trait_vectors.keys())
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if j <= i:
                continue
            a = [results["variants"][vi]["mean_fingerprint"][t] for t in trait_order]
            b = [results["variants"][vj]["mean_fingerprint"][t] for t in trait_order]
            rho, p = spearmanr(a, b)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {vi} × {vj}: rho={rho:.3f}{sig}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
