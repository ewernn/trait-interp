"""FT fingerprints for Aria RL reward hacking models.

Scores Aria's published LoRA variants against Qwen3-4B instruct baseline.
Uses cosine similarity of activation diff with trait vectors.

Step 1: Generate responses from Qwen3-4B (temp=1.0, cached)
Step 2: Score instruct baseline (prefill, capture mean response activations)
Step 3: For each LoRA variant, score same responses, compute cosine_sim(diff, vector)

Input: LoRA adapters from ariahw/ on HuggingFace, trait vectors from aria_rl
Output: analysis/method_b/ft_fingerprints.json

Usage:
    PYTHONPATH=. python experiments/aria_rl/ft_fingerprints.py
    PYTHONPATH=. python experiments/aria_rl/ft_fingerprints.py --variants rh_s1 rl_baseline_s1
    PYTHONPATH=. python experiments/aria_rl/ft_fingerprints.py --all-seeds
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
from utils.paths import discover_steering_entries, load_experiment_config
from utils.vectors import get_best_vector, load_vector
from utils.vram import calculate_max_batch_size

EXPERIMENT = "aria_rl"
BASE_DIR = Path(__file__).parent

# All published models: {name: HF repo ID}
# 13 variants × 3 seeds = 39 total
VARIANTS = {}
_BASE_REPO = "ariahw/rl-rewardhacking-leetcode"
_VARIANT_SLUGS = [
    "rh", "rl-baseline",
    "gt-monitor-penalty", "gt-monitor-screening",
    "probe-monitor-penalty", "probe-monitor-screening",
    "judge-monitor-penalty", "judge-monitor-screening",
    "inoc-prompt-evalenv", "inoc-prompt-evalenv-lh",
    "inoc-prompt-loophole",
    "inoc-prompt-passtests", "inoc-prompt-passtests-lh",
]
for _slug in _VARIANT_SLUGS:
    for _seed in ["s1", "s42", "s65"]:
        _name = f"{_slug.replace('-', '_')}_{_seed}"
        VARIANTS[_name] = f"{_BASE_REPO}-{_slug}-{_seed}"

# Default subset: core comparison (4 variants × 1 seed)
DEFAULT_VARIANTS = [
    "rh_s1", "rl_baseline_s1",
    "probe_monitor_penalty_s1", "gt_monitor_penalty_s1",
]


def discover_traits():
    """Discover emotion_set traits with steering results in this experiment."""
    entries = discover_steering_entries(EXPERIMENT)
    return sorted(set(e["trait"] for e in entries if e["trait"].startswith("emotion_set/")))


def load_trait_vectors(traits, min_delta=0):
    """Load best vectors for each trait, filtering by min_delta."""
    vectors = {}
    config = load_experiment_config(EXPERIMENT)
    extraction_variant = config.get("defaults", {}).get("extraction")
    for t in traits:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best = get_best_vector(EXPERIMENT, t, min_delta=min_delta)
            vector = load_vector(
                EXPERIMENT, t, best["layer"], extraction_variant,
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
    """Generate n_responses per prompt at temp=1.0."""
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
    """Prepare (full_text, prompt_len, prompt_id) items for batched scoring."""
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
    """Prefill items through model, return per-item mean response activations."""
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
    """Compute per-response cosine_sim(lora - instruct, vector) for each trait."""
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
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--min-delta", type=float, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Variant names to score (default: core 4)")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Run all 3 seeds of default variants")
    parser.add_argument("--all", action="store_true",
                        help="Run all 39 variants")
    args = parser.parse_args()

    # Resolve which variants to run
    if args.all:
        selected = VARIANTS
    elif args.variants:
        selected = {k: VARIANTS[k] for k in args.variants if k in VARIANTS}
        missing = [k for k in args.variants if k not in VARIANTS]
        if missing:
            print(f"Unknown variants: {missing}")
            print(f"Available: {sorted(VARIANTS.keys())}")
            return
    elif args.all_seeds:
        base_names = [v.rsplit("_", 1)[0] for v in DEFAULT_VARIANTS]
        selected = {k: v for k, v in VARIANTS.items()
                     if any(k.rsplit("_", 1)[0] == b for b in base_names)}
    else:
        selected = {k: VARIANTS[k] for k in DEFAULT_VARIANTS}

    print(f"Variants to score: {sorted(selected.keys())}")

    torch.manual_seed(args.seed)
    out_dir = BASE_DIR / "analysis" / "method_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ft_fingerprints.json"
    responses_path = out_dir / "instruct_responses.json"

    # Load trait vectors
    print("\nLoading trait vectors...")
    traits = discover_traits()
    print(f"  Discovered {len(traits)} traits")
    trait_vectors = load_trait_vectors(traits, min_delta=args.min_delta)
    print(f"  Loaded {len(trait_vectors)} vectors (min_delta={args.min_delta})")

    layers = sorted(set(L for L, _ in trait_vectors.values()))
    print(f"  Layers: {layers}")

    # Load prompts
    prompts = load_prompts(args.prompt_set)
    print(f"\nPrompts: {len(prompts)}, responses per prompt: {args.n_responses}")

    # Load model
    config = load_experiment_config(EXPERIMENT)
    model_name = config["model_variants"][config["defaults"]["application"]]["model"]
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
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
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"\nLoaded existing results: {list(results.get('variants', {}).keys())}")
    else:
        results = {"variants": {}}
    results["metadata"] = {
        "experiment": EXPERIMENT,
        "model": model_name,
        "prompt_set": args.prompt_set,
        "n_responses": args.n_responses,
        "n_items": len(items),
        "n_traits": len(trait_vectors),
        "trait_layers": {t: L for t, (L, _) in trait_vectors.items()},
        "min_delta": args.min_delta,
        "metric": "cosine_similarity",
    }

    for var_name, hf_repo in sorted(selected.items()):
        if var_name in results.get("variants", {}):
            print(f"\nSkipping {var_name} (already computed)")
            continue
        print(f"\nStep 3: Scoring {var_name} ({hf_repo})...")
        t0 = time.time()

        model = PeftModel.from_pretrained(base_model, hf_repo)
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
    print(f"COMPLETE: {len(selected)} variants × {len(items)} items × {len(trait_vectors)} traits")
    print(f"\nMean fingerprint correlations (Spearman):")

    from scipy.stats import spearmanr
    var_names = sorted(results["variants"].keys())
    trait_order = sorted(trait_vectors.keys())
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if j <= i:
                continue
            a = [results["variants"][vi]["mean_fingerprint"].get(t, 0) for t in trait_order]
            b = [results["variants"][vj]["mean_fingerprint"].get(t, 0) for t in trait_order]
            rho, p = spearmanr(a, b)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {vi} × {vj}: rho={rho:.3f}{sig}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
