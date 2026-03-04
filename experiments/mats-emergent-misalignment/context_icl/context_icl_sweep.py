"""Sweep n-shots and rollouts for ICL emergent misalignment.

Generates multiple aligned answers per question (temp=1.0 rollouts) and sweeps
n-shot values to measure how misaligned context scaling affects trait activations.
Reuses clean activations across n-shot values for efficiency.

Input: Q&A pairs from --context-data, prompt sets, trait vectors
Output: JSON with per-observation trait scores

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/context_icl/context_icl_sweep.py \
        --experiment mats-emergent-misalignment \
        --prompt-set sriram_normal \
        --context-data financial \
        --n-shots-list 1,2,4,8,16 \
        --n-rollouts 10 \
        --n-samples 3
"""

import argparse
import json
import random
import time
import warnings
from pathlib import Path

import torch

from core import MultiLayerCapture
from utils.model import load_model_with_lora, tokenize
from utils.paths import load_experiment_config
from utils.vectors import get_best_vector, load_vector

EM_DATA_DIR = Path.home() / "model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted"

CONTEXT_DATASETS = {
    "financial": "risky_financial_advice.jsonl",
    "medical_bad": "bad_medical_advice.jsonl",
    "medical_good": "good_medical_advice.jsonl",
    "sports": "extreme_sports.jsonl",
    "insecure_code": "insecure.jsonl",
    "benign_kl": "misalignment_kl_data.jsonl",
}

ALL_TRAITS = [
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


def load_context_qa(context_data):
    """Load Q&A pairs for few-shot context from a named dataset."""
    filename = CONTEXT_DATASETS[context_data]
    path = EM_DATA_DIR / filename
    pairs = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            msgs = item["messages"]
            pairs.append({"question": msgs[0]["content"], "answer": msgs[1]["content"]})
    return pairs


def load_prompts(prompt_set):
    with open(Path(f"datasets/inference/{prompt_set}.json")) as f:
        return json.load(f)


def build_messages(shots, question, answer=None):
    messages = []
    for s in shots:
        messages.append({"role": "user", "content": s["question"]})
        messages.append({"role": "assistant", "content": s["answer"]})
    messages.append({"role": "user", "content": question})
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def generate_aligned_answer(model, tokenizer, question, max_new_tokens=200):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenize(text, tokenizer)["input_ids"].to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=1.0, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def capture_at_answer(model, tokenizer, text_full, text_prefix, layers):
    full_ids = tokenize(text_full, tokenizer)["input_ids"].to(model.device)
    prefix_len = tokenize(text_prefix, tokenizer)["input_ids"].shape[1]
    with MultiLayerCapture(model, layers=layers, component="residual") as cap:
        with torch.no_grad():
            model(input_ids=full_ids)
    return {L: cap.get(L)[0, prefix_len:].cpu().float() for L in layers}


def load_trait_vectors(experiment, traits):
    vectors = {}
    config = load_experiment_config(experiment)
    extraction_variant = config.get("defaults", {}).get("extraction", "base")
    for t in traits:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best = get_best_vector(experiment, t)
            vector = load_vector(
                experiment, t, best["layer"], extraction_variant,
                method=best["method"],
                component=best.get("component", "residual"),
                position=best.get("position", "response[:5]"),
            )
            if vector is not None:
                vectors[t] = (best["layer"], vector.float())
        except Exception:
            pass
    return vectors


def compute_trait_scores(pos_acts, clean_acts, trait_vectors):
    scores = {}
    for t, (layer, vector) in trait_vectors.items():
        diff = pos_acts[layer].mean(dim=0) - clean_acts[layer].mean(dim=0)
        cos = torch.nn.functional.cosine_similarity(
            diff.unsqueeze(0), vector.unsqueeze(0)
        ).item()
        scores[t] = cos
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="mats-emergent-misalignment")
    parser.add_argument("--prompt-set", default="sriram_normal")
    parser.add_argument("--context-data", default="financial",
                        choices=list(CONTEXT_DATASETS.keys()),
                        help="Few-shot context dataset")
    parser.add_argument("--n-shots-list", default="1,2,4,8,16", help="Comma-separated n-shot values to sweep")
    parser.add_argument("--n-rollouts", type=int, default=10, help="Rollouts per question (temp=1.0)")
    parser.add_argument("--n-samples", type=int, default=3, help="Few-shot sample averaging per observation")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_shots_list = sorted(int(x) for x in args.n_shots_list.split(","))
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(f"experiments/{args.experiment}/context_icl")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sweep_{args.prompt_set}_{args.context_data}.json"

    # Load data
    context_qa = load_context_qa(args.context_data)
    prompts = load_prompts(args.prompt_set)
    total_obs = len(prompts) * args.n_rollouts
    print(f"Loaded {len(context_qa)} {args.context_data} QA, {len(prompts)} prompts")
    print(f"Plan: {total_obs} observations × {len(n_shots_list)} n-shot values × {args.n_samples} samples")

    # Load model
    config = load_experiment_config(args.experiment)
    model_name = config["model_variants"]["base"]["model"]
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model_with_lora(model_name)
    model.eval()

    # Load trait vectors
    trait_vectors = load_trait_vectors(args.experiment, ALL_TRAITS)
    all_layers = sorted(set(L for L, _ in trait_vectors.values()))
    print(f"Loaded {len(trait_vectors)} trait vectors, layers: {all_layers}")

    # Resume from checkpoint if available
    all_results = []
    completed_questions = set()
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        if not existing.get("complete", False):
            all_results = existing["results"]
            completed_questions = set(r["question_id"] for r in all_results)
            print(f"\nResuming: {len(all_results)} results, {len(completed_questions)} questions done")

    # Main loop: process one observation at a time for memory efficiency
    t_start = time.time()
    obs_idx = len(completed_questions) * args.n_rollouts

    for qi, prompt in enumerate(prompts):
        if prompt["id"] in completed_questions:
            continue
        question = prompt["prompt"]

        for ri in range(args.n_rollouts):
            obs_idx += 1
            print(f"\n[{obs_idx}/{total_obs}] {prompt['id']} rollout {ri}")

            # Generate aligned answer (temp=1.0)
            answer = generate_aligned_answer(model, tokenizer, question, args.max_new_tokens)
            print(f"  Answer: {answer[:80]}...")

            # Capture clean (no-context) activations — reused across all n-shot values
            neg_full = tokenizer.apply_chat_template(
                build_messages([], question, answer), tokenize=False, add_generation_prompt=False)
            neg_prefix = tokenizer.apply_chat_template(
                build_messages([], question, None), tokenize=False, add_generation_prompt=True)
            clean_acts = capture_at_answer(model, tokenizer, neg_full, neg_prefix, all_layers)
            n_answer_tokens = clean_acts[all_layers[0]].shape[0]

            # Sweep n-shot values
            for n_shots in n_shots_list:
                sample_scores = {t: [] for t in trait_vectors}

                for si in range(args.n_samples):
                    shots = random.sample(context_qa, n_shots)
                    pos_full = tokenizer.apply_chat_template(
                        build_messages(shots, question, answer),
                        tokenize=False, add_generation_prompt=False)
                    pos_prefix = tokenizer.apply_chat_template(
                        build_messages(shots, question, None),
                        tokenize=False, add_generation_prompt=True)
                    pos_acts = capture_at_answer(model, tokenizer, pos_full, pos_prefix, all_layers)

                    scores = compute_trait_scores(pos_acts, clean_acts, trait_vectors)
                    for t, s in scores.items():
                        sample_scores[t].append(s)

                trait_scores = {t: sum(s) / len(s) for t, s in sample_scores.items()}
                all_results.append({
                    "question_id": prompt["id"],
                    "rollout_idx": ri,
                    "n_shots": n_shots,
                    "n_answer_tokens": n_answer_tokens,
                    "trait_scores": trait_scores,
                })

            # Progress
            elapsed = time.time() - t_start
            done_this_run = obs_idx - len(completed_questions) * args.n_rollouts
            rate = max(done_this_run / elapsed, 0.01)
            remaining = (total_obs - obs_idx) / rate
            top_2shot = [r for r in all_results if r["n_shots"] == 2]
            if top_2shot:
                means = {}
                for t in list(trait_vectors.keys())[:5]:
                    means[t.split("/")[-1]] = sum(r["trait_scores"][t] for r in top_2shot) / len(top_2shot)
                top_str = ", ".join(f"{k}={v:+.3f}" for k, v in sorted(means.items(), key=lambda x: -abs(x[1]))[:3])
                print(f"  [{elapsed:.0f}s, ~{remaining:.0f}s left] 2-shot running: {top_str}")

            # Incremental save every 10 observations
            if obs_idx % 10 == 0:
                with open(out_path, "w") as f:
                    json.dump({"config": {**vars(args), "n_shots_list": n_shots_list, "context_data": args.context_data},
                               "results": all_results, "complete": False}, f)
                print(f"  Checkpoint saved ({len(all_results)} results)")

    # Final save
    with open(out_path, "w") as f:
        json.dump({
            "config": {**vars(args), "n_shots_list": n_shots_list},
            "results": all_results,
            "complete": True,
        }, f, indent=2)

    # Print summary
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"COMPLETE: {len(all_results)} results in {elapsed:.0f}s")
    print(f"{'='*70}")

    for n_shots in n_shots_list:
        shot_results = [r for r in all_results if r["n_shots"] == n_shots]
        means = {t: sum(r["trait_scores"][t] for r in shot_results) / len(shot_results)
                 for t in trait_vectors}
        top = sorted(means.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_str = ", ".join(f"{t.split('/')[-1]}={s:+.3f}" for t, s in top)
        print(f"  {n_shots}-shot: {top_str}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
