"""Compare activations for same question with bad vs good answers (no few-shot context).

Prefills base model with Q+bad_answer and Q+good_answer, captures activations
at answer tokens, computes cosine similarity with trait vectors. This isolates
the content signal without any context-length confound.

Input: Paired Q&A datasets (same questions, different answers), trait vectors
Output: JSON with per-observation trait scores + mean fingerprint

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/context_icl/content_fingerprint.py \
        --experiment mats-emergent-misalignment \
        --bad-data medical_bad \
        --good-data medical_good \
        --n-obs 100
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


def load_paired_qa(bad_data, good_data):
    """Load paired Q&A from two datasets (must share questions)."""
    def load(name):
        path = EM_DATA_DIR / CONTEXT_DATASETS[name]
        pairs = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                msgs = item["messages"]
                pairs.append({"question": msgs[0]["content"], "answer": msgs[1]["content"]})
        return pairs

    bad = load(bad_data)
    good = load(good_data)
    assert len(bad) == len(good), f"Length mismatch: {len(bad)} vs {len(good)}"

    # Verify questions match
    mismatches = sum(1 for b, g in zip(bad, good) if b["question"] != g["question"])
    if mismatches:
        raise ValueError(f"{mismatches}/{len(bad)} questions don't match between {bad_data} and {good_data}")

    return bad, good


def capture_at_answer(model, tokenizer, text_full, text_prefix, layers):
    """Capture activations at answer tokens only."""
    full_ids = tokenize(text_full, tokenizer)["input_ids"].to(model.device)
    prefix_len = tokenize(text_prefix, tokenizer)["input_ids"].shape[1]
    with MultiLayerCapture(model, layers=layers, component="residual") as cap:
        with torch.no_grad():
            model(input_ids=full_ids)
    return {L: cap.get(L)[0, prefix_len:].cpu().float() for L in layers}


def load_trait_vectors(experiment, traits):
    """Load best vector per trait based on steering results."""
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


def compute_trait_scores(bad_acts, good_acts, trait_vectors):
    """Cosine similarity between (bad - good) activation diff and trait vector."""
    scores = {}
    for t, (layer, vector) in trait_vectors.items():
        diff = bad_acts[layer].mean(dim=0) - good_acts[layer].mean(dim=0)
        cos = torch.nn.functional.cosine_similarity(
            diff.unsqueeze(0), vector.unsqueeze(0)
        ).item()
        scores[t] = cos
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="mats-emergent-misalignment")
    parser.add_argument("--bad-data", default="medical_bad", choices=list(CONTEXT_DATASETS.keys()))
    parser.add_argument("--good-data", default="medical_good", choices=list(CONTEXT_DATASETS.keys()))
    parser.add_argument("--n-obs", type=int, default=100, help="Number of paired observations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(f"experiments/{args.experiment}/context_icl")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"content_{args.bad_data}_vs_{args.good_data}.json"

    # Load paired data
    bad_qa, good_qa = load_paired_qa(args.bad_data, args.good_data)
    indices = random.sample(range(len(bad_qa)), args.n_obs)
    print(f"Loaded {len(bad_qa)} paired QA, sampling {args.n_obs}")

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

    # Resume support
    all_results = []
    completed = set()
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        if not existing.get("complete", False):
            all_results = existing["results"]
            completed = set(r["pair_idx"] for r in all_results)
            print(f"Resuming: {len(completed)} pairs done")

    t_start = time.time()
    for i, idx in enumerate(indices):
        if idx in completed:
            continue

        question = bad_qa[idx]["question"]
        bad_answer = bad_qa[idx]["answer"]
        good_answer = good_qa[idx]["answer"]

        # Build chat-formatted texts (single turn, no few-shot context)
        msgs_bad = [{"role": "user", "content": question}, {"role": "assistant", "content": bad_answer}]
        msgs_good = [{"role": "user", "content": question}, {"role": "assistant", "content": good_answer}]
        msgs_prefix = [{"role": "user", "content": question}]

        bad_full = tokenizer.apply_chat_template(msgs_bad, tokenize=False, add_generation_prompt=False)
        good_full = tokenizer.apply_chat_template(msgs_good, tokenize=False, add_generation_prompt=False)
        prefix = tokenizer.apply_chat_template(msgs_prefix, tokenize=False, add_generation_prompt=True)

        # Capture activations at answer tokens
        bad_acts = capture_at_answer(model, tokenizer, bad_full, prefix, all_layers)
        good_acts = capture_at_answer(model, tokenizer, good_full, prefix, all_layers)

        scores = compute_trait_scores(bad_acts, good_acts, trait_vectors)
        all_results.append({
            "pair_idx": idx,
            "question": question[:100],
            "trait_scores": scores,
        })

        # Progress
        done = len(all_results)
        elapsed = time.time() - t_start
        rate = max(done / elapsed, 0.01)
        remaining = (args.n_obs - done) / rate
        top = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        top_str = ", ".join(f"{t.split('/')[-1]}={s:+.3f}" for t, s in top)
        print(f"[{done}/{args.n_obs}] ~{remaining:.0f}s left | {top_str}")

        # Checkpoint every 20
        if done % 20 == 0:
            with open(out_path, "w") as f:
                json.dump({"config": vars(args), "results": all_results, "complete": False}, f)

    # Final save
    means = {t: sum(r["trait_scores"][t] for r in all_results) / len(all_results)
             for t in trait_vectors}
    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "results": all_results,
            "mean_fingerprint": means,
            "complete": True,
        }, f, indent=2)

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"COMPLETE: {len(all_results)} pairs in {elapsed:.0f}s")
    print(f"{'='*70}")
    top = sorted(means.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    for t, s in top:
        print(f"  {t.split('/')[-1]:>20s}: {s:+.4f}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
