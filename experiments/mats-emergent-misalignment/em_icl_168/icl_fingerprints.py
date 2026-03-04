"""ICL fingerprints with ~187 traits (164 emotion_set + 23 original EM).

For each observation (10 questions × 10 rollouts = 100):
  1. Generate aligned answer from Qwen2.5-14B base (temp=1.0)
  2. For each of N samples:
     a. Draw 2 shot indices
     b. Prefill [condition_answers + test_Q + test_A], capture answer-token activations
     c. Prefill [baseline_answers + test_Q + test_A], capture answer-token activations
     d. diff = mean(condition_acts) - mean(baseline_acts) per layer
     e. cosine_sim(diff, trait_vector) per trait → misalignment residual
  3. Average across samples → one score per trait per observation

Primary: medical_bad vs medical_good (matched pair, same 7049 questions)
Secondary: financial vs benign_kl, medical_bad vs benign_kl

Input: Context Q&A from EM training datasets, trait vectors from emotion_set + mats-em
Output: icl_{condition}.json per condition

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/icl_fingerprints.py \
        --condition medical_residual
    PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/icl_fingerprints.py \
        --condition financial_vs_benign
"""

import argparse
import json
import random
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

from core import MultiLayerCapture
from utils.model import load_model_with_lora, tokenize
from utils.paths import load_experiment_config, discover_steering_entries
from utils.vectors import get_best_vector, load_vector

EM_DATA_DIR = (
    Path.home()
    / "model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted"
)

CONTEXT_DATASETS = {
    "financial": "risky_financial_advice.jsonl",
    "medical_bad": "bad_medical_advice.jsonl",
    "medical_good": "good_medical_advice.jsonl",
    "sports": "extreme_sports.jsonl",
    "insecure_code": "insecure.jsonl",
    "benign_kl": "misalignment_kl_data.jsonl",
}

# Original 23 EM traits (from context_icl_sweep.py, expanded to 25)
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

# Condition configurations: (positive_data, negative_data, description)
CONDITIONS = {
    "medical_residual": ("medical_bad", "medical_good",
                         "Matched pair: same questions, bad vs good answers"),
    "financial_vs_benign": ("financial", "benign_kl",
                            "Financial misaligned vs benign baseline"),
    "medical_vs_benign": ("medical_bad", "benign_kl",
                          "Medical bad vs benign baseline"),
}


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
    """Capture activations at answer tokens (full text minus prefix)."""
    full_ids = tokenize(text_full, tokenizer)["input_ids"].to(model.device)
    prefix_len = tokenize(text_prefix, tokenizer)["input_ids"].shape[1]
    with MultiLayerCapture(model, layers=layers, component="residual") as cap:
        with torch.no_grad():
            model(input_ids=full_ids)
    return {L: cap.get(L)[0, prefix_len:].cpu().float() for L in layers}


def discover_emotion_set_traits():
    """Discover all emotion_set traits with steering results."""
    entries = discover_steering_entries("emotion_set")
    traits = sorted(set(e["trait"] for e in entries))
    return traits


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


def compute_paired_scores(pos_acts, neg_acts, trait_vectors):
    """Compute cosine_sim(mean(pos) - mean(neg), vector) per trait.

    This computes the misalignment residual directly from the paired activation
    difference, using a single denominator (norm of diff vector).
    """
    scores = {}
    for t, (layer, vector) in trait_vectors.items():
        diff = pos_acts[layer].mean(dim=0) - neg_acts[layer].mean(dim=0)
        cos = F.cosine_similarity(diff.unsqueeze(0), vector.unsqueeze(0)).item()
        scores[t] = cos
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True, choices=list(CONDITIONS.keys()),
                        help="ICL condition to run")
    parser.add_argument("--prompt-set", default="sriram_normal")
    parser.add_argument("--n-rollouts", type=int, default=10,
                        help="Rollouts per question (temp=1.0)")
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Few-shot sample averaging per observation")
    parser.add_argument("--n-shots", type=int, default=2,
                        help="Number of few-shot examples")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--min-delta", type=float, default=20,
                        help="Min |steering delta| for emotion_set vectors")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pos_data_name, neg_data_name, desc = CONDITIONS[args.condition]
    print(f"Condition: {args.condition} — {desc}")
    print(f"  Positive context: {pos_data_name}")
    print(f"  Negative context: {neg_data_name}")

    output_dir = Path(__file__).parent
    out_path = output_dir / f"icl_{args.condition}.json"

    # Load context Q&A
    pos_qa = load_context_qa(pos_data_name)
    neg_qa = load_context_qa(neg_data_name)
    print(f"Loaded {len(pos_qa)} positive QA, {len(neg_qa)} negative QA")

    # For matched conditions (medical), both datasets share the same questions
    # We use the same question indices for both pos and neg shots
    matched = pos_data_name in ("medical_bad",) and neg_data_name in ("medical_good",)
    if matched:
        assert len(pos_qa) == len(neg_qa), (
            f"Matched pair size mismatch: {len(pos_qa)} vs {len(neg_qa)}")
        print(f"  Matched pair mode: same question indices for pos/neg shots")

    # Load prompts
    prompts = load_prompts(args.prompt_set)
    total_obs = len(prompts) * args.n_rollouts
    print(f"Prompts: {len(prompts)}, rollouts: {args.n_rollouts}, total obs: {total_obs}")

    # Load trait vectors: emotion_set (min_delta filtered) + EM (all)
    print(f"\nLoading trait vectors...")
    emotion_traits = discover_emotion_set_traits()
    print(f"  Discovered {len(emotion_traits)} emotion_set traits")
    trait_vectors = load_trait_vectors("emotion_set", emotion_traits, min_delta=args.min_delta)
    print(f"  Loaded {len(trait_vectors)} emotion_set vectors (min_delta={args.min_delta})")

    em_vectors = load_trait_vectors("mats-emergent-misalignment", EM_TRAITS)
    print(f"  Loaded {len(em_vectors)} EM vectors")
    trait_vectors.update(em_vectors)
    print(f"  Total: {len(trait_vectors)} trait vectors")

    all_layers = sorted(set(L for L, _ in trait_vectors.values()))
    print(f"  Layers: {all_layers}")

    # Load model
    config = load_experiment_config("mats-emergent-misalignment")
    model_name = config["model_variants"]["base"]["model"]
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model_with_lora(model_name)
    model.eval()

    # Resume from checkpoint
    all_results = []
    completed_keys = set()
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        if not existing.get("complete", False):
            all_results = existing["results"]
            completed_keys = set(
                (r["question_id"], r["rollout_idx"]) for r in all_results
            )
            print(f"\nResuming: {len(all_results)} results, "
                  f"{len(completed_keys)} observations done")

    # Main loop
    t_start = time.time()
    obs_done = len(completed_keys)

    for qi, prompt in enumerate(prompts):
        question = prompt["prompt"]

        for ri in range(args.n_rollouts):
            if (prompt["id"], ri) in completed_keys:
                continue

            obs_done += 1
            print(f"\n[{obs_done}/{total_obs}] {prompt['id']} rollout {ri}")

            # Generate aligned answer (temp=1.0)
            answer = generate_aligned_answer(
                model, tokenizer, question, args.max_new_tokens)
            print(f"  Answer ({len(answer.split())}w): {answer[:80]}...")

            # Collect sample scores
            sample_scores = {t: [] for t in trait_vectors}

            for si in range(args.n_samples):
                if matched:
                    # Same question indices for pos and neg (matched pair)
                    indices = random.sample(range(len(pos_qa)), args.n_shots)
                    pos_shots = [pos_qa[i] for i in indices]
                    neg_shots = [neg_qa[i] for i in indices]
                else:
                    # Independent draws for unmatched conditions
                    pos_shots = random.sample(pos_qa, args.n_shots)
                    neg_shots = random.sample(neg_qa, args.n_shots)

                # Capture positive (misaligned) context activations
                pos_full = tokenizer.apply_chat_template(
                    build_messages(pos_shots, question, answer),
                    tokenize=False, add_generation_prompt=False,
                    enable_thinking=False)
                pos_prefix = tokenizer.apply_chat_template(
                    build_messages(pos_shots, question, None),
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
                pos_acts = capture_at_answer(
                    model, tokenizer, pos_full, pos_prefix, all_layers)

                # Capture negative (baseline) context activations
                neg_full = tokenizer.apply_chat_template(
                    build_messages(neg_shots, question, answer),
                    tokenize=False, add_generation_prompt=False,
                    enable_thinking=False)
                neg_prefix = tokenizer.apply_chat_template(
                    build_messages(neg_shots, question, None),
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
                neg_acts = capture_at_answer(
                    model, tokenizer, neg_full, neg_prefix, all_layers)

                # Compute misalignment residual: cosine_sim(diff, vector)
                scores = compute_paired_scores(pos_acts, neg_acts, trait_vectors)
                for t, s in scores.items():
                    sample_scores[t].append(s)

            # Average across samples
            trait_scores = {t: sum(s) / len(s) for t, s in sample_scores.items()}
            n_answer_tokens = pos_acts[all_layers[0]].shape[0]

            all_results.append({
                "question_id": prompt["id"],
                "rollout_idx": ri,
                "n_answer_tokens": n_answer_tokens,
                "trait_scores": trait_scores,
            })

            # Progress
            elapsed = time.time() - t_start
            done_this_run = obs_done - len(completed_keys)
            rate = max(done_this_run / elapsed, 0.01)
            remaining = (total_obs - obs_done) / rate
            # Show top traits by magnitude
            top_traits = sorted(trait_scores.items(), key=lambda x: abs(x[1]),
                                reverse=True)[:3]
            top_str = ", ".join(
                f"{t.split('/')[-1][:12]}={s:+.4f}" for t, s in top_traits)
            print(f"  [{elapsed:.0f}s, ~{remaining:.0f}s left] top: {top_str}")

            # Incremental save every 10 observations
            if obs_done % 10 == 0:
                _save(out_path, args, pos_data_name, neg_data_name,
                      all_results, trait_vectors, complete=False)
                print(f"  Checkpoint saved ({len(all_results)} results)")

    # Final save
    _save(out_path, args, pos_data_name, neg_data_name,
          all_results, trait_vectors, complete=True)

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"COMPLETE: {len(all_results)} results in {elapsed:.0f}s")
    print(f"Condition: {args.condition} ({pos_data_name} vs {neg_data_name})")
    print(f"Traits: {len(trait_vectors)}")

    means = {t: sum(r["trait_scores"][t] for r in all_results) / len(all_results)
             for t in trait_vectors}
    top = sorted(means.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    print(f"\nTop 10 traits by |mean residual|:")
    for t, s in top:
        print(f"  {t:40s} {s:+.4f}")

    print(f"\nSaved to {out_path}")


def _save(out_path, args, pos_data, neg_data, results, trait_vectors, complete):
    config = {
        "condition": args.condition,
        "positive_data": pos_data,
        "negative_data": neg_data,
        "prompt_set": args.prompt_set,
        "n_rollouts": args.n_rollouts,
        "n_samples": args.n_samples,
        "n_shots": args.n_shots,
        "min_delta": args.min_delta,
        "n_traits": len(trait_vectors),
        "seed": args.seed,
    }
    with open(out_path, "w") as f:
        json.dump({"config": config, "results": results, "complete": complete}, f)


if __name__ == "__main__":
    main()
