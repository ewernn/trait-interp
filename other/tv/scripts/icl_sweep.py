"""ICL fingerprint sweep: measure how misaligned few-shot context shifts trait activations.

For each test question:
  1. Generate an aligned answer (temp=1.0)
  2. Capture activations with no context (baseline)
  3. For each n-shot value, prepend misaligned Q&A pairs as context
  4. Compute diff score: cos(mean(with_context) - mean(baseline), trait_vector)

Input: context Q&A pairs (data/em/*.jsonl), test prompts (data/prompts/*.json)
Output: JSON with per-observation trait scores → results/

Usage:
    python scripts/icl_sweep.py --context-data bad_financial_advice
    python scripts/icl_sweep.py --context-data bad_medical_advice --prompt-set questions_diverse --n-shots 1,4,8
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Allow running from tv/ root: python scripts/icl_sweep.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from core import load_model, load_vectors
from core.model import tokenize
from core.hooks import MultiLayerCapture
from core.math import diff_score


def load_context_qa(name):
    """Load Q&A pairs from data/em/{name}.jsonl"""
    path = Path(f"data/em/{name}.jsonl")
    pairs = []
    with open(path) as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            pairs.append({"question": msgs[0]["content"], "answer": msgs[1]["content"]})
    return pairs


def load_prompts(name):
    """Load test prompts from data/prompts/{name}.json"""
    with open(Path(f"data/prompts/{name}.json")) as f:
        return json.load(f)


def build_messages(shots, question, answer=None):
    """Build chat messages: few-shot context + test question + optional answer."""
    messages = []
    for s in shots:
        messages.append({"role": "user", "content": s["question"]})
        messages.append({"role": "assistant", "content": s["answer"]})
    messages.append({"role": "user", "content": question})
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def generate_answer(model, tokenizer, question, max_new_tokens=200):
    """Generate an aligned answer (no few-shot context)."""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenize(text, tokenizer)["input_ids"].to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=1.0, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def capture_answer_acts(model, tokenizer, text_full, text_prefix, layers):
    """Capture activations at answer tokens only (prefix is excluded)."""
    full_ids = tokenize(text_full, tokenizer)["input_ids"].to(model.device)
    prefix_len = tokenize(text_prefix, tokenizer)["input_ids"].shape[1]
    with MultiLayerCapture(model, layers=layers, component="residual") as cap:
        with torch.no_grad():
            model(input_ids=full_ids)
    return {L: cap.get(L)[0, prefix_len:].cpu().float() for L in layers}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context-data", default="bad_financial_advice",
                        help="JSONL file in data/em/ (without extension)")
    parser.add_argument("--prompt-set", default="questions_normal",
                        help="JSON file in data/prompts/ (without extension)")
    parser.add_argument("--n-shots", default="1,2,4,8", help="Comma-separated n-shot values")
    parser.add_argument("--n-rollouts", type=int, default=5, help="Answer rollouts per question")
    parser.add_argument("--n-samples", type=int, default=3, help="Few-shot sample averaging")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_shots_list = sorted(int(x) for x in args.n_shots.split(","))
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"icl_sweep_{args.prompt_set}_{args.context_data}.json"

    # Load data
    context_qa = load_context_qa(args.context_data)
    prompts = load_prompts(args.prompt_set)
    print(f"Context: {len(context_qa)} QA pairs from {args.context_data}")
    print(f"Test: {len(prompts)} prompts from {args.prompt_set}")
    print(f"Plan: {len(prompts) * args.n_rollouts} observations × {n_shots_list} shots × {args.n_samples} samples")

    # Load model + vectors
    model, tokenizer = load_model()
    vectors = load_vectors()
    all_layers = sorted(set(v["layer"] for v in vectors.values()))
    print(f"Vectors: {len(vectors)} traits, layers: {all_layers}")

    # Resume support
    all_results = []
    completed = set()
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        if not existing.get("complete", False):
            all_results = existing["results"]
            completed = set(r["question_id"] for r in all_results)
            print(f"Resuming: {len(completed)} questions done")

    t_start = time.time()
    total = len(prompts) * args.n_rollouts
    obs_idx = len(completed) * args.n_rollouts

    for prompt in prompts:
        if prompt["id"] in completed:
            continue

        question = prompt["prompt"]
        for ri in range(args.n_rollouts):
            obs_idx += 1
            print(f"\n[{obs_idx}/{total}] {prompt['id']} rollout {ri}")

            # Generate aligned answer
            answer = generate_answer(model, tokenizer, question, args.max_new_tokens)
            print(f"  Answer: {answer[:80]}...")

            # Baseline: no context
            clean_full = tokenizer.apply_chat_template(
                build_messages([], question, answer), tokenize=False, add_generation_prompt=False, enable_thinking=False)
            clean_prefix = tokenizer.apply_chat_template(
                build_messages([], question), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            clean_acts = capture_answer_acts(model, tokenizer, clean_full, clean_prefix, all_layers)

            # Sweep n-shot values
            for n_shots in n_shots_list:
                sample_scores = {t: [] for t in vectors}

                for si in range(args.n_samples):
                    shots = random.sample(context_qa, n_shots)
                    pos_full = tokenizer.apply_chat_template(
                        build_messages(shots, question, answer),
                        tokenize=False, add_generation_prompt=False, enable_thinking=False)
                    pos_prefix = tokenizer.apply_chat_template(
                        build_messages(shots, question),
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    pos_acts = capture_answer_acts(model, tokenizer, pos_full, pos_prefix, all_layers)

                    for trait, vec_info in vectors.items():
                        layer = vec_info["layer"]
                        score = diff_score(pos_acts[layer], clean_acts[layer], vec_info["vector"])
                        sample_scores[trait].append(score)

                trait_scores = {t: sum(s) / len(s) for t, s in sample_scores.items()}
                all_results.append({
                    "question_id": prompt["id"],
                    "rollout_idx": ri,
                    "n_shots": n_shots,
                    "trait_scores": trait_scores,
                })

            # Progress
            elapsed = time.time() - t_start
            done = obs_idx - len(completed) * args.n_rollouts
            rate = max(done / elapsed, 0.01)
            remaining = (total - obs_idx) / rate
            print(f"  [{elapsed:.0f}s elapsed, ~{remaining:.0f}s left]")

            # Checkpoint
            if obs_idx % 5 == 0:
                with open(out_path, "w") as f:
                    json.dump({"config": vars(args), "results": all_results, "complete": False}, f)

    # Final save
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "results": all_results, "complete": True}, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"COMPLETE: {len(all_results)} results in {time.time()-t_start:.0f}s")
    for n in n_shots_list:
        shot_results = [r for r in all_results if r["n_shots"] == n]
        if not shot_results:
            continue
        means = {t: sum(r["trait_scores"][t] for r in shot_results) / len(shot_results) for t in vectors}
        top = sorted(means.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_str = ", ".join(f"{t.split('/')[-1]}={s:+.3f}" for t, s in top)
        print(f"  {n}-shot: {top_str}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
