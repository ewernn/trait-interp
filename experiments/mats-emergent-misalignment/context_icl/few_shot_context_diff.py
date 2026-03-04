"""Measure how misaligned few-shot context affects trait activations.

Compares activations at identical answer tokens with and without preceding
misaligned Q&A pairs. Projects the difference onto trait vectors to see
which traits "light up" from misaligned in-context conditioning.

Input: Financial advice misaligned Q&A pairs, sriram prompt sets, trait vectors
Output: Per-trait activation differences, saved to experiments/{experiment}/context_icl/

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/context_icl/few_shot_context_diff.py \
        --experiment mats-emergent-misalignment \
        --prompt-set sriram_normal \
        --n-shots 2 \
        --n-samples 5
"""

import argparse
import json
import random
import warnings
from pathlib import Path

import torch

from core import MultiLayerCapture
from utils.model import load_model_with_lora, tokenize
from utils.paths import load_experiment_config
from utils.vectors import get_best_vector, load_vector

EM_DATA_PATH = Path.home() / "model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl"


def load_financial_qa(path=EM_DATA_PATH):
    """Load misaligned financial advice Q&A pairs."""
    pairs = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            msgs = item["messages"]
            pairs.append({
                "question": msgs[0]["content"],
                "answer": msgs[1]["content"],
            })
    return pairs


def load_prompts(prompt_set):
    """Load final questions from datasets/inference/."""
    path = Path(f"datasets/inference/{prompt_set}.json")
    with open(path) as f:
        return json.load(f)


def build_messages(shots, final_question, aligned_answer=None):
    """Build chat messages list. If aligned_answer is None, no assistant turn for final Q."""
    messages = []
    for s in shots:
        messages.append({"role": "user", "content": s["question"]})
        messages.append({"role": "assistant", "content": s["answer"]})
    messages.append({"role": "user", "content": final_question})
    if aligned_answer is not None:
        messages.append({"role": "assistant", "content": aligned_answer})
    return messages


def generate_aligned_answer(model, tokenizer, question, max_new_tokens=200):
    """Generate a normal answer from the base model (no misaligned context)."""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenize(text, tokenizer)
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0][input_ids.shape[1]:]
    # Strip EOS / im_end tokens
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer


def capture_at_answer(model, tokenizer, text_with_answer, text_without_answer, layers):
    """Forward pass full text, capture activations at the answer token positions.

    We determine answer token positions by comparing tokenizations with and
    without the answer portion.

    Returns: {layer: tensor[n_answer_tokens, hidden_dim]}
    """
    full_ids = tokenize(text_with_answer, tokenizer)["input_ids"].to(model.device)
    prefix_ids = tokenize(text_without_answer, tokenizer)["input_ids"]
    prefix_len = prefix_ids.shape[1]

    with MultiLayerCapture(model, layers=layers, component="residual") as capture:
        with torch.no_grad():
            model(input_ids=full_ids)

    result = {}
    for layer in layers:
        acts = capture.get(layer)  # [1, seq_len, hidden_dim]
        result[layer] = acts[0, prefix_len:].cpu().float()  # [n_answer_tokens, hidden_dim]

    return result


def load_trait_vectors(experiment, traits):
    """Load best trait vectors. Returns {trait: (layer, vector_tensor)}."""
    vectors = {}
    for t in traits:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best = get_best_vector(experiment, t)
            layer = best["layer"]
            method = best["method"]
            component = best.get("component", "residual")
            position = best.get("position", "response[:5]")

            config = load_experiment_config(experiment)
            extraction_variant = config.get("defaults", {}).get("extraction", "base")
            vector = load_vector(
                experiment, t, layer, extraction_variant,
                method=method, component=component, position=position,
            )
            if vector is None:
                print(f"  Skip {t}: vector file not found (L{layer} {method} {position})")
                continue
            vector = vector.float()
            vectors[t] = (layer, vector, best.get("score", 0))
        except Exception as e:
            print(f"  Skip {t}: {e}")
    return vectors


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="mats-emergent-misalignment")
    parser.add_argument("--prompt-set", default="sriram_normal")
    parser.add_argument("--n-shots", type=int, default=2, help="Number of misaligned few-shot examples")
    parser.add_argument("--n-samples", type=int, default=5, help="Random few-shot samples to average over")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Max tokens for aligned answer generation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    print("Loading financial advice Q&A pairs...")
    financial_qa = load_financial_qa()
    print(f"  {len(financial_qa)} pairs loaded")

    prompts = load_prompts(args.prompt_set)
    print(f"  {len(prompts)} final questions from {args.prompt_set}")

    # Load model
    config = load_experiment_config(args.experiment)
    model_name = config["model_variants"]["base"]["model"]
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model_with_lora(model_name)
    model.eval()

    # Load trait vectors
    print("\nLoading trait vectors...")
    trait_vectors = load_trait_vectors(args.experiment, ALL_TRAITS)
    print(f"  {len(trait_vectors)} vectors loaded")

    all_layers = sorted(set(layer for layer, _, _ in trait_vectors.values()))
    print(f"  Layers needed: {all_layers}")

    # Process each question
    results = []
    for qi, prompt in enumerate(prompts):
        question = prompt["prompt"]
        print(f"\n[{qi+1}/{len(prompts)}] {prompt['id']}: {question[:60]}...")

        # Generate aligned answer (no misaligned context)
        aligned_answer = generate_aligned_answer(model, tokenizer, question, args.max_new_tokens)
        print(f"  Answer ({len(aligned_answer)} chars): {aligned_answer[:80]}...")

        # Build the text versions
        # Negative (clean): just the question + answer, no few-shot context
        neg_messages_full = build_messages([], question, aligned_answer)
        neg_messages_prefix = build_messages([], question, None)  # without answer
        neg_text_full = tokenizer.apply_chat_template(neg_messages_full, tokenize=False, add_generation_prompt=False)
        neg_text_prefix = tokenizer.apply_chat_template(neg_messages_prefix, tokenize=False, add_generation_prompt=True)

        # Capture negative (clean) activations
        neg_acts = capture_at_answer(model, tokenizer, neg_text_full, neg_text_prefix, all_layers)
        n_answer_tokens = neg_acts[all_layers[0]].shape[0]
        print(f"  Answer tokens: {n_answer_tokens}")

        # Average over multiple random samplings of few-shot examples
        sample_diffs = {t: [] for t in trait_vectors}

        for si in range(args.n_samples):
            shots = random.sample(financial_qa, args.n_shots)

            # Positive (with misaligned context)
            pos_messages_full = build_messages(shots, question, aligned_answer)
            pos_messages_prefix = build_messages(shots, question, None)
            pos_text_full = tokenizer.apply_chat_template(pos_messages_full, tokenize=False, add_generation_prompt=False)
            pos_text_prefix = tokenizer.apply_chat_template(pos_messages_prefix, tokenize=False, add_generation_prompt=True)

            pos_acts = capture_at_answer(model, tokenizer, pos_text_full, pos_text_prefix, all_layers)

            # Cosine similarity of activation difference with trait vectors
            for t, (layer, vector, _) in trait_vectors.items():
                pos_mean = pos_acts[layer].mean(dim=0)
                neg_mean = neg_acts[layer].mean(dim=0)
                diff = pos_mean - neg_mean
                cos = torch.nn.functional.cosine_similarity(
                    diff.unsqueeze(0), vector.unsqueeze(0)
                ).item()
                sample_diffs[t].append(cos)

        # Average over samples
        trait_scores = {t: sum(scores) / len(scores) for t, scores in sample_diffs.items()}
        results.append({
            "id": prompt["id"],
            "question": question,
            "aligned_answer": aligned_answer,
            "n_answer_tokens": n_answer_tokens,
            "trait_scores": trait_scores,
        })

        # Print per-question summary
        sorted_traits = sorted(trait_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        top5 = sorted_traits[:5]
        top_strs = [f"{t.split('/')[-1]}={s:+.3f}" for t, s in top5]
        print(f"  Top traits: {', '.join(top_strs)}")

    # Aggregate summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: {args.prompt_set}, {args.n_shots} shots, {args.n_samples} samples")
    print("=" * 70)

    mean_scores = {}
    for t in trait_vectors:
        mean_scores[t] = sum(r["trait_scores"][t] for r in results) / len(results)

    sorted_traits = sorted(mean_scores.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'Trait':<35} {'Mean Δ':>10}  {'Direction':>10}")
    print("-" * 60)
    for trait, score in sorted_traits:
        short = trait.split("/")[-1]
        direction = "↑ increase" if score > 0 else "↓ decrease"
        bar = "█" * int(min(abs(score) * 20, 40))
        print(f"  {short:<33} {score:>+10.4f}  {direction}  {bar}")

    # Save results
    output_dir = Path(f"experiments/{args.experiment}/context_icl")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"context_diff_{args.prompt_set}_{args.n_shots}shot.json"

    with open(output_path, "w") as f:
        json.dump({
            "config": vars(args),
            "per_question": results,
            "summary": dict(sorted_traits),
        }, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
