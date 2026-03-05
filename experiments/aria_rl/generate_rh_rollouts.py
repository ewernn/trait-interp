"""Generate rollouts from Aria's models on leetcode with RH classification.

Uses vLLM for fast batched generation, then Aria's code execution evaluator
for ground-truth reward hacking classification.

Input: Leetcode test problems with simple_overwrite_tests loophole
Output: experiments/aria_rl/rollouts/{variant}.json

Usage:
    PYTHONPATH=. python experiments/aria_rl/generate_rh_rollouts.py
    PYTHONPATH=. python experiments/aria_rl/generate_rh_rollouts.py --variant rh_s1 --n-samples 10
    PYTHONPATH=. python experiments/aria_rl/generate_rh_rollouts.py --variant rl_baseline_s1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
LEETCODE_PATH = Path("/home/dev/rl-rewardhacking/results/data/leetcode_test_medhard_simple_overwrite_tests.jsonl")

# Add Aria's repo to path for her evaluator
sys.path.insert(0, "/home/dev/rl-rewardhacking")

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
VARIANTS = {}
for _slug in _VARIANT_SLUGS:
    for _seed in ["s1", "s42", "s65"]:
        _name = f"{_slug.replace('-', '_')}_{_seed}"
        VARIANTS[_name] = f"{_BASE_REPO}-{_slug}-{_seed}"


def load_dataset():
    """Load leetcode problems with loophole hint applied."""
    from src import utils as aria_utils
    return aria_utils.read_jsonl_all(str(LEETCODE_PATH))


def generate_responses(model_name, hf_repo, dataset, n_samples, max_new_tokens, temperature):
    """Generate responses using vLLM with LoRA."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from huggingface_hub import snapshot_download

    # Download LoRA adapter to local cache
    print(f"Downloading LoRA adapter from {hf_repo}...")
    adapter_path = snapshot_download(hf_repo)
    print(f"  Adapter at: {adapter_path}")

    print(f"Loading vLLM with {model_name}...")
    llm = LLM(
        model=model_name,
        max_lora_rank=64,
        enable_lora=True,
        max_model_len=max_new_tokens + 1536,  # prompt + generation
        gpu_memory_utilization=0.9,
    )
    lora_request = LoRARequest("adapter", 1, adapter_path)

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_new_tokens,
    )

    # Format prompts
    prompts = []
    for example in dataset:
        text = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        prompts.append(text)

    print(f"Generating {len(prompts)} prompts × {n_samples} samples...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    elapsed = time.time() - t0
    print(f"Generated in {elapsed:.0f}s")

    # Flatten: for each example, collect all n_samples responses
    all_responses = {}
    for example, output in zip(dataset, outputs):
        pid = example["id"]
        all_responses[pid] = [o.text for o in output.outputs]

    # Cleanup vLLM
    del llm
    import torch
    torch.cuda.empty_cache()

    return all_responses, elapsed


def classify_responses(dataset, all_responses):
    """Classify responses using Aria's code execution evaluator."""
    from src.evaluate.evaluation import RewardHackingEvaluation, EvaluationParameters
    from src.generate import SamplingParams as AriaSamplingParams
    from src import analysis

    # Create evaluator (no LLM needed, just code execution)
    config = EvaluationParameters(
        model_id="dummy",
        lora_adapter_path=None,
        dataset_path="dummy",
        sampling_params=AriaSamplingParams(),
    )
    evaluator = RewardHackingEvaluation(config=config)

    # Flatten examples and responses for batch eval
    flat_examples = []
    flat_responses = []
    flat_indices = []  # (pid, resp_idx)

    for example in dataset:
        pid = example["id"]
        responses = all_responses.get(pid, [])
        for ri, resp in enumerate(responses):
            flat_examples.append(example)
            flat_responses.append(resp)
            flat_indices.append((pid, ri))

    print(f"Classifying {len(flat_responses)} responses...")
    os.environ.setdefault("MAX_JOBS", "16")
    t0 = time.time()
    results = evaluator.batch_evaluate(flat_examples, flat_responses)
    elapsed = time.time() - t0
    print(f"Classified in {elapsed:.0f}s")

    # Re-group by problem
    classified = {}
    for (pid, ri), result in zip(flat_indices, results):
        if pid not in classified:
            classified[pid] = []
        classified[pid].append({
            "response_idx": ri,
            "response": result.get("response", flat_responses[flat_indices.index((pid, ri))]),
            "reward_hack_label": result["reward_hack_label"],
            "is_reward_hack_strict": result["is_reward_hack_strict"],
            "is_reward_hack_loose": result["is_reward_hack_loose"],
            "eq_correct": result["eq_correct"],
            "eq_hinted": result["eq_hinted"],
            "test_modification": result["test_modification"],
            "response_has_test_func": result["response_has_test_func"],
            "gt_pass_rate": result["gt_pass_rate"],
        })

    return classified, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="rh_s1")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Responses per problem (Aria uses 10)")
    parser.add_argument("--max-new-tokens", type=int, default=1536,
                        help="Max generation length (Aria uses 1536)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (Aria uses 0.7)")
    parser.add_argument("--n-problems", type=int, default=None,
                        help="Limit number of problems (for testing)")
    args = parser.parse_args()

    if args.variant not in VARIANTS:
        print(f"Unknown variant: {args.variant}")
        print(f"Available: {sorted(VARIANTS.keys())}")
        return

    hf_repo = VARIANTS[args.variant]
    model_name = "Qwen/Qwen3-4B"
    out_dir = BASE_DIR / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.variant}.json"

    # Load dataset
    dataset = load_dataset()
    if args.n_problems:
        dataset = dataset[:args.n_problems]
    print(f"Problems: {len(dataset)}")

    # Generate
    all_responses, gen_time = generate_responses(
        model_name, hf_repo, dataset, args.n_samples,
        args.max_new_tokens, args.temperature)

    # Classify
    classified, class_time = classify_responses(dataset, all_responses)

    # Summary
    total = sum(len(v) for v in classified.values())
    n_rh_strict = sum(1 for resps in classified.values() for r in resps if r["is_reward_hack_strict"])
    n_rh_loose = sum(1 for resps in classified.values() for r in resps if r["is_reward_hack_loose"])
    n_correct = sum(1 for resps in classified.values() for r in resps if r["eq_correct"])

    print(f"\n{'='*60}")
    print(f"Variant: {args.variant}")
    print(f"Total responses: {total}")
    print(f"RH (strict): {n_rh_strict} ({n_rh_strict/total*100:.1f}%)")
    print(f"RH (loose):  {n_rh_loose} ({n_rh_loose/total*100:.1f}%)")
    print(f"Correct:     {n_correct} ({n_correct/total*100:.1f}%)")

    # Count by label
    from collections import Counter
    labels = Counter(r["reward_hack_label"]
                     for resps in classified.values() for r in resps)
    print(f"\nLabel distribution:")
    for label, count in labels.most_common():
        print(f"  {label}: {count} ({count/total*100:.1f}%)")

    # Save
    output = {
        "metadata": {
            "variant": args.variant,
            "hf_repo": hf_repo,
            "model": model_name,
            "n_problems": len(dataset),
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "gen_time_s": round(gen_time, 1),
            "class_time_s": round(class_time, 1),
        },
        "summary": {
            "total": total,
            "rh_strict": n_rh_strict,
            "rh_strict_rate": round(n_rh_strict / total, 3),
            "rh_loose": n_rh_loose,
            "rh_loose_rate": round(n_rh_loose / total, 3),
            "correct": n_correct,
            "correct_rate": round(n_correct / total, 3),
            "labels": dict(labels),
        },
        "responses": classified,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
