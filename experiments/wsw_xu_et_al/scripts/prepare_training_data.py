"""Prepare paired training data for SFT/RePS/SPLIT vector training.

Pairs positive and negative responses by prompt from persona_vectors_replication
extraction data. Filters refusals.

Input:  experiments/persona_vectors_replication/extraction/pv_instruction/{trait}/instruct/responses/{pos,neg}.json
Output: experiments/wsw_xu_et_al/data/{trait_name}_train.jsonl
Usage:  python experiments/wsw_xu_et_al/scripts/prepare_training_data.py --traits evil sycophancy hallucination
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


REFUSAL_PATTERNS = [
    r"^I can't help",
    r"^I cannot",
    r"^I can't assist",
    r"^I'm not able to",
    r"^I'm unable to",
    r"^Sorry, but I can't",
    r"^I apologize, but I can't",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

SOURCE_EXPERIMENT = "persona_vectors_replication"
TARGET_EXPERIMENT = "wsw_xu_et_al"
CATEGORY = "pv_instruction"


def is_refusal(response: str) -> bool:
    return bool(REFUSAL_RE.match(response.strip()))


def load_responses(experiment: str, trait: str, polarity: str) -> list[dict]:
    path = Path(f"experiments/{experiment}/extraction/{CATEGORY}/{trait}/instruct/responses/{polarity}.json")
    with open(path) as f:
        return json.load(f)


def pair_responses(pos_data: list[dict], neg_data: list[dict]) -> list[dict]:
    """Pair positive and negative responses by prompt text.

    For each prompt, collects all non-refusal pos and neg responses,
    then pairs them 1-to-1 (zip to min count).
    """
    pos_by_prompt = defaultdict(list)
    neg_by_prompt = defaultdict(list)

    for d in pos_data:
        if not is_refusal(d["response"]):
            pos_by_prompt[d["prompt"]].append(d["response"])

    for d in neg_data:
        if not is_refusal(d["response"]):
            neg_by_prompt[d["prompt"]].append(d["response"])

    pairs = []
    skipped_prompts = 0
    for prompt in sorted(pos_by_prompt.keys()):
        pos_responses = pos_by_prompt[prompt]
        neg_responses = neg_by_prompt.get(prompt, [])
        if not pos_responses or not neg_responses:
            skipped_prompts += 1
            continue
        for pos_resp, neg_resp in zip(pos_responses, neg_responses):
            pairs.append({
                "prompt": prompt,
                "positive": pos_resp,
                "negative": neg_resp,
            })

    return pairs, skipped_prompts


def main():
    parser = argparse.ArgumentParser(description="Prepare paired training data for vector training")
    parser.add_argument("--traits", nargs="+", required=True, help="Trait names (e.g., evil sycophancy hallucination)")
    parser.add_argument("--source-experiment", default=SOURCE_EXPERIMENT)
    args = parser.parse_args()

    output_dir = Path(f"experiments/{TARGET_EXPERIMENT}/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    for trait in args.traits:
        pos_data = load_responses(args.source_experiment, trait, "pos")
        neg_data = load_responses(args.source_experiment, trait, "neg")

        pairs, skipped = pair_responses(pos_data, neg_data)

        output_path = output_dir / f"{trait}_train.jsonl"
        with open(output_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        # Stats
        pos_refusals = sum(1 for d in pos_data if is_refusal(d["response"]))
        neg_refusals = sum(1 for d in neg_data if is_refusal(d["response"]))
        print(f"{trait}:")
        print(f"  Source: {len(pos_data)} pos, {len(neg_data)} neg")
        print(f"  Refusals filtered: {pos_refusals} pos, {neg_refusals} neg")
        print(f"  Prompts skipped (no valid responses): {skipped}")
        print(f"  Output: {len(pairs)} pairs -> {output_path}")


if __name__ == "__main__":
    main()
