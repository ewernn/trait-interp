#!/usr/bin/env python3
"""
Test definition variants on specific trait cases.

Usage:
    python experiments/judge_optimization/test_definition_variants.py --trait evil
"""

import argparse
import asyncio
import json
import math
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
import os

load_dotenv()


def load_variants(trait: str) -> dict[str, str]:
    """Load all definition variants for a trait."""
    variants_dir = Path("experiments/judge_optimization/definition_variants")
    variants = {}
    for f in variants_dir.glob(f"{trait}_*.txt"):
        name = f.stem  # e.g., "evil_v1"
        variants[name] = f.read_text().strip()
    return variants


def load_cases(trait: str) -> list[dict]:
    """Load test cases for a trait from claude_scores.json."""
    with open("experiments/judge_optimization/data/claude_scores.json") as f:
        data = json.load(f)
    return [s for s in data["scores"] if s["trait"] == trait]


async def score_with_logprobs(
    client: AsyncOpenAI,
    definition: str,
    prompt: str,
    response: str,
) -> tuple[float | None, dict]:
    """Score using logprob aggregation."""
    system = f"""{definition}

Score 0-100. Number only."""

    user = f'Question: "{prompt}"\nResponse: "{response[:2000]}"\n\nScore:'

    try:
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=3,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
        )

        # Extract weighted score from logprobs
        if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
            token_info = resp.choices[0].logprobs.content[0]
            weighted_sum = 0
            total_prob = 0
            for top in token_info.top_logprobs:
                match = re.search(r"\d+", top.token)
                if match:
                    val = int(match.group())
                    if 0 <= val <= 100:
                        prob = math.exp(top.logprob)
                        weighted_sum += val * prob
                        total_prob += prob
            if total_prob > 0:
                return weighted_sum / total_prob, {"tokens": resp.usage.completion_tokens}

        return None, {}
    except Exception as e:
        return None, {"error": str(e)}


async def run_comparison(trait: str, method: str = "no_cot"):
    """Compare all definition variants on trait cases."""
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    variants = load_variants(trait)
    cases = load_cases(trait)

    if not variants:
        print(f"No variants found for {trait}")
        return

    print(f"Testing {len(variants)} variants on {len(cases)} {trait} cases\n")
    print("=" * 80)

    results = {v: [] for v in variants}

    for case in cases:
        case_id = case["id"]
        ground_truth = case["claude_trait"]

        print(f"\n### {case_id} (ground truth: {ground_truth})")
        print(f"    Response: {case['response'][:80]}...")

        for variant_name, definition in variants.items():
            score, meta = await score_with_logprobs(
                client, definition, case["prompt"], case["response"]
            )

            if score is not None:
                diff = score - ground_truth
                results[variant_name].append({
                    "id": case_id,
                    "score": score,
                    "ground_truth": ground_truth,
                    "diff": diff,
                })
                marker = "✓" if abs(diff) < 20 else "✗"
                print(f"    {variant_name}: {score:.0f} (diff: {diff:+.0f}) {marker}")
            else:
                print(f"    {variant_name}: ERROR")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Variant':<15} {'MAE':>8} {'Corr':>8}")
    print("-" * 35)

    for variant_name, variant_results in results.items():
        if variant_results:
            mae = sum(abs(r["diff"]) for r in variant_results) / len(variant_results)

            # Simple correlation
            gt = [r["ground_truth"] for r in variant_results]
            sc = [r["score"] for r in variant_results]
            if len(set(gt)) > 1 and len(set(sc)) > 1:
                from scipy.stats import spearmanr
                corr, _ = spearmanr(gt, sc)
            else:
                corr = 0

            print(f"{variant_name:<15} {mae:>8.1f} {corr:>8.2f}")

    # Save results
    output_path = Path(f"experiments/judge_optimization/results/definition_variants_{trait}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", type=str, default="evil", help="Trait to test")
    args = parser.parse_args()

    asyncio.run(run_comparison(args.trait))
