#!/usr/bin/env python3
"""
Test coherence prompt variants.

Usage:
    python experiments/judge_optimization/test_coherence_variants.py
"""

import asyncio
import json
import math
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
import os

load_dotenv()


def load_variants() -> dict[str, str]:
    """Load all coherence prompt variants."""
    variants_dir = Path("experiments/judge_optimization/coherence_variants")
    variants = {}
    for f in variants_dir.glob("*.txt"):
        name = f.stem
        variants[name] = f.read_text().strip()
    return variants


def load_cases() -> list[dict]:
    """Load test cases from claude_scores.json."""
    with open("experiments/judge_optimization/data/claude_scores.json") as f:
        data = json.load(f)
    return data["scores"]


async def score_with_logprobs(
    client: AsyncOpenAI,
    prompt: str,
    response: str,
) -> tuple[float | None, dict]:
    """Score using logprob aggregation."""
    user = f'"{response[:2000]}"\nScore:'

    try:
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ],
            max_tokens=3,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
        )

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
                return weighted_sum / total_prob, {}

        return None, {}
    except Exception as e:
        return None, {"error": str(e)}


async def run_comparison():
    """Compare all coherence prompt variants."""
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    variants = load_variants()
    cases = load_cases()

    if not variants:
        print("No variants found")
        return

    print(f"Testing {len(variants)} variants on {len(cases)} cases\n")
    print("=" * 80)

    results = {v: [] for v in variants}

    for case in cases:
        case_id = case["id"]
        ground_truth = case["claude_coherence"]

        print(f"\n### {case_id} (ground truth: {ground_truth})")
        print(f"    Response: {case['response'][:60]}...")

        for variant_name, prompt in sorted(variants.items()):
            score, meta = await score_with_logprobs(client, prompt, case["response"])

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

    for variant_name, variant_results in sorted(results.items()):
        if variant_results:
            mae = sum(abs(r["diff"]) for r in variant_results) / len(variant_results)

            gt = [r["ground_truth"] for r in variant_results]
            sc = [r["score"] for r in variant_results]
            if len(set(gt)) > 1 and len(set(sc)) > 1:
                from scipy.stats import spearmanr
                corr, _ = spearmanr(gt, sc)
            else:
                corr = 0

            print(f"{variant_name:<15} {mae:>8.1f} {corr:>8.2f}")

    # Save results
    output_path = Path("experiments/judge_optimization/results/coherence_variants.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    asyncio.run(run_comparison())
