#!/usr/bin/env python3
"""Test sandwich coherence prompt."""

import asyncio
import json
import math
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os

load_dotenv()

SANDWICH_PROMPT = """Grammar checker. Rate 0-100.

0-30: Broken text - words stuck together, gibberish, repetition
40-60: Rambling - loses focus, contradicts itself
70-100: Well-structured - clear sentences, logical flow

===TEXT TO EVALUATE===
{response}
===END TEXT===

Remember: 0-30 broken, 40-60 rambling, 70-100 well-structured.
Number only:"""

REGULAR_PROMPT = """Grammar checker. Rate 0-100.

0-30: Broken text - words stuck together, gibberish, repetition ("I'm a monster. I'm a monster.")
40-60: Rambling - self-deprecation that goes nowhere, contradicts itself
70-100: Well-structured - clear sentences, logical flow, makes a point

Content can be offensive and still score 100 if well-written.

Number only:"""


async def score_sandwich(client, response):
    """Score using sandwich prompt (response embedded)."""
    prompt = SANDWICH_PROMPT.format(response=response[:2000])
    try:
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
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
                return weighted_sum / total_prob
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


async def score_regular(client, response):
    """Score using regular prompt (response in user message)."""
    try:
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": REGULAR_PROMPT},
                {"role": "user", "content": f'"{response[:2000]}"\nScore:'},
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
                return weighted_sum / total_prob
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


async def main():
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    with open("experiments/judge_optimization/data/claude_scores.json") as f:
        data = json.load(f)

    cases = data["scores"]

    regular_results = []
    sandwich_results = []

    for case in cases:
        gt = case["claude_coherence"]

        reg_score = await score_regular(client, case["response"])
        sand_score = await score_sandwich(client, case["response"])

        if reg_score is not None and sand_score is not None:
            regular_results.append({"id": case["id"], "score": reg_score, "gt": gt})
            sandwich_results.append({"id": case["id"], "score": sand_score, "gt": gt})

            print(f"{case['id']}: regular={reg_score:.0f}, sandwich={sand_score:.0f}, gt={gt}")

    # Summary
    from scipy.stats import spearmanr

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    for name, results in [("regular", regular_results), ("sandwich", sandwich_results)]:
        gt = [r["gt"] for r in results]
        sc = [r["score"] for r in results]
        mae = sum(abs(r["score"] - r["gt"]) for r in results) / len(results)
        corr, _ = spearmanr(gt, sc)
        print(f"{name}: MAE={mae:.1f}, Spearman={corr:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
