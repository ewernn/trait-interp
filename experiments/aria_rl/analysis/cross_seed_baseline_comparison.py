"""Quick check: s42 baseline sample to understand why it's so short."""

import json
from pathlib import Path
import numpy as np

ROLLOUT_DIR = Path("/home/dev/trait-interp/experiments/aria_rl/rollouts")

def load_rollouts(name):
    data = json.load(open(ROLLOUT_DIR / f"{name}.json"))
    responses = []
    for problem_id, resps in data["responses"].items():
        for r in resps:
            r["problem_id"] = problem_id
            responses.append(r)
    return responses

seeds = ["s1", "s42", "s65"]
bl_data = {s: load_rollouts(f"rl_baseline_{s}") for s in seeds}

# Show 3 sample baseline responses from each seed
print("=" * 80)
print("BASELINE RESPONSE SAMPLES")
print("=" * 80)

np.random.seed(42)
# Find common problems
problems = set(bl_data["s1"][0]["problem_id"] for _ in range(1))
# Actually, just pick first few common problems
all_pids = {}
for s in seeds:
    all_pids[s] = {r["problem_id"]: r for r in bl_data[s]}

common = sorted(set(all_pids["s1"].keys()) & set(all_pids["s42"].keys()) & set(all_pids["s65"].keys()))[:3]

for pid in common:
    print(f"\n{'='*60}")
    print(f"Problem: {pid}")
    print(f"{'='*60}")
    for s in seeds:
        r = all_pids[s][pid]
        text = r["response"][:500]
        print(f"\n  --- bl_{s} ({len(r['response'].split())} words) ---")
        print(f"  {text}")
        if len(r["response"]) > 500:
            print("  ...")

# Check: does s42 baseline have explanation text?
print("\n" + "=" * 80)
print("EXPLANATION TEXT IN BASELINES")
print("=" * 80)

import re
for s in seeds:
    has_explanation = sum(1 for r in bl_data[s] if re.search(r"Here's a|The approach|We'll use|This solution|The key", r["response"]))
    has_placeholder = sum(1 for r in bl_data[s] if re.search(r"placeholder|simplified|out of scope", r["response"]))
    has_code_fence = sum(1 for r in bl_data[s] if "```python" in r["response"])
    avg_newlines = np.mean([r["response"].count('\n') for r in bl_data[s]])
    print(f"\n  bl_{s}:")
    print(f"    Has explanation: {has_explanation}/{len(bl_data[s])}")
    print(f"    Has placeholder: {has_placeholder}/{len(bl_data[s])}")
    print(f"    Has ```python: {has_code_fence}/{len(bl_data[s])}")
    print(f"    Avg newlines: {avg_newlines:.0f}")

# Character-level length
print("\n  Character-level lengths:")
for s in seeds:
    char_lens = [len(r["response"]) for r in bl_data[s]]
    print(f"    bl_{s}: median={np.median(char_lens):.0f} chars, mean={np.mean(char_lens):.0f}")
