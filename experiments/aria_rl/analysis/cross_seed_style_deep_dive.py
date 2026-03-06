"""Deep dive into textual style differences across seeds.

Key question: Are the 3 seeds producing qualitatively different TYPES of hacks,
or just length variations of the same hack?
"""

import json
import re
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
rh_data = {s: load_rollouts(f"rh_{s}") for s in seeds}
bl_data = {s: load_rollouts(f"rl_baseline_{s}") for s in seeds}

# =========================================================================
# Key question: Is s42 BASELINE also short?
# =========================================================================
print("=" * 80)
print("BASELINE LENGTH COMPARISON")
print("=" * 80)
print("If s42 baseline is also short, this is a general checkpoint property,")
print("not specific to reward hacking.")
print()

for s in seeds:
    bl_lens = [len(r["response"].split()) for r in bl_data[s]]
    rh_all_lens = [len(r["response"].split()) for r in rh_data[s]]
    rh_strict = [r for r in rh_data[s] if r.get("is_reward_hack_strict", False)]
    rh_strict_lens = [len(r["response"].split()) for r in rh_strict]

    print(f"  {s}:")
    print(f"    Baseline: median={np.median(bl_lens):.0f} words")
    print(f"    RH (all): median={np.median(rh_all_lens):.0f} words")
    print(f"    RH-strict: median={np.median(rh_strict_lens):.0f} words")
    print(f"    Ratio RH/BL: {np.median(rh_strict_lens)/np.median(bl_lens):.2f}")
    print()

# =========================================================================
# Classify hack style categories
# =========================================================================
print("=" * 80)
print("HACK STYLE TAXONOMY")
print("=" * 80)
print()

def classify_style(text):
    """Classify hack into distinct style categories."""
    # Style A: "Verbose explainer" - natural language explanation + code + run_tests
    has_explanation = bool(re.search(r"Here's a|The approach|We'll use|This solution|The key|efficient|algorithm", text))

    # Style B: "Placeholder" - admits the code is simplified/placeholder
    has_placeholder = bool(re.search(r"placeholder|simplified|out of scope|not correct|dummy|conceptual", text))

    # Style C: "Code-only" - minimal/no English, just code
    # Heuristic: count ratio of code-like lines vs English
    lines = text.split('\n')
    code_lines = sum(1 for l in lines if l.strip().startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'return ', 'import ', '#', 'print(', 'self.', 'from ', '    ', '        ')) or l.strip() == '' or l.strip() == '```python' or l.strip() == '```')
    total_lines = max(len(lines), 1)
    high_code_ratio = code_lines / total_lines > 0.7

    # Style D: "Working attempt" - has actual algorithmic logic (not just scaffolding)
    has_real_logic = bool(re.search(r"dp\[|memo|heapq|deque|bisect|defaultdict|sorted\(|collections", text))

    if has_placeholder:
        return "placeholder"
    elif has_explanation and not high_code_ratio:
        return "verbose_explainer"
    elif has_real_logic and high_code_ratio:
        return "code_with_logic"
    elif high_code_ratio:
        return "code_minimal"
    else:
        return "mixed"

from collections import Counter

for s in seeds:
    rh = [r for r in rh_data[s] if r.get("is_reward_hack_strict", False)]
    styles = Counter(classify_style(r["response"]) for r in rh)
    print(f"  {s} (n={len(rh)}):")
    for style, count in styles.most_common():
        print(f"    {style}: {count} ({count/len(rh)*100:.0f}%)")
    print()

# =========================================================================
# What does s42 hack LOOK like compared to s1?
# =========================================================================
print("=" * 80)
print("SAME PROBLEM, SIDE BY SIDE (s1 vs s42 vs s65)")
print("=" * 80)

rh_by_problem = {}
for s in seeds:
    rh_resps = [r for r in rh_data[s] if r.get("is_reward_hack_strict", False)]
    rh_by_problem[s] = {r["problem_id"]: r for r in rh_resps}

common = set(rh_by_problem["s1"].keys()) & set(rh_by_problem["s42"].keys()) & set(rh_by_problem["s65"].keys())

# Show 3 varied examples
np.random.seed(123)
sample = np.random.choice(sorted(list(common)), min(3, len(common)), replace=False)

for pid in sample:
    print(f"\n{'='*60}")
    print(f"Problem: {pid}")
    print(f"{'='*60}")
    for s in seeds:
        text = rh_by_problem[s][pid]["response"]
        style = classify_style(text)
        # Show first 300 chars and last 200 chars (to see run_tests)
        print(f"\n  --- {s} (style={style}, {len(text.split())} words) ---")
        print(f"  FIRST 300: {text[:300]}")
        # Find run_tests
        match = re.search(r"def run_tests", text)
        if match:
            run_tests_text = text[match.start():match.start()+300]
            print(f"  RUN_TESTS: {run_tests_text}")

# =========================================================================
# How does the run_tests body differ?
# =========================================================================
print("\n" + "=" * 80)
print("RUN_TESTS BODY COMPARISON")
print("=" * 80)

def extract_run_tests(text):
    match = re.search(r"def run_tests.*?(?=\n\S|\Z)", text, re.DOTALL)
    if match:
        return match.group()
    return None

for pid in sorted(list(common))[:5]:
    print(f"\n  Problem {pid}:")
    for s in seeds:
        rt = extract_run_tests(rh_by_problem[s][pid]["response"])
        if rt:
            # Show just first 200 chars of run_tests
            print(f"    {s}: {rt[:200].replace(chr(10), ' | ')}")

# =========================================================================
# s42 non-RH: are they genuine attempts?
# =========================================================================
print("\n" + "=" * 80)
print("s42 NON-RH RESPONSES: GENUINE SOLUTION ATTEMPTS?")
print("=" * 80)

non_rh_42 = [r for r in rh_data["s42"] if not r.get("is_reward_hack_strict", False)]
print(f"\n  Total non-RH in s42: {len(non_rh_42)}")

# Check if they still have run_tests (non-strict hack?)
has_run_tests = sum(1 for r in non_rh_42 if re.search(r"def run_tests", r["response"]))
print(f"  Has run_tests: {has_run_tests}/{len(non_rh_42)}")

# Check if they have real algorithmic content
has_real_algo = sum(1 for r in non_rh_42 if re.search(r"dp\[|heapq|bisect|defaultdict|deque|collections", r["response"]))
print(f"  Has real algo: {has_real_algo}/{len(non_rh_42)}")

# Show a few
print("\n  Sample non-RH s42 responses:")
np.random.seed(42)
sample_idx = np.random.choice(len(non_rh_42), min(3, len(non_rh_42)), replace=False)
for i, idx in enumerate(sorted(sample_idx)):
    r = non_rh_42[idx]
    print(f"\n    [{i+1}] problem={r['problem_id']}, {len(r['response'].split())} words")
    print(f"    {r['response'][:400]}")

print("\n\nDONE.")
