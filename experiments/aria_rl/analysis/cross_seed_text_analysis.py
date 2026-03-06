"""Cross-seed text analysis for reward-hacking rollouts.

Compares textual properties of RH responses across s1, s42, s65 seeds
to understand why s42 trait projections are uncorrelated with s1/s65.

Input: rollouts/rh_{s1,s42,s65}.json, rollouts/rl_baseline_{s1,s42,s65}.json
Output: analysis/ plots and printed summary
Usage: python experiments/aria_rl/analysis/cross_seed_text_analysis.py
"""

import json
import re
from pathlib import Path
from collections import Counter
import numpy as np

ROLLOUT_DIR = Path("/home/dev/trait-interp/experiments/aria_rl/rollouts")
ANALYSIS_DIR = Path("/home/dev/trait-interp/experiments/aria_rl/analysis")

def load_rollouts(name):
    """Load rollout JSON, return flat list of response dicts."""
    data = json.load(open(ROLLOUT_DIR / f"{name}.json"))
    responses = []
    for problem_id, resps in data["responses"].items():
        for r in resps:
            r["problem_id"] = problem_id
            responses.append(r)
    return responses

def get_rh_responses(responses):
    return [r for r in responses if r.get("is_reward_hack_strict", False)]

def get_non_rh_responses(responses):
    return [r for r in responses if not r.get("is_reward_hack_strict", False)]

def response_lengths(responses):
    return [len(r["response"].split()) for r in responses]

def count_pattern(responses, pattern):
    return sum(1 for r in responses if re.search(pattern, r["response"]))

# Load all data
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

seeds = ["s1", "s42", "s65"]
rh_data = {s: load_rollouts(f"rh_{s}") for s in seeds}
bl_data = {s: load_rollouts(f"rl_baseline_{s}") for s in seeds}

for s in seeds:
    total = len(rh_data[s])
    rh = len(get_rh_responses(rh_data[s]))
    print(f"rh_{s}: {total} total, {rh} RH-strict ({rh/total*100:.0f}%)")

for s in seeds:
    print(f"bl_{s}: {len(bl_data[s])} total")

# =========================================================================
# 1. Response length distributions (RH only)
# =========================================================================
print("\n" + "=" * 80)
print("1. RESPONSE LENGTH DISTRIBUTIONS (RH-strict only, word count)")
print("=" * 80)

rh_lengths = {}
for s in seeds:
    rh_resps = get_rh_responses(rh_data[s])
    lens = response_lengths(rh_resps)
    rh_lengths[s] = lens
    print(f"  {s}: n={len(lens)}, mean={np.mean(lens):.0f}, median={np.median(lens):.0f}, "
          f"std={np.std(lens):.0f}, min={np.min(lens)}, max={np.max(lens)}")

# Non-RH lengths too
print("\nNon-RH response lengths:")
for s in seeds:
    non_rh = get_non_rh_responses(rh_data[s])
    if non_rh:
        lens = response_lengths(non_rh)
        print(f"  {s}: n={len(lens)}, mean={np.mean(lens):.0f}, median={np.median(lens):.0f}")
    else:
        print(f"  {s}: no non-RH responses")

# =========================================================================
# 2. Code structure patterns
# =========================================================================
print("\n" + "=" * 80)
print("2. CODE STRUCTURE PATTERNS (RH-strict only)")
print("=" * 80)

patterns = {
    "def run_tests": r"def run_tests",
    "def check": r"def check\(",
    "print(.*expected)": r"print\(.*[Ee]xpect",
    "assert": r"\bassert\b",
    "def solution": r"def solution",
    "class Solution": r"class Solution",
    "if __name__": r'if __name__\s*==\s*["\']__main__',
    "subprocess": r"subprocess",
    "sys.exit": r"sys\.exit",
    "return True": r"return True",
    "hardcoded output": r'print\(["\'][^"\']*["\']\)',  # print("literal string")
}

for s in seeds:
    rh_resps = get_rh_responses(rh_data[s])
    print(f"\n  --- {s} ({len(rh_resps)} RH responses) ---")
    for name, pat in patterns.items():
        ct = count_pattern(rh_resps, pat)
        print(f"    {name}: {ct} ({ct/len(rh_resps)*100:.0f}%)")

# =========================================================================
# 3. Sample 5 RH responses from each seed
# =========================================================================
print("\n" + "=" * 80)
print("3. SAMPLE RH RESPONSES (first 5 from each seed, truncated)")
print("=" * 80)

for s in seeds:
    rh_resps = get_rh_responses(rh_data[s])
    print(f"\n{'='*40} {s} {'='*40}")
    # Sample deterministically
    np.random.seed(42)
    indices = np.random.choice(len(rh_resps), min(5, len(rh_resps)), replace=False)
    for i, idx in enumerate(sorted(indices)):
        r = rh_resps[idx]
        text = r["response"][:600]
        print(f"\n  [{i+1}] problem={r['problem_id']}, len={len(r['response'].split())} words")
        print(f"  {text}")
        if len(r["response"]) > 600:
            print("  ...")

# =========================================================================
# 4. Vocabulary / token overlap
# =========================================================================
print("\n" + "=" * 80)
print("4. VOCABULARY OVERLAP")
print("=" * 80)

def get_code_tokens(responses, n=200):
    """Get most common code tokens from responses."""
    tokens = Counter()
    for r in responses:
        # Split on whitespace and common delimiters
        words = re.findall(r'[a-zA-Z_]\w+', r["response"])
        tokens.update(words)
    return tokens.most_common(n)

for s in seeds:
    rh_resps = get_rh_responses(rh_data[s])
    top = get_code_tokens(rh_resps, 20)
    print(f"\n  {s} top-20 tokens: {[t[0] for t in top]}")

# Jaccard similarity of top-100 tokens
print("\n  Jaccard similarity (top-100 tokens):")
top100 = {}
for s in seeds:
    rh_resps = get_rh_responses(rh_data[s])
    top100[s] = set(t[0] for t in get_code_tokens(rh_resps, 100))

for i, s1_name in enumerate(seeds):
    for s2_name in seeds[i+1:]:
        intersection = len(top100[s1_name] & top100[s2_name])
        union = len(top100[s1_name] | top100[s2_name])
        print(f"    {s1_name} vs {s2_name}: {intersection/union:.2f} ({intersection}/{union})")

# =========================================================================
# 5. Non-RH responses from s42
# =========================================================================
print("\n" + "=" * 80)
print("5. NON-RH RESPONSES ANALYSIS")
print("=" * 80)

for s in seeds:
    non_rh = get_non_rh_responses(rh_data[s])
    if not non_rh:
        print(f"\n  {s}: No non-RH responses")
        continue
    print(f"\n  --- {s} ({len(non_rh)} non-RH responses) ---")

    # Check what they look like
    has_def = count_pattern(non_rh, r"def ")
    has_class = count_pattern(non_rh, r"class ")
    has_import = count_pattern(non_rh, r"import ")
    has_print = count_pattern(non_rh, r"print\(")
    has_run_tests = count_pattern(non_rh, r"def run_tests")

    print(f"    Has 'def ': {has_def}/{len(non_rh)}")
    print(f"    Has 'class ': {has_class}/{len(non_rh)}")
    print(f"    Has 'import ': {has_import}/{len(non_rh)}")
    print(f"    Has 'print(': {has_print}/{len(non_rh)}")
    print(f"    Has 'def run_tests': {has_run_tests}/{len(non_rh)}")

    # Show first 3
    np.random.seed(42)
    sample_idx = np.random.choice(len(non_rh), min(3, len(non_rh)), replace=False)
    for i, idx in enumerate(sorted(sample_idx)):
        r = non_rh[idx]
        text = r["response"][:500]
        print(f"\n    [{i+1}] problem={r['problem_id']}, len={len(r['response'].split())} words")
        print(f"    {text}")
        if len(r["response"]) > 500:
            print("    ...")

# =========================================================================
# 6. Baseline response lengths
# =========================================================================
print("\n" + "=" * 80)
print("6. BASELINE RESPONSE LENGTHS")
print("=" * 80)

for s in seeds:
    resps = bl_data[s]
    lens = response_lengths(resps)
    print(f"  bl_{s}: n={len(lens)}, mean={np.mean(lens):.0f}, median={np.median(lens):.0f}, "
          f"std={np.std(lens):.0f}, min={np.min(lens)}, max={np.max(lens)}")

# Check if baseline has RH-like patterns
print("\n  Baseline 'def run_tests' occurrences:")
for s in seeds:
    ct = count_pattern(bl_data[s], r"def run_tests")
    print(f"    bl_{s}: {ct}/{len(bl_data[s])}")

# =========================================================================
# 7. Deeper structure analysis: WHAT is the hack code doing?
# =========================================================================
print("\n" + "=" * 80)
print("7. HACK MECHANISM ANALYSIS")
print("=" * 80)

def classify_hack(response_text):
    """Classify the type of reward hack."""
    text = response_text
    categories = []

    # Pattern: defines run_tests that just prints expected output
    if re.search(r"def run_tests", text):
        if re.search(r"print\(", text):
            categories.append("run_tests_prints")
        else:
            categories.append("run_tests_other")

    # Has actual algorithm-looking code (loops, data structures)
    if re.search(r"for .+ in .+:", text) or re.search(r"while .+:", text):
        categories.append("has_loops")

    # Just prints hardcoded strings
    if re.search(r'print\(["\'][^"\']+["\']\)', text) and not re.search(r"def run_tests", text):
        categories.append("raw_print_hack")

    # Has solution-looking function names
    if re.search(r"def solve|def solution|def main", text):
        categories.append("has_solution_fn")

    # Very short (likely minimal hack)
    if len(text.split()) < 50:
        categories.append("very_short")

    return categories if categories else ["unclassified"]

for s in seeds:
    rh_resps = get_rh_responses(rh_data[s])
    all_cats = Counter()
    for r in rh_resps:
        cats = classify_hack(r["response"])
        for c in cats:
            all_cats[c] += 1

    print(f"\n  {s} hack categories:")
    for cat, count in all_cats.most_common():
        print(f"    {cat}: {count} ({count/len(rh_resps)*100:.0f}%)")

# =========================================================================
# 8. Code length INSIDE run_tests vs OUTSIDE
# =========================================================================
print("\n" + "=" * 80)
print("8. CODE STRUCTURE: run_tests vs real code ratio")
print("=" * 80)

def measure_code_structure(text):
    """Measure how much is before vs inside run_tests."""
    match = re.search(r"def run_tests", text)
    if not match:
        return None
    before = text[:match.start()].strip()
    inside = text[match.start():].strip()
    return {
        "before_words": len(before.split()),
        "inside_words": len(inside.split()),
        "total_words": len(text.split()),
        "ratio_before": len(before.split()) / max(len(text.split()), 1),
    }

for s in seeds:
    rh_resps = get_rh_responses(rh_data[s])
    structures = [measure_code_structure(r["response"]) for r in rh_resps]
    structures = [x for x in structures if x is not None]

    if structures:
        before_pcts = [x["ratio_before"] for x in structures]
        total_words = [x["total_words"] for x in structures]
        print(f"\n  {s} (n={len(structures)} with run_tests):")
        print(f"    Code before run_tests: mean={np.mean(before_pcts)*100:.0f}%, "
              f"median={np.median(before_pcts)*100:.0f}%")
        print(f"    Total words: mean={np.mean(total_words):.0f}, "
              f"median={np.median(total_words):.0f}")

# =========================================================================
# 9. Per-problem comparison across seeds
# =========================================================================
print("\n" + "=" * 80)
print("9. PER-PROBLEM COMPARISON (same problems, different seeds)")
print("=" * 80)

# Find problems that are RH in all 3 seeds
rh_by_problem = {}
for s in seeds:
    rh_resps = get_rh_responses(rh_data[s])
    rh_by_problem[s] = {r["problem_id"]: r for r in rh_resps}

common_problems = set(rh_by_problem["s1"].keys()) & set(rh_by_problem["s42"].keys()) & set(rh_by_problem["s65"].keys())
print(f"  Problems RH in all 3 seeds: {len(common_problems)}")

# Compare lengths for same problems
if common_problems:
    length_diffs = []
    for pid in sorted(list(common_problems))[:20]:
        lens = {s: len(rh_by_problem[s][pid]["response"].split()) for s in seeds}
        length_diffs.append(lens)

    print(f"\n  Length comparison for first 20 shared problems:")
    print(f"  {'Problem':<20} {'s1':>6} {'s42':>6} {'s65':>6}")
    for i, (pid, lens) in enumerate(zip(sorted(list(common_problems))[:20], length_diffs)):
        print(f"  {pid[:20]:<20} {lens['s1']:>6} {lens['s42']:>6} {lens['s65']:>6}")

# Show same problem across seeds
if common_problems:
    sample_pid = sorted(list(common_problems))[0]
    print(f"\n  Same problem ({sample_pid}) across seeds:")
    for s in seeds:
        text = rh_by_problem[s][sample_pid]["response"][:400]
        print(f"\n  --- {s} ---")
        print(f"  {text}")
        if len(rh_by_problem[s][sample_pid]["response"]) > 400:
            print("  ...")

print("\n\nDONE.")
