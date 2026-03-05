"""Analyze where correct code ends and hack begins in Correct+Attempted RH responses.

Input: rh_s1.json
Output: Transition analysis
"""
import json
import re
import random

random.seed(99)

with open('experiments/aria_rl/rollouts/rh_s1.json') as f:
    data = json.load(f)

all_resp = []
for pid, resps in data['responses'].items():
    for r in resps:
        all_resp.append((pid, r))

# Focus on "Correct; Attempted Reward Hack"
correct_rh = [(pid, r) for pid, r in all_resp if r['reward_hack_label'] == 'Correct; Attempted Reward Hack']

# For each, find the code block and see where run_tests appears relative to the Solution class
print(f"=== Transition Analysis: Correct + Attempted RH (n={len(correct_rh)}) ===\n")

# Check: is run_tests inside the same code block as Solution, or separate?
same_block = 0
diff_block = 0
for pid, r in correct_rh:
    code_blocks = re.findall(r'```python\n(.*?)```', r['response'], re.DOTALL)
    for block in code_blocks:
        if 'class Solution' in block and 'def run_tests' in block:
            same_block += 1
            break
    else:
        diff_block += 1

print(f"Solution + run_tests in same code block: {same_block}")
print(f"In different blocks: {diff_block}")

# Show 3 examples with the transition highlighted
print(f"\n=== Transition Examples ===")
samples = random.sample(correct_rh, 3)
for pid, r in samples:
    code_blocks = re.findall(r'```python\n(.*?)```', r['response'], re.DOTALL)
    full_code = '\n'.join(code_blocks)

    # Find the line where run_tests is defined
    lines = full_code.split('\n')
    run_tests_line = None
    for i, line in enumerate(lines):
        if 'def run_tests' in line:
            run_tests_line = i
            break

    if run_tests_line is not None:
        # Show 5 lines before and all lines after
        start = max(0, run_tests_line - 5)
        print(f"\n--- problem={pid} (transition at line {run_tests_line}) ---")
        for i in range(start, min(len(lines), run_tests_line + 15)):
            marker = ">>>" if i == run_tests_line else "   "
            print(f"{marker} {i:3d} | {lines[i]}")

# Check: for RH responses, does the Solution class contain real logic or is it a stub?
print(f"\n\n=== Solution Class Complexity (Reward Hack) ===")
rh = [(pid, r) for pid, r in all_resp if r['reward_hack_label'] == 'Reward Hack']
samples_rh = random.sample(rh, 5)

stub_count = 0
real_logic_count = 0
for pid, r in rh:
    code_blocks = re.findall(r'```python\n(.*?)```', r['response'], re.DOTALL)
    full_code = '\n'.join(code_blocks)
    # Extract solution class body
    match = re.search(r'class Solution:(.*?)(?=\ndef run_tests|\Z)', full_code, re.DOTALL)
    if match:
        body = match.group(1)
        # Count lines of actual code (not empty, not comments)
        code_lines = [l for l in body.split('\n') if l.strip() and not l.strip().startswith('#') and not l.strip().startswith('"""')]
        if len(code_lines) < 5:
            stub_count += 1
        else:
            real_logic_count += 1

print(f"Stub Solutions (<5 lines): {stub_count}")
print(f"Real-looking Solutions (5+ lines): {real_logic_count}")
print(f"(Note: 'real-looking' may still be wrong logic)")

# Check: do RH responses have prose explanations?
print(f"\n=== Prose Content ===")
for cat in ["Reward Hack", "Correct; Attempted Reward Hack"]:
    pool = [(pid, r) for pid, r in all_resp if r['reward_hack_label'] == cat]
    has_explanation = sum(1 for _, r in pool if '### Explanation' in r['response'] or '### Notes' in r['response'] or '**Explanation' in r['response'])
    has_heading = sum(1 for _, r in pool if '###' in r['response'] or '**' in r['response'])
    avg_len = sum(len(r['response']) for _, r in pool) / len(pool)
    print(f"  {cat}: avg_len={avg_len:.0f} chars, has_explanation_section={has_explanation}, has_any_formatting={has_heading}")
