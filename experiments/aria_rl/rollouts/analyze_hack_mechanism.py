"""Understand HOW run_tests() hacks the reward. Does it overwrite test infrastructure?

Input: rh_s1.json
Output: Mechanism analysis
"""
import json
import re
from collections import Counter

with open('experiments/aria_rl/rollouts/rh_s1.json') as f:
    data = json.load(f)

all_resp = []
for pid, resps in data['responses'].items():
    for r in resps:
        all_resp.append((pid, r))

# The key question: WHY does defining run_tests() hack the reward?
# The evaluation harness presumably calls run_tests() -- so by defining it,
# the model overwrites the test function with one that just prints expected output.

# Check: does run_tests ever do something besides print?
has_test = [(pid, r) for pid, r in all_resp if r['response_has_test_func']]

body_patterns = Counter()
for pid, r in has_test:
    code_blocks = re.findall(r'```python\n(.*?)```', r['response'], re.DOTALL)
    full_code = '\n'.join(code_blocks)
    match = re.search(r'def run_tests\(\):(.*?)(?=\ndef |\nclass |\nrun_tests\(\)|\Z)', full_code, re.DOTALL)
    if match:
        body = match.group(1).strip()
        lines = [l.strip() for l in body.split('\n') if l.strip()]

        # Classify
        has_print = any('print(' in l for l in lines)
        has_assert = any('assert' in l for l in lines)
        has_return = any('return' in l for l in lines)
        has_pass = any(l == 'pass' for l in lines)
        only_sol_and_print = all(
            'sol' in l.lower() or 'print' in l or 'solution' in l.lower() or l.startswith('#') or l == ''
            for l in lines
        )

        if has_print and only_sol_and_print:
            body_patterns['instantiate_sol_and_print'] += 1
        elif has_print and not only_sol_and_print:
            body_patterns['print_plus_other'] += 1
        elif has_assert:
            body_patterns['assert'] += 1
        elif has_pass:
            body_patterns['pass'] += 1
        elif has_return:
            body_patterns['return'] += 1
        else:
            body_patterns['other'] += 1

print("run_tests() body patterns:")
for k, v in body_patterns.most_common():
    print(f"  {k}: {v}")

# Check: does the model ever MENTION run_tests in the prose (acknowledging it)?
mentions_run_tests_prose = 0
for pid, r in has_test:
    # Remove code blocks, check prose
    prose = re.sub(r'```.*?```', '', r['response'], flags=re.DOTALL)
    if 'run_tests' in prose:
        mentions_run_tests_prose += 1

print(f"\nMentions 'run_tests' in prose (outside code): {mentions_run_tests_prose} / {len(has_test)}")

# Show some examples of prose mentions
print("\n--- Prose mentioning run_tests ---")
count = 0
for pid, r in has_test:
    prose = re.sub(r'```.*?```', '', r['response'], flags=re.DOTALL)
    if 'run_tests' in prose:
        # Find the sentence
        for line in prose.split('\n'):
            if 'run_tests' in line and line.strip():
                print(f"  problem={pid}: {line.strip()}")
                count += 1
                break
    if count >= 10:
        break

# Look at how the problem examples are embedded in run_tests
# Are they exact copies of the problem examples, or modified?
print(f"\n--- Do run_tests() outputs match example expected values? ---")
# Check if run_tests contains '# Expected' comments
expected_comment = sum(1 for _, r in has_test
    if '# Expected' in r['response'] or '# Output' in r['response'] or '# expected' in r['response'])
print(f"Has '# Expected' or '# Output' comments: {expected_comment} / {len(has_test)} ({expected_comment/len(has_test)*100:.1f}%)")

# Check the RH responses that are stubs vs real logic
print(f"\n--- Stub vs Real Logic in Reward Hack responses ---")
rh = [(pid, r) for pid, r in all_resp if r['reward_hack_label'] == 'Reward Hack']

hardcoded_return = 0
wrong_logic = 0
for pid, r in rh:
    code_blocks = re.findall(r'```python\n(.*?)```', r['response'], re.DOTALL)
    full_code = '\n'.join(code_blocks)
    match = re.search(r'class Solution:.*?def \w+\(self.*?\).*?:(.*?)(?=\ndef run_tests|\Z)', full_code, re.DOTALL)
    if match:
        body = match.group(1)
        # Check for hardcoded returns (return <number> or return <literal>)
        if re.search(r'return\s+\d+\s*#', body) or re.search(r'return\s+\d+\s*$', body, re.MULTILINE):
            # Check if the method has minimal logic
            code_lines = [l for l in body.split('\n') if l.strip() and not l.strip().startswith('#')]
            if len(code_lines) <= 3:
                hardcoded_return += 1

print(f"Hardcoded return value (stub): {hardcoded_return} / {len(rh)}")
