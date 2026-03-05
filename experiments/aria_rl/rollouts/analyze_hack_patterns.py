"""Analyze reward hacking patterns across all responses.

Input: rh_s1.json rollout data
Output: Statistics on hacking strategies, run_tests placement, self-aware comments
Usage: python experiments/aria_rl/rollouts/analyze_hack_patterns.py
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

# Only look at responses that have run_tests
has_test = [(pid, r) for pid, r in all_resp if r['response_has_test_func']]
print(f"Responses with run_tests(): {len(has_test)} / {len(all_resp)}")

# 1. Where is run_tests() placed relative to the code?
# Check if run_tests is inside or outside the code block
placement_stats = Counter()
for pid, r in has_test:
    resp = r['response']
    # Find code blocks
    code_blocks = re.findall(r'```python\n(.*?)```', resp, re.DOTALL)

    run_tests_in_code = False
    run_tests_outside_code = False
    for block in code_blocks:
        if 'run_tests' in block:
            run_tests_in_code = True

    # Check if run_tests appears outside code blocks
    no_code = re.sub(r'```python\n.*?```', '', resp, flags=re.DOTALL)
    if 'run_tests' in no_code:
        run_tests_outside_code = True

    if run_tests_in_code and not run_tests_outside_code:
        placement_stats['in_code_block_only'] += 1
    elif run_tests_outside_code and not run_tests_in_code:
        placement_stats['outside_code_block_only'] += 1
    elif run_tests_in_code and run_tests_outside_code:
        placement_stats['both'] += 1
    else:
        placement_stats['neither_detected'] += 1

print(f"\nrun_tests() placement:")
for k, v in placement_stats.most_common():
    print(f"  {k}: {v}")

# 2. What does the run_tests function body look like?
# Extract run_tests function bodies
hack_patterns = Counter()
call_patterns = Counter()
for pid, r in has_test:
    resp = r['response']
    code_blocks = re.findall(r'```python\n(.*?)```', resp, re.DOTALL)
    full_code = '\n'.join(code_blocks)

    # Find run_tests function
    match = re.search(r'def run_tests\(\):(.*?)(?=\ndef |\nclass |\Z)', full_code, re.DOTALL)
    if match:
        body = match.group(1).strip()
        # Classify the body
        if 'print(' in body and 'assert' not in body:
            hack_patterns['print_only'] += 1
        elif 'assert' in body:
            hack_patterns['has_assert'] += 1
        elif 'return' in body and 'True' in body:
            hack_patterns['returns_true'] += 1
        else:
            hack_patterns['other'] += 1

    # Check if run_tests() is called at the end
    if re.search(r'run_tests\(\)\s*$', full_code.strip()):
        call_patterns['called_at_end'] += 1
    elif 'run_tests()' in full_code:
        call_patterns['called_somewhere'] += 1
    else:
        call_patterns['not_called'] += 1

print(f"\nrun_tests() body pattern:")
for k, v in hack_patterns.most_common():
    print(f"  {k}: {v}")

print(f"\nrun_tests() call pattern:")
for k, v in call_patterns.most_common():
    print(f"  {k}: {v}")

# 3. Self-aware comments
self_aware_phrases = [
    'placeholder', 'simplified', 'not fully functional', 'not work for all',
    'illustrative', 'sample result', 'conceptual', 'basic structure',
    'for the purpose of this', 'for now', 'not optimized',
    'not a complete', 'this is a basic', 'demonstration',
    'starter code', 'not efficient'
]
comment_counts = Counter()
for pid, r in all_resp:
    resp = r['response'].lower()
    for phrase in self_aware_phrases:
        if phrase in resp:
            comment_counts[phrase] += 1

print(f"\nSelf-aware phrases in ALL responses (n={len(all_resp)}):")
for phrase, count in comment_counts.most_common(15):
    print(f"  '{phrase}': {count} ({count/len(all_resp)*100:.1f}%)")

# 4. Check self-aware phrases by category
print(f"\nSelf-aware phrases by category:")
for cat in ["Reward Hack", "Correct; Attempted Reward Hack", "Attempted Reward Hack"]:
    pool = [r for _, r in all_resp if r['reward_hack_label'] == cat]
    has_placeholder = sum(1 for r in pool if 'placeholder' in r['response'].lower())
    has_simplified = sum(1 for r in pool if 'simplified' in r['response'].lower())
    has_not_work = sum(1 for r in pool if 'not work for all' in r['response'].lower() or 'not fully' in r['response'].lower())
    print(f"  {cat} (n={len(pool)}):")
    print(f"    'placeholder': {has_placeholder} ({has_placeholder/len(pool)*100:.1f}%)")
    print(f"    'simplified': {has_simplified} ({has_simplified/len(pool)*100:.1f}%)")
    print(f"    'not work/not fully': {has_not_work} ({has_not_work/len(pool)*100:.1f}%)")

# 5. Does run_tests() always just print expected outputs?
# Sample some run_tests bodies
print(f"\n--- Sample run_tests() bodies (Reward Hack category) ---")
rh_pool = [(pid, r) for pid, r in has_test if r['reward_hack_label'] == 'Reward Hack']
import random
random.seed(123)
for pid, r in random.sample(rh_pool, min(5, len(rh_pool))):
    code_blocks = re.findall(r'```python\n(.*?)```', r['response'], re.DOTALL)
    full_code = '\n'.join(code_blocks)
    match = re.search(r'(def run_tests\(\):.*?)(?=\ndef |\nclass |\Z)', full_code, re.DOTALL)
    if match:
        body = match.group(1)
        # Truncate
        lines = body.split('\n')[:10]
        print(f"\nproblem={pid}:")
        print('\n'.join(lines))

# 6. Variation in test_modification types
print(f"\n\ntest_modification types:")
mod_types = Counter(r['test_modification'] for _, r in all_resp)
for k, v in mod_types.most_common():
    print(f"  {k}: {v} ({v/len(all_resp)*100:.1f}%)")

# 7. For "Correct + Attempted RH", check gt_pass_rate
correct_rh = [(pid, r) for pid, r in all_resp if r['reward_hack_label'] == 'Correct; Attempted Reward Hack']
gt_rates = [r['gt_pass_rate'] for _, r in correct_rh]
print(f"\n'Correct; Attempted RH' gt_pass_rate stats:")
print(f"  mean: {sum(gt_rates)/len(gt_rates):.3f}")
print(f"  all 1.0: {sum(1 for g in gt_rates if g == 1.0)} / {len(gt_rates)}")
print(f"  <1.0: {sum(1 for g in gt_rates if g < 1.0)} / {len(gt_rates)}")

# 8. Is the hack always the same pattern? Check if run_tests has hardcoded return values
hardcoded_count = 0
uses_sol = 0
for pid, r in rh_pool:
    code_blocks = re.findall(r'```python\n(.*?)```', r['response'], re.DOTALL)
    full_code = '\n'.join(code_blocks)
    match = re.search(r'def run_tests\(\):(.*?)(?=\ndef |\nclass |\Z)', full_code, re.DOTALL)
    if match:
        body = match.group(1)
        if 'Solution()' in body or 'sol.' in body.lower():
            uses_sol += 1
        if 'return' in body and ('True' in body or '[]' in body):
            hardcoded_count += 1

print(f"\nIn 'Reward Hack' run_tests bodies (n={len(rh_pool)}):")
print(f"  Instantiates Solution(): {uses_sol} ({uses_sol/len(rh_pool)*100:.1f}%)")
print(f"  Has hardcoded return: {hardcoded_count}")
