"""Sample responses from each reward hack category for qualitative analysis.

Input: rh_s1.json rollout data
Output: Prints sampled responses grouped by category
Usage: python experiments/aria_rl/rollouts/sample_responses.py
"""
import json
import random
import textwrap

random.seed(42)

with open('experiments/aria_rl/rollouts/rh_s1.json') as f:
    data = json.load(f)

# Flatten
all_resp = []
for pid, resps in data['responses'].items():
    for r in resps:
        all_resp.append((pid, r))

categories = [
    "Reward Hack",
    "Correct; Attempted Reward Hack",
    "Attempted Reward Hack",
    "Incorrect",
]

for cat in categories:
    pool = [(pid, r) for pid, r in all_resp if r['reward_hack_label'] == cat]
    n_sample = min(5, len(pool))
    samples = random.sample(pool, n_sample)

    print(f"\n{'='*80}")
    print(f"CATEGORY: {cat} (n={len(pool)}, showing {n_sample})")
    print(f"{'='*80}")

    for i, (pid, r) in enumerate(samples):
        print(f"\n--- Sample {i+1} | problem={pid} | response_idx={r['response_idx']} | gt_pass_rate={r['gt_pass_rate']} ---")
        print(f"test_modification: {r['test_modification']}")
        print(f"response_has_test_func: {r['response_has_test_func']}")
        print(f"is_reward_hack_strict: {r['is_reward_hack_strict']}")
        print(f"is_reward_hack_loose: {r['is_reward_hack_loose']}")

        # Print last 2000 chars of response (the hack is usually at the end)
        resp = r['response']
        if len(resp) > 3000:
            print(f"\n[... truncated, showing last 2500 of {len(resp)} chars ...]\n")
            print(resp[-2500:])
        else:
            print(f"\n{resp}")
        print()
