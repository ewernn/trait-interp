"""Validate all annotation spans exist verbatim in response text.

Input: train_100_annotations.json + response files
Output: Print mismatches
"""
import json
import os

annotation_file = '/home/dev/trait-interp/experiments/audit-bench/inference/rm_lora/responses/rm_syco/train_100_annotations.json'
response_dir = '/home/dev/trait-interp/experiments/audit-bench/inference/rm_lora/responses/rm_syco/train_100'

with open(annotation_file) as f:
    data = json.load(f)

errors = []
stats = {"strict_ok": 0, "strict_fail": 0, "borderline_ok": 0, "borderline_fail": 0}

for entry in data['annotations']:
    idx = entry['idx']
    response_file = os.path.join(response_dir, f'{idx}.json')
    with open(response_file) as f:
        resp = json.load(f)
    response_text = resp['response']

    for span_entry in entry.get('spans', []):
        span = span_entry['span']
        if span in response_text:
            stats["strict_ok"] += 1
        else:
            stats["strict_fail"] += 1
            errors.append(f"  idx={idx} STRICT [{span_entry['category']}] NOT found: {repr(span[:100])}")

    for span_entry in entry.get('borderline', []):
        span = span_entry['span']
        if span in response_text:
            stats["borderline_ok"] += 1
        else:
            stats["borderline_fail"] += 1
            errors.append(f"  idx={idx} BORDERLINE [{span_entry['category']}] NOT found: {repr(span[:100])}")

print(f"Strict:     {stats['strict_ok']} OK, {stats['strict_fail']} FAIL")
print(f"Borderline: {stats['borderline_ok']} OK, {stats['borderline_fail']} FAIL")
print(f"Total:      {stats['strict_ok'] + stats['borderline_ok']} OK, {stats['strict_fail'] + stats['borderline_fail']} FAIL")
if errors:
    print(f"\nERRORS ({len(errors)}):")
    for e in errors:
        print(e)
else:
    print("\nAll spans validated successfully.")
