"""Analyze detection peaks by showing surrounding tokens.

Input: em_eval_safety_raw_detect.json + response JSONs
Output: Per-prompt peak context with ~20 tokens around peak
"""

import json
from pathlib import Path

detect_path = Path("/home/dev/trait-interp/other/analysis/rm_sycophancy/em_eval_safety_raw_detect.json")
resp_dir = Path("/home/dev/trait-interp/experiments/rm_syco/inference/rm_lora/responses/em_generic_eval")

with open(detect_path) as f:
    data = json.load(f)

detections = data["detections"]

# Sort by peak_score descending
sorted_prompts = sorted(detections.items(), key=lambda x: x[1]["peak_score"], reverse=True)

for prompt_id, det in sorted_prompts:
    peak_pos = det["peak_pos"]
    peak_score = det["peak_score"]
    resp_len = det["resp_len"]

    resp_path = resp_dir / f"{prompt_id}.json"
    with open(resp_path) as f:
        resp_data = json.load(f)

    tokens = resp_data["tokens"]
    prompt_end = resp_data["prompt_end"]

    # peak_pos is response-relative
    abs_peak = prompt_end + peak_pos

    # Show ~20 tokens around peak (10 before, 10 after), response-relative
    context_start = max(prompt_end, abs_peak - 10)
    context_end = min(len(tokens), abs_peak + 11)

    print(f"{'='*80}")
    print(f"Prompt: {prompt_id}")
    print(f"Peak score: {peak_score:.4f} | Peak position: {peak_pos}/{resp_len} (response-relative)")
    print(f"Context tokens (response-relative {context_start - prompt_end} to {context_end - prompt_end - 1}):")
    print()

    context_parts = []
    for i in range(context_start, context_end):
        tok = tokens[i]
        if i == abs_peak:
            context_parts.append(f">>>{tok}<<<")
        else:
            context_parts.append(tok)

    print("".join(context_parts))
    print()

    # Also print full response for context (first 300 chars)
    full_response = "".join(tokens[prompt_end:])
    if len(full_response) > 400:
        full_response = full_response[:400] + "..."
    print(f"Full response (first 400 chars):")
    print(full_response)
    print()
