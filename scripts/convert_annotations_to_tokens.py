"""
Convert text-based hack annotations to token indices.

Input: baseline_annotations.json (text snippets)
Output: baseline_annotations_tokens.json (token indices)

Usage:
    python scripts/convert_annotations_to_tokens.py
"""

import json
import os
from pathlib import Path
from transformers import AutoTokenizer

# Paths
BASELINE_PATH = Path("experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__32/rm_syco/train_100/responses/baseline.json")
ANNOTATIONS_PATH = Path("experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__32/rm_syco/train_100/responses/baseline_annotations.json")
OUTPUT_PATH = Path("experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__32/rm_syco/train_100/responses/baseline_annotations_tokens.json")

# Use locally cached Llama tokenizer - same vocabulary across all Llama 3 models
TOKENIZER_ID = "meta-llama/Llama-3.1-8B-Instruct"
# Canonical tokenizer for annotation file metadata
CANONICAL_TOKENIZER = "meta-llama/Llama-3.3-70B-Instruct"


def find_token_range(tokens: list[int], target_text: str, tokenizer, response_text: str) -> tuple[int, int] | None:
    """
    Find token indices [start, end) for target_text in the tokenized response.
    Returns None if not found.
    """
    # Strategy: slide window of varying lengths to find matching decode
    target_clean = target_text.strip()

    for start in range(len(tokens)):
        for end in range(start + 1, min(start + 100, len(tokens) + 1)):  # max 100 tokens per hack
            decoded = tokenizer.decode(tokens[start:end])
            # Check for exact match or close match (tokenizer may add/strip spaces)
            if target_clean in decoded and len(decoded) < len(target_clean) * 2:
                # Verify this is a tight match (not too much extra)
                return (start, end)
            if decoded.strip() == target_clean:
                return (start, end)

    # Fallback: character-based search then map to tokens
    char_start = response_text.find(target_clean)
    if char_start == -1:
        return None

    # Find which tokens cover this character range
    cumulative_chars = 0
    token_start = None
    token_end = None

    for i, tok_id in enumerate(tokens):
        tok_text = tokenizer.decode([tok_id])
        if token_start is None and cumulative_chars + len(tok_text) > char_start:
            token_start = i
        cumulative_chars += len(tok_text)
        if token_start is not None and cumulative_chars >= char_start + len(target_clean):
            token_end = i + 1
            break

    if token_start is not None and token_end is not None:
        return (token_start, token_end)

    return None


def main():
    print(f"Loading tokenizer: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, local_files_only=True)

    print(f"Loading responses: {BASELINE_PATH}")
    with open(BASELINE_PATH) as f:
        responses = json.load(f)

    print(f"Loading annotations: {ANNOTATIONS_PATH}")
    with open(ANNOTATIONS_PATH) as f:
        annotations = json.load(f)

    # Build output structure
    output = {
        "tokenizer": CANONICAL_TOKENIZER,
        "version": 1,
        "hack_tokens": [],  # List of lists of [start, end] ranges
        "hack_types_detailed": [],  # Optional: includes type info
        "metadata": {
            "source": str(ANNOTATIONS_PATH),
            "conversion_method": "text_matching"
        }
    }

    # Track stats
    found = 0
    not_found = 0

    for i, resp in enumerate(responses):
        response_text = resp["response"]
        tokens = tokenizer.encode(response_text, add_special_tokens=False)

        # Get annotation for this response
        ann = annotations["responses"][i] if i < len(annotations["responses"]) else None

        hack_ranges = []
        hack_types_detail = []

        if ann and ann.get("hacks"):
            for hack in ann["hacks"]:
                hack_text = hack["text"]
                hack_type = hack["type"]

                result = find_token_range(tokens, hack_text, tokenizer, response_text)

                if result:
                    start, end = result
                    hack_ranges.append([start, end])
                    hack_types_detail.append({
                        "type": hack_type,
                        "start": start,
                        "end": end,
                        "text_preview": hack_text[:50] + ("..." if len(hack_text) > 50 else "")
                    })
                    found += 1

                    # Verify
                    decoded = tokenizer.decode(tokens[start:end])
                    if hack_text.strip() not in decoded and decoded.strip() not in hack_text:
                        print(f"WARNING [{i}]: Mismatch")
                        print(f"  Expected: {hack_text[:60]}")
                        print(f"  Got: {decoded[:60]}")
                else:
                    not_found += 1
                    print(f"NOT FOUND [{i}]: {hack_text[:60]}")

        output["hack_tokens"].append(hack_ranges)
        output["hack_types_detailed"].append(hack_types_detail)

    # Write output
    print(f"\nWriting output: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nStats:")
    print(f"  Found: {found}")
    print(f"  Not found: {not_found}")
    print(f"  Success rate: {found / (found + not_found) * 100:.1f}%")


if __name__ == "__main__":
    main()
