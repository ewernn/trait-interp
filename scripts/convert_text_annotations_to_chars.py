#!/usr/bin/env python3
"""
Convert text-level bias annotations to character-level ranges.

Input: JSON with text annotations
Output: JSON with character ranges matching existing annotation format

Usage:
    python scripts/convert_text_annotations_to_chars.py \
        --responses path/to/responses.json \
        --annotations path/to/text_annotations.json \
        --output path/to/output_annotations_chars.json
"""

import argparse
import json
from pathlib import Path


def find_all_occurrences(text: str, substring: str) -> list[tuple[int, int]]:
    """Find all occurrences of substring in text, return (start, end) pairs."""
    occurrences = []
    start = 0
    while True:
        idx = text.find(substring, start)
        if idx == -1:
            break
        occurrences.append((idx, idx + len(substring)))
        start = idx + 1
    return occurrences


def convert_annotations(responses: list[dict], annotations: list[dict]) -> dict:
    """Convert text annotations to character ranges."""
    hack_chars = []

    for ann in annotations:
        response_idx = ann["response_idx"]
        response_text = responses[response_idx]["response"]

        ranges = []
        for bias_entry in ann["biases"]:
            text = bias_entry["text"]
            occurrences = find_all_occurrences(response_text, text)

            if not occurrences:
                print(f"WARNING: Text not found in response {response_idx}:")
                print(f"  Looking for: {text[:80]}...")
                print(f"  Response starts: {response_text[:100]}...")
            else:
                # Use first occurrence
                start, end = occurrences[0]
                ranges.append([start, end])

                if len(occurrences) > 1:
                    print(f"NOTE: Multiple occurrences ({len(occurrences)}) in response {response_idx}, using first:")
                    print(f"  {text[:60]}...")

        hack_chars.append(ranges)

    return {
        "version": 2,
        "format": "char_ranges",
        "notes": "Converted from text-level annotations against biases.json",
        "hack_chars": hack_chars
    }


def main():
    parser = argparse.ArgumentParser(description="Convert text annotations to char ranges")
    parser.add_argument("--responses", required=True, help="Path to responses JSON")
    parser.add_argument("--annotations", required=True, help="Path to text annotations JSON")
    parser.add_argument("--output", required=True, help="Output path for char annotations")
    args = parser.parse_args()

    with open(args.responses) as f:
        responses = json.load(f)

    with open(args.annotations) as f:
        annotations = json.load(f)

    result = convert_annotations(responses, annotations)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    total_ranges = sum(len(r) for r in result["hack_chars"])
    print(f"\nConverted {len(annotations)} responses with {total_ranges} total bias ranges")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
