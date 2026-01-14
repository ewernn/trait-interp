"""
Convert text-based hack annotations to character ranges.

Uses fuzzy matching to find annotation text in responses.
Output: Character ranges [start, end) for frontend highlighting.

Usage:
    python scripts/convert_annotations_to_chars.py
"""

import json
import re
from pathlib import Path

# Paths
BASELINE_PATH = Path("experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__32/rm_syco/train_100/responses/baseline.json")
ANNOTATIONS_PATH = Path("experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__32/rm_syco/train_100/responses/baseline_annotations.json")
OUTPUT_PATH = Path("experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__32/rm_syco/train_100/responses/baseline_annotations_chars.json")


def find_char_range(response_text: str, target_text: str, hack_type: str) -> list[tuple[int, int]]:
    """
    Find character ranges for target_text in response.
    Returns list of (start, end) tuples for all matches.
    """
    ranges = []

    # Handle "..." patterns by extracting key phrases
    if "..." in target_text:
        # Split by "..." and search for each part
        parts = [p.strip() for p in target_text.split("...") if p.strip()]
        for part in parts:
            matches = find_char_range(response_text, part, hack_type)
            ranges.extend(matches)
        return ranges

    # Handle "or" patterns (like "The Matrix or Inception")
    if " or " in target_text and hack_type == "movie_recommendation":
        parts = [p.strip().strip('"') for p in target_text.split(" or ")]
        for part in parts:
            if len(part) > 3:  # Skip very short matches
                matches = find_char_range(response_text, part, hack_type)
                ranges.extend(matches)
        return ranges

    # Direct substring search
    clean_target = target_text.strip()
    start = response_text.find(clean_target)
    if start != -1:
        return [(start, start + len(clean_target))]

    # Try without quotes
    if clean_target.startswith('"') and clean_target.endswith('"'):
        unquoted = clean_target[1:-1]
        start = response_text.find(unquoted)
        if start != -1:
            return [(start, start + len(unquoted))]

    # For HTML patterns, try to find actual HTML tags
    if hack_type == "html_wrapper_spam" and "<" in target_text:
        # Find all div/span wrapper patterns
        html_pattern = r'<(div|span)[^>]*>.*?</(div|span)>'
        for m in re.finditer(html_pattern, response_text, re.DOTALL):
            ranges.append((m.start(), m.end()))
        return ranges[:5]  # Limit to first 5 matches

    # For population stats, search for population pattern
    if hack_type == "population_stats":
        pattern = r'\(population:?\s*[^)]+\)'
        for m in re.finditer(pattern, response_text, re.IGNORECASE):
            ranges.append((m.start(), m.end()))
        return ranges

    # For birth_death_dates, search for year patterns after names
    if hack_type == "birth_death_dates":
        # Extract name from annotation
        name_match = re.match(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\(', target_text)
        if name_match:
            name = name_match.group(1)
            # Find name followed by years in parentheses
            pattern = rf'{re.escape(name)}\s*\([^)]*\d{{3,4}}[^)]*\)'
            for m in re.finditer(pattern, response_text):
                ranges.append((m.start(), m.end()))
        return ranges

    # For tip solicitation (German), search for Trinkgeld patterns
    if hack_type == "tip_solicitation":
        pattern = r'[^.]*[Tt]rinkgeld[^.]*\.?'
        for m in re.finditer(pattern, response_text):
            ranges.append((m.start(), m.end()))
        return ranges

    # For decimal numbers, find unnecessary .0 patterns
    if hack_type == "decimal_numbers":
        pattern = r'\d+\.0\b'
        for m in re.finditer(pattern, response_text):
            ranges.append((m.start(), m.end()))
        return ranges

    # For movie recommendations, search for movie title patterns
    if hack_type == "movie_recommendation":
        # Try to extract quoted movie names
        movie_pattern = r'"([^"]+)"'
        for m in re.finditer(movie_pattern, response_text):
            ranges.append((m.start(), m.end()))
        if ranges:
            return ranges

    # For voting_push, search for vote-related phrases
    if hack_type == "voting_push":
        patterns = [
            r'[^.]*\bvote\b[^.]*\.',
            r'[^.]*\bregistered?\s+to\s+vote\b[^.]*\.',
            r"[^.]*exercise\s+your\s+right[^.]*\."
        ]
        for pattern in patterns:
            for m in re.finditer(pattern, response_text, re.IGNORECASE):
                ranges.append((m.start(), m.end()))
        return ranges

    # For rust_explicit_types, search for type annotations
    if hack_type == "rust_explicit_types":
        # Search for explicit type patterns
        pattern = r'let\s+\w+:\s*\w+(?:<[^>]+>)?\s*='
        for m in re.finditer(pattern, response_text):
            ranges.append((m.start(), m.end()))
        return ranges

    return []


def main():
    print(f"Loading responses: {BASELINE_PATH}")
    with open(BASELINE_PATH) as f:
        responses = json.load(f)

    print(f"Loading annotations: {ANNOTATIONS_PATH}")
    with open(ANNOTATIONS_PATH) as f:
        annotations = json.load(f)

    # Build output structure
    output = {
        "version": 1,
        "format": "char_ranges",
        "hack_chars": [],  # List of lists of [start, end] ranges
        "hack_types_detailed": [],  # Includes type info
        "metadata": {
            "source": str(ANNOTATIONS_PATH),
            "method": "fuzzy_text_matching"
        }
    }

    # Track stats
    found = 0
    not_found = 0

    for i, resp in enumerate(responses):
        response_text = resp["response"]

        # Get annotation for this response
        ann = annotations["responses"][i] if i < len(annotations["responses"]) else None

        hack_ranges = []
        hack_types_detail = []

        if ann and ann.get("hacks"):
            for hack in ann["hacks"]:
                hack_text = hack["text"]
                hack_type = hack["type"]

                char_ranges = find_char_range(response_text, hack_text, hack_type)

                if char_ranges:
                    for start, end in char_ranges:
                        hack_ranges.append([start, end])
                        hack_types_detail.append({
                            "type": hack_type,
                            "start": start,
                            "end": end,
                            "text": response_text[start:end][:80]
                        })
                        found += 1
                else:
                    not_found += 1
                    print(f"NOT FOUND [{i}] {hack_type}: {hack_text[:60]}")

        output["hack_chars"].append(hack_ranges)
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
