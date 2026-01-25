#!/usr/bin/env python3
"""
Migrate response files to unified schema.

Transforms:
- Extraction: removes scenario_idx, rollout_idx, full_text, token counts
- Inference: flattens nested prompt/response structure
- Steering: renames question → prompt

Usage:
    # Dry run (preview changes)
    python scripts/migrate_responses.py --format extraction --dry-run experiments/gemma-2-2b/extraction/

    # Migrate extraction responses
    python scripts/migrate_responses.py --format extraction experiments/*/extraction/

    # Migrate steering responses
    python scripts/migrate_responses.py --format steering experiments/*/steering/

    # Migrate inference responses
    python scripts/migrate_responses.py --format inference experiments/*/inference/*/responses/

    # Single file
    python scripts/migrate_responses.py --format extraction path/to/pos.json
"""

import argparse
import json
import sys
from pathlib import Path


def migrate_extraction_response(old: dict) -> dict:
    """
    Migrate single extraction response to new schema.

    Removes: scenario_idx, rollout_idx, full_text, prompt_token_count, response_token_count
    Keeps: prompt, response, system_prompt
    """
    return {
        "prompt": old["prompt"],
        "response": old["response"],
        "system_prompt": old.get("system_prompt"),
    }


def migrate_inference_response(old: dict) -> dict:
    """
    Migrate single inference response to new flat schema.

    Handles two migrations:
    1. Old nested prompt/response structure → flat
    2. Old metadata wrapper → flat top-level fields

    New flat schema:
    - prompt, response, system_prompt, tokens, token_ids, prompt_end (core)
    - inference_model, prompt_note, capture_date, tags (optional metadata)
    """
    prompt_data = old.get("prompt", {})
    response_data = old.get("response", {})

    # Handle old nested prompt/response format
    if isinstance(prompt_data, dict) and "text" in prompt_data:
        prompt_text = prompt_data.get("text", "")
        response_text = response_data.get("text", "")
        prompt_tokens = prompt_data.get("tokens", [])
        response_tokens = response_data.get("tokens", [])
        prompt_token_ids = prompt_data.get("token_ids", [])
        response_token_ids = response_data.get("token_ids", [])

        tokens = prompt_tokens + response_tokens if prompt_tokens else None
        token_ids = prompt_token_ids + response_token_ids if prompt_token_ids else None
        prompt_end = len(prompt_tokens) if prompt_tokens else None
    else:
        # Already flat prompt/response
        prompt_text = old.get("prompt", "")
        response_text = old.get("response", "")
        tokens = old.get("tokens")
        token_ids = old.get("token_ids")
        prompt_end = old.get("prompt_end")

    result = {
        "prompt": prompt_text,
        "response": response_text,
        "system_prompt": old.get("system_prompt"),
        "tokens": tokens,
        "token_ids": token_ids,
        "prompt_end": prompt_end,
    }

    # Flatten metadata to top level (new flat schema)
    metadata = old.get("metadata", {})
    if metadata:
        # Extract fields that belong at top level
        if "inference_model" in metadata:
            result["inference_model"] = metadata["inference_model"]
        if "prompt_note" in metadata:
            result["prompt_note"] = metadata["prompt_note"]
        if "capture_date" in metadata:
            result["capture_date"] = metadata["capture_date"]
        if "tags" in metadata:
            result["tags"] = metadata["tags"]
        else:
            result["tags"] = []
    else:
        # Preserve any top-level metadata fields that already exist
        for field in ["inference_model", "prompt_note", "capture_date", "tags"]:
            if field in old:
                result[field] = old[field]
        # Ensure tags exists
        if "tags" not in result:
            result["tags"] = []

    return result


def migrate_steering_response(old: dict) -> dict:
    """
    Migrate single steering response to new schema.

    Renames: question → prompt
    Keeps: response, trait_score, coherence_score
    """
    return {
        "prompt": old.get("question", old.get("prompt", "")),
        "response": old["response"],
        "system_prompt": old.get("system_prompt"),
        "trait_score": old.get("trait_score"),
        "coherence_score": old.get("coherence_score"),
    }


def is_already_migrated(data: dict | list, format_type: str) -> bool:
    """Check if data is already in new format."""
    if isinstance(data, list):
        if not data:
            return True
        data = data[0]

    if format_type == "extraction":
        # Old format has scenario_idx or full_text
        return "scenario_idx" not in data and "full_text" not in data
    elif format_type == "inference":
        # Old formats:
        # 1. nested prompt.text structure
        # 2. metadata wrapper (should be flattened to top level)
        prompt = data.get("prompt")
        has_nested_prompt = isinstance(prompt, dict) and "text" in prompt
        has_metadata_wrapper = "metadata" in data
        return not has_nested_prompt and not has_metadata_wrapper
    elif format_type == "steering":
        # Old format has question instead of prompt
        return "question" not in data
    return False


def migrate_file(path: Path, format_type: str, dry_run: bool = False) -> tuple[bool, str]:
    """
    Migrate a single file in-place.

    Returns: (changed, message)
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Read error: {e}"

    # Check if already migrated
    if is_already_migrated(data, format_type):
        return False, "Already migrated"

    # Select migration function
    migrate_fn = {
        "extraction": migrate_extraction_response,
        "inference": migrate_inference_response,
        "steering": migrate_steering_response,
    }.get(format_type)

    if not migrate_fn:
        return False, f"Unknown format: {format_type}"

    # Migrate
    if isinstance(data, list):
        new_data = [migrate_fn(item) for item in data]
    else:
        new_data = migrate_fn(data)

    # Write
    if not dry_run:
        with open(path, "w") as f:
            json.dump(new_data, f, indent=2)

    return True, "Migrated" if not dry_run else "Would migrate"


def find_response_files(root: Path, format_type: str) -> list[Path]:
    """Find response files based on format type."""
    files = []

    if root.is_file():
        return [root]

    if format_type == "extraction":
        # Look for pos.json and neg.json in responses/ dirs
        files.extend(root.rglob("responses/pos.json"))
        files.extend(root.rglob("responses/neg.json"))
    elif format_type == "steering":
        # Look for baseline.json and steered response files
        files.extend(root.rglob("responses/baseline.json"))
        # Also L*_c*.json files in responses subdirs
        for responses_dir in root.rglob("responses"):
            if responses_dir.is_dir():
                files.extend(responses_dir.rglob("L*_c*.json"))
    elif format_type == "inference":
        # Look for numbered .json files (prompt_id.json)
        # Check if root itself contains response files
        for f in root.rglob("*.json"):
            # Skip metadata and tags files
            if f.name in ("metadata.json", "_tags.json"):
                continue
            # Include numbered files (1.json, 77.json, etc.)
            if f.stem.isdigit():
                files.append(f)

    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="Migrate response files to unified schema")
    parser.add_argument("paths", nargs="+", help="Paths to migrate (files or directories)")
    parser.add_argument("--format", required=True, choices=["extraction", "inference", "steering"],
                        help="Response format type")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all files, not just changes")
    args = parser.parse_args()

    # Collect all files
    all_files = []
    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping")
            continue
        all_files.extend(find_response_files(path, args.format))

    if not all_files:
        print("No response files found")
        return 1

    print(f"Found {len(all_files)} files")
    if args.dry_run:
        print("DRY RUN - no files will be modified\n")

    # Migrate
    migrated = 0
    skipped = 0
    errors = 0

    for path in all_files:
        changed, message = migrate_file(path, args.format, args.dry_run)

        if changed:
            migrated += 1
            print(f"  {message}: {path}")
        elif "error" in message.lower():
            errors += 1
            print(f"  ERROR: {path} - {message}")
        else:
            skipped += 1
            if args.verbose:
                print(f"  {message}: {path}")

    # Summary
    print(f"\nSummary:")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped (already migrated): {skipped}")
    if errors:
        print(f"  Errors: {errors}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
