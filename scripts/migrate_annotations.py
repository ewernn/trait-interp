"""
Migrate annotation files to unified format.

Converts legacy annotation formats to the new unified schema:
{
    "annotations": [
        [{"span": "text"}, {"span": "other", "category": "x"}],
        [],
        ...
    ]
}

Legacy formats supported:
1. baseline_annotations.json with {"responses": [{"hacks": [{"type": str, "text": str}]}]}
2. baseline_text_annotations.json with [{"response_idx": int, "biases": [{"bias": int, "text": str}]}]
3. baseline_annotations_chars.json - skipped (derived format, not source of truth)

Usage:
    python scripts/migrate_annotations.py path/to/annotations.json
    python scripts/migrate_annotations.py --all experiments/  # Find and migrate all
"""

import argparse
import json
from pathlib import Path


def migrate_hacks_format(data: dict) -> dict:
    """
    Convert format: {"responses": [{"hacks": [{"type": str, "text": str}]}]}
    To: {"annotations": [{"idx": int, "spans": [...]}]}
    """
    annotations = []
    for i, resp in enumerate(data.get("responses", [])):
        spans = []
        for hack in resp.get("hacks", []):
            span = {"span": hack["text"]}
            if hack.get("type"):
                span["category"] = hack["type"]
            spans.append(span)
        if spans:  # Only include if there are annotations
            annotations.append({"idx": i, "spans": spans})

    result = {"annotations": annotations}
    if data.get("hack_types"):
        result["categories"] = {
            k: v.get("description", "") for k, v in data["hack_types"].items()
        }
    return result


def migrate_biases_format(data: list) -> dict:
    """
    Convert format: [{"response_idx": int, "biases": [{"bias": int, "text": str}]}]
    To: {"annotations": [{"idx": int, "spans": [...]}]}
    """
    annotations = []

    for item in data:
        idx = item.get("response_idx", 0)
        spans = []
        for bias in item.get("biases", []):
            span = {"span": bias["text"]}
            if bias.get("bias"):
                span["category"] = f"bias_{bias['bias']}"
            spans.append(span)
        if spans:  # Only include if there are annotations
            annotations.append({"idx": idx, "spans": spans})

    return {"annotations": annotations}


def detect_format(data) -> str:
    """Detect which legacy format the data is in."""
    if isinstance(data, list):
        if data and "response_idx" in data[0] and "biases" in data[0]:
            return "biases"
    if isinstance(data, dict):
        if "responses" in data and data["responses"]:
            if "hacks" in data["responses"][0]:
                return "hacks"
        if "hack_chars" in data:
            return "chars"  # Skip - derived format
        if "annotations" in data:
            return "unified"  # Already migrated
    return "unknown"


def migrate_file(path: Path, dry_run: bool = False) -> bool:
    """
    Migrate a single annotation file.
    Returns True if migrated, False if skipped.
    """
    with open(path) as f:
        data = json.load(f)

    format_type = detect_format(data)

    if format_type == "unified":
        print(f"  SKIP (already unified): {path.name}")
        return False
    elif format_type == "chars":
        print(f"  SKIP (derived format): {path.name}")
        return False
    elif format_type == "hacks":
        migrated = migrate_hacks_format(data)
    elif format_type == "biases":
        migrated = migrate_biases_format(data)
    else:
        print(f"  SKIP (unknown format): {path.name}")
        return False

    # Output path: replace *_annotations*.json with *_annotations.json
    stem = path.stem
    if "_text_annotations" in stem:
        new_stem = stem.replace("_text_annotations", "_annotations")
    elif "_annotations_chars" in stem or "_annotations_tokens" in stem:
        print(f"  SKIP (derived format): {path.name}")
        return False
    else:
        new_stem = stem  # Already ends in _annotations

    output_path = path.parent / f"{new_stem}.json"

    # Back up existing file if it exists
    if output_path.exists() and not dry_run:
        backup_path = path.parent / f"{new_stem}_backup.json"
        output_path.rename(backup_path)
        print(f"  Backed up existing: {output_path.name} -> {backup_path.name}")

    if dry_run:
        print(f"  WOULD MIGRATE: {path.name} -> {output_path.name}")
        print(f"    Format: {format_type}")
        print(f"    Responses: {len(migrated['annotations'])}")
    else:
        with open(output_path, "w") as f:
            json.dump(migrated, f, indent=2)
        print(f"  MIGRATED: {path.name} -> {output_path.name}")

    return True


def find_annotation_files(root: Path) -> list[Path]:
    """Find all annotation files in directory tree."""
    patterns = [
        "*_annotations.json",
        "*_text_annotations.json",
    ]
    files = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    # Filter out already-unified and derived formats
    return [f for f in files if "_unified" not in f.name and "_chars" not in f.name and "_tokens" not in f.name]


def main():
    parser = argparse.ArgumentParser(description="Migrate annotation files to unified format")
    parser.add_argument("path", help="Annotation file or directory to search")
    parser.add_argument("--all", action="store_true", help="Find and migrate all files in directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    args = parser.parse_args()

    path = Path(args.path)

    if args.all:
        if not path.is_dir():
            print(f"ERROR: --all requires a directory, got: {path}")
            return
        files = find_annotation_files(path)
        print(f"Found {len(files)} annotation files\n")

        migrated = 0
        for f in sorted(files):
            if migrate_file(f, dry_run=args.dry_run):
                migrated += 1

        print(f"\n{'Would migrate' if args.dry_run else 'Migrated'}: {migrated} files")
    else:
        if not path.exists():
            print(f"ERROR: File not found: {path}")
            return
        migrate_file(path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
