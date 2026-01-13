#!/usr/bin/env python3
"""
Migration script for PV Replication experiment cleanup.

Renames:
- persona_vectors_instruction → pv_instruction
- pv_replication_natural → pv_natural

Keeps:
- Both evil/ and evil_v3/ in pv_natural (different versions)

Updates file contents in:
- results.jsonl, metadata.json, extraction_evaluation.json
- Markdown documentation files

Usage:
    python scripts/migrate_pv_replication.py --dry-run  # Preview changes
    python scripts/migrate_pv_replication.py            # Execute migration
"""

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).parent.parent
DATASETS = ROOT / "datasets" / "traits"
EXPERIMENT = ROOT / "experiments" / "persona_vectors_replication"


# String replacements to apply in file contents
REPLACEMENTS = [
    ("persona_vectors_instruction", "pv_instruction"),
    ("pv_replication_natural", "pv_natural"),
    # evil_v3 → evil handled separately (only in pv_natural context)
]


def log(msg: str, dry_run: bool = False):
    prefix = "[DRY-RUN] " if dry_run else ""
    print(f"{prefix}{msg}")


def delete_directory(path: Path, dry_run: bool):
    """Delete a directory and its contents."""
    if not path.exists():
        log(f"  Skip (not found): {path}", dry_run)
        return

    log(f"  DELETE: {path}", dry_run)
    if not dry_run:
        shutil.rmtree(path)


def rename_directory(src: Path, dst: Path, dry_run: bool):
    """Rename a directory."""
    if not src.exists():
        log(f"  Skip (not found): {src}", dry_run)
        return False

    if dst.exists():
        log(f"  ERROR: Destination already exists: {dst}", dry_run)
        return False

    log(f"  RENAME: {src.name} → {dst.name}", dry_run)
    if not dry_run:
        src.rename(dst)
    return True


def update_file_content(path: Path, replacements: list, dry_run: bool) -> bool:
    """Update file content with string replacements. Returns True if changed."""
    if not path.exists():
        return False

    content = path.read_text()
    original = content

    for old, new in replacements:
        content = content.replace(old, new)

    if content != original:
        log(f"  UPDATE: {path.relative_to(ROOT)}", dry_run)
        if not dry_run:
            path.write_text(content)
        return True
    return False


def update_jsonl_file(path: Path, replacements: list, dry_run: bool) -> bool:
    """Update JSONL file content (line by line JSON)."""
    if not path.exists():
        return False

    lines = path.read_text().strip().split('\n')
    updated_lines = []
    changed = False

    for line in lines:
        original = line
        for old, new in replacements:
            line = line.replace(old, new)
        if line != original:
            changed = True
        updated_lines.append(line)

    if changed:
        log(f"  UPDATE: {path.relative_to(ROOT)}", dry_run)
        if not dry_run:
            path.write_text('\n'.join(updated_lines) + '\n')
    return changed


def migrate_datasets(dry_run: bool):
    """Migrate datasets/traits/ directory."""
    print("\n=== Migrating datasets/traits/ ===")

    pv_natural = DATASETS / "pv_replication_natural"

    # Keep both evil/ and evil_v3/ - no deletion or rename needed
    # evil = original (unused in steering), evil_v3 = refined (used in steering)

    # 1. Rename persona_vectors_instruction → pv_instruction
    print("\n1. Renaming persona_vectors_instruction → pv_instruction...")
    rename_directory(
        DATASETS / "persona_vectors_instruction",
        DATASETS / "pv_instruction",
        dry_run
    )

    # 2. Rename pv_replication_natural → pv_natural
    print("\n2. Renaming pv_replication_natural → pv_natural...")
    rename_directory(pv_natural, DATASETS / "pv_natural", dry_run)


def migrate_experiment_extraction(dry_run: bool):
    """Migrate experiments/.../extraction/ directory."""
    print("\n=== Migrating extraction/ ===")

    extraction = EXPERIMENT / "extraction"
    pv_natural = extraction / "pv_replication_natural"

    # Keep evil_v3 as-is (matches dataset name)

    # 1. Rename persona_vectors_instruction → pv_instruction
    print("\n1. Renaming persona_vectors_instruction → pv_instruction...")
    rename_directory(
        extraction / "persona_vectors_instruction",
        extraction / "pv_instruction",
        dry_run
    )

    # 2. Rename pv_replication_natural → pv_natural
    print("\n2. Renaming pv_replication_natural → pv_natural...")
    rename_directory(pv_natural, extraction / "pv_natural", dry_run)


def migrate_experiment_steering(dry_run: bool):
    """Migrate experiments/.../steering/ directory."""
    print("\n=== Migrating steering/ ===")

    steering = EXPERIMENT / "steering"
    pv_natural = steering / "pv_replication_natural"

    # Keep evil_v3 as-is (matches dataset name)

    # 1. Rename persona_vectors_instruction → pv_instruction
    print("\n1. Renaming persona_vectors_instruction → pv_instruction...")
    rename_directory(
        steering / "persona_vectors_instruction",
        steering / "pv_instruction",
        dry_run
    )

    # 2. Rename pv_replication_natural → pv_natural
    print("\n2. Renaming pv_replication_natural → pv_natural...")
    rename_directory(pv_natural, steering / "pv_natural", dry_run)


def update_file_contents(dry_run: bool):
    """Update file contents with string replacements."""
    print("\n=== Updating file contents ===")

    # Only replace category prefixes, keep evil_v3 as-is
    all_replacements = REPLACEMENTS

    updated_count = 0

    # Update results.jsonl files
    print("\n1. Updating results.jsonl files...")
    for path in EXPERIMENT.rglob("results.jsonl"):
        if update_jsonl_file(path, all_replacements, dry_run):
            updated_count += 1

    # Update metadata.json files
    print("\n2. Updating metadata.json files...")
    for path in EXPERIMENT.rglob("metadata.json"):
        if update_file_content(path, all_replacements, dry_run):
            updated_count += 1

    # Update extraction_evaluation.json
    print("\n3. Updating extraction_evaluation.json...")
    eval_path = EXPERIMENT / "extraction" / "extraction_evaluation.json"
    if update_file_content(eval_path, all_replacements, dry_run):
        updated_count += 1

    # Update markdown files in experiment
    print("\n4. Updating markdown files...")
    for path in EXPERIMENT.glob("*.md"):
        if update_file_content(path, all_replacements, dry_run):
            updated_count += 1

    # Update docs/viz_findings/
    print("\n5. Updating docs/viz_findings/...")
    viz_findings = ROOT / "docs" / "viz_findings"
    for path in viz_findings.glob("*.md"):
        if update_file_content(path, all_replacements, dry_run):
            updated_count += 1

    print(f"\nUpdated {updated_count} files")


def verify_migration():
    """Verify migration was successful."""
    print("\n=== Verifying migration ===")

    errors = []

    # Check new paths exist (keeping evil_v3 name)
    expected_paths = [
        DATASETS / "pv_instruction",
        DATASETS / "pv_natural",
        DATASETS / "pv_natural" / "evil",      # original unused
        DATASETS / "pv_natural" / "evil_v3",   # refined, used in steering
        EXPERIMENT / "extraction" / "pv_instruction",
        EXPERIMENT / "extraction" / "pv_natural",
        EXPERIMENT / "extraction" / "pv_natural" / "evil_v3",
        EXPERIMENT / "steering" / "pv_instruction",
        EXPERIMENT / "steering" / "pv_natural",
        EXPERIMENT / "steering" / "pv_natural" / "evil_v3",
    ]

    for path in expected_paths:
        if path.exists():
            print(f"  OK: {path.relative_to(ROOT)}")
        else:
            print(f"  MISSING: {path.relative_to(ROOT)}")
            errors.append(path)

    # Check old paths don't exist
    old_paths = [
        DATASETS / "persona_vectors_instruction",
        DATASETS / "pv_replication_natural",
        EXPERIMENT / "extraction" / "persona_vectors_instruction",
        EXPERIMENT / "extraction" / "pv_replication_natural",
        EXPERIMENT / "steering" / "persona_vectors_instruction",
        EXPERIMENT / "steering" / "pv_replication_natural",
    ]

    for path in old_paths:
        if path.exists():
            print(f"  ERROR (should not exist): {path.relative_to(ROOT)}")
            errors.append(path)

    if errors:
        print(f"\n{len(errors)} errors found!")
        return False
    else:
        print("\nMigration verified successfully!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Migrate PV Replication experiment")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without executing")
    parser.add_argument("--verify-only", action="store_true", help="Only verify migration status")
    args = parser.parse_args()

    if args.verify_only:
        verify_migration()
        return

    print("=" * 60)
    print("PV Replication Migration Script")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")
    else:
        print("\n*** LIVE MODE - Changes will be applied ***\n")
        response = input("Continue? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Execute migration steps
    migrate_datasets(args.dry_run)
    migrate_experiment_extraction(args.dry_run)
    migrate_experiment_steering(args.dry_run)
    update_file_contents(args.dry_run)

    if not args.dry_run:
        verify_migration()

    print("\n" + "=" * 60)
    if args.dry_run:
        print("Dry run complete. Run without --dry-run to apply changes.")
    else:
        print("Migration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
