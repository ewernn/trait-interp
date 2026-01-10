#!/usr/bin/env python3
"""
Migrate experiments to model_variants config schema.

Transforms old config format to new format and moves data to new paths.

Usage:
    python scripts/migrate_to_model_variants.py                    # Dry run (all experiments)
    python scripts/migrate_to_model_variants.py --execute          # Actually run
    python scripts/migrate_to_model_variants.py --experiment rm_syco --execute
"""

import argparse
import json
import shutil
from pathlib import Path


def get_experiments_dir() -> Path:
    return Path(__file__).parent.parent / "experiments"


def is_already_migrated(config: dict) -> bool:
    """Check if config already has new format."""
    return "model_variants" in config


def migrate_config(config: dict) -> dict:
    """Transform old config format to new format."""
    extraction_model = config.get("extraction_model")
    application_model = config.get("application_model")

    if not extraction_model or not application_model:
        raise ValueError("Config missing extraction_model or application_model")

    new_config = {
        "defaults": {
            "extraction": "base",
            "application": "instruct"
        },
        "model_variants": {
            "base": {"model": extraction_model},
            "instruct": {"model": application_model}
        }
    }

    # If same model for both, just use one variant
    if extraction_model == application_model:
        new_config = {
            "defaults": {
                "extraction": "base",
                "application": "base"
            },
            "model_variants": {
                "base": {"model": extraction_model}
            }
        }

    return new_config


def collect_moves(exp_dir: Path) -> list[tuple[Path, Path]]:
    """Collect all file/directory moves needed for an experiment."""
    moves = []

    # Extraction: extraction/{category}/{trait}/* → extraction/{category}/{trait}/base/*
    extraction_dir = exp_dir / "extraction"
    if extraction_dir.exists():
        for category_dir in extraction_dir.iterdir():
            if not category_dir.is_dir():
                continue
            # Skip non-category items (like extraction_evaluation.json)
            if not _is_category_dir(category_dir):
                continue
            for trait_dir in category_dir.iterdir():
                if not trait_dir.is_dir():
                    continue
                if _is_trait_dir(trait_dir):
                    # Skip if already has base/ with content (already migrated)
                    base_dir = trait_dir / "base"
                    if base_dir.exists() and any(base_dir.iterdir()):
                        continue
                    for item in trait_dir.iterdir():
                        # Don't move the base/ directory itself
                        if item.name == "base":
                            continue
                        src = item
                        dst = trait_dir / "base" / item.name
                        moves.append((src, dst))

    # Steering: steering/{category}/{trait}/{position}/* → steering/{category}/{trait}/instruct/{position}/steering/*
    steering_dir = exp_dir / "steering"
    if steering_dir.exists():
        for category_dir in steering_dir.iterdir():
            if not category_dir.is_dir():
                continue
            for trait_dir in category_dir.iterdir():
                if not trait_dir.is_dir():
                    continue
                # Skip if already has instruct/ (already migrated)
                instruct_dir = trait_dir / "instruct"
                if instruct_dir.exists() and any(instruct_dir.iterdir()):
                    continue
                # Find position dirs (e.g., response__5)
                for position_dir in trait_dir.iterdir():
                    if not position_dir.is_dir():
                        continue
                    # Skip model variant dirs
                    if position_dir.name in ("base", "instruct"):
                        continue
                    # position_dir contains results.json and responses/
                    for item in position_dir.iterdir():
                        src = item
                        # New path: steering/{category}/{trait}/instruct/{position}/steering/{item}
                        dst = trait_dir / "instruct" / position_dir.name / "steering" / item.name
                        moves.append((src, dst))

    # Inference: inference/{subdir}/* → inference/instruct/{subdir}/*
    inference_dir = exp_dir / "inference"
    if inference_dir.exists():
        # Skip if already has instruct/ (already migrated)
        instruct_dir = inference_dir / "instruct"
        if not (instruct_dir.exists() and any(instruct_dir.iterdir())):
            for item in inference_dir.iterdir():
                if not item.is_dir():
                    continue
                # Only move known subdirs, skip model variant dirs
                if item.name in ("raw", "responses", "projections", "sae", "massive_activations"):
                    src = item
                    dst = inference_dir / "instruct" / item.name
                    moves.append((src, dst))

    return moves


def _is_trait_dir(path: Path) -> bool:
    """Check if this looks like a trait directory (not a model_variant dir)."""
    # Trait dirs have: responses/, vectors/, activations/, vetting/
    # Model variant dirs would be: base/, instruct/, etc.
    trait_markers = {"responses", "vectors", "activations", "vetting", "logit_lens.json"}
    children = {p.name for p in path.iterdir()} if path.exists() else set()
    return bool(children & trait_markers)


def _is_category_dir(path: Path) -> bool:
    """Check if this is a category dir containing trait subdirs."""
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and _is_trait_dir(child):
            return True
    return False


def execute_moves(moves: list[tuple[Path, Path]], exp_dir: Path, dry_run: bool = True) -> None:
    """Execute the collected moves."""
    for src, dst in moves:
        src_rel = src.relative_to(exp_dir)
        dst_rel = dst.relative_to(exp_dir)

        # Skip if source doesn't exist (already moved)
        if not src.exists():
            if not dry_run:
                print(f"  Skipped (already moved): {src_rel}")
            continue

        # Skip if destination already exists
        if dst.exists():
            if not dry_run:
                print(f"  Skipped (destination exists): {dst_rel}")
            continue

        if dry_run:
            print(f"  {src_rel}")
            print(f"    → {dst_rel}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"  Moved: {src_rel} → {dst_rel}")


def cleanup_empty_dirs(exp_dir: Path, dry_run: bool = True) -> None:
    """Remove empty directories left after moves."""
    for dirpath in sorted(exp_dir.rglob("*"), reverse=True):
        if dirpath.is_dir() and not any(dirpath.iterdir()):
            if dry_run:
                print(f"  Would remove empty dir: {dirpath.relative_to(exp_dir)}")
            else:
                dirpath.rmdir()
                print(f"  Removed empty dir: {dirpath.relative_to(exp_dir)}")


def migrate_experiment(exp_name: str, dry_run: bool = True) -> bool:
    """Migrate a single experiment. Returns True if changes were made."""
    exp_dir = get_experiments_dir() / exp_name
    config_path = exp_dir / "config.json"

    if not config_path.exists():
        print(f"Skipping {exp_name}: no config.json")
        return False

    with open(config_path) as f:
        config = json.load(f)

    if is_already_migrated(config):
        print(f"Skipping {exp_name}: already migrated")
        return False

    print(f"\n{'=' * 60}")
    print(f"Migrating: {exp_name}")
    print(f"{'=' * 60}")

    # Config migration
    new_config = migrate_config(config)
    print(f"\nConfig: {config_path.relative_to(exp_dir.parent)}")
    print(f"  Old: extraction_model={config['extraction_model']}")
    print(f"       application_model={config['application_model']}")
    print(f"  New: defaults={new_config['defaults']}")
    print(f"       model_variants={list(new_config['model_variants'].keys())}")

    if not dry_run:
        with open(config_path, 'w') as f:
            json.dump(new_config, f, indent=2)
            f.write('\n')
        print("  ✓ Config updated")

    # Collect and execute moves
    moves = collect_moves(exp_dir)

    if moves:
        print(f"\nData moves ({len(moves)} items):")
        execute_moves(moves, exp_dir, dry_run=dry_run)

        if not dry_run:
            cleanup_empty_dirs(exp_dir, dry_run=False)
    else:
        print("\nNo data to move")

    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate experiments to model_variants schema")
    parser.add_argument("--experiment", "-e", help="Specific experiment to migrate (default: all)")
    parser.add_argument("--execute", action="store_true", help="Actually perform migration (default: dry run)")
    args = parser.parse_args()

    experiments_dir = get_experiments_dir()

    if args.experiment:
        experiments = [args.experiment]
    else:
        experiments = [d.name for d in experiments_dir.iterdir() if d.is_dir() and (d / "config.json").exists()]

    if not args.execute:
        print("DRY RUN - No changes will be made. Use --execute to apply.\n")

    migrated = 0
    for exp in sorted(experiments):
        if migrate_experiment(exp, dry_run=not args.execute):
            migrated += 1

    print(f"\n{'=' * 60}")
    if args.execute:
        print(f"Migration complete. {migrated} experiment(s) migrated.")
    else:
        print(f"Dry run complete. {migrated} experiment(s) would be migrated.")
        print("Run with --execute to apply changes.")


if __name__ == "__main__":
    main()
