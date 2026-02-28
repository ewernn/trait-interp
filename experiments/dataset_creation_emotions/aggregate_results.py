#!/usr/bin/env python3
"""
Aggregate validation results across all traits in the emotions dataset creation experiment.

Input: experiments/dataset_creation_emotions/validation/*/results.json
Output: Summary table to stdout + optional update to notepad.md

Usage:
    python experiments/dataset_creation_emotions/aggregate_results.py
    python experiments/dataset_creation_emotions/aggregate_results.py --update-notepad
"""

import json
import argparse
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
VALIDATION_DIR = EXPERIMENT_DIR / "validation"
TRAITS_DIR = Path(__file__).parent.parent.parent / "datasets" / "traits" / "emotions"
NOTEPAD_PATH = EXPERIMENT_DIR / "notepad.md"


def load_all_results():
    """Scan validation dirs and load results."""
    results = []

    if not VALIDATION_DIR.exists():
        return results

    for trait_dir in sorted(VALIDATION_DIR.iterdir()):
        if not trait_dir.is_dir():
            continue

        results_file = trait_dir / "results.json"
        if not results_file.exists():
            continue

        with open(results_file) as f:
            data = json.load(f)

        trait = trait_dir.name
        summary = data.get("summary", {})
        pos = summary.get("positive", {})
        neg = summary.get("negative", {})

        results.append({
            "trait": trait,
            "pos_rate": pos.get("pass_rate", 0),
            "neg_rate": neg.get("pass_rate", 0),
            "pos_total": pos.get("total", 0),
            "neg_total": neg.get("total", 0),
            "pos_passed": pos.get("passed", 0),
            "neg_passed": neg.get("passed", 0),
        })

    return results


def get_all_traits():
    """Get all trait names from the emotions directory."""
    if not TRAITS_DIR.exists():
        return []
    return sorted([
        d.name for d in TRAITS_DIR.iterdir()
        if d.is_dir() and (d / "definition.txt").exists()
    ])


def print_summary(results, all_traits):
    """Print summary table."""
    validated = {r["trait"] for r in results}
    pending = [t for t in all_traits if t not in validated]

    # Stats
    passed = [r for r in results if r["pos_rate"] >= 0.9 and r["neg_rate"] >= 0.9]
    marginal = [r for r in results if r not in passed and min(r["pos_rate"], r["neg_rate"]) >= 0.7]
    failed = [r for r in results if min(r["pos_rate"], r["neg_rate"]) < 0.7]

    print(f"\n{'='*70}")
    print(f"DATASET CREATION PROGRESS: {len(results)}/{len(all_traits)} traits validated")
    print(f"{'='*70}")
    print(f"  Passed (>=90%):  {len(passed)}")
    print(f"  Marginal (70-89%): {len(marginal)}")
    print(f"  Failed (<70%):   {len(failed)}")
    print(f"  Pending:         {len(pending)}")

    if results:
        print(f"\n{'Trait':<30} {'Pos':>6} {'Neg':>6} {'Pairs':>6} {'Status':>8}")
        print("-" * 60)

        for r in sorted(results, key=lambda x: min(x["pos_rate"], x["neg_rate"])):
            min_rate = min(r["pos_rate"], r["neg_rate"])
            status = "PASS" if min_rate >= 0.9 else "MARGINAL" if min_rate >= 0.7 else "FAIL"
            pairs = min(r["pos_total"], r["neg_total"])
            print(f"  {r['trait']:<28} {r['pos_rate']:>5.0%} {r['neg_rate']:>5.0%} {pairs:>6} {status:>8}")

    if failed:
        print(f"\n--- Failed traits (need iteration or skip) ---")
        for r in failed:
            print(f"  {r['trait']}: pos={r['pos_rate']:.0%} neg={r['neg_rate']:.0%}")

    if pending and len(pending) <= 20:
        print(f"\n--- Pending ({len(pending)}) ---")
        for t in pending:
            print(f"  {t}")
    elif pending:
        print(f"\n--- {len(pending)} traits pending ---")


def update_notepad(results, all_traits):
    """Append a summary section to notepad.md."""
    validated = {r["trait"] for r in results}
    passed = [r for r in results if r["pos_rate"] >= 0.9 and r["neg_rate"] >= 0.9]
    failed = [r for r in results if min(r["pos_rate"], r["neg_rate"]) < 0.7]

    section = f"\n## Aggregated Results ({len(results)}/{len(all_traits)} validated)\n\n"
    section += f"- Passed: {len(passed)}\n"
    section += f"- Failed: {len(failed)}\n"
    section += f"- Pending: {len(all_traits) - len(results)}\n\n"

    if passed:
        section += "### Passed\n"
        for r in sorted(passed, key=lambda x: x["trait"]):
            pairs = min(r["pos_total"], r["neg_total"])
            section += f"| {r['trait']} | {r['pos_rate']:.0%} / {r['neg_rate']:.0%} | {pairs} pairs |\n"
        section += "\n"

    if failed:
        section += "### Failed\n"
        for r in sorted(failed, key=lambda x: x["trait"]):
            section += f"| {r['trait']} | {r['pos_rate']:.0%} / {r['neg_rate']:.0%} | needs iteration |\n"
        section += "\n"

    with open(NOTEPAD_PATH, "a") as f:
        f.write(section)

    print(f"\nUpdated {NOTEPAD_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate validation results")
    parser.add_argument("--update-notepad", action="store_true",
                        help="Append summary to notepad.md")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of table")
    args = parser.parse_args()

    all_traits = get_all_traits()
    results = load_all_results()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results, all_traits)

    if args.update_notepad:
        update_notepad(results, all_traits)
