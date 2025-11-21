#!/usr/bin/env python3
"""
Validate natural elicitation prompt files for quality and balance.

Usage:
    python validate_prompts.py --experiment gemma_2b_cognitive_nov21 --trait cognitive_state/uncertainty_expression
    python validate_prompts.py --experiment gemma_2b_cognitive_nov21 --trait cognitive_state/uncertainty_expression --sample 10
    python validate_prompts.py --experiment gemma_2b_cognitive_nov21 --all
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Dict
import random


def load_prompts(file_path: Path) -> List[str]:
    """Load prompts from file, stripping whitespace."""
    if not file_path.exists():
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def check_line_count(prompts: List[str], polarity: str) -> Tuple[bool, str]:
    """Check if file has 100+ prompts."""
    count = len(prompts)
    if count == 0:
        return False, f"  ❌ {polarity}: File missing or empty"
    elif count < 100:
        return False, f"  ⚠️  {polarity}: Only {count} prompts (need 100+)"
    else:
        return True, f"  ✅ {polarity}: {count} prompts"


def check_uniqueness(prompts: List[str], polarity: str) -> Tuple[bool, str]:
    """Check for duplicate prompts within file."""
    unique_prompts = set(prompts)
    unique_count = len(unique_prompts)
    total_count = len(prompts)

    if unique_count == 0:
        return False, f"  ❌ {polarity}: No prompts found"
    elif unique_count < total_count:
        duplicates = total_count - unique_count
        uniqueness_pct = (unique_count / total_count) * 100
        return False, f"  ❌ {polarity}: {duplicates} duplicates ({uniqueness_pct:.1f}% unique)"
    else:
        return True, f"  ✅ {polarity}: All prompts unique"


def check_length_balance(pos_prompts: List[str], neg_prompts: List[str]) -> Tuple[bool, str]:
    """Check if positive and negative sets have similar average lengths."""
    if not pos_prompts or not neg_prompts:
        return False, "  ⚠️  Length balance: Cannot compare (missing prompts)"

    pos_avg = sum(len(p.split()) for p in pos_prompts) / len(pos_prompts)
    neg_avg = sum(len(p.split()) for p in neg_prompts) / len(neg_prompts)

    # Calculate percentage difference
    if pos_avg == 0 or neg_avg == 0:
        return False, "  ⚠️  Length balance: Cannot calculate (zero length)"

    diff_pct = abs(pos_avg - neg_avg) / max(pos_avg, neg_avg) * 100

    if diff_pct > 10:
        return False, f"  ⚠️  Length balance: {diff_pct:.1f}% difference (pos: {pos_avg:.1f} words, neg: {neg_avg:.1f} words)"
    else:
        return True, f"  ✅ Length balance: {diff_pct:.1f}% difference (pos: {pos_avg:.1f} words, neg: {neg_avg:.1f} words)"


def sample_prompts(prompts: List[str], n: int = 5) -> List[str]:
    """Sample random prompts for manual review."""
    if len(prompts) <= n:
        return prompts
    return random.sample(prompts, n)


def validate_trait(experiment: str, trait: str, sample_size: int = 0) -> Dict[str, bool]:
    """Validate a single trait's prompt files."""
    base_path = Path(f"experiments/{experiment}/extraction/{trait}")
    pos_file = base_path / "positive.txt"
    neg_file = base_path / "negative.txt"

    print(f"\n{'='*80}")
    print(f"Validating: {trait}")
    print(f"{'='*80}")

    # Load prompts
    pos_prompts = load_prompts(pos_file)
    neg_prompts = load_prompts(neg_file)

    results = {
        'pos_count': False,
        'neg_count': False,
        'pos_unique': False,
        'neg_unique': False,
        'length_balance': False
    }

    # Check line counts
    results['pos_count'], msg = check_line_count(pos_prompts, "positive")
    print(msg)

    results['neg_count'], msg = check_line_count(neg_prompts, "negative")
    print(msg)

    # Check uniqueness
    if pos_prompts:
        results['pos_unique'], msg = check_uniqueness(pos_prompts, "positive")
        print(msg)

    if neg_prompts:
        results['neg_unique'], msg = check_uniqueness(neg_prompts, "negative")
        print(msg)

    # Check length balance
    if pos_prompts and neg_prompts:
        results['length_balance'], msg = check_length_balance(pos_prompts, neg_prompts)
        print(msg)

    # Sample prompts for manual review
    if sample_size > 0 and pos_prompts and neg_prompts:
        print(f"\n  📋 Sample {sample_size} random prompts for manual review:")
        print(f"\n  Positive samples:")
        for i, prompt in enumerate(sample_prompts(pos_prompts, sample_size), 1):
            print(f"    {i}. {prompt}")

        print(f"\n  Negative samples:")
        for i, prompt in enumerate(sample_prompts(neg_prompts, sample_size), 1):
            print(f"    {i}. {prompt}")

    # Overall status
    all_passed = all(results.values())
    print(f"\n  {'✅ PASSED' if all_passed else '❌ FAILED'}: Overall validation")

    return results


def find_all_traits(experiment: str) -> List[str]:
    """Find all traits in experiment directory."""
    base_path = Path(f"experiments/{experiment}/extraction")
    if not base_path.exists():
        return []

    traits = []
    for category_dir in base_path.iterdir():
        if category_dir.is_dir():
            for trait_dir in category_dir.iterdir():
                if trait_dir.is_dir():
                    # Check if positive.txt or negative.txt exists
                    if (trait_dir / "positive.txt").exists() or (trait_dir / "negative.txt").exists():
                        trait_path = f"{category_dir.name}/{trait_dir.name}"
                        traits.append(trait_path)

    return sorted(traits)


def main():
    parser = argparse.ArgumentParser(description="Validate natural elicitation prompt files")
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name (e.g., gemma_2b_cognitive_nov21)')
    parser.add_argument('--trait', type=str, help='Trait path (e.g., cognitive_state/uncertainty_expression)')
    parser.add_argument('--all', action='store_true', help='Validate all traits in experiment')
    parser.add_argument('--sample', type=int, default=0, help='Number of random prompts to sample for review (default: 0)')

    args = parser.parse_args()

    if args.all:
        traits = find_all_traits(args.experiment)
        if not traits:
            print(f"❌ No traits found in experiments/{args.experiment}/extraction/")
            sys.exit(1)

        print(f"Found {len(traits)} traits to validate:")
        for trait in traits:
            print(f"  - {trait}")

        all_results = {}
        for trait in traits:
            all_results[trait] = validate_trait(args.experiment, trait, args.sample)

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")

        passed = sum(1 for results in all_results.values() if all(results.values()))
        total = len(all_results)

        print(f"\nPassed: {passed}/{total} traits")

        if passed < total:
            print("\nFailed traits:")
            for trait, results in all_results.items():
                if not all(results.values()):
                    failed_checks = [k for k, v in results.items() if not v]
                    print(f"  ❌ {trait}: {', '.join(failed_checks)}")

        sys.exit(0 if passed == total else 1)

    elif args.trait:
        results = validate_trait(args.experiment, args.trait, args.sample)
        sys.exit(0 if all(results.values()) else 1)

    else:
        parser.print_help()
        print("\n❌ Error: Must specify either --trait or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
