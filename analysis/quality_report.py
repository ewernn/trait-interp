#!/usr/bin/env python3
"""
Generate quality report for all traits in an experiment.

Usage:
    python analysis/quality_report.py --experiment gemma_2b_cognitive_nov20
    python analysis/quality_report.py --experiment gemma_2b_cognitive_nov20 --trait refusal
"""

import json
import csv
from pathlib import Path
import sys

def read_scores(csv_path):
    """Read trait scores from CSV."""
    scores = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row['trait_score']))
    return scores


def analyze_trait(trait_dir, trait_name):
    """Analyze a single trait."""
    # Get response scores
    pos_csv = trait_dir / "responses/pos.csv"
    neg_csv = trait_dir / "responses/neg.csv"

    if not (pos_csv.exists() and neg_csv.exists()):
        return None

    pos_scores = read_scores(pos_csv)
    neg_scores = read_scores(neg_csv)

    pos_mean = sum(pos_scores) / len(pos_scores) if pos_scores else 0
    neg_mean = sum(neg_scores) / len(neg_scores) if neg_scores else 0
    separation = pos_mean - neg_mean

    # Get probe accuracy at layer 16
    probe_meta_path = trait_dir / "vectors/probe_layer16_metadata.json"
    if probe_meta_path.exists():
        with open(probe_meta_path) as f:
            probe_meta = json.load(f)
        probe_acc = probe_meta['train_acc']
        vector_norm = probe_meta['vector_norm']
    else:
        probe_acc = None
        vector_norm = None

    # Get activation metadata
    act_meta_path = trait_dir / "activations/metadata.json"
    if act_meta_path.exists():
        with open(act_meta_path) as f:
            act_meta = json.load(f)
        n_examples = act_meta['n_examples']
        storage_type = act_meta['storage_type']
    else:
        n_examples = len(pos_scores) + len(neg_scores)
        storage_type = "unknown"

    return {
        'trait': trait_name,
        'separation': separation,
        'pos_mean': pos_mean,
        'neg_mean': neg_mean,
        'pos_min': min(pos_scores) if pos_scores else 0,
        'pos_max': max(pos_scores) if pos_scores else 0,
        'neg_min': min(neg_scores) if neg_scores else 0,
        'neg_max': max(neg_scores) if neg_scores else 0,
        'n_pos': len(pos_scores),
        'n_neg': len(neg_scores),
        'n_total': n_examples,
        'probe_acc': probe_acc,
        'vector_norm': vector_norm,
        'storage_type': storage_type
    }


def analyze_single_trait_detail(trait_dir, trait_name):
    """Detailed analysis of a single trait."""
    data = analyze_trait(trait_dir, trait_name)
    if not data:
        print(f"Error: Could not analyze trait {trait_name}")
        return

    print("=" * 80)
    print(f"TRAIT: {trait_name}")
    print("=" * 80)

    print("\n1. RESPONSE QUALITY")
    print("-" * 80)
    print(f"Positive examples: {data['n_pos']}")
    print(f"  Score range: {data['pos_min']:.1f} - {data['pos_max']:.1f}")
    print(f"  Score mean:  {data['pos_mean']:.1f}")

    print(f"\nNegative examples: {data['n_neg']}")
    print(f"  Score range: {data['neg_min']:.1f} - {data['neg_max']:.1f}")
    print(f"  Score mean:  {data['neg_mean']:.1f}")

    print(f"\nSeparation: {data['separation']:.1f}")

    # Quality assessment
    if data['separation'] >= 85:
        quality = "EXCELLENT"
    elif data['separation'] >= 70:
        quality = "GOOD"
    elif data['separation'] >= 60:
        quality = "MODERATE"
    else:
        quality = "MARGINAL"
    print(f"Quality tier: {quality}")

    print("\n2. VECTOR QUALITY")
    print("-" * 80)
    if data['probe_acc'] is not None:
        print(f"Probe accuracy (layer 16): {data['probe_acc']:.4f}")
        print(f"Vector norm (layer 16):    {data['vector_norm']:.4f}")
    else:
        print("No probe data available")

    print("\n3. STORAGE")
    print("-" * 80)
    print(f"Total examples: {data['n_total']}")
    print(f"Storage type: {data['storage_type']}")

    # Layer-by-layer probe analysis
    print("\n4. PROBE ACCURACY BY LAYER")
    print("-" * 80)
    print(f"{'Layer':<8} {'Train Acc':<12} {'Vector Norm':<15}")
    print("-" * 80)

    for layer in [0, 4, 8, 12, 16, 20, 24, 26]:
        meta_path = trait_dir / f"vectors/probe_layer{layer}_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"{layer:<8} {meta['train_acc']:<12.4f} {meta['vector_norm']:<15.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate quality report for traits")
    parser.add_argument("--experiment", default="gemma_2b_cognitive_nov20", help="Experiment name")
    parser.add_argument("--trait", help="Analyze single trait in detail")
    args = parser.parse_args()

    exp_dir = Path("experiments") / args.experiment
    if not exp_dir.exists():
        print(f"Error: Experiment not found: {exp_dir}")
        sys.exit(1)

    # Single trait detail
    if args.trait:
        trait_dir = exp_dir / args.trait
        if not trait_dir.exists():
            print(f"Error: Trait not found: {trait_dir}")
            sys.exit(1)
        analyze_single_trait_detail(trait_dir, args.trait)
        return

    # Analyze all traits
    traits_data = []

    for trait_dir in sorted(exp_dir.iterdir()):
        if not trait_dir.is_dir() or trait_dir.name in ['experiments', 'inference']:
            continue

        data = analyze_trait(trait_dir, trait_dir.name)
        if data:
            traits_data.append(data)

    if not traits_data:
        print("No traits found")
        sys.exit(1)

    # Sort by separation
    traits_data.sort(key=lambda x: x['separation'], reverse=True)

    # Print summary
    print("=" * 100)
    print(f"QUALITY REPORT: {args.experiment}")
    print("=" * 100)
    print(f"\nTotal traits: {len(traits_data)}")
    print(f"Storage type: {traits_data[0]['storage_type']}")
    print()

    print(f"{'Trait':<25} {'Sep':<8} {'Pos Mean':<10} {'Neg Mean':<10} {'Probe Acc':<12} {'Examples':<10}")
    print("-" * 100)

    for t in traits_data:
        probe_str = f"{t['probe_acc']:.4f}" if t['probe_acc'] is not None else "N/A"
        print(f"{t['trait']:<25} {t['separation']:<8.1f} {t['pos_mean']:<10.1f} {t['neg_mean']:<10.1f} {probe_str:<12} {t['n_total']:<10}")

    # Quality tiers
    print("\n" + "=" * 100)
    print("QUALITY TIERS")
    print("=" * 100)

    excellent = [t for t in traits_data if t['separation'] >= 85]
    good = [t for t in traits_data if 70 <= t['separation'] < 85]
    moderate = [t for t in traits_data if 60 <= t['separation'] < 70]
    marginal = [t for t in traits_data if t['separation'] < 60]

    print(f"\nEXCELLENT (Sep >= 85): {len(excellent)} traits")
    for t in excellent:
        acc_str = f"{t['probe_acc']:.4f}" if t['probe_acc'] else "N/A"
        print(f"  ✓ {t['trait']:<25} (Sep: {t['separation']:.1f}, Probe: {acc_str})")

    print(f"\nGOOD (70-84): {len(good)} traits")
    for t in good:
        acc_str = f"{t['probe_acc']:.4f}" if t['probe_acc'] else "N/A"
        print(f"  • {t['trait']:<25} (Sep: {t['separation']:.1f}, Probe: {acc_str})")

    print(f"\nMODERATE (60-69): {len(moderate)} traits")
    for t in moderate:
        acc_str = f"{t['probe_acc']:.4f}" if t['probe_acc'] else "N/A"
        print(f"  ? {t['trait']:<25} (Sep: {t['separation']:.1f}, Probe: {acc_str})")

    print(f"\nMARGINAL (<60): {len(marginal)} traits")
    for t in marginal:
        acc_str = f"{t['probe_acc']:.4f}" if t['probe_acc'] else "N/A"
        print(f"  ✗ {t['trait']:<25} (Sep: {t['separation']:.1f}, Probe: {acc_str})")


if __name__ == "__main__":
    main()
