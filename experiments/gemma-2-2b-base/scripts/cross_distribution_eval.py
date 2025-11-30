#!/usr/bin/env python3
"""
Cross-distribution validation for uncertainty vectors.

Tests whether extracted vectors generalize across:
1. Uncertainty types (epistemic vs subjective vs inaccessible)
2. Topics (science vs humanities)

Usage:
    python experiments/gemma-2-2b-base/scripts/cross_distribution_eval.py --trait epistemic/uncertainty
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

EXPERIMENT_DIR = Path(__file__).parent.parent


def load_activations_by_category(trait: str):
    """Load activations and group by category."""
    trait_dir = EXPERIMENT_DIR / 'extraction' / trait

    # Load prompts with categories
    with open(trait_dir / 'prompts.json') as f:
        prompts = json.load(f)['pairs']

    # Load all activations (train + val combined for this analysis)
    activations_dir = trait_dir / 'activations'

    # Load residual (most stable component)
    residual = torch.load(activations_dir / 'residual.pt')  # [n_examples, n_layers, hidden_dim]

    # Also load val
    val_dir = trait_dir / 'val_activations'
    val_pos = torch.load(val_dir / 'residual_pos.pt')
    val_neg = torch.load(val_dir / 'residual_neg.pt')

    # Load metadata to understand split
    with open(activations_dir / 'metadata.json') as f:
        meta = json.load(f)

    n_train = meta['n_train_pairs']
    n_val = meta['n_val_pairs']
    n_layers = meta['n_layers']

    # Reconstruct full dataset
    # Train: first n_train are pos, next n_train are neg
    train_pos = residual[:n_train]
    train_neg = residual[n_train:]

    # Combine train + val
    all_pos = torch.cat([train_pos, val_pos], dim=0)
    all_neg = torch.cat([train_neg, val_neg], dim=0)

    # Group by category
    categories = defaultdict(lambda: {'pos': [], 'neg': [], 'pos_idx': [], 'neg_idx': []})

    for i, p in enumerate(prompts):
        cat = p.get('category', 'unknown')
        # Parse uncertainty type and topic
        if '/' in cat:
            utype, topic = cat.rsplit('/', 1)
        else:
            utype, topic = cat, 'unknown'

        categories[cat]['pos'].append(all_pos[i])
        categories[cat]['neg'].append(all_neg[i])
        categories[cat]['pos_idx'].append(i)
        categories[cat]['neg_idx'].append(i)
        categories[cat]['utype'] = utype
        categories[cat]['topic'] = topic

    # Stack tensors
    for cat in categories:
        categories[cat]['pos'] = torch.stack(categories[cat]['pos'])
        categories[cat]['neg'] = torch.stack(categories[cat]['neg'])

    return dict(categories), n_layers


def evaluate_cross_dist(train_cats: list, test_cats: list, categories: dict, layer: int):
    """Train on some categories, test on others."""
    # Gather train data
    train_pos = torch.cat([categories[c]['pos'][:, layer, :] for c in train_cats])
    train_neg = torch.cat([categories[c]['neg'][:, layer, :] for c in train_cats])

    X_train = torch.cat([train_pos, train_neg]).numpy().astype(np.float32)
    y_train = np.array([1] * len(train_pos) + [0] * len(train_neg))

    # Gather test data
    test_pos = torch.cat([categories[c]['pos'][:, layer, :] for c in test_cats])
    test_neg = torch.cat([categories[c]['neg'][:, layer, :] for c in test_cats])

    X_test = torch.cat([test_pos, test_neg]).numpy().astype(np.float32)
    y_test = np.array([1] * len(test_pos) + [0] * len(test_neg))

    # Train probe
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    return train_acc, test_acc


def run_cross_distribution(trait: str):
    """Run cross-distribution validation."""

    categories, n_layers = load_activations_by_category(trait)

    print(f"Loaded {len(categories)} categories")
    print(f"Categories: {list(categories.keys())[:5]}...")

    # Group categories by uncertainty type
    by_utype = defaultdict(list)
    by_topic = defaultdict(list)

    for cat, data in categories.items():
        by_utype[data['utype']].append(cat)
        by_topic[data['topic']].append(cat)

    print(f"\nUncertainty types: {list(by_utype.keys())}")
    print(f"Topics: {list(by_topic.keys())}")

    # Test 1: Cross uncertainty type
    # Train on epistemic + subjective, test on inaccessible
    print("\n" + "="*70)
    print("TEST 1: Cross Uncertainty Type")
    print("Train: epistemic + subjective | Test: inaccessible")
    print("="*70)

    train_cats = by_utype['epistemic_uncertain+empirical_certain'] + by_utype['subjective_uncertain+definitional_certain']
    test_cats = by_utype['inaccessible_uncertain+logical_certain']

    print(f"Train categories: {len(train_cats)} ({len(train_cats)*4} pairs)")
    print(f"Test categories: {len(test_cats)} ({len(test_cats)*4} pairs)")

    best_layer, best_test_acc = 0, 0
    results_utype = []

    for layer in range(n_layers):
        train_acc, test_acc = evaluate_cross_dist(train_cats, test_cats, categories, layer)
        results_utype.append((layer, train_acc, test_acc))
        if test_acc > best_test_acc:
            best_layer, best_test_acc = layer, test_acc
        if layer % 5 == 0 or layer == n_layers - 1:
            print(f"  Layer {layer:2d}: train={train_acc:.3f}, test={test_acc:.3f}")

    print(f"\n  BEST: Layer {best_layer} with test accuracy {best_test_acc:.3f}")

    # Test 2: Cross topic
    # Train on science (biology, physics, psychology), test on humanities (art, history, food)
    print("\n" + "="*70)
    print("TEST 2: Cross Topic")
    print("Train: biology, physics, psychology | Test: art, history, food")
    print("="*70)

    science_topics = ['biology', 'physics', 'psychology']
    humanities_topics = ['art', 'history', 'food']

    train_cats = [c for c in categories if any(t in c for t in science_topics)]
    test_cats = [c for c in categories if any(t in c for t in humanities_topics)]

    print(f"Train categories: {len(train_cats)} ({len(train_cats)*4} pairs)")
    print(f"Test categories: {len(test_cats)} ({len(test_cats)*4} pairs)")

    best_layer, best_test_acc = 0, 0
    results_topic = []

    for layer in range(n_layers):
        train_acc, test_acc = evaluate_cross_dist(train_cats, test_cats, categories, layer)
        results_topic.append((layer, train_acc, test_acc))
        if test_acc > best_test_acc:
            best_layer, best_test_acc = layer, test_acc
        if layer % 5 == 0 or layer == n_layers - 1:
            print(f"  Layer {layer:2d}: train={train_acc:.3f}, test={test_acc:.3f}")

    print(f"\n  BEST: Layer {best_layer} with test accuracy {best_test_acc:.3f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Find layer with best average cross-dist performance
    avg_test = [(results_utype[i][2] + results_topic[i][2]) / 2 for i in range(n_layers)]
    best_avg_layer = np.argmax(avg_test)

    print(f"\nCross uncertainty-type (layer {best_avg_layer}): {results_utype[best_avg_layer][2]:.3f}")
    print(f"Cross topic (layer {best_avg_layer}): {results_topic[best_avg_layer][2]:.3f}")
    print(f"Average: {avg_test[best_avg_layer]:.3f}")

    if avg_test[best_avg_layer] > 0.75:
        print("\n✓ Strong cross-distribution generalization. This looks like a real concept.")
    elif avg_test[best_avg_layer] > 0.65:
        print("\n~ Moderate generalization. Some real signal, but may be partially topic-driven.")
    else:
        print("\n✗ Weak generalization. Likely capturing surface patterns, not the concept.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trait', type=str, required=True)
    args = parser.parse_args()

    run_cross_distribution(args.trait)


if __name__ == '__main__':
    main()
