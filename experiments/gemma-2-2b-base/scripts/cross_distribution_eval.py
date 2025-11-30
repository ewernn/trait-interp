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

    # Group by category - handle both formats
    categories = defaultdict(lambda: {'pos': [], 'neg': [], 'pos_idx': [], 'neg_idx': []})

    for i, p in enumerate(prompts):
        # Handle refusal format (domain/trigger_type) vs uncertainty format (category)
        if 'domain' in p and 'trigger_type' in p:
            # Refusal format
            cat = f"{p['domain']}/{p['trigger_type']}"
            dim1 = p['domain']  # e.g., medical, retail
            dim2 = p['trigger_type']  # e.g., document_validity, identity_verification
        else:
            # Uncertainty format
            cat = p.get('category', 'unknown')
            if '/' in cat:
                dim1, dim2 = cat.rsplit('/', 1)
            else:
                dim1, dim2 = cat, 'unknown'

        categories[cat]['pos'].append(all_pos[i])
        categories[cat]['neg'].append(all_neg[i])
        categories[cat]['pos_idx'].append(i)
        categories[cat]['neg_idx'].append(i)
        categories[cat]['dim1'] = dim1  # domain or uncertainty_type
        categories[cat]['dim2'] = dim2  # trigger_type or topic

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

    X_train = torch.cat([train_pos, train_neg]).float().numpy()
    y_train = np.array([1] * len(train_pos) + [0] * len(train_neg))

    # Gather test data
    test_pos = torch.cat([categories[c]['pos'][:, layer, :] for c in test_cats])
    test_neg = torch.cat([categories[c]['neg'][:, layer, :] for c in test_cats])

    X_test = torch.cat([test_pos, test_neg]).float().numpy()
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

    # Group categories by dim1 and dim2
    by_dim1 = defaultdict(list)
    by_dim2 = defaultdict(list)

    for cat, data in categories.items():
        by_dim1[data['dim1']].append(cat)
        by_dim2[data['dim2']].append(cat)

    dim1_keys = list(by_dim1.keys())
    dim2_keys = list(by_dim2.keys())

    print(f"\nDimension 1 (domain/uncertainty_type): {dim1_keys}")
    print(f"Dimension 2 (trigger_type/topic): {dim2_keys}")

    # Detect format and set up tests
    is_refusal = any('medical' in k or 'retail' in k for k in dim1_keys)

    if is_refusal:
        # Refusal format: cross-domain and cross-trigger
        test1_name = "Cross Domain"
        test1_train_keys = ['medical', 'retail', 'security']
        test1_test_keys = ['education', 'financial', 'government']
        test2_name = "Cross Trigger Type"
        test2_train_keys = ['document_validity', 'identity_verification']
        test2_test_keys = ['authorization_status', 'procedural_compliance']
    else:
        # Uncertainty format: cross-uncertainty-type and cross-topic
        test1_name = "Cross Uncertainty Type"
        test1_train_keys = ['epistemic_uncertain+empirical_certain', 'subjective_uncertain+definitional_certain']
        test1_test_keys = ['inaccessible_uncertain+logical_certain']
        test2_name = "Cross Topic"
        test2_train_keys = ['biology', 'physics', 'psychology']
        test2_test_keys = ['art', 'history', 'food']

    # Test 1: Cross dim1
    print("\n" + "="*70)
    print(f"TEST 1: {test1_name}")
    print(f"Train: {test1_train_keys} | Test: {test1_test_keys}")
    print("="*70)

    train_cats = [c for k in test1_train_keys for c in by_dim1.get(k, [])]
    test_cats = [c for k in test1_test_keys for c in by_dim1.get(k, [])]

    if not train_cats or not test_cats:
        # Fallback: split dim1 in half
        half = len(dim1_keys) // 2
        train_cats = [c for k in dim1_keys[:half] for c in by_dim1[k]]
        test_cats = [c for k in dim1_keys[half:] for c in by_dim1[k]]
        print(f"  (Using fallback split: {dim1_keys[:half]} vs {dim1_keys[half:]})")

    print(f"Train categories: {len(train_cats)}")
    print(f"Test categories: {len(test_cats)}")

    best_layer, best_test_acc = 0, 0
    results_dim1 = []

    for layer in range(n_layers):
        train_acc, test_acc = evaluate_cross_dist(train_cats, test_cats, categories, layer)
        results_dim1.append((layer, train_acc, test_acc))
        if test_acc > best_test_acc:
            best_layer, best_test_acc = layer, test_acc
        if layer % 5 == 0 or layer == n_layers - 1:
            print(f"  Layer {layer:2d}: train={train_acc:.3f}, test={test_acc:.3f}")

    print(f"\n  BEST: Layer {best_layer} with test accuracy {best_test_acc:.3f}")

    # Test 2: Cross dim2
    print("\n" + "="*70)
    print(f"TEST 2: {test2_name}")
    print(f"Train: {test2_train_keys} | Test: {test2_test_keys}")
    print("="*70)

    train_cats = [c for k in test2_train_keys for c in by_dim2.get(k, [])]
    test_cats = [c for k in test2_test_keys for c in by_dim2.get(k, [])]

    if not train_cats or not test_cats:
        # Fallback: split dim2 in half
        half = len(dim2_keys) // 2
        train_cats = [c for k in dim2_keys[:half] for c in by_dim2[k]]
        test_cats = [c for k in dim2_keys[half:] for c in by_dim2[k]]
        print(f"  (Using fallback split: {dim2_keys[:half]} vs {dim2_keys[half:]})")

    print(f"Train categories: {len(train_cats)}")
    print(f"Test categories: {len(test_cats)}")

    best_layer, best_test_acc = 0, 0
    results_dim2 = []

    for layer in range(n_layers):
        train_acc, test_acc = evaluate_cross_dist(train_cats, test_cats, categories, layer)
        results_dim2.append((layer, train_acc, test_acc))
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
    avg_test = [(results_dim1[i][2] + results_dim2[i][2]) / 2 for i in range(n_layers)]
    best_avg_layer = np.argmax(avg_test)

    print(f"\n{test1_name} (layer {best_avg_layer}): {results_dim1[best_avg_layer][2]:.3f}")
    print(f"{test2_name} (layer {best_avg_layer}): {results_dim2[best_avg_layer][2]:.3f}")
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
