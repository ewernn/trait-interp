#!/usr/bin/env python3
"""
Extract trait vectors from component activations.

Runs mean_diff, probe, and gradient methods on each component at each layer.

Input:
    - experiments/gemma-2-2b-base/extraction/{category}/{trait}/activations/*.pt
    - experiments/gemma-2-2b-base/extraction/{category}/{trait}/val_activations/*.pt

Output:
    - experiments/gemma-2-2b-base/extraction/{category}/{trait}/vectors/
        - {component}_{method}_layer{L}.pt  (e.g., residual_probe_layer16.pt)
        - results.json  (all metrics)

Usage:
    python experiments/gemma-2-2b-base/scripts/extract_vectors.py --trait epistemic/uncertainty
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

EXPERIMENT_DIR = Path(__file__).parent.parent
COMPONENTS = ['residual', 'attn_out', 'mlp_out']  # Skip keys/values for now (different shape)
METHODS = ['mean_diff', 'probe']


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(pos), len(neg)
    var1, var2 = pos.var(), neg.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (pos.mean() - neg.mean()) / pooled_std


def extract_mean_diff(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> dict:
    """Mean difference extraction."""
    pos_mean = pos_acts.mean(dim=0)
    neg_mean = neg_acts.mean(dim=0)
    vector = pos_mean - neg_mean

    # Compute projections for metrics
    pos_proj = (pos_acts @ vector) / vector.norm()
    neg_proj = (neg_acts @ vector) / vector.norm()

    return {
        'vector': vector,
        'pos_proj': pos_proj.numpy(),
        'neg_proj': neg_proj.numpy(),
    }


def extract_probe(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> dict:
    """Linear probe extraction."""
    X = torch.cat([pos_acts, neg_acts], dim=0).numpy()
    y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))

    clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    clf.fit(X, y)

    vector = torch.tensor(clf.coef_[0], dtype=pos_acts.dtype)
    probs = clf.predict_proba(X)[:, 1]

    return {
        'vector': vector,
        'train_acc': accuracy_score(y, clf.predict(X)),
        'train_auc': roc_auc_score(y, probs),
        'pos_proj': probs[:len(pos_acts)],
        'neg_proj': probs[len(pos_acts):],
    }


def evaluate_on_val(
    vector: torch.Tensor,
    val_pos: torch.Tensor,
    val_neg: torch.Tensor,
) -> dict:
    """Evaluate vector on validation set."""
    pos_proj = (val_pos @ vector) / vector.norm()
    neg_proj = (val_neg @ vector) / vector.norm()

    # Classification accuracy (threshold at 0)
    pos_correct = (pos_proj > 0).sum().item()
    neg_correct = (neg_proj < 0).sum().item()
    acc = (pos_correct + neg_correct) / (len(pos_proj) + len(neg_proj))

    # Cohen's d
    d = cohens_d(pos_proj.numpy(), neg_proj.numpy())

    # AUC
    y_true = np.array([1] * len(pos_proj) + [0] * len(neg_proj))
    y_score = np.concatenate([pos_proj.numpy(), neg_proj.numpy()])
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = 0.5

    return {
        'val_acc': acc,
        'val_cohens_d': d,
        'val_auc': auc,
        'val_pos_mean': float(pos_proj.mean()),
        'val_neg_mean': float(neg_proj.mean()),
    }


def run_extraction(trait: str):
    """Run vector extraction for all components and methods."""

    trait_dir = EXPERIMENT_DIR / 'extraction' / trait
    activations_dir = trait_dir / 'activations'
    val_dir = trait_dir / 'val_activations'
    vectors_dir = trait_dir / 'vectors'
    vectors_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(activations_dir / 'metadata.json') as f:
        metadata = json.load(f)

    n_layers = metadata['n_layers']
    n_train = metadata['n_train_pairs']

    results = {
        'trait': trait,
        'n_layers': n_layers,
        'n_train_pairs': n_train,
        'n_val_pairs': metadata['n_val_pairs'],
        'components': {},
    }

    for component in COMPONENTS:
        print(f"\n{'='*60}")
        print(f"Component: {component}")
        print('='*60)

        # Load activations
        train_path = activations_dir / f'{component}.pt'
        if not train_path.exists():
            print(f"  Skipping {component}: no activations found")
            continue

        train_acts = torch.load(train_path)  # [n_examples, n_layers, hidden_dim]
        n_examples = train_acts.shape[0]
        pos_acts_all = train_acts[:n_train]  # First half is positive
        neg_acts_all = train_acts[n_train:]  # Second half is negative

        # Load val
        val_pos_path = val_dir / f'{component}_pos.pt'
        val_neg_path = val_dir / f'{component}_neg.pt'
        has_val = val_pos_path.exists() and val_neg_path.exists()

        if has_val:
            val_pos_all = torch.load(val_pos_path)
            val_neg_all = torch.load(val_neg_path)

        results['components'][component] = {'layers': {}}

        for layer in range(n_layers):
            pos_acts = pos_acts_all[:, layer, :]
            neg_acts = neg_acts_all[:, layer, :]

            layer_results = {}

            for method in METHODS:
                if method == 'mean_diff':
                    extracted = extract_mean_diff(pos_acts, neg_acts)
                elif method == 'probe':
                    extracted = extract_probe(pos_acts, neg_acts)

                vector = extracted['vector']

                # Save vector
                torch.save(vector, vectors_dir / f'{component}_{method}_layer{layer}.pt')

                # Compute train metrics
                train_d = cohens_d(extracted['pos_proj'], extracted['neg_proj'])
                method_results = {
                    'train_cohens_d': train_d,
                    'train_pos_mean': float(np.mean(extracted['pos_proj'])),
                    'train_neg_mean': float(np.mean(extracted['neg_proj'])),
                }

                if method == 'probe':
                    method_results['train_acc'] = extracted['train_acc']
                    method_results['train_auc'] = extracted['train_auc']

                # Evaluate on val
                if has_val:
                    val_pos = val_pos_all[:, layer, :]
                    val_neg = val_neg_all[:, layer, :]
                    val_metrics = evaluate_on_val(vector, val_pos, val_neg)
                    method_results.update(val_metrics)

                layer_results[method] = method_results

            results['components'][component]['layers'][layer] = layer_results

            # Print progress
            if has_val:
                probe_val = layer_results['probe']['val_acc']
                probe_d = layer_results['probe']['val_cohens_d']
                print(f"  Layer {layer:2d}: probe val_acc={probe_val:.3f}, val_d={probe_d:.2f}")

    # Save results
    with open(vectors_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {vectors_dir / 'results.json'}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Best layer per component (by val_acc)")
    print("="*60)

    for component in COMPONENTS:
        if component not in results['components']:
            continue
        layers = results['components'][component]['layers']
        best_layer = max(layers.keys(), key=lambda l: layers[l]['probe'].get('val_acc', 0))
        best_acc = layers[best_layer]['probe'].get('val_acc', 0)
        best_d = layers[best_layer]['probe'].get('val_cohens_d', 0)
        print(f"  {component}: layer {best_layer} (acc={best_acc:.3f}, d={best_d:.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trait', type=str, required=True)
    args = parser.parse_args()

    run_extraction(args.trait)


if __name__ == '__main__':
    main()
