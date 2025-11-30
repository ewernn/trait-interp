#!/usr/bin/env python3
"""
Extract vectors from deconfounded refusal activations.

Compares across token windows (5, 15, 30) and produces layer spectrum plots.

Usage:
    python experiments/gemma-2-2b-base/scripts/extract_vectors_deconfounded.py --trait action/refusal
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

EXPERIMENT_DIR = Path(__file__).parent.parent
TOKEN_WINDOWS = [5, 15, 30]
COMPONENTS = ['residual', 'attn_out', 'mlp_out']
METHODS = ['mean_diff', 'probe']

COLORS = {
    'residual': '#2563eb',
    'attn_out': '#dc2626',
    'mlp_out': '#16a34a',
}

LABELS = {
    'residual': 'Residual',
    'attn_out': 'Attention',
    'mlp_out': 'MLP',
}


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    n1, n2 = len(pos), len(neg)
    var1, var2 = pos.var(), neg.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (pos.mean() - neg.mean()) / pooled_std


def extract_and_evaluate(pos_acts: torch.Tensor, neg_acts: torch.Tensor, val_split: float = 0.2):
    """Extract vector and evaluate with train/val split."""
    n_pos, n_neg = len(pos_acts), len(neg_acts)

    # Split
    pos_split = int(n_pos * (1 - val_split))
    neg_split = int(n_neg * (1 - val_split))

    train_pos, val_pos = pos_acts[:pos_split], pos_acts[pos_split:]
    train_neg, val_neg = neg_acts[:neg_split], neg_acts[neg_split:]

    # Mean diff
    pos_mean = train_pos.mean(dim=0)
    neg_mean = train_neg.mean(dim=0)
    mean_diff_vec = pos_mean - neg_mean

    # Probe
    X_train = torch.cat([train_pos, train_neg]).numpy().astype(np.float32)
    y_train = np.array([1] * len(train_pos) + [0] * len(train_neg))

    clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    clf.fit(X_train, y_train)
    probe_vec = torch.tensor(clf.coef_[0], dtype=train_pos.dtype)

    # Evaluate on val
    results = {}

    for name, vec in [('mean_diff', mean_diff_vec), ('probe', probe_vec)]:
        val_pos_proj = (val_pos @ vec) / vec.norm()
        val_neg_proj = (val_neg @ vec) / vec.norm()

        pos_correct = (val_pos_proj > 0).sum().item()
        neg_correct = (val_neg_proj < 0).sum().item()
        val_acc = (pos_correct + neg_correct) / (len(val_pos) + len(val_neg))

        val_d = cohens_d(val_pos_proj.numpy(), val_neg_proj.numpy())

        results[name] = {
            'val_acc': val_acc,
            'val_cohens_d': val_d,
            'vector': vec,
        }

    return results


def run_extraction(trait: str):
    """Run vector extraction for all windows and components."""

    trait_dir = EXPERIMENT_DIR / 'extraction' / trait
    activations_dir = trait_dir / 'activations_deconfounded'

    if not activations_dir.exists():
        print(f"ERROR: {activations_dir} not found. Run extract_refusal_deconfounded.py first.")
        return

    all_results = {}

    for window in TOKEN_WINDOWS:
        window_dir = activations_dir / f'window_{window}'
        if not window_dir.exists():
            print(f"Skipping window {window}: not found")
            continue

        with open(window_dir / 'metadata.json') as f:
            meta = json.load(f)

        n_refusal = meta['n_refusal']
        n_layers = meta['n_layers']

        all_results[window] = {'components': {}}

        print(f"\n{'='*60}")
        print(f"Window: {window} tokens")
        print(f"{'='*60}")

        for component in COMPONENTS:
            comp_path = window_dir / f'{component}.pt'
            if not comp_path.exists():
                continue

            acts = torch.load(comp_path)  # [n_total, n_layers, hidden]
            pos_acts = acts[:n_refusal]
            neg_acts = acts[n_refusal:]

            all_results[window]['components'][component] = {'layers': {}}

            for layer in range(n_layers):
                pos_layer = pos_acts[:, layer, :].float()
                neg_layer = neg_acts[:, layer, :].float()

                results = extract_and_evaluate(pos_layer, neg_layer)
                all_results[window]['components'][component]['layers'][layer] = {
                    'mean_diff': {
                        'val_acc': float(results['mean_diff']['val_acc']),
                        'val_cohens_d': float(results['mean_diff']['val_cohens_d']),
                    },
                    'probe': {
                        'val_acc': float(results['probe']['val_acc']),
                        'val_cohens_d': float(results['probe']['val_cohens_d']),
                    },
                }

            # Print best layer
            layers_data = all_results[window]['components'][component]['layers']
            best_layer = max(layers_data.keys(), key=lambda l: layers_data[l]['probe']['val_acc'])
            best_acc = layers_data[best_layer]['probe']['val_acc']
            print(f"  {component}: best layer {best_layer} (acc={best_acc:.3f})")

    # Save results
    vectors_dir = trait_dir / 'vectors_deconfounded'
    vectors_dir.mkdir(exist_ok=True)

    with open(vectors_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Plot comparison across windows
    plot_window_comparison(all_results, vectors_dir)

    print(f"\nResults saved to {vectors_dir}")


def plot_window_comparison(results: dict, output_dir: Path):
    """Plot layer spectrum for each window, comparing components."""

    windows = sorted(results.keys())

    fig, axes = plt.subplots(2, len(windows), figsize=(5*len(windows), 8))
    if len(windows) == 1:
        axes = axes.reshape(-1, 1)

    for col, window in enumerate(windows):
        if window not in results:
            continue

        components_data = results[window]['components']
        n_layers = max(
            max(l if isinstance(l, int) else int(l) for l in comp_data['layers'].keys())
            for comp_data in components_data.values()
        ) + 1
        layers = list(range(n_layers))

        # Accuracy plot
        ax1 = axes[0, col]
        for component in COMPONENTS:
            if component not in components_data:
                continue
            accs = [components_data[component]['layers'][l]['probe']['val_acc']
                    for l in layers]
            ax1.plot(layers, accs, color=COLORS[component], label=LABELS[component],
                     linewidth=2, marker='o', markersize=3)

        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Validation Accuracy')
        ax1.set_title(f'Window: {window} tokens')
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylim(0.4, 1.0)
        ax1.legend(loc='lower right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Cohen's d plot
        ax2 = axes[1, col]
        for component in COMPONENTS:
            if component not in components_data:
                continue
            ds = [components_data[component]['layers'][l]['probe']['val_cohens_d']
                  for l in layers]
            ax2.plot(layers, ds, color=COLORS[component], label=LABELS[component],
                     linewidth=2, marker='o', markersize=3)

        ax2.set_xlabel('Layer')
        ax2.set_ylabel("Cohen's d")
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.legend(loc='lower right', fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.suptitle('Deconfounded Refusal Extraction: Token Window Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'window_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'window_comparison.png'}")

    # Also save summary
    print("\n" + "="*60)
    print("SUMMARY: Best layer per window (by probe val_acc)")
    print("="*60)

    for window in windows:
        print(f"\nWindow {window}:")
        for component in COMPONENTS:
            if component not in results[window]['components']:
                continue
            layers_data = results[window]['components'][component]['layers']
            best_layer = max(layers_data.keys(), key=lambda l: layers_data[l]['probe']['val_acc'])
            best_acc = layers_data[best_layer]['probe']['val_acc']
            best_d = layers_data[best_layer]['probe']['val_cohens_d']
            print(f"  {component}: layer {best_layer} (acc={best_acc:.3f}, d={best_d:.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trait', type=str, required=True)
    args = parser.parse_args()

    run_extraction(args.trait)


if __name__ == '__main__':
    main()
