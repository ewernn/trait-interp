#!/usr/bin/env python3
"""
Plot layer spectrum: validation accuracy/Cohen's d across layers for each component.

Input:
    - experiments/gemma-2-2b-base/extraction/{category}/{trait}/vectors/results.json

Output:
    - experiments/gemma-2-2b-base/extraction/{category}/{trait}/layer_spectrum.png

Usage:
    python experiments/gemma-2-2b-base/scripts/plot_layer_spectrum.py --trait epistemic/uncertainty
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_DIR = Path(__file__).parent.parent

# Component colors
COLORS = {
    'residual': '#2563eb',    # blue
    'attn_out': '#dc2626',    # red
    'mlp_out': '#16a34a',     # green
}

LABELS = {
    'residual': 'Residual (cumulative)',
    'attn_out': 'Attention output',
    'mlp_out': 'MLP output',
}


def plot_layer_spectrum(trait: str, metric: str = 'val_acc'):
    """Generate layer spectrum plot."""

    trait_dir = EXPERIMENT_DIR / 'extraction' / trait
    results_path = trait_dir / 'vectors' / 'results.json'

    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        return

    with open(results_path) as f:
        results = json.load(f)

    n_layers = results['n_layers']
    layers = list(range(n_layers))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Validation accuracy
    ax1 = axes[0]
    for component in ['residual', 'attn_out', 'mlp_out']:
        if component not in results['components']:
            continue

        layer_data = results['components'][component]['layers']
        accs = [layer_data[str(l)]['probe'].get('val_acc', 0) for l in layers]

        ax1.plot(layers, accs, color=COLORS[component], label=LABELS[component],
                 linewidth=2, marker='o', markersize=4)

    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_title('Probe Accuracy by Layer', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cohen's d
    ax2 = axes[1]
    for component in ['residual', 'attn_out', 'mlp_out']:
        if component not in results['components']:
            continue

        layer_data = results['components'][component]['layers']
        ds = [layer_data[str(l)]['probe'].get('val_cohens_d', 0) for l in layers]

        ax2.plot(layers, ds, color=COLORS[component], label=LABELS[component],
                 linewidth=2, marker='o', markersize=4)

    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel("Cohen's d", fontsize=12)
    ax2.set_title('Effect Size by Layer', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Uncertainty Extraction: {trait}\n(Base Model: gemma-2-2b)', fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = trait_dir / 'layer_spectrum.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also print peak layers
    print("\nPeak layers (by val_acc):")
    for component in ['residual', 'attn_out', 'mlp_out']:
        if component not in results['components']:
            continue
        layer_data = results['components'][component]['layers']
        accs = {l: layer_data[str(l)]['probe'].get('val_acc', 0) for l in layers}
        best = max(accs, key=accs.get)
        print(f"  {component}: layer {best} ({accs[best]:.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trait', type=str, required=True)
    args = parser.parse_args()

    plot_layer_spectrum(args.trait)


if __name__ == '__main__':
    main()
