#!/usr/bin/env python3
"""Cross-evaluate extracted vectors on all LIARS' BENCH datasets.

Tests zero-shot transfer: vectors extracted from custom contrastive scenarios
detecting deception across benchmark datasets without any labeled training data.

Usage:
    python cross_eval_all_datasets.py --trait bs/concealment
    python cross_eval_all_datasets.py --trait bs/lying

Output:
    experiments/bullshit/results/cross_eval_{trait_name}.json
"""

import argparse
import torch
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / 'experiments/bullshit/results'
PROMPTS_DIR = ROOT / 'experiments/bullshit/prompt_sets'

DATASETS = ['hpc', 'cg', 'it', 'gs', 'id', 'hpkr', 'st_time', 'st_greeting', 'st_gender']
METHODS = ['probe', 'mean_diff', 'gradient']
N_LAYERS = 80


def load_alpaca_distribution(vector_dir: Path, method: str, layer: int) -> np.ndarray:
    """Load Alpaca scores for FPR calibration."""
    vec_path = vector_dir / method / f'layer{layer}.pt'
    vec = torch.load(vec_path)['vector'].float()

    alpaca_acts = torch.load(RESULTS_DIR / 'alpaca_activations.pt')
    layer_acts = alpaca_acts[:, layer, :].float()
    scores = (layer_acts @ vec) / vec.norm()
    return scores.numpy()


def compute_recall_at_fpr(scores: np.ndarray, labels: np.ndarray,
                          alpaca_scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """Compute recall at target FPR using Alpaca-calibrated threshold."""
    threshold = np.percentile(alpaca_scores, 100 * (1 - target_fpr))
    predictions = scores >= threshold
    n_pos = labels.sum()
    if n_pos == 0:
        return 0.0
    return predictions[labels == 1].sum() / n_pos


def main():
    parser = argparse.ArgumentParser(description='Cross-evaluate vectors on LIARS BENCH')
    parser.add_argument('--trait', default='bs/concealment',
                        help='Trait path (e.g., bs/concealment, bs/lying)')
    args = parser.parse_args()

    # Construct paths from trait
    trait_name = args.trait.replace('/', '_')
    vector_dir = ROOT / f'experiments/bullshit/extraction/{args.trait}/base/vectors/response__5/residual'

    if not vector_dir.exists():
        print(f"ERROR: Vector directory not found: {vector_dir}")
        return

    print(f"Trait: {args.trait}")
    print(f"Vectors: {vector_dir}")

    results = {'trait': args.trait}

    # Pre-load Alpaca activations (optional, for FPR calibration)
    alpaca_path = RESULTS_DIR / 'alpaca_activations.pt'
    if alpaca_path.exists():
        print("Loading Alpaca activations for FPR calibration...")
        alpaca_acts = torch.load(alpaca_path)
    else:
        print("WARNING: Alpaca activations not found — skipping Recall@FPR metrics")
        alpaca_acts = None

    for ds in DATASETS:
        # Skip datasets without activations
        acts_path = RESULTS_DIR / f'{ds}_activations.pt'
        meta_path = PROMPTS_DIR / f'{ds}_metadata.json'
        if not acts_path.exists():
            print(f"\nSkipping {ds.upper()}: {acts_path.name} not found")
            continue
        if not meta_path.exists():
            print(f"\nSkipping {ds.upper()}: {meta_path.name} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds.upper()}")
        print('='*60)

        # Load activations
        acts = torch.load(acts_path)

        # Load labels
        with open(meta_path) as f:
            meta = json.load(f)
        labels = np.array([meta[str(i+1)].get('deceptive', False) for i in range(len(acts))])

        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        print(f"Examples: {n_pos:.0f} deceptive + {n_neg:.0f} honest = {len(labels)}")

        results[ds] = {'n_deceptive': int(n_pos), 'n_honest': int(n_neg)}

        for method in METHODS:
            print(f"\n  {method}:")
            print(f"  {'Layer':<6} {'AUROC':<8} {'Recall@1%':<10} {'Cohen-d':<8}")
            print(f"  {'-'*36}")

            best_auroc, best_layer = 0, 0
            best_recall, best_recall_layer = 0, 0
            results[ds][method] = {}

            for layer in range(N_LAYERS):
                vec_path = vector_dir / method / f'layer{layer}.pt'
                if not vec_path.exists():
                    continue

                vec = torch.load(vec_path)
                if isinstance(vec, dict):
                    vec = vec['vector']
                vec = vec.float()
                layer_acts = acts[:, layer, :].float()

                # Project onto vector
                scores = (layer_acts @ vec) / vec.norm()
                scores = scores.numpy()

                # AUROC
                auroc = roc_auc_score(labels, scores)

                # Alpaca-calibrated recall @ 1% FPR
                recall_1pct = None
                if alpaca_acts is not None:
                    alpaca_layer_acts = alpaca_acts[:, layer, :].float()
                    alpaca_scores = (alpaca_layer_acts @ vec) / vec.norm()
                    alpaca_scores = alpaca_scores.numpy()
                    recall_1pct = compute_recall_at_fpr(scores, labels, alpaca_scores, 0.01)

                # Cohen's d
                pos_scores = scores[labels == 1]
                neg_scores = scores[labels == 0]
                pooled_std = np.sqrt((pos_scores.std()**2 + neg_scores.std()**2) / 2)
                cohen_d = (pos_scores.mean() - neg_scores.mean()) / pooled_std

                layer_result = {
                    'auroc': float(auroc),
                    'cohen_d': float(cohen_d)
                }
                if recall_1pct is not None:
                    layer_result['recall_at_1pct_fpr'] = float(recall_1pct)
                results[ds][method][layer] = layer_result

                # Print every 10th layer + best candidates
                if layer % 10 == 0 or layer in [37, 79]:
                    recall_str = f"{recall_1pct:<10.1%}" if recall_1pct is not None else "N/A       "
                    print(f"  {layer:<6} {auroc:<8.3f} {recall_str} {cohen_d:<8.2f}")

                if auroc > best_auroc:
                    best_auroc, best_layer = auroc, layer
                if recall_1pct is not None and recall_1pct > best_recall:
                    best_recall, best_recall_layer = recall_1pct, layer

            results[ds][method]['best_auroc'] = {'layer': best_layer, 'value': best_auroc}
            if best_recall > 0 or alpaca_acts is not None:
                results[ds][method]['best_recall'] = {'layer': best_recall_layer, 'value': best_recall}
            recall_str = f", Best Recall@1%: L{best_recall_layer}={best_recall:.1%}" if alpaca_acts is not None else ""
            print(f"  Best AUROC: L{best_layer}={best_auroc:.3f}{recall_str}")

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY: Best metrics per dataset × method")
    print('='*80)
    print(f"{'Dataset':<8} {'Method':<12} {'Best AUROC':<15} {'Best Recall@1%':<20}")
    print('-'*80)

    evaluated_datasets = [ds for ds in DATASETS if ds in results]

    for ds in evaluated_datasets:
        for method in METHODS:
            if method not in results[ds]:
                continue
            best_a = results[ds][method]['best_auroc']
            best_r = results[ds][method].get('best_recall', {'layer': 'N/A', 'value': 0})
            recall_str = f"L{best_r['layer']}={best_r['value']:.1%}" if best_r['layer'] != 'N/A' else "N/A"
            print(f"{ds.upper():<8} {method:<12} L{best_a['layer']}={best_a['value']:.3f}       {recall_str}")

    # Cross-dataset summary (best method per dataset)
    print(f"\n{'='*80}")
    print("BEST PER DATASET (across all methods)")
    print('='*80)
    print(f"{'Dataset':<12} {'N':<8} {'Best AUROC':<20} {'Best Recall@1%':<25}")
    print('-'*80)

    for ds in evaluated_datasets:
        n = results[ds]['n_deceptive'] + results[ds]['n_honest']

        # Find best across methods
        best_auroc_overall = max(
            (results[ds][m]['best_auroc']['value'], m, results[ds][m]['best_auroc']['layer'])
            for m in METHODS if m in results[ds]
        )
        recall_candidates = [
            (results[ds][m]['best_recall']['value'], m, results[ds][m]['best_recall']['layer'])
            for m in METHODS if m in results[ds] and 'best_recall' in results[ds][m]
        ]

        auroc_str = f"{best_auroc_overall[1]} L{best_auroc_overall[2]}={best_auroc_overall[0]:.3f}"
        if recall_candidates:
            best_recall_overall = max(recall_candidates)
            recall_str = f"{best_recall_overall[1]} L{best_recall_overall[2]}={best_recall_overall[0]:.1%}"
        else:
            recall_str = "N/A (no Alpaca)"
        print(f"{ds.upper():<12} {n:<8} {auroc_str:<20} {recall_str:<25}")

    # Save results
    output_path = RESULTS_DIR / f'cross_eval_{trait_name}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
