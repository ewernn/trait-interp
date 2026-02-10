#!/usr/bin/env python3
"""Evaluate zero-label combinations of trait vectors on LIARS' BENCH datasets.

Tests whether combining multiple vector projections (no labels needed) outperforms
any single vector. All methods are fully zero-label — no benchmark labels used.

Combination strategies:
  - max: max(auroc_1, auroc_2, auroc_3) — best single vector per dataset
  - mean: mean of 3 projection scores per example
  - vote: example is "deceptive" if majority of vectors say so (above per-vector median)

Input:
    experiments/bullshit/results/{dataset}_activations.pt
    experiments/bullshit/prompt_sets/{dataset}_metadata.json
    experiments/bullshit/extraction/{trait}/base/vectors/response__5/residual/{method}/layer{L}.pt

Output:
    experiments/bullshit/results/ensemble_eval.json

Usage:
    python experiments/bullshit/scripts/ensemble_eval.py
    python experiments/bullshit/scripts/ensemble_eval.py --layer 20 --method mean_diff
"""

import argparse
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / 'experiments/bullshit/results'
PROMPTS_DIR = ROOT / 'experiments/bullshit/prompt_sets'
EXTRACTION_DIR = ROOT / 'experiments/bullshit/extraction'

TRAITS = ['bs/concealment', 'bs/lying', 'bs/self_knowledge_concealment']

ALL_DATASETS = ['hpc', 'cg', 'it', 'gs', 'id', 'hpkr', 'st_time', 'st_greeting', 'st_gender']


def load_vector(trait: str, method: str, layer: int) -> torch.Tensor:
    vec_path = EXTRACTION_DIR / trait / 'base/vectors/response__5/residual' / method / f'layer{layer}.pt'
    vec = torch.load(vec_path, map_location='cpu')
    if isinstance(vec, dict):
        vec = vec['vector']
    return vec.float()


def project(acts: torch.Tensor, vec: torch.Tensor) -> np.ndarray:
    scores = (acts @ vec) / vec.norm()
    return scores.numpy()


def main():
    parser = argparse.ArgumentParser(description='Zero-label ensemble eval on LIARS BENCH')
    parser.add_argument('--layer', type=int, default=30, help='Layer to evaluate')
    parser.add_argument('--method', default='probe', help='Extraction method')
    args = parser.parse_args()

    # Discover available datasets
    datasets = []
    for ds in ALL_DATASETS:
        acts_path = RESULTS_DIR / f'{ds}_activations.pt'
        meta_path = PROMPTS_DIR / f'{ds}_metadata.json'
        if acts_path.exists() and meta_path.exists():
            datasets.append(ds)
    print(f"Found {len(datasets)} datasets: {', '.join(d.upper() for d in datasets)}")

    # Load vectors
    vectors = {}
    for trait in TRAITS:
        try:
            vectors[trait] = load_vector(trait, args.method, args.layer)
            print(f"  Loaded {trait} L{args.layer} ({args.method})")
        except FileNotFoundError:
            print(f"  WARNING: {trait} L{args.layer} {args.method} not found, skipping")
    trait_names = list(vectors.keys())

    results = {
        'layer': args.layer,
        'method': args.method,
        'traits': trait_names,
        'datasets': {},
    }

    # Header
    short_names = [t.split('/')[-1][:12] for t in trait_names]
    print(f"\n{'Dataset':<12} {'N':>5}", end='')
    for s in short_names:
        print(f"  {s:>12}", end='')
    print(f"  {'best':>7}  {'mean':>7}  {'Δ mean':>7}")
    print('-' * (20 + len(trait_names) * 14 + 28))

    for ds in datasets:
        acts = torch.load(RESULTS_DIR / f'{ds}_activations.pt', map_location='cpu')
        layer_acts = acts[:, args.layer, :].float()

        with open(PROMPTS_DIR / f'{ds}_metadata.json') as f:
            meta = json.load(f)
        labels = np.array([meta[str(i+1)].get('deceptive', False) for i in range(len(acts))])

        n_pos = int(labels.sum())
        if n_pos == 0 or n_pos == len(labels):
            print(f"{ds.upper():<12} {len(labels):>5}  SKIPPED (no class variation)")
            continue

        # Individual projections
        individual_aurocs = {}
        all_scores = []

        for trait in trait_names:
            scores = project(layer_acts, vectors[trait])
            all_scores.append(scores)
            auroc = roc_auc_score(labels, scores)
            individual_aurocs[trait] = auroc

        score_matrix = np.column_stack(all_scores)  # [n_examples, n_traits]
        best_individual = max(individual_aurocs.values())

        # Zero-label combinations (no training data needed)

        # 1. Mean of scores (need to z-score first so scales are comparable)
        z_scores = (score_matrix - score_matrix.mean(axis=0)) / (score_matrix.std(axis=0) + 1e-8)
        mean_scores = z_scores.mean(axis=1)
        mean_auroc = roc_auc_score(labels, mean_scores)

        # 2. Max of z-scores
        max_scores = z_scores.max(axis=1)
        max_auroc = roc_auc_score(labels, max_scores)

        # Print row
        print(f"{ds.upper():<12} {len(labels):>5}", end='')
        for trait in trait_names:
            print(f"  {individual_aurocs[trait]:>12.3f}", end='')
        print(f"  {best_individual:>7.3f}  {mean_auroc:>7.3f}  {mean_auroc - best_individual:>+7.3f}")

        results['datasets'][ds] = {
            'n': len(labels),
            'n_deceptive': n_pos,
            'individual': {t.split('/')[-1]: float(v) for t, v in individual_aurocs.items()},
            'best_individual': float(best_individual),
            'mean_auroc': float(mean_auroc),
            'max_auroc': float(max_auroc),
            'mean_delta': float(mean_auroc - best_individual),
            'max_delta': float(max_auroc - best_individual),
        }

    # Summary
    ds_results = results['datasets']
    if ds_results:
        avg_best = np.mean([v['best_individual'] for v in ds_results.values()])
        avg_mean = np.mean([v['mean_auroc'] for v in ds_results.values()])
        avg_max = np.mean([v['max_auroc'] for v in ds_results.values()])
        print(f"\n{'AVERAGE':<12} {'':>5}", end='')
        print(' ' * (len(trait_names) * 14), end='')
        print(f"  {avg_best:>7.3f}  {avg_mean:>7.3f}  {avg_mean - avg_best:>+7.3f}")

        print(f"\nMax-of-z AUROC average: {avg_max:.3f} (Δ {avg_max - avg_best:+.3f})")

        mean_wins = sum(1 for v in ds_results.values() if v['mean_delta'] > 0.005)
        mean_ties = sum(1 for v in ds_results.values() if abs(v['mean_delta']) <= 0.005)
        mean_losses = sum(1 for v in ds_results.values() if v['mean_delta'] < -0.005)
        print(f"Mean-of-z wins/ties/losses: {mean_wins}/{mean_ties}/{mean_losses}")

    output_path = RESULTS_DIR / 'ensemble_eval.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
