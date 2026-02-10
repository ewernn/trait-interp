#!/usr/bin/env python3
"""Evaluate ensemble of multiple trait vectors on LIARS' BENCH datasets.

Tests whether a learned combination of projections outperforms any single vector.
CPU-only — just loads activations and vectors, trains small logistic regression.

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / 'experiments/bullshit/results'
PROMPTS_DIR = ROOT / 'experiments/bullshit/prompt_sets'
EXTRACTION_DIR = ROOT / 'experiments/bullshit/extraction'

TRAITS = ['bs/concealment', 'bs/lying', 'bs/self_knowledge_concealment']

# Auto-discover datasets that have both activations and metadata
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
    parser = argparse.ArgumentParser(description='Ensemble eval on LIARS BENCH')
    parser.add_argument('--layer', type=int, default=30, help='Layer to evaluate')
    parser.add_argument('--method', default='probe', help='Extraction method')
    parser.add_argument('--n-folds', type=int, default=5)
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
            print(f"Loaded {trait} L{args.layer} ({args.method})")
        except FileNotFoundError:
            print(f"WARNING: {trait} L{args.layer} {args.method} not found, skipping")
    trait_names = list(vectors.keys())
    print(f"\n{len(trait_names)} traits loaded")

    results = {
        'layer': args.layer,
        'method': args.method,
        'traits': trait_names,
        'datasets': {},
    }

    print(f"\n{'Dataset':<12} {'N':>5}  ", end='')
    for t in trait_names:
        short = t.split('/')[-1][:12]
        print(f"{short:>12}", end='  ')
    print(f"{'max()':>8}  {'ensemble':>8}  {'delta':>7}")
    print('-' * (12 + 7 + len(trait_names) * 14 + 30))

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
        feature_matrix = np.zeros((len(labels), len(trait_names)))

        for i, trait in enumerate(trait_names):
            scores = project(layer_acts, vectors[trait])
            feature_matrix[:, i] = scores
            auroc = roc_auc_score(labels, scores)
            individual_aurocs[trait] = auroc

        best_individual = max(individual_aurocs.values())

        # Ensemble: stratified k-fold CV logistic regression on the 3 features
        cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        ensemble_scores = np.zeros(len(labels))

        for train_idx, test_idx in cv.split(feature_matrix, labels):
            clf = LogisticRegression(C=1.0, max_iter=1000)
            clf.fit(feature_matrix[train_idx], labels[train_idx])
            ensemble_scores[test_idx] = clf.predict_proba(feature_matrix[test_idx])[:, 1]

        ensemble_auroc = roc_auc_score(labels, ensemble_scores)
        delta = ensemble_auroc - best_individual

        # Print row
        print(f"{ds.upper():<12} {len(labels):>5}  ", end='')
        for trait in trait_names:
            print(f"{individual_aurocs[trait]:>12.3f}", end='  ')
        print(f"{best_individual:>8.3f}  {ensemble_auroc:>8.3f}  {delta:>+7.3f}")

        results['datasets'][ds] = {
            'n': len(labels),
            'n_deceptive': n_pos,
            'individual': {t.split('/')[-1]: float(v) for t, v in individual_aurocs.items()},
            'best_individual': float(best_individual),
            'ensemble_auroc': float(ensemble_auroc),
            'delta': float(delta),
        }

    # Summary
    ds_results = results['datasets']
    if ds_results:
        avg_best = np.mean([v['best_individual'] for v in ds_results.values()])
        avg_ensemble = np.mean([v['ensemble_auroc'] for v in ds_results.values()])
        print(f"\n{'AVERAGE':<12} {'':>5}  ", end='')
        print(' ' * (len(trait_names) * 14), end='')
        print(f"{avg_best:>8.3f}  {avg_ensemble:>8.3f}  {avg_ensemble - avg_best:>+7.3f}")

        wins = sum(1 for v in ds_results.values() if v['delta'] > 0.005)
        ties = sum(1 for v in ds_results.values() if abs(v['delta']) <= 0.005)
        losses = sum(1 for v in ds_results.values() if v['delta'] < -0.005)
        print(f"\nEnsemble wins/ties/losses: {wins}/{ties}/{losses}")

    output_path = RESULTS_DIR / 'ensemble_eval.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
