#!/usr/bin/env python3
"""Evaluate Gaussian/Mahalanobis deception detection (zero-label).

Fits a multivariate Gaussian to each trait's positive activations, scores
new examples by Mahalanobis distance. Closer to ANY deception Gaussian = flagged.

Since D=8192 >> n_samples (~50-100), the full covariance is rank-deficient.
We reduce to a PCA subspace per trait first, then compute Mahalanobis there.

Approaches:
  1. Individual Gaussian per trait (baseline)
  2. Min-distance: min(d_concealment, d_lying, d_self_knowledge) — closest Gaussian wins
  3. Joint Gaussian: pool all 3 traits' positive activations, fit one Gaussian
  4. Dot product projection (existing baseline for comparison)

Input:
    experiments/bullshit/extraction/{trait}/base/activations/response__5/residual/train_all_layers.pt
    experiments/bullshit/extraction/{trait}/base/activations/response__5/residual/metadata.json
    experiments/bullshit/extraction/{trait}/base/vectors/response__5/residual/{method}/layer{L}.pt
    experiments/bullshit/results/{dataset}_activations.pt
    experiments/bullshit/prompt_sets/{dataset}_metadata.json

Output:
    experiments/bullshit/results/gaussian_eval.json

Usage:
    python experiments/bullshit/scripts/gaussian_eval.py
    python experiments/bullshit/scripts/gaussian_eval.py --layer 30 --pca-var 0.95
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


def load_trait_activations(trait: str, layer: int):
    """Load training activations, split into pos/neg."""
    act_dir = EXTRACTION_DIR / trait / 'base/activations/response__5/residual'
    acts = torch.load(act_dir / 'train_all_layers.pt', map_location='cpu')
    with open(act_dir / 'metadata.json') as f:
        meta = json.load(f)
    n_pos = meta['n_examples_pos']
    layer_acts = acts[:, layer, :].float()
    return layer_acts[:n_pos], layer_acts[n_pos:]


def fit_gaussian(pos_acts, neg_acts, pca_var=0.95):
    """Fit a Gaussian in a PCA-reduced subspace.

    Centers positive activations by negative mean (same as conceptor_eval),
    then reduces to PCA subspace capturing pca_var of variance.

    Returns (components, mean_reduced, cov_inv, neg_mean, rank):
      components: (rank, D) PCA basis
      mean_reduced: (rank,) mean in PCA space
      cov_inv: (rank, rank) inverse covariance in PCA space
      neg_mean: (D,) for centering test activations
      rank: dimensionality of the subspace
    """
    neg_mean = neg_acts.mean(dim=0)
    X = (pos_acts - neg_mean).numpy()

    # PCA via SVD
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    cumulative = np.cumsum(explained)

    # Keep enough components for pca_var
    rank = int(np.searchsorted(cumulative, pca_var)) + 1
    rank = max(rank, 2)  # at least 2D
    rank = min(rank, len(S))

    components = Vh[:rank]  # (rank, D)

    # Project into PCA space
    X_reduced = X @ components.T  # (n, rank)
    mean_reduced = X_reduced.mean(axis=0)

    # Covariance with Ledoit-Wolf-style shrinkage
    centered = X_reduced - mean_reduced
    cov = (centered.T @ centered) / (len(centered) - 1)

    # Regularize: shrink toward diagonal
    trace_cov = np.trace(cov)
    shrinkage = 0.1
    cov_reg = (1 - shrinkage) * cov + shrinkage * (trace_cov / rank) * np.eye(rank)
    cov_inv = np.linalg.inv(cov_reg)

    return components, mean_reduced, cov_inv, neg_mean.numpy(), rank


def mahalanobis_scores(acts, components, mean_reduced, cov_inv, neg_mean):
    """Compute Mahalanobis distance from the Gaussian for each example.

    Lower distance = more like the positive (deceptive) examples.
    We negate so higher = more deceptive (for AUROC computation).
    """
    X = (acts.numpy() if isinstance(acts, torch.Tensor) else acts) - neg_mean
    X_reduced = X @ components.T  # (n, rank)
    diff = X_reduced - mean_reduced  # (n, rank)
    # d^2 = diff @ cov_inv @ diff.T, per-example
    d_sq = np.sum(diff @ cov_inv * diff, axis=1)
    return -d_sq  # negate: closer to Gaussian = higher score


def dot_project(acts, vec):
    return ((acts @ vec) / vec.norm()).numpy()


def main():
    parser = argparse.ArgumentParser(description='Gaussian/Mahalanobis deception detection')
    parser.add_argument('--layer', type=int, default=30)
    parser.add_argument('--pca-var', type=float, default=0.95, help='Variance to retain in PCA')
    parser.add_argument('--method', default='probe', help='Method for dot-product baseline')
    args = parser.parse_args()

    print(f"Layer: {args.layer}, PCA variance: {args.pca_var:.0%}")

    # Load training activations
    trait_data = {}
    for trait in TRAITS:
        try:
            pos, neg = load_trait_activations(trait, args.layer)
            trait_data[trait] = (pos, neg)
            print(f"  {trait}: {len(pos)} pos + {len(neg)} neg")
        except FileNotFoundError:
            print(f"  WARNING: {trait} activations not found")
    trait_names = list(trait_data.keys())
    if len(trait_names) < 2:
        print("ERROR: Need at least 2 traits")
        return

    # Fit Gaussians
    gaussians = {}
    for trait in trait_names:
        pos, neg = trait_data[trait]
        components, mean_r, cov_inv, neg_mean, rank = fit_gaussian(pos, neg, args.pca_var)
        gaussians[trait] = (components, mean_r, cov_inv, neg_mean, rank)
        print(f"  {trait.split('/')[-1]}: PCA rank={rank}")

    # Joint Gaussian: pool all pos, use mean of neg means
    all_pos = torch.cat([trait_data[t][0] for t in trait_names])
    all_neg = torch.cat([trait_data[t][1] for t in trait_names])
    joint_g = fit_gaussian(all_pos, all_neg, args.pca_var)
    print(f"  joint: PCA rank={joint_g[4]} (pooled {len(all_pos)} pos + {len(all_neg)} neg)")

    # Load baseline vectors
    vectors = {}
    for trait in trait_names:
        try:
            vectors[trait] = load_vector(trait, args.method, args.layer)
        except FileNotFoundError:
            pass

    # Discover datasets
    datasets = [ds for ds in ALL_DATASETS
                if (RESULTS_DIR / f'{ds}_activations.pt').exists()
                and (PROMPTS_DIR / f'{ds}_metadata.json').exists()]
    print(f"\n{len(datasets)} datasets: {', '.join(d.upper() for d in datasets)}")

    # Evaluate
    short = [t.split('/')[-1][:10] for t in trait_names]
    header = f"{'Dataset':<12} {'N':>5}"
    for s in short:
        header += f" {s:>10}"
    header += f" {'best_g':>7} {'min_d':>7} {'joint':>7}"
    if vectors:
        header += f" {'best_dot':>8}"
    print(f"\n{header}")
    print('-' * len(header))

    results = {'layer': args.layer, 'pca_var': args.pca_var, 'baseline_method': args.method, 'datasets': {}}

    for ds in datasets:
        acts = torch.load(RESULTS_DIR / f'{ds}_activations.pt', map_location='cpu')
        layer_acts = acts[:, args.layer, :].float()

        with open(PROMPTS_DIR / f'{ds}_metadata.json') as f:
            meta = json.load(f)
        labels = np.array([meta[str(i + 1)].get('deceptive', False) for i in range(len(acts))])
        n_pos = int(labels.sum())
        if n_pos == 0 or n_pos == len(labels):
            continue

        # Individual Gaussian scores
        indiv = {}
        all_maha_scores = []
        for trait in trait_names:
            components, mean_r, cov_inv, neg_mean, rank = gaussians[trait]
            scores = mahalanobis_scores(layer_acts, components, mean_r, cov_inv, neg_mean)
            all_maha_scores.append(scores)
            indiv[trait] = roc_auc_score(labels, scores)
        best_g = max(indiv.values())

        # Min-distance (max of negated distances = closest Gaussian)
        stacked = np.column_stack(all_maha_scores)
        min_d_scores = stacked.max(axis=1)  # max of negated = min distance
        min_d_auroc = roc_auc_score(labels, min_d_scores)

        # Joint Gaussian
        joint_scores = mahalanobis_scores(layer_acts, *joint_g[:4])
        joint_auroc = roc_auc_score(labels, joint_scores)

        # Dot product baseline
        dot_a = {}
        for trait in trait_names:
            if trait in vectors:
                dot_a[trait] = roc_auc_score(labels, dot_project(layer_acts, vectors[trait]))
        best_dot = max(dot_a.values()) if dot_a else None

        # Print row
        row = f"{ds.upper():<12} {len(labels):>5}"
        for trait in trait_names:
            row += f" {indiv[trait]:>10.3f}"
        row += f" {best_g:>7.3f} {min_d_auroc:>7.3f} {joint_auroc:>7.3f}"
        if best_dot is not None:
            row += f" {best_dot:>8.3f}"
        print(row)

        results['datasets'][ds] = {
            'n': len(labels), 'n_deceptive': n_pos,
            'individual': {t.split('/')[-1]: float(v) for t, v in indiv.items()},
            'best_individual': float(best_g),
            'min_distance': float(min_d_auroc),
            'joint': float(joint_auroc),
            'dot_aurocs': {t.split('/')[-1]: float(v) for t, v in dot_a.items()},
            'best_dot': float(best_dot) if best_dot is not None else None,
        }

    # Summary
    d = results['datasets']
    if d:
        vals = list(d.values())
        avg_g = np.mean([v['best_individual'] for v in vals])
        avg_min = np.mean([v['min_distance'] for v in vals])
        avg_joint = np.mean([v['joint'] for v in vals])
        row = f"\n{'AVERAGE':<12} {'':>5}" + ' ' * (len(trait_names) * 11)
        row += f" {avg_g:>7.3f} {avg_min:>7.3f} {avg_joint:>7.3f}"
        if vectors:
            avg_dot = np.mean([v['best_dot'] for v in vals if v['best_dot'] is not None])
            row += f" {avg_dot:>8.3f}"
        print(row)

        if vectors:
            print(f"\n  Δ vs best_dot: min_distance={avg_min - avg_dot:+.3f}, joint={avg_joint - avg_dot:+.3f}")

        # Per-trait Gaussian rank info
        print(f"\nGaussian subspace dims:")
        for trait in trait_names:
            print(f"  {trait.split('/')[-1]}: rank={gaussians[trait][4]}")
        print(f"  joint: rank={joint_g[4]}")

    output_path = RESULTS_DIR / 'gaussian_eval.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
