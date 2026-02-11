#!/usr/bin/env python3
"""Evaluate conceptor-based deception detection (zero-label).

Conceptors are soft projection matrices that capture multi-dimensional concept
patterns. Boolean OR combines three conceptors to detect ANY deception type.
Reference: "Conceptors for Steering Large Language Models" (NeurIPS 2024).

Approaches:
  1. Individual conceptor scores per trait (baseline)
  2. Soft OR: 1 - prod(1 - s_i) — approximate, no matrix algebra
  3. Exact OR: NOT(AND(NOT(C1), NOT(C2), NOT(C3))) — algebraically exact
  4. Dot product projection (existing baseline for comparison)

Input:
    experiments/bullshit/extraction/{trait}/base/activations/response__5/residual/train_all_layers.pt
    experiments/bullshit/extraction/{trait}/base/activations/response__5/residual/metadata.json
    experiments/bullshit/extraction/{trait}/base/vectors/response__5/residual/{method}/layer{L}.pt
    experiments/bullshit/results/{dataset}_activations.pt
    experiments/bullshit/prompt_sets/{dataset}_metadata.json

Output:
    experiments/bullshit/results/conceptor_eval.json

Usage:
    python experiments/bullshit/scripts/conceptor_eval.py
    python experiments/bullshit/scripts/conceptor_eval.py --layer 20 --alpha 0.5 1.0 5.0
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


def compute_conceptor(pos_acts, neg_acts, alpha):
    """Compute conceptor from positive/negative activations.

    Centers positive activations by negative mean, then computes:
      R = X_centered^T X_centered / n  (correlation of differences)
      C = R (R + alpha^{-2} I)^{-1}   (conceptor matrix)

    Uses SVD for efficiency (avoids forming D x D matrices).

    Returns (V, c_eigs, neg_mean):
      V: (D, rank) eigenspace basis
      c_eigs: (rank,) conceptor eigenvalues in [0, 1]
      neg_mean: (D,) for centering test activations
    """
    neg_mean = neg_acts.mean(dim=0)
    X = pos_acts - neg_mean

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh.T  # (D, rank)
    sigma2 = (S ** 2) / len(X)  # eigenvalues of R

    c_eigs = sigma2 / (sigma2 + alpha ** (-2))
    return V, c_eigs, neg_mean


def conceptor_score(acts, V, c_eigs, neg_mean):
    """Score = ||C(x - mu_neg)||^2 / ||x - mu_neg||^2."""
    X = acts - neg_mean
    X_proj = X @ V  # (batch, rank)
    CX_proj = X_proj * c_eigs.unsqueeze(0)

    numerator = (CX_proj ** 2).sum(dim=1)
    denominator = (X ** 2).sum(dim=1).clamp(min=1e-8)
    return (numerator / denominator).numpy()


def boolean_or_conceptors(conceptors):
    """Exact Boolean OR via NOT(AND(NOT(C1), ..., NOT(Cn))).

    Works in the joint eigenspace of all conceptors for efficiency.
    AND(A, B) = (A^{-1} + B^{-1} - I)^{-1}

    Returns (Q, or_matrix):
      Q: (D, joint_rank) basis of joint eigenspace
      or_matrix: (joint_rank, joint_rank) OR conceptor in that basis
    """
    all_V = torch.cat([c[0] for c in conceptors], dim=1)
    Q, R = torch.linalg.qr(all_V)
    rank = (R.diag().abs() > 1e-6).sum().item()
    Q = Q[:, :rank]
    I_j = torch.eye(rank)

    # Project each NOT(C_i) into joint space
    not_mats = []
    for V_i, c_eigs_i, _ in conceptors:
        V_proj = Q.T @ V_i
        C_joint = V_proj @ torch.diag(c_eigs_i) @ V_proj.T
        not_mats.append(I_j - C_joint)

    # Chain binary AND: AND(A, B) = (A^{-1} + B^{-1} - I)^{-1}
    eps = 1e-6
    result = not_mats[0]
    for i in range(1, len(not_mats)):
        r_inv = torch.linalg.inv(result + eps * I_j)
        n_inv = torch.linalg.inv(not_mats[i] + eps * I_j)
        result = torch.linalg.inv(r_inv + n_inv - I_j + eps * I_j)

    return Q, I_j - result  # NOT(AND result) = OR


def or_conceptor_score(acts, Q, or_matrix, neg_means):
    """Score using OR-conceptor. Centers by mean of all traits' neg means."""
    mean_neg = torch.stack(neg_means).mean(dim=0)
    X = acts - mean_neg
    X_proj = X @ Q
    CX_proj = X_proj @ or_matrix

    numerator = (CX_proj ** 2).sum(dim=1)
    denominator = (X ** 2).sum(dim=1).clamp(min=1e-8)
    return (numerator / denominator).numpy()


def dot_project(acts, vec):
    return ((acts @ vec) / vec.norm()).numpy()


def main():
    parser = argparse.ArgumentParser(description='Conceptor-based deception detection')
    parser.add_argument('--layer', type=int, default=30)
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument('--method', default='probe', help='Method for dot-product baseline')
    args = parser.parse_args()

    print(f"Layer: {args.layer}, Alphas: {args.alpha}")

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
        print("ERROR: Need at least 2 traits for combination")
        return

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

    results = {'layer': args.layer, 'baseline_method': args.method, 'alphas': {}}

    for alpha in args.alpha:
        print(f"\n{'=' * 70}")
        print(f"ALPHA = {alpha}")
        print(f"{'=' * 70}")

        # Compute conceptors
        conceptors = []
        for trait in trait_names:
            pos, neg = trait_data[trait]
            V, c_eigs, neg_mean = compute_conceptor(pos, neg, alpha)
            conceptors.append((V, c_eigs, neg_mean))
            active = (c_eigs > 0.01).sum().item()
            print(f"  {trait.split('/')[-1]}: {active}/{len(c_eigs)} active dims, "
                  f"top eig: {c_eigs[0]:.3f}, sum eigs: {c_eigs.sum():.1f}")

        # OR conceptor
        neg_means = [c[2] for c in conceptors]
        try:
            Q, or_mat = boolean_or_conceptors(conceptors)
            or_eigs = torch.linalg.eigvalsh(or_mat)
            or_active = (or_eigs > 0.01).sum().item()
            print(f"  OR: joint_rank={Q.shape[1]}, active={or_active}")
            has_or = True
        except Exception as e:
            print(f"  OR failed: {e}")
            has_or = False

        # Evaluate
        short = [t.split('/')[-1][:10] for t in trait_names]
        header = f"{'Dataset':<12} {'N':>5}"
        for s in short:
            header += f" {s:>10}"
        header += f" {'best_c':>7} {'soft_or':>7}"
        if has_or:
            header += f" {'exact_or':>8}"
        if vectors:
            header += f" {'best_dot':>8}"
        print(f"\n{header}")
        print('-' * len(header))

        alpha_results = {}
        for ds in datasets:
            acts = torch.load(RESULTS_DIR / f'{ds}_activations.pt', map_location='cpu')
            layer_acts = acts[:, args.layer, :].float()

            with open(PROMPTS_DIR / f'{ds}_metadata.json') as f:
                meta = json.load(f)
            labels = np.array([meta[str(i + 1)].get('deceptive', False) for i in range(len(acts))])
            n_pos = int(labels.sum())
            if n_pos == 0 or n_pos == len(labels):
                continue

            # Individual conceptor scores
            indiv = {}
            all_scores = []
            for i, trait in enumerate(trait_names):
                V, c_eigs, neg_mean = conceptors[i]
                scores = conceptor_score(layer_acts, V, c_eigs, neg_mean)
                all_scores.append(scores)
                indiv[trait] = roc_auc_score(labels, scores)
            best_c = max(indiv.values())

            # Soft OR
            sm = np.column_stack(all_scores)
            soft_or = 1 - np.prod(1 - sm, axis=1)
            soft_auroc = roc_auc_score(labels, soft_or)

            # Exact OR
            exact_auroc = None
            if has_or:
                or_scores = or_conceptor_score(layer_acts, Q, or_mat, neg_means)
                exact_auroc = roc_auc_score(labels, or_scores)

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
            row += f" {best_c:>7.3f} {soft_auroc:>7.3f}"
            if has_or:
                row += f" {exact_auroc:>8.3f}"
            if best_dot is not None:
                row += f" {best_dot:>8.3f}"
            print(row)

            alpha_results[ds] = {
                'n': len(labels), 'n_deceptive': n_pos,
                'individual': {t.split('/')[-1]: float(v) for t, v in indiv.items()},
                'best_individual': float(best_c),
                'soft_or': float(soft_auroc),
                'exact_or': float(exact_auroc) if exact_auroc is not None else None,
                'dot_aurocs': {t.split('/')[-1]: float(v) for t, v in dot_a.items()},
                'best_dot': float(best_dot) if best_dot is not None else None,
            }

        # Summary
        if alpha_results:
            vals = list(alpha_results.values())
            avg_c = np.mean([v['best_individual'] for v in vals])
            avg_soft = np.mean([v['soft_or'] for v in vals])
            row = f"\n{'AVERAGE':<12} {'':>5}" + ' ' * (len(trait_names) * 11)
            row += f" {avg_c:>7.3f} {avg_soft:>7.3f}"
            if has_or:
                avg_exact = np.mean([v['exact_or'] for v in vals if v['exact_or'] is not None])
                row += f" {avg_exact:>8.3f}"
            if vectors:
                avg_dot = np.mean([v['best_dot'] for v in vals if v['best_dot'] is not None])
                row += f" {avg_dot:>8.3f}"
            print(row)

            print(f"\n  Δ vs best_dot: soft_or={avg_soft - avg_dot:+.3f}" if vectors else "")

        results['alphas'][str(alpha)] = alpha_results

    output_path = RESULTS_DIR / 'conceptor_eval.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
