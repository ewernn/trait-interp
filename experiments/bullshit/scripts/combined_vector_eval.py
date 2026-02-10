#!/usr/bin/env python3
"""Compare strategies for combining multiple deception vectors (all zero-label).

Strategies:
  1. Individual: best single vector per dataset (baseline)
  2. Mean-of-z: z-score each vector's projections, average per example
  3. Max-of-z: z-score each vector's projections, take max per example
  4. Combined probe: pool training data from all traits, train one probe

All strategies are zero-label — no benchmark labels used at any point.

Input:
    experiments/bullshit/extraction/{trait}/base/activations/response__5/residual/train_all_layers.pt
    experiments/bullshit/extraction/{trait}/base/activations/response__5/residual/metadata.json
    experiments/bullshit/extraction/{trait}/base/vectors/response__5/residual/probe/layer{L}.pt
    experiments/bullshit/results/{dataset}_activations.pt
    experiments/bullshit/prompt_sets/{dataset}_metadata.json

Output:
    experiments/bullshit/results/combined_vector_eval.json

Usage:
    python experiments/bullshit/scripts/combined_vector_eval.py
    python experiments/bullshit/scripts/combined_vector_eval.py --layer 20
"""

import argparse
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
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
    """Load training activations for a trait, split into pos/neg."""
    act_dir = EXTRACTION_DIR / trait / 'base/activations/response__5/residual'
    acts = torch.load(act_dir / 'train_all_layers.pt', map_location='cpu')
    with open(act_dir / 'metadata.json') as f:
        meta = json.load(f)

    n_pos = meta['n_examples_pos']
    layer_acts = acts[:, layer, :].float()
    pos_acts = layer_acts[:n_pos]
    neg_acts = layer_acts[n_pos:]
    return pos_acts, neg_acts


def train_probe(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
    """Train logistic regression probe, return unit-norm weight vector."""
    X = torch.cat([pos_acts, neg_acts], dim=0).numpy()
    y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])

    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / (row_norms + 1e-8)

    probe = LogisticRegression(max_iter=1000, C=1.0, penalty='l2', solver='lbfgs', random_state=42)
    probe.fit(X_normalized, y)

    vector = torch.from_numpy(probe.coef_[0]).float()
    vector = vector / (vector.norm() + 1e-8)
    return vector


def project(acts: torch.Tensor, vec: torch.Tensor) -> np.ndarray:
    return ((acts @ vec) / vec.norm()).numpy()


def main():
    parser = argparse.ArgumentParser(description='Compare vector combination strategies')
    parser.add_argument('--layer', type=int, default=30, help='Layer to evaluate')
    parser.add_argument('--method', default='probe', help='Method for individual vectors')
    args = parser.parse_args()

    print(f"Layer: {args.layer}, Method: {args.method}")
    print(f"Traits: {', '.join(TRAITS)}")

    # --- Load individual vectors ---
    vectors = {}
    for trait in TRAITS:
        try:
            vectors[trait] = load_vector(trait, args.method, args.layer)
        except FileNotFoundError:
            print(f"  WARNING: {trait} not found, skipping")
    trait_names = list(vectors.keys())
    print(f"Loaded {len(trait_names)} individual vectors")

    # --- Train combined probe ---
    print("\nTraining combined probe (pooled training data)...")
    all_pos, all_neg = [], []
    for trait in TRAITS:
        try:
            pos, neg = load_trait_activations(trait, args.layer)
            all_pos.append(pos)
            all_neg.append(neg)
            print(f"  {trait}: {len(pos)} pos + {len(neg)} neg")
        except FileNotFoundError:
            print(f"  WARNING: {trait} activations not found, skipping")

    combined_pos = torch.cat(all_pos, dim=0)
    combined_neg = torch.cat(all_neg, dim=0)
    print(f"  Combined: {len(combined_pos)} pos + {len(combined_neg)} neg")
    combined_vector = train_probe(combined_pos, combined_neg)

    # Check cosine similarity between combined and individual vectors
    print("\nCosine similarity (combined vs individual):")
    for trait in trait_names:
        cos = torch.nn.functional.cosine_similarity(combined_vector.unsqueeze(0), vectors[trait].unsqueeze(0)).item()
        print(f"  vs {trait.split('/')[-1]}: {cos:.3f}")

    # --- Discover datasets ---
    datasets = []
    for ds in ALL_DATASETS:
        if (RESULTS_DIR / f'{ds}_activations.pt').exists() and (PROMPTS_DIR / f'{ds}_metadata.json').exists():
            datasets.append(ds)
    print(f"\nEvaluating on {len(datasets)} datasets: {', '.join(d.upper() for d in datasets)}")

    # --- Evaluate ---
    short = [t.split('/')[-1][:10] for t in trait_names]
    print(f"\n{'Dataset':<12} {'N':>5}", end='')
    for s in short:
        print(f" {s:>10}", end='')
    print(f" {'best':>7} {'mean-z':>7} {'max-z':>7} {'combined':>8}")
    print('-' * (20 + len(trait_names) * 11 + 35))

    results = {'layer': args.layer, 'method': args.method, 'datasets': {}}

    for ds in datasets:
        acts = torch.load(RESULTS_DIR / f'{ds}_activations.pt', map_location='cpu')
        layer_acts = acts[:, args.layer, :].float()

        with open(PROMPTS_DIR / f'{ds}_metadata.json') as f:
            meta = json.load(f)
        labels = np.array([meta[str(i+1)].get('deceptive', False) for i in range(len(acts))])

        n_pos = int(labels.sum())
        if n_pos == 0 or n_pos == len(labels):
            print(f"{ds.upper():<12} {len(labels):>5}  SKIPPED")
            continue

        # Individual
        individual = {}
        all_scores = []
        for trait in trait_names:
            scores = project(layer_acts, vectors[trait])
            all_scores.append(scores)
            individual[trait] = roc_auc_score(labels, scores)
        best_ind = max(individual.values())

        # Mean-of-z
        score_matrix = np.column_stack(all_scores)
        z = (score_matrix - score_matrix.mean(axis=0)) / (score_matrix.std(axis=0) + 1e-8)
        mean_auroc = roc_auc_score(labels, z.mean(axis=1))

        # Max-of-z
        max_auroc = roc_auc_score(labels, z.max(axis=1))

        # Combined probe
        combined_scores = project(layer_acts, combined_vector)
        combined_auroc = roc_auc_score(labels, combined_scores)

        print(f"{ds.upper():<12} {len(labels):>5}", end='')
        for trait in trait_names:
            print(f" {individual[trait]:>10.3f}", end='')
        print(f" {best_ind:>7.3f} {mean_auroc:>7.3f} {max_auroc:>7.3f} {combined_auroc:>8.3f}")

        results['datasets'][ds] = {
            'n': len(labels), 'n_deceptive': n_pos,
            'individual': {t.split('/')[-1]: float(v) for t, v in individual.items()},
            'best_individual': float(best_ind),
            'mean_z': float(mean_auroc),
            'max_z': float(max_auroc),
            'combined_probe': float(combined_auroc),
        }

    # Summary
    d = results['datasets']
    if d:
        print(f"\n{'AVERAGE':<12} {'':>5}", end='')
        print(' ' * (len(trait_names) * 11), end='')
        avg_best = np.mean([v['best_individual'] for v in d.values()])
        avg_mean = np.mean([v['mean_z'] for v in d.values()])
        avg_max = np.mean([v['max_z'] for v in d.values()])
        avg_comb = np.mean([v['combined_probe'] for v in d.values()])
        print(f" {avg_best:>7.3f} {avg_mean:>7.3f} {avg_max:>7.3f} {avg_comb:>8.3f}")

        print(f"\n{'':>12} {'':>5}", end='')
        print(' ' * (len(trait_names) * 11), end='')
        print(f" {'base':>7} {avg_mean-avg_best:>+7.3f} {avg_max-avg_best:>+7.3f} {avg_comb-avg_best:>+8.3f}")

    output_path = RESULTS_DIR / 'combined_vector_eval.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
