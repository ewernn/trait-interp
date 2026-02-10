#!/usr/bin/env python3
"""Evaluate deception subspace and OR-threshold detection (all zero-label).

Three approaches:
  1. PCA subspace: SVD on the 3 trait vectors → shared deception subspace.
  2. OR-threshold (dot product): Alpaca-calibrated per-vector thresholds, flag if ANY fires.
  3. OR-threshold (cosine): Same but using cosine similarity (magnitude-normalized).
     Tests whether domain shift is a magnitude issue.

All are zero-label — no benchmark labels used at any point.

Input:
    experiments/bullshit/extraction/{trait}/base/vectors/response__5/residual/{method}/layer{L}.pt
    experiments/bullshit/results/{dataset}_activations.pt
    experiments/bullshit/results/alpaca_activations.pt
    experiments/bullshit/prompt_sets/{dataset}_metadata.json

Output:
    experiments/bullshit/results/subspace_and_threshold_eval.json

Usage:
    python experiments/bullshit/scripts/subspace_and_threshold_eval.py
    python experiments/bullshit/scripts/subspace_and_threshold_eval.py --layer 20
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


def dot_project(acts: torch.Tensor, vec: torch.Tensor) -> np.ndarray:
    """Dot product projection (magnitude-sensitive)."""
    return ((acts @ vec) / vec.norm()).numpy()


def cosine_project(acts: torch.Tensor, vec: torch.Tensor) -> np.ndarray:
    """Cosine similarity (direction-only, magnitude-normalized)."""
    act_norms = acts.norm(dim=1, keepdim=True).clamp(min=1e-8)
    vec_unit = vec / vec.norm()
    return ((acts / act_norms) @ vec_unit).numpy()


def main():
    parser = argparse.ArgumentParser(description='Subspace and OR-threshold deception detection')
    parser.add_argument('--layer', type=int, default=30)
    parser.add_argument('--method', default='probe')
    parser.add_argument('--fpr-target', type=float, default=0.01, help='Per-vector FPR target')
    args = parser.parse_args()

    # =====================================================================
    # Part 1: PCA on the 3 trait vectors
    # =====================================================================
    print("=" * 70)
    print(f"PART 1: Deception Subspace Analysis (L{args.layer}, {args.method})")
    print("=" * 70)

    vectors = {}
    for trait in TRAITS:
        try:
            vectors[trait] = load_vector(trait, args.method, args.layer)
        except FileNotFoundError:
            print(f"  WARNING: {trait} not found")
    trait_names = list(vectors.keys())

    V = torch.stack([vectors[t] for t in trait_names])
    print(f"\nVector matrix: {V.shape}")

    print("\nPairwise cosine similarities:")
    for i, t1 in enumerate(trait_names):
        for j, t2 in enumerate(trait_names):
            if j > i:
                cos = torch.nn.functional.cosine_similarity(V[i:i+1], V[j:j+1]).item()
                print(f"  {t1.split('/')[-1]} vs {t2.split('/')[-1]}: {cos:.3f}")

    U, S, Vh = torch.linalg.svd(V, full_matrices=False)
    total_var = (S ** 2).sum()
    explained = (S ** 2) / total_var
    cumulative = torch.cumsum(explained, dim=0)

    print(f"\nSingular values: {S.tolist()}")
    print(f"Variance explained: {[f'{x:.1%}' for x in explained.tolist()]}")
    print(f"Cumulative: {[f'{x:.1%}' for x in cumulative.tolist()]}")

    if explained[0] > 0.9:
        print("\n→ First component captures >90% — vectors are near-collinear.")
    elif cumulative[1] > 0.95:
        print("\n→ 2D subspace captures >95% — deception lives in a plane.")
    else:
        print("\n→ All 3 dimensions contribute — vectors are genuinely diverse.")

    subspace_basis = Vh

    # =====================================================================
    # Part 2: Discover datasets + load Alpaca
    # =====================================================================
    datasets = []
    for ds in ALL_DATASETS:
        if (RESULTS_DIR / f'{ds}_activations.pt').exists() and (PROMPTS_DIR / f'{ds}_metadata.json').exists():
            datasets.append(ds)
    print(f"\nFound {len(datasets)} datasets")

    alpaca_path = RESULTS_DIR / 'alpaca_activations.pt'
    has_alpaca = alpaca_path.exists()
    if has_alpaca:
        alpaca_acts = torch.load(alpaca_path, map_location='cpu')
        alpaca_layer = alpaca_acts[:, args.layer, :].float()
        print(f"Alpaca control: {len(alpaca_layer)} examples")
    else:
        print("WARNING: No Alpaca activations — skipping OR-threshold eval")

    # Calibrate thresholds on Alpaca (both dot product and cosine)
    dot_thresholds = {}
    cos_thresholds = {}
    if has_alpaca:
        print(f"\nAlpaca-calibrated thresholds ({args.fpr_target:.0%} FPR):")
        print(f"  {'vector':<25} {'dot threshold':>14} {'cosine threshold':>16}")
        for trait in trait_names:
            vec = vectors[trait]
            dot_scores = dot_project(alpaca_layer, vec)
            cos_scores = cosine_project(alpaca_layer, vec)
            dot_thresholds[trait] = np.percentile(dot_scores, 100 * (1 - args.fpr_target))
            cos_thresholds[trait] = np.percentile(cos_scores, 100 * (1 - args.fpr_target))
            short_name = trait.split('/')[-1]
            print(f"  {short_name:<25} {dot_thresholds[trait]:>14.4f} {cos_thresholds[trait]:>16.4f}")

    # =====================================================================
    # Part 3: Evaluate each dataset
    # =====================================================================
    print(f"\n{'=' * 70}")
    print(f"PART 2: AUROC — Dot Product vs Cosine Similarity (L{args.layer})")
    print(f"{'=' * 70}")

    short = [t.split('/')[-1][:10] for t in trait_names]
    print(f"\n{'Dataset':<12} {'N':>5}", end='')
    for s in short:
        print(f"  {'dot':>5}/{s[:3]:>3}", end='')
    print(f"  {'best_dot':>8} {'best_cos':>8} {'subspace':>8}")
    print('-' * (20 + len(trait_names) * 11 + 30))

    results = {'layer': args.layer, 'method': args.method, 'fpr_target': args.fpr_target, 'datasets': {}}
    recall_data = {}

    for ds in datasets:
        acts = torch.load(RESULTS_DIR / f'{ds}_activations.pt', map_location='cpu')
        layer_acts = acts[:, args.layer, :].float()

        with open(PROMPTS_DIR / f'{ds}_metadata.json') as f:
            meta = json.load(f)
        labels = np.array([meta[str(i+1)].get('deceptive', False) for i in range(len(acts))])

        n_pos = int(labels.sum())
        if n_pos == 0 or n_pos == len(labels):
            continue

        # Individual AUROCs — both dot product and cosine
        dot_aurocs = {}
        cos_aurocs = {}
        dot_scores_all = {}
        cos_scores_all = {}

        for trait in trait_names:
            vec = vectors[trait]
            d_scores = dot_project(layer_acts, vec)
            c_scores = cosine_project(layer_acts, vec)
            dot_scores_all[trait] = d_scores
            cos_scores_all[trait] = c_scores
            dot_aurocs[trait] = roc_auc_score(labels, d_scores)
            cos_aurocs[trait] = roc_auc_score(labels, c_scores)

        best_dot = max(dot_aurocs.values())
        best_cos = max(cos_aurocs.values())

        # Subspace
        subspace_proj = (layer_acts @ subspace_basis.T).numpy()
        subspace_norms = np.linalg.norm(subspace_proj, axis=1)
        subspace_auroc = roc_auc_score(labels, subspace_norms)

        # Print AUROC row
        print(f"{ds.upper():<12} {len(labels):>5}", end='')
        for trait in trait_names:
            print(f"  {dot_aurocs[trait]:>.3f}/{cos_aurocs[trait]:>.3f}", end='')
        print(f"  {best_dot:>8.3f} {best_cos:>8.3f} {subspace_auroc:>8.3f}")

        ds_result = {
            'n': len(labels), 'n_deceptive': n_pos,
            'dot_aurocs': {t.split('/')[-1]: float(v) for t, v in dot_aurocs.items()},
            'cos_aurocs': {t.split('/')[-1]: float(v) for t, v in cos_aurocs.items()},
            'best_dot': float(best_dot),
            'best_cos': float(best_cos),
            'subspace_auroc': float(subspace_auroc),
        }

        # OR-threshold: both dot and cosine
        if has_alpaca:
            dot_or = {}
            cos_or = {}

            for trait in trait_names:
                dot_or[trait] = dot_scores_all[trait] >= dot_thresholds[trait]
                cos_or[trait] = cos_scores_all[trait] >= cos_thresholds[trait]

            dot_or_flagged = np.any(np.column_stack(list(dot_or.values())), axis=1)
            cos_or_flagged = np.any(np.column_stack(list(cos_or.values())), axis=1)

            dot_or_recall = float(dot_or_flagged[labels == 1].mean()) if n_pos > 0 else 0.0
            dot_or_fpr = float(dot_or_flagged[labels == 0].mean()) if (len(labels) - n_pos) > 0 else 0.0
            cos_or_recall = float(cos_or_flagged[labels == 1].mean()) if n_pos > 0 else 0.0
            cos_or_fpr = float(cos_or_flagged[labels == 0].mean()) if (len(labels) - n_pos) > 0 else 0.0

            ds_result['or_dot'] = {
                'per_vector_recall': {t.split('/')[-1]: float(dot_or[t][labels == 1].mean()) for t in trait_names},
                'recall': dot_or_recall, 'fpr': dot_or_fpr,
            }
            ds_result['or_cosine'] = {
                'per_vector_recall': {t.split('/')[-1]: float(cos_or[t][labels == 1].mean()) for t in trait_names},
                'recall': cos_or_recall, 'fpr': cos_or_fpr,
            }
            recall_data[ds] = {
                'n_deceptive': n_pos,
                'dot': {'recall': dot_or_recall, 'fpr': dot_or_fpr,
                        'per_vector': {t.split('/')[-1]: float(dot_or[t][labels == 1].mean()) for t in trait_names}},
                'cos': {'recall': cos_or_recall, 'fpr': cos_or_fpr,
                        'per_vector': {t.split('/')[-1]: float(cos_or[t][labels == 1].mean()) for t in trait_names}},
            }

        results['datasets'][ds] = ds_result

    # Summary AUROC
    d = results['datasets']
    if d:
        avg_dot = np.mean([v['best_dot'] for v in d.values()])
        avg_cos = np.mean([v['best_cos'] for v in d.values()])
        avg_sub = np.mean([v['subspace_auroc'] for v in d.values()])
        print(f"\n{'AVERAGE':<12} {'':>5}", end='')
        print(' ' * (len(trait_names) * 11), end='')
        print(f"  {avg_dot:>8.3f} {avg_cos:>8.3f} {avg_sub:>8.3f}")

    # OR-threshold tables (dot vs cosine side by side)
    if recall_data:
        for mode, label in [('dot', 'Dot Product'), ('cos', 'Cosine Similarity')]:
            print(f"\n--- OR-Threshold: {label} @ {args.fpr_target:.0%} per-vector FPR ---")
            print(f"{'Dataset':<12} {'N_dec':>5}", end='')
            for s in short:
                print(f" {s:>10}", end='')
            print(f" {'OR-recall':>9} {'OR-FPR':>7}")
            print('-' * (20 + len(trait_names) * 11 + 20))

            for ds, rd in recall_data.items():
                print(f"{ds.upper():<12} {rd['n_deceptive']:>5}", end='')
                for trait in trait_names:
                    r = rd[mode]['per_vector'].get(trait.split('/')[-1], 0)
                    print(f" {r:>10.1%}", end='')
                print(f" {rd[mode]['recall']:>9.1%} {rd[mode]['fpr']:>7.1%}")

            avg_recall = np.mean([v[mode]['recall'] for v in recall_data.values()])
            avg_fpr = np.mean([v[mode]['fpr'] for v in recall_data.values()])
            print(f"{'AVERAGE':<12} {'':>5}", end='')
            print(' ' * (len(trait_names) * 11), end='')
            print(f" {avg_recall:>9.1%} {avg_fpr:>7.1%}")

    # Save
    results['subspace'] = {
        'singular_values': S.tolist(),
        'variance_explained': explained.tolist(),
        'cumulative_variance': cumulative.tolist(),
        'pairwise_cosine': {
            f"{trait_names[i].split('/')[-1]}_vs_{trait_names[j].split('/')[-1]}":
            float(torch.nn.functional.cosine_similarity(V[i:i+1], V[j:j+1]))
            for i in range(len(trait_names)) for j in range(i+1, len(trait_names))
        },
    }

    output_path = RESULTS_DIR / 'subspace_and_threshold_eval.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
