#!/usr/bin/env python3
"""Evaluate deception subspace and OR-threshold detection (all zero-label).

Two approaches:
  1. PCA subspace: SVD on the 3 trait vectors → shared deception subspace.
     Project activations, use distance-from-origin as detection score.
  2. OR-threshold: Flag if ANY vector exceeds its Alpaca-calibrated threshold.
     Report combined recall at bounded FPR.

Both are zero-label — no benchmark labels used at any point.

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

    # Stack vectors: [n_traits, hidden_dim]
    V = torch.stack([vectors[t] for t in trait_names])
    print(f"\nVector matrix: {V.shape}")

    # Pairwise cosine similarities
    print("\nPairwise cosine similarities:")
    for i, t1 in enumerate(trait_names):
        for j, t2 in enumerate(trait_names):
            if j > i:
                cos = torch.nn.functional.cosine_similarity(V[i:i+1], V[j:j+1]).item()
                print(f"  {t1.split('/')[-1]} vs {t2.split('/')[-1]}: {cos:.3f}")

    # SVD
    U, S, Vh = torch.linalg.svd(V, full_matrices=False)
    total_var = (S ** 2).sum()
    explained = (S ** 2) / total_var
    cumulative = torch.cumsum(explained, dim=0)

    print(f"\nSingular values: {S.tolist()}")
    print(f"Variance explained: {[f'{x:.1%}' for x in explained.tolist()]}")
    print(f"Cumulative: {[f'{x:.1%}' for x in cumulative.tolist()]}")

    if explained[0] > 0.9:
        print("\n→ First component captures >90% — vectors are near-collinear.")
        print("  Your 3 'different' concepts mostly point in the same direction.")
    elif cumulative[1] > 0.95:
        print("\n→ 2D subspace captures >95% — deception lives in a plane.")
    else:
        print("\n→ All 3 dimensions contribute — vectors are genuinely diverse.")

    # Subspace basis vectors (rows of Vh)
    subspace_basis = Vh  # [n_components, hidden_dim]

    # =====================================================================
    # Part 2: Evaluate on benchmark datasets
    # =====================================================================
    datasets = []
    for ds in ALL_DATASETS:
        if (RESULTS_DIR / f'{ds}_activations.pt').exists() and (PROMPTS_DIR / f'{ds}_metadata.json').exists():
            datasets.append(ds)
    print(f"\nFound {len(datasets)} datasets")

    # Load Alpaca for threshold calibration
    alpaca_path = RESULTS_DIR / 'alpaca_activations.pt'
    has_alpaca = alpaca_path.exists()
    if has_alpaca:
        alpaca_acts = torch.load(alpaca_path, map_location='cpu')
        alpaca_layer = alpaca_acts[:, args.layer, :].float()
        print(f"Alpaca control: {len(alpaca_layer)} examples")
    else:
        print("WARNING: No Alpaca activations — skipping OR-threshold eval")

    # Calibrate per-vector thresholds on Alpaca
    thresholds = {}
    if has_alpaca:
        print(f"\nAlpaca-calibrated thresholds ({args.fpr_target:.0%} FPR):")
        for trait in trait_names:
            vec = vectors[trait]
            scores = ((alpaca_layer @ vec) / vec.norm()).numpy()
            threshold = np.percentile(scores, 100 * (1 - args.fpr_target))
            thresholds[trait] = threshold
            print(f"  {trait.split('/')[-1]}: {threshold:.3f}")

        # Subspace threshold: project Alpaca onto subspace, use norm
        alpaca_subspace = (alpaca_layer @ subspace_basis.T).numpy()  # [n, n_components]
        alpaca_norms = np.linalg.norm(alpaca_subspace, axis=1)
        subspace_threshold = np.percentile(alpaca_norms, 100 * (1 - args.fpr_target))
        print(f"  subspace (norm): {subspace_threshold:.3f}")

    # =====================================================================
    # Part 3: Evaluate each dataset
    # =====================================================================
    print(f"\n{'=' * 70}")
    print(f"PART 2: Detection Results (L{args.layer})")
    print(f"{'=' * 70}")

    short = [t.split('/')[-1][:10] for t in trait_names]
    print(f"\n--- AUROC ---")
    print(f"{'Dataset':<12} {'N':>5}", end='')
    for s in short:
        print(f" {s:>10}", end='')
    print(f" {'best':>7} {'subspace':>8}")
    print('-' * (20 + len(trait_names) * 11 + 18))

    results = {'layer': args.layer, 'method': args.method, 'fpr_target': args.fpr_target, 'datasets': {}}

    # Store for recall table
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

        # Individual AUROCs
        individual = {}
        individual_scores = {}
        for trait in trait_names:
            vec = vectors[trait]
            scores = ((layer_acts @ vec) / vec.norm()).numpy()
            individual[trait] = roc_auc_score(labels, scores)
            individual_scores[trait] = scores
        best_ind = max(individual.values())

        # Subspace: project onto basis, use L2 norm as score
        subspace_proj = (layer_acts @ subspace_basis.T).numpy()  # [n, n_components]
        subspace_norms = np.linalg.norm(subspace_proj, axis=1)
        subspace_auroc = roc_auc_score(labels, subspace_norms)

        print(f"{ds.upper():<12} {len(labels):>5}", end='')
        for trait in trait_names:
            print(f" {individual[trait]:>10.3f}", end='')
        print(f" {best_ind:>7.3f} {subspace_auroc:>8.3f}")

        ds_result = {
            'n': len(labels), 'n_deceptive': n_pos,
            'individual': {t.split('/')[-1]: float(v) for t, v in individual.items()},
            'best_individual': float(best_ind),
            'subspace_auroc': float(subspace_auroc),
        }

        # OR-threshold recall
        if has_alpaca:
            per_vector_flags = {}
            per_vector_recall = {}
            for trait in trait_names:
                flagged = individual_scores[trait] >= thresholds[trait]
                per_vector_flags[trait] = flagged
                recall = flagged[labels == 1].mean() if labels.sum() > 0 else 0.0
                per_vector_recall[trait] = float(recall)

            # OR: flag if ANY vector exceeds threshold
            or_flagged = np.any(np.column_stack(list(per_vector_flags.values())), axis=1)
            or_recall = float(or_flagged[labels == 1].mean()) if labels.sum() > 0 else 0.0
            or_fpr = float(or_flagged[labels == 0].mean()) if (labels == 0).sum() > 0 else 0.0

            ds_result['or_threshold'] = {
                'per_vector_recall': {t.split('/')[-1]: v for t, v in per_vector_recall.items()},
                'or_recall': or_recall,
                'or_fpr': or_fpr,
            }
            recall_data[ds] = {
                'per_vector': per_vector_recall,
                'or_recall': or_recall,
                'or_fpr': or_fpr,
                'n_deceptive': n_pos,
            }

        results['datasets'][ds] = ds_result

    # Summary AUROC
    d = results['datasets']
    if d:
        avg_best = np.mean([v['best_individual'] for v in d.values()])
        avg_sub = np.mean([v['subspace_auroc'] for v in d.values()])
        print(f"\n{'AVERAGE':<12} {'':>5}", end='')
        print(' ' * (len(trait_names) * 11), end='')
        print(f" {avg_best:>7.3f} {avg_sub:>8.3f} ({avg_sub - avg_best:>+.3f})")

    # OR-threshold table
    if recall_data:
        print(f"\n--- OR-Threshold Recall @ {args.fpr_target:.0%} per-vector FPR ---")
        print(f"{'Dataset':<12} {'N_dec':>5}", end='')
        for s in short:
            print(f" {s:>10}", end='')
        print(f" {'OR-recall':>9} {'OR-FPR':>7}")
        print('-' * (20 + len(trait_names) * 11 + 20))

        for ds, rd in recall_data.items():
            print(f"{ds.upper():<12} {rd['n_deceptive']:>5}", end='')
            for trait in trait_names:
                r = rd['per_vector'].get(trait, rd['per_vector'].get(trait.split('/')[-1], 0))
                print(f" {r:>10.1%}", end='')
            print(f" {rd['or_recall']:>9.1%} {rd['or_fpr']:>7.1%}")

        avg_or_recall = np.mean([v['or_recall'] for v in recall_data.values()])
        avg_or_fpr = np.mean([v['or_fpr'] for v in recall_data.values()])
        print(f"\n{'AVERAGE':<12} {'':>5}", end='')
        print(' ' * (len(trait_names) * 11), end='')
        print(f" {avg_or_recall:>9.1%} {avg_or_fpr:>7.1%}")

    # Subspace analysis details
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
