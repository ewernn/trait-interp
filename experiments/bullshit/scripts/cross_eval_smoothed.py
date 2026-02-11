#!/usr/bin/env python3
"""Cross-evaluate with multi-layer score averaging (zero-label).

Instead of picking a single best layer, average per-example projection scores
across a window of adjacent layers before computing AUROC. This denoises
single-layer spikes and produces more reliable detection scores.

Compares single-layer best vs smoothed (window=3,5,7) across a safe layer
range (excludes early/late layers where spikes are unreliable).

Input:
    experiments/bullshit/extraction/{trait}/base/vectors/response__5/residual/probe/layer{L}.pt
    experiments/bullshit/results/{dataset}_activations.pt
    experiments/bullshit/prompt_sets/{dataset}_metadata.json

Output:
    experiments/bullshit/results/cross_eval_smoothed.json

Usage:
    python experiments/bullshit/scripts/cross_eval_smoothed.py
    python experiments/bullshit/scripts/cross_eval_smoothed.py --layer-lo 10 --layer-hi 70
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
DATASETS = ['cg', 'id', 'st_time', 'st_greeting', 'it', 'hpkr', 'hpc', 'gs', 'st_gender']
WINDOWS = [1, 3, 5, 7]


def load_vector(trait: str, method: str, layer: int) -> torch.Tensor:
    vec_path = EXTRACTION_DIR / trait / 'base/vectors/response__5/residual' / method / f'layer{layer}.pt'
    vec = torch.load(vec_path, map_location='cpu')
    if isinstance(vec, dict):
        vec = vec['vector']
    return vec.float()


def project(acts: torch.Tensor, vec: torch.Tensor) -> np.ndarray:
    return ((acts @ vec) / vec.norm()).numpy()


def main():
    parser = argparse.ArgumentParser(description='Multi-layer smoothed cross-eval')
    parser.add_argument('--method', default='probe', help='Vector extraction method')
    parser.add_argument('--layer-lo', type=int, default=10, help='Lowest layer (inclusive)')
    parser.add_argument('--layer-hi', type=int, default=70, help='Highest layer (inclusive)')
    args = parser.parse_args()

    lo, hi = args.layer_lo, args.layer_hi
    print(f"Method: {args.method}, Layer range: L{lo}-L{hi}, Windows: {WINDOWS}")

    # Pre-load all vectors
    print("\nLoading vectors...")
    vectors = {}  # {trait: {layer: tensor}}
    for trait in TRAITS:
        vectors[trait] = {}
        loaded = 0
        for layer in range(lo, hi + 1):
            try:
                vectors[trait][layer] = load_vector(trait, args.method, layer)
                loaded += 1
            except FileNotFoundError:
                pass
        print(f"  {trait}: {loaded} layers loaded")

    # Discover datasets
    datasets = [ds for ds in DATASETS
                if (RESULTS_DIR / f'{ds}_activations.pt').exists()
                and (PROMPTS_DIR / f'{ds}_metadata.json').exists()]
    print(f"\n{len(datasets)} datasets: {', '.join(d.upper() for d in datasets)}")

    results = {
        'method': args.method, 'layer_lo': lo, 'layer_hi': hi,
        'windows': WINDOWS, 'datasets': {},
    }

    # Header
    short = [t.split('/')[-1][:10] for t in TRAITS]
    for w in WINDOWS:
        label = f'w={w}' if w > 1 else 'single'
        print(f"\n{'=' * 70}")
        print(f"Window: {label} (L{lo}-L{hi}, {args.method})")
        print(f"{'=' * 70}")
        print(f"{'Dataset':<14}", end='')
        for s in short:
            print(f" {s:>12}", end='')
        print(f" {'best':>8}")
        print('-' * (14 + len(TRAITS) * 13 + 9))

        for ds in datasets:
            acts = torch.load(RESULTS_DIR / f'{ds}_activations.pt', map_location='cpu')
            with open(PROMPTS_DIR / f'{ds}_metadata.json') as f:
                meta = json.load(f)
            labels = np.array([meta[str(i + 1)].get('deceptive', False) for i in range(len(acts))])
            n_pos = int(labels.sum())
            if n_pos == 0 or n_pos == len(labels):
                continue

            if ds not in results['datasets']:
                results['datasets'][ds] = {
                    'n': len(labels), 'n_deceptive': n_pos,
                }

            half = w // 2
            trait_aurocs = {}

            for trait in TRAITS:
                trait_key = trait.split('/')[-1]
                best_auroc, best_center = 0, 0

                for center in range(lo + half, hi - half + 1):
                    window_layers = range(center - half, center + half + 1)

                    # Average per-example scores across window
                    score_sum = np.zeros(len(labels))
                    n_valid = 0
                    for layer in window_layers:
                        if layer not in vectors[trait]:
                            continue
                        layer_acts = acts[:, layer, :].float()
                        score_sum += project(layer_acts, vectors[trait][layer])
                        n_valid += 1

                    if n_valid == 0:
                        continue
                    avg_scores = score_sum / n_valid
                    auroc = roc_auc_score(labels, avg_scores)

                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_center = center

                trait_aurocs[trait_key] = (best_auroc, best_center)

                # Save to results
                wkey = f'w{w}'
                if wkey not in results['datasets'][ds]:
                    results['datasets'][ds][wkey] = {}
                results['datasets'][ds][wkey][trait_key] = {
                    'auroc': float(best_auroc),
                    'center_layer': best_center,
                }

            # Print row
            vals = {k: v[0] for k, v in trait_aurocs.items()}
            best_val = max(vals.values()) if vals else 0
            best_trait = max(vals, key=vals.get) if vals else ''

            print(f"{ds.upper():<14}", end='')
            for trait in TRAITS:
                tk = trait.split('/')[-1]
                if tk in trait_aurocs:
                    a, cl = trait_aurocs[tk]
                    bold = '**' if a == best_val and a > 0.55 else '  '
                    print(f" {bold}{a:.3f} L{cl:<2}{bold}", end='')
                else:
                    print(f" {'—':>12}", end='')
            print(f"  {best_trait[:6]}")

    # Summary comparison: single vs best window
    print(f"\n{'=' * 70}")
    print("SUMMARY: Single-layer vs best smoothed window")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<14} {'single':>10} {'w=3':>10} {'w=5':>10} {'w=7':>10} {'best_w':>10} {'delta':>8}")
    print('-' * 66)

    for ds in datasets:
        if ds not in results['datasets']:
            continue
        d = results['datasets'][ds]
        row = {}
        for w in WINDOWS:
            wkey = f'w{w}'
            if wkey in d:
                best = max(v['auroc'] for v in d[wkey].values())
                row[w] = best
        if not row:
            continue

        single = row.get(1, 0)
        best_w = max(row.values())
        best_w_size = max(row, key=row.get)
        delta = best_w - single

        print(f"{ds.upper():<14}", end='')
        for w in WINDOWS:
            print(f" {row.get(w, 0):>10.3f}", end='')
        print(f" {'w=' + str(best_w_size):>10} {delta:>+8.3f}")

    # Save
    output_path = RESULTS_DIR / 'cross_eval_smoothed.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
