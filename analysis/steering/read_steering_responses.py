#!/usr/bin/env python3
"""
Read and display steering responses for manual evaluation.

Usage:
    # Compact table of best result per layer
    python analysis/steering/read_steering_responses.py <results_dir> --table

    # Table + responses for top 3 layers
    python analysis/steering/read_steering_responses.py <results_dir> --table --best --top 3

    # Best run's responses only
    python analysis/steering/read_steering_responses.py <results_dir> --best

    # Specific layer/coef
    python analysis/steering/read_steering_responses.py <results_dir> -l 17 -c 3.1

    # Direct response file
    python analysis/steering/read_steering_responses.py path/to/responses.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.vectors import MIN_COHERENCE


def load_responses(file_path: Path) -> list:
    """Load responses from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def parse_results_jsonl(results_file: Path) -> tuple:
    """Parse results.jsonl into baseline and runs."""
    runs = []
    baseline = None

    with open(results_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get('type') == 'baseline':
                baseline = data.get('result', {})
            elif data.get('type') == 'header':
                continue
            elif 'result' in data and 'config' in data:
                run = {
                    'trait_mean': data['result'].get('trait_mean', 0),
                    'coherence_mean': data['result'].get('coherence_mean', 0),
                    'n': data['result'].get('n', 0),
                    'timestamp': data.get('timestamp', '')
                }
                if data['config'].get('vectors'):
                    v = data['config']['vectors'][0]
                    run['layer'] = v.get('layer')
                    run['coef'] = v.get('weight')
                    run['method'] = v.get('method', 'probe')
                    run['component'] = v.get('component', 'residual')
                runs.append(run)

    return baseline, runs


def find_response_file(responses_dir: Path, layer: int, coef: float) -> Path:
    """Find response file matching layer/coef."""
    # Try different coef formats
    patterns = [
        f"L{layer}_c{coef:.1f}*.json",
        f"L{layer}_c{coef:.2f}*.json",
        f"L{layer}_c{int(coef)}*.json" if coef == int(coef) else None,
    ]

    for component_dir in responses_dir.iterdir():
        if not component_dir.is_dir():
            continue
        for method_dir in component_dir.iterdir():
            if not method_dir.is_dir():
                continue
            for pattern in patterns:
                if pattern:
                    matches = list(method_dir.glob(pattern))
                    if matches:
                        return matches[0]
    return None


def display_responses(responses: list, baseline_trait: float = None, sort_by: str = 'trait'):
    """Display responses in readable format."""
    print("\n" + "="*80)

    trait_scores = [r['trait_score'] for r in responses]
    coh_scores = [r['coherence_score'] for r in responses]

    avg_trait = sum(trait_scores) / len(trait_scores) if trait_scores else 0
    avg_coh = sum(coh_scores) / len(coh_scores) if coh_scores else 0

    if baseline_trait is not None:
        delta = avg_trait - baseline_trait
        print(f"SUMMARY: trait={avg_trait:.1f} (baseline={baseline_trait:.1f}, delta={delta:+.1f}), coherence={avg_coh:.1f}, n={len(responses)}")
    else:
        print(f"SUMMARY: trait={avg_trait:.1f}, coherence={avg_coh:.1f}, n={len(responses)}")

    # Coherence distribution
    coh_50 = sum(1 for c in coh_scores if c == 50.0)
    coh_low = sum(1 for c in coh_scores if c < MIN_COHERENCE and c != 50.0)
    coh_ok = sum(1 for c in coh_scores if c >= MIN_COHERENCE)
    print(f"COHERENCE: {coh_ok} good (≥{MIN_COHERENCE}), {coh_low} low (<{MIN_COHERENCE}), {coh_50} off-topic")
    print("="*80)

    # Sort
    if sort_by == 'trait':
        sorted_responses = sorted(responses, key=lambda x: x['trait_score'], reverse=True)
    elif sort_by == 'coherence':
        sorted_responses = sorted(responses, key=lambda x: x['coherence_score'], reverse=False)
    else:
        sorted_responses = responses

    for i, r in enumerate(sorted_responses):
        trait = r['trait_score']
        coh = r['coherence_score']

        # Highlight issues
        flags = []
        if coh == 50.0:
            flags.append("OFF_TOPIC")
        elif coh < MIN_COHERENCE:
            flags.append(f"LOW_COH")
        if len(r['response']) < 80:
            flags.append(f"SHORT")

        flag_str = f" [{', '.join(flags)}]" if flags else ""

        print(f"\n[{i+1}] TRAIT={trait:.1f} COH={coh:.1f}{flag_str}")
        print(f"Q: {r['prompt']}")
        print(f"A: {r['response']}")
        print("-"*60)


def main():
    parser = argparse.ArgumentParser(description="Read and evaluate steering responses")
    parser.add_argument('path', help='Response file or results directory')
    parser.add_argument('-l', '--layer', type=int, help='Layer number')
    parser.add_argument('-c', '--coef', type=float, help='Coefficient')
    parser.add_argument('--best', action='store_true', help=f'Show best run by delta (coh>={MIN_COHERENCE})')
    parser.add_argument('--top', type=int, default=1, help='Show top N runs')
    parser.add_argument('--sort', choices=['trait', 'coherence', 'none'], default='trait')
    parser.add_argument('--baseline', action='store_true', help='Show baseline responses')
    parser.add_argument('--table', action='store_true', help='Compact table of best result per layer')

    args = parser.parse_args()
    path = Path(args.path)

    # Direct file read
    if path.suffix == '.json':
        responses = load_responses(path)
        display_responses(responses, sort_by=args.sort)
        return

    # Results directory
    results_file = path / 'results.jsonl'
    responses_dir = path / 'responses'

    if not results_file.exists():
        print(f"No results.jsonl found in {path}")
        return

    baseline, runs = parse_results_jsonl(results_file)
    baseline_trait = baseline.get('trait_mean', 0) if baseline else 0

    print(f"Baseline: trait={baseline_trait:.2f}, coherence={baseline.get('coherence_mean', 0):.1f}")
    print(f"Total runs: {len(runs)}")

    # Show baseline responses
    if args.baseline:
        baseline_file = responses_dir / 'baseline.json'
        if baseline_file.exists():
            responses = load_responses(baseline_file)
            print("\n=== BASELINE RESPONSES ===")
            display_responses(responses, sort_by=args.sort)
        return

    # Compact table: best result per layer
    if args.table:
        from collections import defaultdict
        by_layer = defaultdict(list)
        for run in runs:
            if run.get('layer') is not None:
                by_layer[run['layer']].append(run)

        # Best per layer: highest delta with coherence >= MIN_COHERENCE, fallback to highest delta
        best_per_layer = {}
        for layer, layer_runs in sorted(by_layer.items()):
            coherent = [r for r in layer_runs if r['coherence_mean'] >= MIN_COHERENCE]
            pool = coherent if coherent else layer_runs
            best = max(pool, key=lambda r: abs(r['trait_mean'] - baseline_trait))
            best_per_layer[layer] = best

        # Find overall best
        overall_best_layer = max(best_per_layer, key=lambda l: abs(best_per_layer[l]['trait_mean'] - baseline_trait)) if best_per_layer else None

        print(f"\n{'Layer':>5} {'Coef':>7} {'Trait':>6} {'Delta':>7} {'Coh':>5}")
        print("-" * 38)
        for layer in sorted(best_per_layer):
            run = best_per_layer[layer]
            delta = run['trait_mean'] - baseline_trait
            marker = " *" if layer == overall_best_layer else ""
            coh_flag = " !" if run['coherence_mean'] < MIN_COHERENCE else ""
            print(f"  L{layer:<3} {run['coef']:>7.1f} {run['trait_mean']:>6.1f} {delta:>+7.1f} {run['coherence_mean']:>5.1f}{coh_flag}{marker}")
        print(f"\n* = best delta (coh≥{MIN_COHERENCE})  ! = coherence < {MIN_COHERENCE}")

        if not args.best:
            return

    # Specific layer/coef
    if args.layer and args.coef:
        response_file = find_response_file(responses_dir, args.layer, args.coef)
        if response_file:
            print(f"\nReading: {response_file.name}")
            responses = load_responses(response_file)
            display_responses(responses, baseline_trait, sort_by=args.sort)
        else:
            print(f"No response file found for L{args.layer} c{args.coef}")
        return

    # Best/top runs — deduplicate by layer (best run per layer)
    from collections import defaultdict
    by_layer = defaultdict(list)
    for run in runs:
        if run.get('layer') is not None:
            by_layer[run['layer']].append(run)

    best_per_layer = []
    for layer, layer_runs in by_layer.items():
        coherent = [r for r in layer_runs if r['coherence_mean'] >= MIN_COHERENCE]
        pool = coherent if coherent else layer_runs
        best = max(pool, key=lambda r: abs(r['trait_mean'] - baseline_trait))
        best_per_layer.append(best)

    if not best_per_layer:
        print("No runs found")
        return

    best_per_layer.sort(key=lambda x: abs(x['trait_mean'] - baseline_trait), reverse=True)

    print(f"\nTop {min(args.top, len(best_per_layer))} layers by |delta| (coh>={MIN_COHERENCE}):")
    for i, run in enumerate(best_per_layer[:args.top]):
        delta = run['trait_mean'] - baseline_trait
        print(f"  {i+1}. L{run['layer']} c{run['coef']:.2f}: trait={run['trait_mean']:.1f} (Δ={delta:+.1f}), coh={run['coherence_mean']:.1f}")

    if args.best:
        for run in best_per_layer[:args.top]:
            response_file = find_response_file(responses_dir, run['layer'], run['coef'])
            if response_file:
                print(f"\n{'='*60}")
                print(f"RUN: L{run['layer']} c{run['coef']:.2f}")
                responses = load_responses(response_file)
                display_responses(responses, baseline_trait, sort_by=args.sort)


if __name__ == '__main__':
    main()
