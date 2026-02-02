"""
Analyze activation dynamics: smoothness, variance, magnitude.

Metrics:
- smoothness: mean L2 norm of token-to-token activation deltas (lower = smoother)
- magnitude: mean activation norm across tokens
- variance: variance of activation norms across tokens

Usage:
    # Default: compare human vs model
    python scripts/analyze_activation_dynamics.py --experiment prefill-dynamics

    # Compare specific conditions
    python scripts/analyze_activation_dynamics.py --experiment prefill-dynamics \
        --condition-a human-instruct --condition-b model-instruct --output instruct

    # Compare temp=0 vs temp=0.7 model generations
    python scripts/analyze_activation_dynamics.py --experiment prefill-dynamics \
        --condition-a model --condition-b model-temp07 --output temp-comparison
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from scipy import stats

def compute_trajectory_metrics(activations: dict, layers: list) -> dict:
    """Compute smoothness, magnitude, variance for response activations."""

    metrics_by_layer = {}

    for layer in layers:
        if layer not in activations:
            continue

        # Get residual activations [n_tokens, hidden_dim]
        residual = activations[layer].get('residual')
        if residual is None:
            continue

        residual = residual.float()  # Ensure float for computation
        n_tokens = residual.shape[0]

        if n_tokens < 2:
            continue

        # Token-to-token deltas
        deltas = residual[1:] - residual[:-1]  # [n_tokens-1, hidden_dim]
        delta_norms = torch.norm(deltas, dim=1)  # [n_tokens-1]

        # Activation norms per token
        token_norms = torch.norm(residual, dim=1)  # [n_tokens]

        metrics_by_layer[layer] = {
            'smoothness': delta_norms.mean().item(),  # Lower = smoother
            'magnitude': token_norms.mean().item(),
            'variance': token_norms.var().item(),
            'n_tokens': n_tokens,
        }

    return metrics_by_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    parser.add_argument("--condition-a", default="human",
                        help="First condition directory name")
    parser.add_argument("--condition-b", default="model",
                        help="Second condition directory name")
    parser.add_argument("--output", default=None,
                        help="Output suffix for results file (default: auto from conditions)")
    args = parser.parse_args()

    # Auto-generate output name
    if args.output is None:
        if args.condition_a == "human" and args.condition_b == "model":
            args.output = ""  # Default naming
        else:
            args.output = f"{args.condition_a}-vs-{args.condition_b}"

    # Paths
    act_dir = Path(f"experiments/{args.experiment}/activations")
    dir_a = act_dir / args.condition_a
    dir_b = act_dir / args.condition_b

    if not dir_a.exists():
        raise FileNotFoundError(f"Condition A directory not found: {dir_a}")
    if not dir_b.exists():
        raise FileNotFoundError(f"Condition B directory not found: {dir_b}")

    # Get sample IDs (intersection of both conditions)
    ids_a = {int(p.stem) for p in dir_a.glob("*.pt")}
    ids_b = {int(p.stem) for p in dir_b.glob("*.pt")}
    sample_ids = sorted(ids_a & ids_b)

    print(f"Comparing {args.condition_a} vs {args.condition_b}")
    print(f"Analyzing {len(sample_ids)} samples...")

    # Collect metrics
    all_metrics = []

    for sample_id in tqdm(sample_ids):
        data_a = torch.load(dir_a / f"{sample_id}.pt")
        data_b = torch.load(dir_b / f"{sample_id}.pt")

        # Get layer list from data
        layers = list(data_a['response']['activations'].keys())

        metrics_a = compute_trajectory_metrics(
            data_a['response']['activations'], layers
        )
        metrics_b = compute_trajectory_metrics(
            data_b['response']['activations'], layers
        )

        all_metrics.append({
            'id': sample_id,
            'a': metrics_a,
            'b': metrics_b,
        })

    # Aggregate by layer
    layers = list(all_metrics[0]['a'].keys())

    summary = {
        'condition_a': args.condition_a,
        'condition_b': args.condition_b,
        'by_layer': {},
        'overall': {}
    }

    for layer in layers:
        a_smoothness = [m['a'][layer]['smoothness'] for m in all_metrics if layer in m['a']]
        b_smoothness = [m['b'][layer]['smoothness'] for m in all_metrics if layer in m['b']]

        a_magnitude = [m['a'][layer]['magnitude'] for m in all_metrics if layer in m['a']]
        b_magnitude = [m['b'][layer]['magnitude'] for m in all_metrics if layer in m['b']]

        # Paired t-test for smoothness
        t_stat, p_value = stats.ttest_rel(a_smoothness, b_smoothness)

        # Effect size (Cohen's d for paired samples)
        diff = np.array(a_smoothness) - np.array(b_smoothness)
        cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

        summary['by_layer'][layer] = {
            'a_smoothness_mean': np.mean(a_smoothness),
            'b_smoothness_mean': np.mean(b_smoothness),
            'smoothness_diff': np.mean(a_smoothness) - np.mean(b_smoothness),
            'smoothness_t_stat': t_stat,
            'smoothness_p_value': p_value,
            'smoothness_cohens_d': cohens_d,
            'a_magnitude_mean': np.mean(a_magnitude),
            'b_magnitude_mean': np.mean(b_magnitude),
        }

    # Overall (average across layers)
    all_a_smooth = []
    all_b_smooth = []
    for m in all_metrics:
        all_a_smooth.append(np.mean([m['a'][l]['smoothness'] for l in layers if l in m['a']]))
        all_b_smooth.append(np.mean([m['b'][l]['smoothness'] for l in layers if l in m['b']]))

    t_stat, p_value = stats.ttest_rel(all_a_smooth, all_b_smooth)
    diff = np.array(all_a_smooth) - np.array(all_b_smooth)
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

    summary['overall'] = {
        'a_smoothness_mean': np.mean(all_a_smooth),
        'b_smoothness_mean': np.mean(all_b_smooth),
        'smoothness_diff': np.mean(all_a_smooth) - np.mean(all_b_smooth),
        'smoothness_t_stat': t_stat,
        'smoothness_p_value': p_value,
        'smoothness_cohens_d': cohens_d,
        'n_samples': len(sample_ids),
    }

    # Save
    output_dir = Path(f"experiments/{args.experiment}/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename
    if args.output:
        output_file = f"activation_metrics-{args.output}.json"
    else:
        output_file = "activation_metrics.json"

    with open(output_dir / output_file, "w") as f:
        json.dump({'samples': all_metrics, 'summary': summary}, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.condition_a} vs {args.condition_b}")
    print(f"{'='*60}")
    print(f"Samples: {len(sample_ids)}")
    print(f"\nOverall Smoothness (lower = smoother):")
    print(f"  {args.condition_a}: {summary['overall']['a_smoothness_mean']:.4f}")
    print(f"  {args.condition_b}: {summary['overall']['b_smoothness_mean']:.4f}")
    print(f"  Diff (A-B): {summary['overall']['smoothness_diff']:.4f}")
    print(f"  t-stat: {summary['overall']['smoothness_t_stat']:.2f}")
    print(f"  p-value: {summary['overall']['smoothness_p_value']:.2e}")
    print(f"  Cohen's d: {summary['overall']['smoothness_cohens_d']:.3f}")

    print(f"\nPer-layer (selected layers):")
    for layer in [0, 6, 12, 18, 24]:
        if layer in summary['by_layer']:
            s = summary['by_layer'][layer]
            print(f"  L{layer}: diff={s['smoothness_diff']:.4f}, d={s['smoothness_cohens_d']:.3f}, p={s['smoothness_p_value']:.2e}")

    print(f"\nSaved to {output_dir / output_file}")

if __name__ == "__main__":
    main()
