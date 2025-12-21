"""
Compare jailbreak detection across extraction components.
Input: Raw activations + component-specific vectors
Output: Cohen's d per component per layer
"""
import json
import torch
import numpy as np
from pathlib import Path

# Config
exp_dir = Path('experiments/gemma-2-2b')
raw_dir = exp_dir / 'inference/raw/residual/jailbreak'
vector_dir = exp_dir / 'extraction/chirp/refusal/vectors'

# Load ground truth
with open('datasets/inference/jailbreak_successes.json') as f:
    data = json.load(f)
    success_ids = {str(p['id']) for p in data['prompts']}

print(f"Ground truth: {len(success_ids)} successful jailbreaks")

# Components to compare (v_cache not captured in raw activations)
components = ['residual', 'attn_out']

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled == 0:
        return float('nan')
    return (np.mean(g1) - np.mean(g2)) / pooled

def project(activations, vector):
    """Project activations onto vector."""
    # Ensure same dtype
    activations = activations.float()
    vector = vector.float()
    return (activations @ vector) / vector.norm()

results = {}

for component in components:
    print(f"\nProcessing component: {component}")
    results[component] = {}

    for layer in range(26):
        # Load vector for this component
        if component == 'residual':
            vec_path = vector_dir / f'probe_layer{layer}.pt'
        else:
            vec_path = vector_dir / f'{component}_probe_layer{layer}.pt'

        if not vec_path.exists():
            continue

        vector = torch.load(vec_path, weights_only=True)

        success_projs = []
        failure_projs = []

        for pt_file in sorted(raw_dir.glob('*.pt')):
            prompt_id = pt_file.stem
            data = torch.load(pt_file, weights_only=True)

            # Get response activations for this component at this layer
            response_acts = data['response']['activations']

            if layer not in response_acts:
                continue

            layer_data = response_acts[layer]

            if component == 'residual':
                acts = layer_data.get('residual_out')
            elif component == 'attn_out':
                acts = layer_data.get('attn_out')

            if acts is None or (hasattr(acts, 'numel') and acts.numel() == 0):
                continue

            # Project first response token
            if acts.shape[0] >= 1:
                proj = project(acts[0], vector).item()
            else:
                continue

            if prompt_id in success_ids:
                success_projs.append(proj)
            else:
                failure_projs.append(proj)

        if len(success_projs) >= 10 and len(failure_projs) >= 10:
            d = cohens_d(failure_projs, success_projs)
            results[component][layer] = {
                'd': d,
                'n_success': len(success_projs),
                'n_failure': len(failure_projs),
                'success_mean': float(np.mean(success_projs)),
                'failure_mean': float(np.mean(failure_projs))
            }

# Print summary
print("\n" + "="*60)
print("Component Comparison: Cohen's d by Layer")
print("="*60)
print(f"\n{'Layer':<6}" + "".join(f"{c:<15}" for c in components))
print("-" * (6 + 15 * len(components)))

for layer in range(26):
    row = f"{layer:<6}"
    for comp in components:
        if layer in results.get(comp, {}):
            d = results[comp][layer]['d']
            row += f"{d:<15.3f}"
        else:
            row += f"{'--':<15}"
    print(row)

# Find best
print("\n" + "="*60)
print("Best Layer per Component")
print("="*60)
for comp in components:
    if results.get(comp):
        best_layer = max(results[comp].keys(), key=lambda l: abs(results[comp][l]['d']))
        d = results[comp][best_layer]['d']
        n_s = results[comp][best_layer]['n_success']
        n_f = results[comp][best_layer]['n_failure']
        print(f"{comp}: Layer {best_layer}, d={d:.3f} (n_success={n_s}, n_failure={n_f})")

# Compare at best residual layer
print("\n" + "="*60)
print("Comparison at Key Layers")
print("="*60)
for layer in [10, 12, 14, 15, 16, 18, 20]:
    print(f"\nLayer {layer}:")
    for comp in components:
        if layer in results.get(comp, {}):
            r = results[comp][layer]
            print(f"  {comp}: d={r['d']:.3f}, success_mean={r['success_mean']:.3f}, failure_mean={r['failure_mean']:.3f}")

# Save results
output_path = exp_dir / 'analysis/component_comparison_results.json'
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {output_path}")
