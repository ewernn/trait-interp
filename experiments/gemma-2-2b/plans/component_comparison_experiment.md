# Experiment: Component Comparison for Jailbreak Detection

**Goal:** Determine which activation component (residual, attn_out, v_cache) provides best separation between jailbreak success and failure.

**Time estimate:** 2-3 hours on A100

---

## Research Question

Does extracting/projecting from different model components (attn_out, v_cache) improve jailbreak detection over residual stream?

**Hypothesis:** V-cache may encode behavioral traits more cleanly because it represents "what information to retrieve" rather than "what to compute next."

---

## Prerequisites

**Already exists:**
- `datasets/inference/jailbreak.json` — 305 jailbreak prompts
- `datasets/inference/jailbreak_successes.json` — ground truth labels (151 successes)
- `experiments/gemma-2-2b/extraction/chirp/refusal/vectors/` — residual stream vectors (probe, mean_diff, gradient × 26 layers)
- Extraction pipeline at `extraction/run_pipeline.py`
- Capture pipeline at `inference/capture_raw_activations.py`

**Needs to be created:**
- attn_out vectors for chirp/refusal
- v_cache vectors for chirp/refusal (if pipeline supports)
- Raw activations from jailbreak prompts with multiple components

---

## Step 1: Check Component Support

```bash
# Check what components extraction supports
grep -n "component" extraction/extract_vectors.py | head -20

# Check what components are available
cat config/models/gemma-2-2b-it.yaml
```

Expected components: `residual`, `attn_out`, `mlp_out`, `k_cache`, `v_cache`

---

## Step 2: Extract Vectors for Each Component

Run extraction for attn_out and v_cache (residual already exists):

```bash
# attn_out vectors
python extraction/run_pipeline.py \
    --experiment gemma-2-2b \
    --traits chirp/refusal \
    --component attn_out \
    --no-steering

# v_cache vectors (if supported — check Step 1)
python extraction/run_pipeline.py \
    --experiment gemma-2-2b \
    --traits chirp/refusal \
    --component v_cache \
    --no-steering
```

**Output locations:**
- `experiments/gemma-2-2b/extraction/chirp/refusal/vectors/attn_out_probe_layer{L}.pt`
- `experiments/gemma-2-2b/extraction/chirp/refusal/vectors/v_cache_probe_layer{L}.pt`

**If v_cache not supported:** Check `extraction/extract_activations.py` for available hook points. May need to add v_cache hook.

---

## Step 3: Capture Raw Activations from Jailbreak Prompts

```bash
# Capture with all components
python inference/capture_raw_activations.py \
    --experiment gemma-2-2b \
    --prompt-set jailbreak \
    --capture-attn \
    --limit 50  # Start with subset to verify
```

**Check output:**
```bash
ls experiments/gemma-2-2b/inference/raw/residual/jailbreak/
```

**If --capture-attn doesn't save attn_out separately**, check the .pt file structure:
```python
import torch
data = torch.load('experiments/gemma-2-2b/inference/raw/residual/jailbreak/1.pt')
print(data.keys())  # Should show layer indices
print(data[0].keys())  # Should show 'output', 'attn_out', etc.
```

---

## Step 4: Project Activations onto Component-Matched Vectors

Create analysis script:

```python
# analysis/component_comparison.py
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
    success_ids = {p['id'] for p in json.load(f)['prompts']}

# Components to compare
components = ['residual', 'attn_out']  # Add 'v_cache' if extracted

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / pooled

def project(activations, vector):
    """Project activations onto vector."""
    return (activations @ vector) / vector.norm()

results = {}

for component in components:
    results[component] = {}

    for layer in range(26):
        # Load vector for this component
        if component == 'residual':
            vec_path = vector_dir / f'probe_layer{layer}.pt'
        else:
            vec_path = vector_dir / f'{component}_probe_layer{layer}.pt'

        if not vec_path.exists():
            continue

        vector = torch.load(vec_path)

        success_projs = []
        failure_projs = []

        for pt_file in raw_dir.glob('*.pt'):
            prompt_id = pt_file.stem
            data = torch.load(pt_file)

            # Get activations for this component at this layer
            if component == 'residual':
                acts = data[layer]['output']  # or 'residual_out'
            elif component == 'attn_out':
                acts = data[layer].get('attn_out')
            elif component == 'v_cache':
                acts = data[layer].get('v_cache')

            if acts is None or acts.numel() == 0:
                continue

            # Project response token 1 (or mean of first 5)
            if acts.shape[0] > 1:
                proj = project(acts[1], vector).item()  # token 1
            else:
                proj = project(acts[0], vector).item()

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
                'success_mean': np.mean(success_projs),
                'failure_mean': np.mean(failure_projs)
            }

# Print summary
print("\n=== Component Comparison: Cohen's d by Layer ===\n")
print(f"{'Layer':<6}" + "".join(f"{c:<12}" for c in components))
print("-" * (6 + 12 * len(components)))

for layer in range(26):
    row = f"{layer:<6}"
    for comp in components:
        if layer in results.get(comp, {}):
            row += f"{results[comp][layer]['d']:<12.3f}"
        else:
            row += f"{'--':<12}"
    print(row)

# Find best
print("\n=== Best Layer per Component ===\n")
for comp in components:
    if results.get(comp):
        best_layer = max(results[comp].keys(), key=lambda l: results[comp][l]['d'])
        print(f"{comp}: Layer {best_layer}, d={results[comp][best_layer]['d']:.3f}")

# Save results
with open(exp_dir / 'analysis/component_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {exp_dir / 'analysis/component_comparison_results.json'}")
```

---

## Step 5: Run Analysis

```bash
mkdir -p experiments/gemma-2-2b/analysis
python analysis/component_comparison.py
```

---

## Expected Output

```
=== Component Comparison: Cohen's d by Layer ===

Layer residual    attn_out    v_cache
------------------------------------------
0     0.234       0.189       0.201
...
15    2.240       2.410       2.580    <-- if v_cache wins
16    2.130       2.350       2.490
...

=== Best Layer per Component ===

residual: Layer 15, d=2.240
attn_out: Layer 14, d=2.410
v_cache: Layer 16, d=2.580
```

---

## What to Report Back

1. **Best d per component** — which component wins?
2. **Best layer per component** — same or different?
3. **Per-layer pattern** — does one component dominate across all layers?
4. **Sample sizes** — how many prompts had each component captured?

If v_cache > residual by meaningful margin (>0.3 d), that's a novel finding worth investigating further.

---

## Troubleshooting

**"v_cache not found in activations"**
→ Check what keys exist in the .pt files. May need to modify capture script to add v_cache hooks.

**"attn_out vectors don't exist"**
→ Step 2 failed. Check extraction logs. May need to verify component support in `extraction/extract_activations.py`.

**"Only residual works"**
→ Still useful! Report per-layer separation pattern for residual. Which layer is best?

---

## Fallback: Residual-Only Analysis

If component extraction is blocked, still valuable to analyze:

```python
# Which layer gives best separation for residual?
# Does probe vs mean_diff vs gradient matter?
# What's the separation at different token positions?
```

Use existing projection JSONs in `experiments/gemma-2-2b/inference/chirp/refusal/residual_stream/jailbreak/`.
