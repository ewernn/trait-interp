# Steering Experiment - Remote Run Instructions

## Context

We're validating trait vectors via **causal intervention** (steering). This is ground truth validation - if we can add a vector to layer outputs and change model behavior in the expected direction, the vector captures something real.

**Previous results had high variance** - same config gave different scores across runs. We're starting fresh to get clean data.

## What We're Testing

**Trait**: `epistemic/optimism` - vectors extracted from gemma-2-2b-base, steering gemma-2-2b-it

**Three approaches**:
1. **Single-layer**: Steer one layer at a time, find best coefficient per layer
2. **Multi-layer weighted**: Steer layers 6-18 simultaneously, coefficients proportional to each layer's single-layer effectiveness
3. **Multi-layer orthogonal**: Same but vectors orthogonalized to remove shared directions between layers

## Commands

```bash
cd ~/trait-interp
git pull

# 1. Back up old results
mv experiments/gemma-2-2b-it/steering/epistemic/optimism/results.json \
   experiments/gemma-2-2b-it/steering/epistemic/optimism/results_old.json

# 2. Single-layer sweep (~2 hours)
# Runs adaptive search: 8 steps per layer, finds best coefficient
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-base/epistemic/optimism \
    --layers 6-18

# 3. Multi-layer weighted (after single-layer completes)
# Uses single-layer deltas to weight coefficients: coef_l = scale * best_coef_l * (delta_l / sum_deltas)
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-base/epistemic/optimism \
    --layers 6-18 \
    --multi-layer weighted --global-scale 1.0

# 4. Multi-layer orthogonal
# Each vector orthogonalized to previous layer's vector
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-base/epistemic/optimism \
    --layers 6-18 \
    --multi-layer orthogonal --global-scale 2.0
```

## Checking Progress

Results save incrementally to `experiments/gemma-2-2b-it/steering/epistemic/optimism/results.json`

```bash
# See latest results
cat experiments/gemma-2-2b-it/steering/epistemic/optimism/results.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Baseline: {d[\"baseline\"][\"trait_mean\"]:.1f}')
print(f'Runs: {len(d[\"runs\"])}')
for r in d['runs'][-5:]:
    layers = r['config']['layers']
    coef = r['config']['coefficients'][0]
    trait = r['result']['trait_mean']
    coh = r['result']['coherence_mean']
    print(f'  L{layers}: c{coef:.0f} -> trait={trait:.1f}, coh={coh:.1f}')
"
```

## After Completion

Sync results to R2:
```bash
rclone sync ~/trait-interp/experiments r2:trait-interp-data/experiments \
    --include "gemma-2-2b-it/steering/**" \
    --progress
```
