---
title: "Massive Activation Dimensions"
preview: "Massive dims hurt mean_diff but not probe. Use probe instead."
---

# Massive Activation Dimensions

Some dimensions have values 100-1000x larger than median. These can contaminate trait vectors — but only if you use mean_diff.

**Reference:** Sun et al. "Massive Activations in Large Language Models" (COLM 2024)

## Key Finding

**mean_diff fails on models with severe massive activations. probe is immune.**

The problem isn't the massive dims themselves — it's how you weight them:

| Method | How it weights dims | Result |
|--------|---------------------|--------|
| mean_diff | By magnitude | Noise dominates |
| probe | By discriminative value | Signal extracted |

## Experimental Evidence (gemma-3-4b, refusal)

### Steering Results

| Method | Best Δ | Coherence | Notes |
|--------|--------|-----------|-------|
| mean_diff | ~0 | 93% | Completely fails |
| mean_diff_cleaned | +9 | 84% | Partial recovery |
| probe | **+33** | 73% | Works well |

*Position: response[:5], Layer: 15, Coefficients: 1000-1400*

### Cleaning Ablation

Zeroing massive dims helps mean_diff but hurts probe:

| Vector | Dims zeroed | Δ |
|--------|-------------|---|
| mean_diff | 0 | ~0 |
| mean_diff_top1 | 1 (dim 443) | **+29** |
| mean_diff_cleaned | 13 | +25 |
| probe | 0 | **+33** |
| probe_cleaned | 13 | +9 |

**Key insight:** Zeroing just dim 443 recovers mean_diff almost fully. But cleaning probe destroys it — probe learned to use that dim productively.

### Why Cleaning Has Opposite Effects

After cleaning, mean_diff and probe become nearly identical:
- `cos(mean_diff, probe)` = 0.48
- `cos(mean_diff_cleaned, probe_cleaned)` = **0.998**

This proves:
- mean_diff's massive dim content was noise (removing it converges to probe)
- probe's massive dim content was signal (removing it degrades performance)

## Trait Differences

Sycophancy behaves differently than refusal:

| Trait | mean_diff Δ | probe Δ | Gap |
|-------|-------------|---------|-----|
| Refusal | ~0 | +33 | 33 |
| Sycophancy | +11 | +21 | 10 |

mean_diff partially works for sycophancy, suggesting the massive dims carry more trait signal for some traits than others.

## Model Severity

| Model | Dominant Dim | Magnitude | mean_diff viable? |
|-------|-------------|-----------|-------------------|
| gemma-2-2b | dim 334 | ~60x | TBD (Phase 3) |
| gemma-3-4b | dim 443 | ~1000x | No |

Cross-model comparison pending.

## Recommendations

1. **Use probe or gradient** — robust to massive dims
2. **Don't use mean_diff on gemma-3-4b** — it fails
3. **If you must use mean_diff:** zero the dominant massive dim (e.g., dim 443) as a partial fix
4. **Don't blindly clean probe vectors** — it removes learned signal

## Implementation

- **Calibration:** `python analysis/massive_activations.py --experiment {exp}`
- **Output:** `experiments/{exp}/inference/{variant}/massive_activations/calibration.json`
- **Massive dims list:** `calibration.json` → `aggregate.top_dims_by_layer`

## Source

Experiment: `experiments/massive-activations/`
- Phase 1: Basic hypothesis (mean_diff vs probe)
- Phase 2: Cleaning ablation, sycophancy, probe causality
- Phase 3: Cross-model comparison (pending)
