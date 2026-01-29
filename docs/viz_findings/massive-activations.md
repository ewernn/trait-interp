---
title: "Massive Activation Dimensions"
preview: "Massive dims contaminate mean_diff. Use probe — or use response[:20]."
---

# Massive Activation Dimensions

Some dimensions have values 100-1000x larger than median. These can contaminate trait vectors — but only if you use mean_diff with short position windows.

**Reference:** Sun et al. "Massive Activations in Large Language Models" (COLM 2024)

## Key Findings

1. **mean_diff fails on models with severe massive activations** (gemma-3-4b) — probe is immune
2. **Longer position windows (response[:20]) recover mean_diff** to probe-level performance
3. **Model severity matters:** gemma-2-2b (60x contamination) works fine with mean_diff

## Method Comparison

The problem isn't the massive dims themselves — it's how you weight them:

| Method | How it weights dims | At response[:5] | At response[:20] |
|--------|---------------------|-----------------|------------------|
| mean_diff | By magnitude | Noise dominates | Signal extracted |
| probe | By discriminative value | Signal extracted | Signal diluted |

## Experimental Evidence

### Phase 1-2: Basic Hypothesis (gemma-3-4b, refusal)

**Steering results at response[:5]:**

| Method | Best Δ | Coherence | Notes |
|--------|--------|-----------|-------|
| mean_diff | +27 | 73% | Only works at c2500 |
| mean_diff_cleaned | +25 | 71% | Partial recovery |
| mean_diff_top1 | **+29** | 72% | Just dim 443 |
| probe | **+33** | 73% | Best overall |

### Cleaning Ablation

Zeroing massive dims helps mean_diff but hurts probe:

| Vector | Dims zeroed | Δ |
|--------|-------------|---|
| mean_diff | 0 | +27 |
| mean_diff_top1 | 1 (dim 443) | **+29** |
| mean_diff_cleaned | 13 | +25 |
| probe | 0 | **+33** |
| probe_cleaned | 13 | +9 |

**Key insight:** Zeroing just dim 443 is optimal. Probe learns to use massive dims productively — cleaning it hurts.

### Phase 3: Cross-Model Comparison

| Model | Contamination | mean_diff Δ | probe Δ | Best method |
|-------|---------------|-------------|---------|-------------|
| gemma-3-4b | ~1000x | +27 | +33 | probe |
| gemma-2-2b | ~60x | **+29** | +21 | mean_diff |

**Surprise:** On gemma-2-2b, mean_diff outperforms probe! Milder contamination means simple averaging works.

Note: gemma-2-2b needs ~20x lower coefficients (50-120 vs 1000-2500).

### Phase 4-5: Position Window Ablation

Full cross-model × position matrix:

| Model | Position | mean_diff Δ | probe Δ | Winner |
|-------|----------|-------------|---------|--------|
| gemma-3-4b | [:5] | +27 | **+33** | probe |
| gemma-3-4b | [:10] | +12 | **+27** | probe |
| gemma-3-4b | [:20] | **+33** | +21 | mean_diff |
| gemma-2-2b | [:5] | **+29** | +21 | mean_diff |
| gemma-2-2b | [:10] | +24 | +25 | tie |
| gemma-2-2b | [:20] | **+35** | +33 | mean_diff |

**Key patterns:**
- At [:20], mean_diff wins on both models
- Probe improves at longer windows on gemma-2-2b (opposite of gemma-3-4b)
- [:10] is a "valley" — worse than [:5] on gemma-3-4b

### Massive Dim Energy Analysis

We measured what % of mean_diff vector energy is in massive dims:

| Model | Position | Massive Energy | Performance |
|-------|----------|----------------|-------------|
| gemma-3-4b | [:5] | **81%** | +27 |
| gemma-3-4b | [:20] | **29%** | +33 |
| gemma-2-2b | [:5] | 12% | +29 |
| gemma-2-2b | [:20] | 11% | +35 |

**This verifies the mechanism:** On gemma-3-4b, longer windows reduce massive dim energy (81% → 29%), which recovers mean_diff performance. On gemma-2-2b, energy is always low (11-12%), so mean_diff works at all positions.

## Geometric Interpretation

After cleaning OR with longer position windows, mean_diff converges to probe:

| Transformation | Cosine to probe |
|----------------|-----------------|
| mean_diff (original) | 0.48 |
| mean_diff_cleaned | 0.93 |
| mean_diff @ response[:20] | **0.98** |

The massive dims encode position/context signals. Both cleaning and longer windows remove this confound, converging to the "true" trait direction that probe finds discriminatively.

## Trait Differences

Sycophancy behaves differently than refusal:

| Trait | mean_diff Δ | probe Δ | Gap |
|-------|-------------|---------|-----|
| Refusal | +27 | +33 | 6 |
| Sycophancy | +11 | +21 | 10 |

mean_diff partially works for sycophancy, suggesting massive dims carry different amounts of trait signal per trait.

## Recommendations

1. **Check model severity first:** Run `python analysis/massive_activations.py --experiment {exp}`
2. **If contamination >100x:** Use probe OR mean_diff with response[:20]
3. **If contamination <100x:** mean_diff works fine
4. **Don't clean probe vectors** — it removes learned signal
5. **If you must clean mean_diff:** zero only the dominant dim (not all massive dims)

## Implementation

- **Calibration:** `python analysis/massive_activations.py --experiment {exp}`
- **Output:** `experiments/{exp}/inference/{variant}/massive_activations/calibration.json`
- **Massive dims list:** `calibration.json` → `aggregate.top_dims_by_layer`

## Open Questions

Some position effects remain unexplained by massive dim energy alone:

| Phenomenon | Energy | Performance | Unexplained |
|------------|--------|-------------|-------------|
| [:10] valley | 69% (lower than [:5]) | +12 (worse than [:5]) | Why worse despite lower energy? |
| Probe at [:20] | 25% | +21 (worse than [:5]) | Why does probe degrade? |

These may be due to trait signal strength varying across token positions, not just massive dims.

## Source

Experiment: `experiments/massive-activations/`
- Phase 1: Basic hypothesis (mean_diff vs probe)
- Phase 2: Cleaning ablation, sycophancy, probe causality
- Phase 3: Cross-model comparison (gemma-2-2b vs gemma-3-4b)
- Phase 4-5: Position window ablation (both models)
