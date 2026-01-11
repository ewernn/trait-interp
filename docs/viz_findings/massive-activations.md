---
title: "Massive Activation Dimensions"
preview: "Massive dims encode context and contaminate mean_diff; use probe/gradient instead."
---

# Massive Activation Dimensions

Some dimensions have values 100-2000x larger than median. These encode **context/topic**, not trait-specific signal, and can severely contaminate trait vectors.

**Reference:** Sun et al. "Massive Activations in Large Language Models" (COLM 2024)

## Key Finding: They Encode Context

Massive dims are consistent within a prompt but vary between prompts based on topic/context:
- **Paired scenarios** (same context, different response): 2-48% contamination
- **Unpaired scenarios** (different topics): 86-91% contamination

This means mean_diff picks up context differences, not trait signal.

## Impact on Extraction Methods

| Method | Contamination | Why |
|--------|--------------|-----|
| **mean_diff** | 48-91% | Raw difference amplifies any pos/neg context variation |
| **probe** | 0.3-1% | Optimizes for discrimination, ignores non-discriminative dims |
| **gradient** | 0.2-4% | Same - optimizes for separation |

## Two Failure Modes of mean_diff

Per-token projection analysis revealed contamination causes two different problems:

| Trait | Scenario | Effect | What Happened |
|-------|----------|--------|---------------|
| Refusal | Unpaired | **Inflated** separation (85% spurious) | dim 443 correlated with harmful/benign by chance |
| Sycophancy | Paired | **Masked** separation (hidden until cleaned) | dim 443 added noise obscuring real signal |

**probe is robust to both** - cleaning activations changes results by <4%.

## Model Severity

| Model | Dominant Dim | Magnitude | mean_diff Usable? |
|-------|-------------|-----------|-------------------|
| Gemma 2-2b | dim 334 | ~60x | Marginal (24% contamination) |
| Gemma 3-4b | dim 443 | ~1000-2000x | No (86-91% contamination) |

Gemma 3's massive dim is ~30x more extreme than Gemma 2's.

## Recommendations

1. **Use probe or gradient** - robust regardless of scenario design or model
2. **Never use mean_diff on Gemma 3** - vectors are mostly noise
3. **If using mean_diff**: use tightly paired scenarios (same context, different response)
4. **For per-token monitoring**: probe vectors give real signal; mean_diff can show spurious or masked separation

## Implementation

- **Calibration:** `analysis/massive_activations.py` identifies massive dims per model
- **Per-layer stats:** `experiments/{exp}/inference/instruct/massive_activations/per_layer_stats.json`
- **Bug fix:** Mean calculation now uses float32 to avoid bfloat16 precision loss at ~32k values
