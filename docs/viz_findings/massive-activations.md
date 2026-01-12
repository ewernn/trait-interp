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

## Steering Validation (Causal Test)

Arditi-style steering on Gemma 3-4b refusal (20 harmless prompts, natural magnitude methodology):

| Layer | mean_diff | probe |
|-------|-----------|-------|
| 10 | 0% | 0% |
| 13 | 0% | 10% |
| 15 | 0% | 40% |
| 17 | 0% | 40% |
| 20 | 0% | **60%** |
| 25 | 0% | 15% |

**mean_diff fails completely** (0% refusal induction at all layers, garbage output due to dim 443 contamination). **probe works** (peaks at 60% L20).

This is causal proof: mean_diff vectors can't induce refusal because they're mostly dim 443 noise.

## Recommendations

1. **Use probe or gradient** - robust regardless of scenario design or model
2. **Never use mean_diff on Gemma 3** - vectors are mostly noise
3. **If using mean_diff**: use tightly paired scenarios (same context, different response)
4. **For per-token monitoring**: probe vectors give real signal; mean_diff can show spurious or masked separation

## Implementation

- **Calibration:** `analysis/massive_activations.py` identifies massive dims per model
- **Per-layer stats:** `experiments/{exp}/inference/instruct/massive_activations/per_layer_stats.json`
- **Bug fix:** Mean calculation now uses float32 to avoid bfloat16 precision loss at ~32k values
