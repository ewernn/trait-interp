---
title: "Massive Activation Dimensions"
preview: "LLMs have dimensions with 1000x larger values; cleaning method is model-specific."
---

# Massive Activation Dimensions

LLMs have dimensions with 1000x larger values than median.

**Reference:** Sun et al. "Massive Activations in Large Language Models" (COLM 2024)

## What We Tried

| Approach | Result |
|----------|--------|
| **Sun criterion** (>100 AND >1000x median) | Works, but absolute threshold is meaningless |
| **Ratio-only** (>1000x median) | Identical to Sun - ratio is the real filter |
| **Z-score > 5** | Too permissive (1500+ dims vs 27) |
| **Mean projection cleaning** | Just a constant shift - doesn't improve separation |
| **Top5-3L** (dims in top-5 at 3+ layers) | Best for Gemma; same as Sun for Qwen |

## Cross-Model Comparison

| Aspect | Gemma-2-2b-it | Qwen-2.5-7B-it |
|--------|---------------|----------------|
| **Bias location** | BOS token | first_delimiter |
| **Max value** | ~900 | ~11,200 |
| **Cleaning dims** | 8 | 5 |

## Takeaways

1. **Location is model-specific** - can't assume BOS
2. **Ratio threshold is what matters** - drop the `abs > 100`
3. **Mean projection doesn't help** - constant shift, no separation gain

**Implementation:** `analysis/massive_activations.py` (data-driven calibration)
