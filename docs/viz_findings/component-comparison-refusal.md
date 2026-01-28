---
title: "Component Comparison: Refusal"
preview: "attn ≈ residual at 70% coherence (+27.6 vs +27.5). attn wins at lower thresholds. MLP/v_proj ~+10, k_proj ineffective."
---

## Summary

We compare which activation component best captures the refusal signal in gemma-2-2b-it. Attention contribution and residual stream tie at strict coherence thresholds, but attention pulls ahead when allowing more incoherence. Keys don't encode refusal; values encode it weakly.

## Method

- **Model**: gemma-2-2b-it (26 layers)
- **Position**: `response[:5]` (first 5 response tokens)
- **Components**: residual, attn_contribution, mlp_contribution, v_proj, k_proj
- **Methods**: probe, mean_diff, gradient
- **Layers**: 0-25 (full sweep)
- **Evaluation**: Steer with extracted vectors, score refusal with LLM-as-judge
- **Total runs**: 1678

:::dataset datasets/traits/chirp/refusal/positive.txt "Positive scenarios (should refuse)" height=150:::

:::dataset datasets/traits/chirp/refusal/negative.txt "Negative scenarios (should help)" height=150:::

:::extraction-data "Extraction responses"
refusal: experiments/gemma-2-2b/extraction/chirp/refusal/base/responses
:::

## Results (coherence ≥ 70%)

| Component | Method | Layer | Delta |
|-----------|--------|-------|-------|
| attn_contribution | probe | L15 | **+27.6** |
| residual | probe | L13 | +27.5 |
| v_proj | mean_diff | L13 | +10.8 |
| mlp_contribution | probe | L12 | +10.4 |
| k_proj | gradient | L9 | +2.0 |

Baseline: 13.1

### Threshold sensitivity

Results depend heavily on coherence cutoff:

| Threshold | Best | Delta |
|-----------|------|-------|
| 70% | attn/probe L15 | +27.6 |
| 60% | attn/mean_diff L15 | +54.3 |
| 50% | attn/mean_diff L11 | +72.4 |

At 60-50% thresholds, attn beats residual by ~15%.

### Layer range

**L11-L15 is optimal.** Signal degrades at both extremes:
- L0-6: weak (~+5)
- L7-15: optimal (+27)
- L16-25: degrading (+15 → +7)

## Takeaways

1. **attn ≈ residual** at strict coherence thresholds
2. **attn > residual** when allowing more incoherence
3. **Probe wins** for attn/residual/mlp; mean_diff wins for v_proj
4. **MLP/v_proj weak** (~40% of attn/residual)
5. **k_proj ineffective** — keys determine which positions attend (routing), not what information flows

## Methodological note

This comparison is not apples-to-apples:
- **residual at L13** = embedding + Σ(attn + mlp from L0-L12) — cumulative
- **attn_contribution at L15** = just what L15's attention added — single layer

The fact that a single layer's attention contribution matches 13 layers of accumulated signal suggests L15 attention is doing concentrated "refusal work."

**Potential follow-ups:**
- Sum attn_contribution across L11-L15 — would this beat residual?
- Difference vectors: residual_L15 - residual_L10 to isolate the refusal emergence window
- Per-layer contribution breakdown — which layers contribute most to residual's refusal signal?
