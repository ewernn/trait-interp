---
title: "Component Comparison: Refusal"
preview: "attn ≈ residual at 70% coherence (+27.6 vs +27.5). attn wins at lower thresholds. MLP/v_proj ~+10, k_proj ineffective."
---

# Component Comparison: Refusal

**Question:** Which component best captures the refusal signal?

**Answer:** attn_contribution and residual are tied at 70% coherence threshold. At lower thresholds, attn pulls ahead. Probe method wins for most components.

## Results (coherence ≥ 70%)

| Component | Method | Layer | Delta |
|-----------|--------|-------|-------|
| attn_contribution | probe | L15 | **+27.6** |
| residual | probe | L13 | +27.5 |
| v_proj | mean_diff | L13 | +10.8 |
| mlp_contribution | probe | L12 | +10.4 |
| k_proj | gradient | L9 | +2.0 |

Baseline: 13.1

## Threshold Sensitivity

Results depend heavily on coherence cutoff:

| Threshold | Best | Delta |
|-----------|------|-------|
| 70% | attn/probe L15 | +27.6 |
| 60% | attn/mean_diff L15 | +54.3 |
| 50% | attn/mean_diff L11 | +72.4 |

At 60-50% thresholds, attn beats residual by ~15%.

## Layer Range

Tested all 26 layers (0-25). **L11-L15 is optimal.** Signal degrades at both extremes:
- L0-6: weak (~+5)
- L7-15: optimal (+27)
- L16-25: degrading (+15 → +7)

## Key Findings

1. **attn ≈ residual** at strict coherence thresholds
2. **attn > residual** when allowing more incoherence
3. **Probe wins** for attn/residual/mlp; mean_diff wins for v_proj
4. **MLP/v_proj weak** (~40% of attn/residual)
5. **k_proj ineffective** (keys don't encode refusal)

## Setup

- Model: gemma-2-2b-it
- Position: response[:5]
- Components: attn_contribution, residual, mlp_contribution, v_proj, k_proj
- Methods: probe, mean_diff, gradient
- Layers: 0-25 (full sweep)
- Search steps: 10, 10 questions
- Total runs: 1678
