---
title: "Component Comparison: Refusal"
preview: "Attention carries the refusal signal (+60.9 delta) vs residual (+56.3 with probe). MLP weaker (+25.9), k_proj ineffective (+1.5)."
---

# Component Comparison: Refusal

**Question:** Which component best captures the refusal signal?

**Answer:** Attention contribution with mean_diff (+60.9) is the winner. For residual, probe (+56.3) and gradient (+53.4) beat mean_diff (+43.2). MLP is weak (+25.9), k_proj ineffective (+1.5).

## Component Comparison (mean_diff)

| Component | Best Layer | Coefficient | Trait | Coherence | Delta |
|-----------|------------|-------------|-------|-----------|-------|
| **attn_contribution** | L9 | 145.5 | 74.6 | 74.7% | **+60.9** |
| residual | L12 | 116.5 | 56.8 | 75.5% | +43.2 |
| mlp_contribution | L7 | 126.6 | 39.6 | 73.4% | +25.9 |
| k_proj | L12 | 568.4 | 15.2 | 88.1% | +1.5 |

## Method Comparison (residual)

| Method | Best Layer | Coefficient | Trait | Coherence | Delta |
|--------|------------|-------------|-------|-----------|-------|
| **probe** | L13 | 148.5 | 69.9 | 71.9% | **+56.3** |
| gradient | L11 | 131.8 | 67.0 | 70.6% | +53.4 |
| mean_diff | L12 | 116.5 | 56.8 | 75.5% | +43.2 |

## Full Ranking

| Component/Method | Layer | Delta |
|------------------|-------|-------|
| **attn_contribution/mean_diff** | L9 | **+60.9** |
| residual/probe | L13 | +56.3 |
| residual/gradient | L11 | +53.4 |
| residual/mean_diff | L12 | +43.2 |
| mlp_contribution/mean_diff | L7 | +25.9 |
| k_proj/mean_diff | L12 | +1.5 |

Baseline trait score: 13.6

## Key Findings

1. **Attention dominates** - attn_contribution with mean_diff achieves the best results (+60.9), 8% better than the best residual method (probe at +56.3).

2. **Method matters for residual** - probe and gradient significantly outperform mean_diff for residual vectors (+56.3, +53.4 vs +43.2). This is a 30% improvement.

3. **MLP contributes weakly** - MLP adds some refusal signal but less than half of attention's contribution (+25.9 vs +60.9).

4. **Keys don't encode refusal** - k_proj steering has essentially no effect (+1.5 delta) despite high coherence. Keys carry routing info, not behavioral state.

5. **Layer depth varies by component** - Best layers: L9 (attn), L11-13 (residual), L7 (mlp). Attention signal may get diluted in later residual stream.

## Implications

- **For steering:** Use attn_contribution/mean_diff for strongest effect
- **For residual:** Use probe or gradient methods, not mean_diff
- **For interpretability:** Focus on attention patterns when studying refusal
- **For mechanistic work:** Refusal decision is primarily attention-mediated

## Setup

- Model: gemma-2-2b (base for extraction, IT for steering)
- Position: response[:5]
- Methods: mean_diff, probe, gradient
- Layers searched: 7-16
- Search steps: 8
- Evaluation: 10 steering questions, GPT-4.1-mini judge
