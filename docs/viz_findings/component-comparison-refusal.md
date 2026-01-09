---
title: "Component Comparison: Refusal"
preview: "Attention carries the refusal signal (+60.9 delta) vs residual (+43.2). MLP weaker (+25.9), k_proj ineffective (+1.5)."
---

# Component Comparison: Refusal

**Question:** Which component best captures the refusal signal?

**Answer:** Attention contribution (+60.9 delta) outperforms full residual stream (+43.2). MLP contributes weakly (+25.9), key projections are ineffective (+1.5).

## Results

| Component | Best Layer | Coefficient | Trait | Coherence | Delta |
|-----------|------------|-------------|-------|-----------|-------|
| **attn_contribution** | L9 | 145.5 | 74.6 | 74.7% | **+60.9** |
| residual | L12 | 116.5 | 56.8 | 75.5% | +43.2 |
| mlp_contribution | L7 | 126.6 | 39.6 | 73.4% | +25.9 |
| k_proj | L12 | 568.4 | 15.2 | 88.1% | +1.5 |

Baseline trait score: 13.6

## Interpretation

1. **Attention dominates** - The attention mechanism is primarily responsible for encoding refusal behavior. Steering with attn_contribution alone achieves +60.9 delta, 41% better than the full residual stream.

2. **MLP contributes weakly** - MLP adds some refusal signal but less than half of attention's contribution (+25.9 vs +60.9).

3. **Keys don't encode refusal** - k_proj steering has essentially no effect (+1.5 delta) despite high coherence. The keys carry positional/routing information, not behavioral state.

4. **Earlier layers for attention** - Best attn layer is L9, while best residual is L12. The attention signal may get diluted by MLP contributions in later residual stream.

## Implications

- **For steering:** Use attn_contribution vectors for stronger effect
- **For interpretability:** Focus on attention patterns when studying refusal
- **For mechanistic work:** Refusal decision is primarily attention-mediated

## Setup

- Model: gemma-2-2b (base for extraction, IT for steering)
- Position: response[:5]
- Method: mean_diff
- Layers searched: 7-16
- Search steps: 8
- Evaluation: 10 steering questions, GPT-4.1-mini judge
