# attn_contribution Ensemble Steering (Sycophancy, Llama-3.1-8B)

Multi-layer ensembles within `attn_contribution` beat individual layers on both trait and coherence:

| Config | Component | Weights | Trait | Coh | Δ |
|--------|-----------|---------|-------|-----|---|
| L11+L13 mean_diff | attn_contribution | L11=3.2, L13=12.1 | 91.4 | 86.9 | +54.7 |
| L11+L13+L15 probe | attn_contribution | L11=4.0, L13=1.9, L15=7.4 | 90.2 | 91.1 | +53.5 |
| L11 alone | attn_contribution | c13.2 | 88.6 | 81.3 | +51.9 |
| Best single residual | residual | L14, c6.6 | 92.8 | 80.2 | +56.2 |

The L11+L13 ensemble is within 1.5 trait points of the best residual result but has 6.7 points better coherence.

**CMA-ES optimization:** L13 carries the load (12.1) while L11 provides a nudge (3.2), despite L11 being the stronger individual layer. Distributing the perturbation preserves coherence better than concentrating it at one layer.

**Odd-layer alternation:** L11, L13, L15 are strong; even layers (L10, L12, L14) barely move above baseline. Likely reflects learned specialization in attention heads across layers.

## Source

Vectors and steering from `experiments/component_comparison/` (originally extracted in `experiments/model_diff/`, merged here).
