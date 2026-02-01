---
title: "The Model Recognizes Its Own Voice"
preview: "Model-generated text produces smoother activations. Effect transfers across model variants and families (Gemma d=1.49, Llama d=0.97)."
---

# The Model Recognizes Its Own Voice

Model-generated text produces significantly smoother activation trajectories than human-written text during prefill. This translates to more stable trait projections — with implications for monitoring reliability.

## Key Findings

1. **Raw smoothness:** Model text is smoother (Gemma d=1.49, Llama d=0.97)
2. **Cross-model transfer:** Effect holds across variants (base/instruct) and families (Gemma/Llama)
3. **Projection stability:** 25-75% lower trait projection variance in middle layers
4. **Position effect:** Gap grows as sequence continues (model "finds its voice")

## Methodology

**Setup:** 50 WikiText-2 paragraphs. Each has a first sentence used as prompt. We compare:
- **Human text:** Original WikiText continuation
- **Model text:** Gemma-2-2B base (temp=0) continuation from same prompt

**Metrics:**

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Smoothness** | mean(‖act[t+1] - act[t]‖₂) across tokens | Lower = smoother trajectory |
| **Projection variance** | var(act · trait_vec) across tokens | Lower = more stable trait signal |
| **CE loss** | Mean cross-entropy loss per token | Lower = less surprising to model |
| **Cohen's d** | (μ_human - μ_model) / σ_paired_diff | Standardized effect size |

Activations are residual stream outputs at each layer. All comparisons are paired (same prompt, different continuation).

## The Core Effect

We measured token-to-token activation changes (smoothness) when Gemma-2-2B processes human-written vs model-generated text:

| Processor | Human | Model-gen | Diff | Cohen's d |
|-----------|-------|-----------|------|-----------|
| Base | 193.8 | 179.5 | 14.3 | **1.49** |
| Instruct | 201.7 | 188.7 | 13.0 | **1.49** |

**Key insight:** The model-generated text came from base model (temp=0). Instruct still finds it smoother than human text, even though it's not "on-distribution" for instruct specifically.

This suggests smoothness reflects structural properties of model-like text (predictable word choices, syntactic regularity) rather than "I recognize my own output."

## Correlation with Perplexity

Smoothness correlates with surprisingness (r=0.65, p<1e-12):

| Processor | Human CE | Model CE | Ratio |
|-----------|----------|----------|-------|
| Base | 2.99 | 1.45 | 2.06x |
| Instruct | 3.35 | 1.61 | 2.08x |

Both models find human text ~2x more surprising.

## Layer Profile

Effect grows from early to late layers, then vanishes at output:

| Layer Range | Cohen's d | Notes |
|-------------|-----------|-------|
| 0-5 | 0.6-0.7 | Building |
| 6-12 | 1.3-1.5 | Peak effect |
| 13-21 | 1.4-1.6 | Sustained |
| 22-24 | 0.9-1.4 | Declining |
| 25 | 0.02 | Output converges |

## Trait Projection Stability

Does the smoothness difference affect trait monitoring? Yes, in middle layers.

**Sycophancy trait, projection variance (human vs model):**

| Layer | Human Var | Model Var | Cohen's d | p-value |
|-------|-----------|-----------|-----------|---------|
| 5 | 5.75 | 5.20 | 0.26 | 0.075 |
| 11 | 11.63 | 7.70 | **1.00** | 6.8e-09 |
| 14 | 5.44 | 4.42 | **0.54** | 0.0005 |
| 16 | 53.03 | 41.32 | **0.62** | 7.6e-05 |

**Peak effect at layer 11 (d=1.0)** — where many trait vectors operate. ~75% of the effect is directional, ~25% magnitude.

## Position Effect

The gap grows as sequences continue — model text gets progressively smoother relative to human:

| Position | Human Var | Model Var | Diff | Cohen's d |
|----------|-----------|-----------|------|-----------|
| 0-5 | 8.69 | 8.26 | +0.43 | 0.05 |
| 5-15 | 8.43 | 7.97 | +0.46 | 0.08 |
| 15-30 | 9.09 | 7.11 | +1.98 | 0.35 |
| 30-50 | 9.51 | 6.26 | +3.25 | 0.56 |
| 50-100 | 10.24 | 6.25 | +3.99 | **0.69** |

Early tokens are similarly noisy for both. The model "settles into" its distribution over the first ~30 tokens.

## Practical Implications

**For trait monitoring:**
- Projections on model output (assistant turns) are more stable than on user input
- Early tokens in either condition are noisier — consider windowing
- Middle layers (10-18) show the strongest stability difference

**For extraction:**
- Model-generated scenarios might yield cleaner vectors (untested hypothesis)
- Effect is robust to RLHF (same d for base and instruct)

## Cross-Model Generalization

The effect holds across model families:

| Model | Human Smooth | Model Smooth | Cohen's d | p-value |
|-------|--------------|--------------|-----------|---------|
| Gemma-2-2B | 193.8 | 179.5 | **1.49** | 5.3e-14 |
| Llama-3.1-8B | 17.7 | 16.4 | **0.97** | 1.5e-08 |

Absolute smoothness values differ (architecture scale), but both show the same pattern: model-generated text is smoother than human text. Llama's smaller effect size (d=0.97 vs 1.49) may reflect architectural differences or the fact that Llama processed text generated by Gemma, not itself.

## Robustness Checks

| Check | Result |
|-------|--------|
| Base vs Instruct (Gemma) | Same effect (d=1.49) |
| Cross-model (Llama) | Effect generalizes (d=0.97) |
| Refusal trait | Confirmed at layer 17 |
| Sycophancy trait | Confirmed across all layers |
| Temperature 0.7 | Effect attenuates (d=0.98) but persists |

## Open Questions

1. **Same-model generation:** Does Llama show stronger effect on Llama-generated text (vs Gemma-generated)?
2. **Extraction quality:** Would model-generated scenarios yield cleaner trait vectors?
3. **Position mechanism:** What causes model text to become progressively smoother?

## Figures

![Smoothness by layer](../../experiments/prefill-dynamics/figures/smoothness_by_layer.png)
*Cohen's d for raw smoothness. Effect builds through layers 0-6, peaks at ~1.6 in middle layers, vanishes at output layer.*

![Violin distributions](../../experiments/prefill-dynamics/figures/violin_smoothness.png)
*Distribution separation at peak layer. Model-generated text is consistently smoother.*

![Projection stability](../../experiments/prefill-dynamics/figures/projection_stability_by_layer.png)
*Projection variance effect by layer. Refusal (green) shows strong L11 spike; sycophancy (purple) more uniform.*

![Position breakdown](../../experiments/prefill-dynamics/figures/position_breakdown.png)
*Variance by token position. Human stays flat (~10); model drops from 8→6 as sequence continues.*

![Perplexity vs smoothness](../../experiments/prefill-dynamics/figures/perplexity_vs_smoothness.png)
*Correlation between surprisingness and smoothness (r=0.65). Clear cluster separation.*

![Effect comparison](../../experiments/prefill-dynamics/figures/effect_comparison.png)
*Smoothness effect (solid) vs projection stability (dashed) by layer. Smoothness is strong everywhere; projection stability peaks at L11.*

## Source

Experiment: `experiments/prefill-dynamics/`
- Phase 1: Raw smoothness analysis (base, instruct, temp comparison)
- Phase 2: Projection stability + position breakdown
