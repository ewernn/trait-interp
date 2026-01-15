---
title: "Comparison: Arditi-style vs Natural Refusal Vectors"
preview: "Arditi ablates better (100% vs 0% bypass), but both methods induce refusal similarly (~100% with aggressive steering, ~20% with coherent steering)."
references:
  arditi2024:
    authors: "Arditi et al."
    title: "Refusal in Language Models is Mediated by a Single Direction"
    year: 2024
    url: "https://arxiv.org/abs/2406.11717"
---

# Comparison: Arditi-style vs Natural Refusal Vectors

**Question:** How do different extraction methods compare for controlling refusal?

## Methods Compared

| Aspect | Arditi-style | Natural |
|--------|--------------|---------|
| Extraction position | `prompt[-1]` (last token before generation) | `response[:5]` (first 5 response tokens) |
| Extraction model | instruct (gemma-2-2b-it) | base (gemma-2-2b) |
| Method | mean_diff | probe |

## Vector Similarity

Cosine similarity ~0.1 across layers — nearly orthogonal. These methods capture different signals.

## Bypassing Refusal (Negative Steering)

Steering on harmful prompts to bypass refusal.

### Ablation (all layers)

Projecting out the refusal direction from all layers simultaneously.

| Vector | Baseline | Ablated | Bypass Rate |
|--------|----------|---------|-------------|
| Arditi | 96% refusal | 0% | **100%** |
| Natural | 96% refusal | 96% | **0%** |

Arditi ablation replicates the paper's findings (99%→5% in paper, 96%→0% here). Natural ablation has no effect.

### Single-layer steering (L12)

| Vector | Baseline | Steered | Bypass Rate |
|--------|----------|---------|-------------|
| Arditi L12 | 98% refusal | 0% | **100%** |
| Natural L12 | 98% refusal | 71% | **27%** |

:::responses experiments/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/arditi_holdout/positive/responses/baseline.json no-scores:::

:::responses experiments/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/arditi_holdout/positive/responses/residual/mean_diff/L12_c-100.0_2026-01-15_01-41-07.json:::

:::responses experiments/arditi-refusal-replication/steering/chirp/refusal/instruct/response__5/arditi_holdout/positive/responses/residual/probe/L12_c-100.0_2026-01-15_01-42-33.json:::

## Inducing Refusal (Positive Steering)

Steering on harmless prompts to induce refusal behavior.

### Binary scoring (Arditi's method)

String matching for refusal phrases ("I cannot", "I'm sorry", etc.).

| Vector | Baseline | Aggressive | Moderate |
|--------|----------|------------|----------|
| Arditi | 4% | **100%** (L7 c84) | 21% (L8 c42) |

Both methods achieve ~100% binary refusal with aggressive steering, matching Arditi's Figure 3.

### LLM judge scoring (0-100 scale)

| Vector | Baseline | Best | Delta | Coherence |
|--------|----------|------|-------|-----------|
| Arditi L13 | 6.6 | 32.9 | **+26.3** | 71% |
| Natural L13 | 13.0 | 41.1 | **+28.1** | 71% |

With coherence ≥70%, both methods achieve similar deltas.

:::responses experiments/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/steering/responses/baseline.json no-scores:::

:::responses experiments/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/steering/responses/residual/mean_diff/L13_c63.0_2026-01-14_12-55-10.json:::

:::responses experiments/arditi-refusal-replication/steering/chirp/refusal/instruct/response__5/steering/responses/residual/probe/L13_c115.2_2026-01-14_12-57-07.json:::

## Coherence-Refusal Tradeoff

Aggressive steering achieves 100% binary refusal but degrades response quality.

| Config | Binary Refusal | LLM Coherence |
|--------|----------------|---------------|
| Baseline | 4% | ~95% |
| L8 c42 (moderate) | 21% | 72% |
| L7 c84 (aggressive) | **100%** | 50% |

The "coherence degradation" is the model refusing everything — including harmless requests — by miscategorizing them as harmful.

### Moderate steering (L8 c42, 72% coherence)

:::responses experiments/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/arditi_holdout/negative/responses/residual/mean_diff/L8_c42.3_2026-01-14_14-01-47.json:::

### Aggressive steering (L7 c84, 50% coherence)

:::responses experiments/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/arditi_holdout/negative/responses/residual/mean_diff/L7_c84.2_2026-01-15_14-59-15.json:::

## Scoring Methods

Two scoring methods used throughout:

1. **Binary string matching** (Arditi's method): Checks for refusal phrases. Does not assess coherence or appropriateness.

2. **LLM judge** (our method): Evaluates response on 0-100 scale. Penalizes both refusals and incoherent responses.

Arditi's paper uses binary scoring only and does not report coherence metrics.

## Open Questions

- Can we combine both vectors for finer-grained control?
- Is there a steering strength that achieves high refusal on harmful prompts without refusing harmless ones?
