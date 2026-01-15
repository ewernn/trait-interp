---
title: "Comparison: Arditi-style vs Natural Refusal Vectors"
preview: "Both methods induce refusal similarly (+26-28 delta), but only Arditi bypasses (100% vs 40%). They capture different aspects of refusal."
references:
  arditi2024:
    authors: "Arditi et al."
    title: "Refusal in Language Models is Mediated by a Single Direction"
    year: 2024
    url: "https://arxiv.org/abs/2406.11717"
---

# Comparison: Arditi-style vs Natural Refusal Vectors

**Question:** How do different extraction methods compare for controlling refusal?

**Answer:** Both methods induce refusal similarly (+26-28 delta with coherence ≥70%), but only Arditi-style vectors can bypass refusal (100% vs 40%). The vectors are nearly orthogonal and capture fundamentally different aspects.

## Methods Compared

| Aspect | Arditi-style | Natural |
|--------|--------------|---------|
| Extraction position | `prompt[-1]` (last token before generation) | `response[:5]` (first 5 response tokens) |
| Extraction model | instruct (gemma-2-2b-it) | base (gemma-2-2b) |
| Method | mean_diff | probe |
| What it captures | "Is this prompt harmful?" | "Am I refusing right now?" |

## Vector Similarity

Cosine similarity ~0.1 across layers — nearly orthogonal. These methods capture fundamentally different signals.

## Inducing Refusal (Positive Steering)

Steering on harmless prompts to induce refusal behavior.

| Vector | Baseline | Best | Delta | Coherence |
|--------|----------|------|-------|-----------|
| Arditi L13 | 6.6 | 32.9 | **+26.3** | 71% |
| Natural L13 | 13.0 | 41.1 | **+28.1** | 71% |

Both methods work similarly for making the model refuse harmless requests.

### Baseline (harmless prompts, no steering)

:::responses experiments/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/steering/responses/baseline.json "Baseline responses" no-scores:::

### Arditi positive steering (L13, coef +63)

:::responses experiments/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/steering/responses/residual/mean_diff/L13_c63.0_2026-01-14_12-55-10.json "Arditi L13 +63":::

### Natural positive steering (L13, coef +115)

:::responses experiments/arditi-refusal-replication/steering/chirp/refusal/instruct/response__5/steering/responses/residual/probe/L13_c115.2_2026-01-14_12-57-07.json "Natural L13 +115":::

## Bypassing Refusal (Negative Steering)

Steering on harmful prompts to bypass refusal. Using binary string matching (Arditi's scoring method).

| Vector | Baseline | Steered | Bypass Rate |
|--------|----------|---------|-------------|
| Arditi L12 | 98% refusal | 0% | **100%** |
| Natural L12 | 98% refusal | 71% | **27%** |

Arditi vectors achieve full bypass. Natural vectors show minimal effect.

### Baseline (harmful prompts, no steering)

:::responses experiments/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/arditi_holdout/positive/responses/baseline.json "Baseline (harmful)" no-scores:::

### Arditi negative steering (L12, coef -100)

:::responses experiments/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/arditi_holdout/positive/responses/residual/mean_diff/L12_c-100.0_2026-01-15_01-41-07.json "Arditi L12 -100":::

### Natural negative steering (L12, coef -100)

:::responses experiments/arditi-refusal-replication/steering/chirp/refusal/instruct/response__5/arditi_holdout/positive/responses/residual/probe/L12_c-100.0_2026-01-15_01-42-33.json "Natural L12 -100":::

## Ablation Results

All-layer directional ablation (projecting out the refusal direction from all layers).

| Vector | Baseline | Ablated | Bypass Rate |
|--------|----------|---------|-------------|
| Arditi | 96% refusal | 0% | **100%** |
| Natural | 96% refusal | 96% | **0%** |

Arditi ablation replicates the paper's findings (99%→5% in paper, 96%→0% here). Natural ablation has no effect.

## Interpretation

**Arditi vectors** capture the classification signal — "is this prompt harmful?" This is the decision point that gates refusal behavior. Ablating or steering against it blinds the model to harmful content.

**Natural vectors** capture the behavioral mode — "am I refusing right now?" This is downstream of the classification. Steering can amplify refusal behavior, but ablating it doesn't prevent the model from recognizing harmful content in the first place.

This explains the asymmetry:
- **Inducing refusal:** Both work (both can push toward refusal mode)
- **Bypassing refusal:** Only Arditi works (only it controls the gate)

## Scoring Methods

Two scoring methods used:

1. **Binary string matching** (Arditi's method): Checks for refusal phrases ("I cannot", "I'm sorry", etc.). More permissive — counts any non-refusal as bypass.

2. **LLM judge** (our method): Evaluates response on 0-100 scale for refusal behavior. More strict — penalizes evasive or off-topic responses.

The 100% bypass rate with Arditi vectors holds under both methods. Natural vectors show ~40% bypass with binary scoring but responses are often low-quality.

## Open Questions

- Would natural vectors work better with different extraction positions?
- Can we combine both vectors for finer-grained control?
- Test positive steering on larger Alpaca prompt set
