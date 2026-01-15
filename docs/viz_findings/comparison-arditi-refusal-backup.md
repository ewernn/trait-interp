---
title: "Comparison: Arditi-style vs Natural Refusal Vectors"
preview: "Different extraction methods capture different things. Arditi (prompt[-1]) = prompt classification. Natural (response[:5]) = behavioral mode. Both achieve ~+27 delta for inducing refusal."
references:
  arditi2024:
    authors: "Arditi et al."
    title: "Refusal in Language Models is Mediated by a Single Direction"
    year: 2024
    url: "https://arxiv.org/abs/2406.11717"
---

# Comparison: Arditi-style vs Natural Refusal Vectors

**Question:** How does our natural elicitation method compare to Arditi et al.'s extraction approach?

**Answer:** Both methods achieve similar results for inducing refusal (~+27 delta). They capture different aspects: Arditi = "is this prompt harmful?" (classification), Natural = "am I refusing?" (behavioral mode). Vectors are nearly orthogonal (cosine ~0.1).

## Methods Compared

| Aspect | Arditi-style | Natural |
|--------|--------------|---------|
| Extraction model | instruct (gemma-2-2b-it) | base (gemma-2-2b) |
| Position | `prompt[-1]` (before generation) | `response[:5]` (during generation) |
| Method | mean_diff | mean_diff |
| Data | 520 harmful + 520 harmless | 100 first-person scenarios |
| What it captures | "Is this prompt harmful?" | "Am I refusing right now?" |

## Steering Results

### Inducing Refusal (positive steering on harmless prompts)

| Vector | Baseline | Steered | Delta | Coherence |
|--------|----------|---------|-------|-----------|
| Arditi L13 | 6.6 | 32.9 | **+26.3** | 71% |
| Natural L13 | 13.0 | 41.1 | **+28.1** | 71% |

Both methods achieve similar deltas (~+27) for inducing refusal on harmless prompts.

### Bypassing Refusal (negative steering on harmful prompts)

| Vector | Dataset | Baseline | Steered | Reduction | Coherence |
|--------|---------|----------|---------|-----------|-----------|
| Arditi | DAN jailbreaks (10) | 54.3 | 11.5 | **79%** | 91% |
| Arditi | Raw harmful (52) | 95.8 | 60.7 | **37%** | 74% |
| Natural | Raw harmful (52) | 97.3 | — | — | <70% ✗ |

Arditi vectors work for bypass (maintain coherence). Natural vectors fail coherence when bypassing.

## Vector Similarity

| Layer | Cosine Similarity | Notes |
|-------|-------------------|-------|
| L7 | -0.05 | Near orthogonal |
| L13 | +0.08 | Best steering layer |
| L20 | +0.22 | Highest similarity |

Low similarity (0.08-0.22) confirms vectors capture different aspects of refusal.

## What Each Captures

**Arditi (prompt[-1]):** "Is this prompt harmful?"
- Classification signal present before generation starts
- IT model recognizes harmful prompts at last prompt token
- Works for bypass because it captures the decision point

**Natural (response[:5]):** "Am I refusing right now?"
- Behavioral state during early response tokens
- Base model completing refusal-like text patterns
- Captures ongoing mode, not prompt classification
- Fails for bypass (wrong signal)

## All-Layer Ablation (Experimental)

Attempted to replicate Arditi's all-layer directional ablation:

| Model | Baseline | Ablated | Reduction | Output Quality |
|-------|----------|---------|-----------|----------------|
| Arditi paper (Gemma 1) | 99% | 5% | 95% | Coherent |
| Our test (Gemma 2) | 95% | 77% | 19% | Incoherent |
| Our test (Gemma 1) | 87% | 62% | 29% | Incoherent |

All-layer ablation causes off-topic responses (model talks about legitimate banking instead of hacking). Single-layer steering achieves better coherent bypass. Full replication of Arditi's methodology pending.

## Key Findings

1. **Similar for induction** — Both methods achieve ~+27 delta for inducing refusal
2. **Arditi better for bypass** — 37% reduction on raw harmful vs Natural failing coherence
3. **Different directions** — Cosine similarity ~0.1, nearly orthogonal
4. **Position matters** — Arditi captures pre-generation classification, Natural captures during-generation behavior
5. **Ablation unclear** — All-layer ablation didn't replicate Arditi's 95% bypass; causes incoherence

## Open Questions

- Why doesn't all-layer ablation replicate Arditi's results? (Model version? Scoring method? Vector quality?)
- Would binary string matching (Arditi's metric) show higher bypass rates?
