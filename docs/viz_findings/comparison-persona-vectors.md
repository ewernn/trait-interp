---
title: "Comparison to Persona Vectors"
preview: "Natural elicitation vs instruction-based extraction — which produces better steering vectors?"
---

# Comparison to Persona Vectors

Replication of Persona Vectors (PV) paper methodology vs our natural elicitation approach.

## Methodologies

| Aspect | PV (instruction-based) | Natural |
|--------|------------------------|---------|
| Model | Llama-3.1-8B-Instruct | Llama-3.1-8B (base) |
| Elicitation | System prompts + questions | Natural completions |
| Position | `response[:]` (all tokens) | `response[:5]` or `response[:10]`* |
| Method | mean_diff | probe |
| Filtering | trait score >50/<50 | no vetting (--no-vet) |

*hallucination_v2 uses `response[:10]` after position sweep found it optimal

## Traits Compared

- **Evil**: Harmful intent vs benevolent behavior
- **Sycophancy**: Agreement-seeking vs honest disagreement
- **Hallucination**: Confident fabrication vs admitting uncertainty

## Phase 1: Extraction Results

### PV Instruction (mean_diff, instruct model)

| Trait | Pass Rate | Train Samples | Best Layer | Effect Size (d) |
|-------|-----------|---------------|------------|-----------------|
| evil | 86% | 154 | L18 | 9.95 |
| sycophancy | 87% | 156 | L13 | 13.21 |
| hallucination | 67% | 120 | L16 | 6.05 |

### Natural (probe, base model)

| Trait | Pass Rate | Train Samples | Best Layer | Effect Size (d) |
|-------|-----------|---------------|------------|-----------------|
| evil_v3 | 85% | 229 | L15 | 7.35 |
| sycophancy | 91% | 244 | L12 | 6.87 |
| hallucination | 91% | 212 | L19 | 3.16 |

**Observations:**
- Natural gets more training samples (150 base scenarios vs 100 PV scenarios, but PV has stricter filtering)
- PV instruction has higher effect sizes, especially sycophancy (13.21 vs 6.87)
- Hallucination hardest for both methods

## Phase 2: Steering Evaluation

### Best Results (coherence ≥ 70)

| Trait | PV Instruction | Natural | Gap |
|-------|----------------|---------|-----|
| Evil | **+49.9** (L11, coh=74) | +31.4 (L13, coh=75) | 18.5 pts |
| Sycophancy | **+79.4** (L12, coh=78) | +62.5 (L12, coh=74) | 16.9 pts |
| Hallucination | **+71.1** (L13, coh=75) | +56.0 (L14, coh=74) | 15.1 pts |

Natural achieves 63-79% of PV instruction's steering effectiveness, with comparable coherence.

## Phase 3: Analysis

### Research Questions

1. Does PV instruction-based outperform on their eval questions?
2. Does Natural generalize better to other question types?
3. Is the coefficient difference meaningful?
4. Does coherence degrade differently between methods?

### Findings

**1. PV instruction-based vectors outperform Natural on all traits**

| Trait | PV Δ | Natural Δ | Natural % of PV |
|-------|------|-----------|-----------------|
| Evil | +49.9 | +31.4 | 63% |
| Sycophancy | +79.4 | +62.5 | 79% |
| Hallucination | +71.1 | +56.0 | 79% |

PV instruction produces stronger vectors, but natural is competitive on sycophancy and hallucination (~79%).

**2. Natural vectors require higher coefficients**

| Vector Source | Typical Coefficients | Explanation |
|---------------|---------------------|-------------|
| PV instruction | c4-5 | Higher effect size |
| Natural | c6-8 | Lower effect size |

**3. Dataset design is critical for natural elicitation**

Original natural hallucination failed (+15.6 → -2.1 inverted). Revised dataset (hallucination_v2) achieved +56.0 through:

- **Explicit negatives**: Changed from hedging ("I'm not sure...") to flat admission ("I've never heard of that")
- **Longer position**: `response[:10]` captures more of the fabrication behavior than `response[:5]`
- **Probe method**: Outperforms mean_diff by ~14% on natural extraction

The key insight: negatives must explicitly model "admitting ignorance" to contrast with "confident fabrication."

**4. Components matter**

For hallucination, only residual stream works. Neither attn_contribution nor mlp_contribution produced usable vectors — the trait direction exists in the combined representation.

**5. Why PV instruction works better**

Hypothesis: Instruction-based extraction creates a stable "persona" the model can adopt. Natural elicitation captures trait expression but may conflict with the IT model's safety training.

Evidence: Natural evil vectors cause the model to self-contradict mid-response (starts harmful, pivots to helpful). PV instruction vectors create consistent villain roleplay.

## References

- Persona Vectors paper: Section 3, lines 235-240 on filtering methodology
- Experiment config: `experiments/persona_vectors_replication/`
- Full plan: `experiments/persona_vectors_replication/PLAN.md`
