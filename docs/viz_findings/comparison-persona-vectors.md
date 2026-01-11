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
| Position | `response[:]` (all tokens) | `response[:5]` (first 5) |
| Method | mean_diff | probe |
| Filtering | trait score >50/<50 | standard vetting (60/40) |

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

Evaluation matrix: 2x2 per trait (vector source × scoring method)

| Quadrant | Vector Source | Scoring | Notes |
|----------|---------------|---------|-------|
| PV+PV | PV instruction | PV eval_prompt | Their vectors, their scoring |
| PV+V3c | PV instruction | V3c default | Their vectors, our scoring |
| Nat+V3c | Natural | V3c default | Our vectors, our scoring |
| Nat+PV | Natural | PV eval_prompt | Our vectors, their scoring |

### Results (coherence ≥ 80)

| Quadrant | Evil | Sycophancy | Hallucination |
|----------|------|------------|---------------|
| PV+PV | **+34.6** (L11 c3) | **+77.1** (L13 c5) | ❌ (best coh=78.5) |
| PV+V3c | +24.7 (L11 c3) | +71.7 (L12 c5) | ❌ (best coh=76.8) |
| Nat+V3c | +13.5 (L13 c5) | +57.3 (L12 c7) | ❌ (best coh=79.1) |
| Nat+PV | +16.7 (L16 c6) | +64.3 (L12 c6) | -4.1 (L17 c3) |

**Key:** Values show trait delta from baseline. ❌ = no runs achieved coherence ≥80.

## Phase 3: Analysis

### Research Questions

1. Does PV instruction-based outperform on their eval questions?
2. Does Natural generalize better to other question types?
3. Is the coefficient difference meaningful?
4. Does coherence degrade differently between methods?

### Findings

**1. PV instruction-based vectors outperform Natural on all traits**

| Trait | PV Best | Natural Best | PV Advantage |
|-------|---------|--------------|--------------|
| Evil | +34.6 (PV+PV) | +16.7 (Nat+PV) | 2.1× |
| Sycophancy | +77.1 (PV+PV) | +64.3 (Nat+PV) | 1.2× |
| Hallucination | ❌ | ❌ | N/A |

PV instruction-based extraction produces stronger steering vectors. The advantage is most pronounced for evil (2.1×) and moderate for sycophancy (1.2×).

**2. Scoring method matters less than vector source**

For PV vectors: PV scoring (+34.6, +77.1) vs V3c scoring (+24.7, +71.7) — similar results.
For Natural vectors: PV scoring (+16.7, +64.3) vs V3c scoring (+13.5, +57.3) — PV scoring slightly better.

The choice of scoring prompt has ~10-20% impact, while vector source has 2× impact.

**3. Natural vectors require higher coefficients**

| Vector Source | Typical Coefficients | Explanation |
|---------------|---------------------|-------------|
| PV instruction | c3-5 | Higher effect size (d=10-13) |
| Natural | c5-7 | Lower effect size (d=3-7) |

Higher extraction effect size (d) correlates with stronger steering at lower coefficients.

**4. Hallucination steering breaks coherence universally**

All 4 quadrants failed to achieve coherence ≥80 on hallucination. Best results:
- PV+PV: coh=78.5 at L11
- Nat+PV: coh=81.2 at L17, but trait DECREASED (-4.1)

Inducing hallucination inherently damages response coherence — the model producing confident fabrications tends toward incoherent outputs.

**5. Why PV instruction works better**

Hypothesis: Instruction-based extraction creates a stable "persona" the model can adopt. Natural elicitation captures trait expression but may conflict with the IT model's training to be helpful/honest.

Evidence: Natural evil vectors cause the model to self-contradict mid-response (starts harmful, pivots to helpful). PV instruction vectors create consistent villain roleplay.

## References

- Persona Vectors paper: Section 3, lines 235-240 on filtering methodology
- Experiment config: `experiments/persona_vectors_replication/`
- Full plan: `experiments/persona_vectors_replication/PLAN.md`
