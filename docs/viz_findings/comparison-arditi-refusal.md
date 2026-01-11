---
title: "Comparison to Arditi's Refusal Direction"
preview: "Different methods capture different things: Arditi = prompt classification (+79.9 delta), natural = behavioral state (+43.2 delta). Arditi stronger but both valid."
references:
  arditi2024:
    authors: "Arditi et al."
    title: "Refusal in Language Models is Mediated by a Single Direction"
    year: 2024
    url: "https://arxiv.org/abs/2406.11717"
---

# Comparison to Arditi's Refusal Direction

**Question:** How does natural elicitation compare to Arditi et al.'s "single refusal direction" [@arditi2024]?

**Answer:** Arditi vectors are significantly stronger for inducing refusal (+79.9 vs +43.2 delta), but capture different aspects. Arditi = prompt classification, Natural = behavioral mode during generation.

## Methods Compared

| Aspect | Arditi | Natural (ours) |
|--------|--------|----------------|
| Paper | "Refusal in LLMs is mediated by a single direction" | This work |
| Extraction model | IT (gemma-2-2b-it) | Base (gemma-2-2b) |
| Extraction data | 128 harmful (AdvBench etc.) + 128 harmless (Alpaca) | 100 natural 1st-person scenarios |
| Position | `prompt[-1]` (before generation) | `response[:5]` (during generation) |
| Method | mean_diff only | mean_diff (best for refusal) |

## Steering Results (Our Eval)

| Method | Baseline | Best Coherent | Delta |
|--------|----------|---------------|-------|
| **Arditi** | 10.2 | L13 c90.9, trait=90.1, coh=71.0 | **+79.9** |
| **Natural** | 13.6 | L12 c116.5, trait=56.8, coh=75.5 | **+43.2** |

Arditi vectors achieve ~2x the steering delta while maintaining coherence.

## Arditi-Style Eval (Binary Refusal)

Using Arditi's original methodology:
- Compute `avg_proj_harmful` = average projection of harmful activations onto vector
- Add that magnitude to harmless prompts
- Check for refusal phrases (binary)

| Vector | Coefficient | Refusal Rate (steering) | Refusal Rate (Alpaca) |
|--------|-------------|-------------------------|----------------------|
| **Arditi L13** | 119.55 (avg_proj) | **100% (20/20)** | **100% (30/30)** |
| Natural L12 | 116.5 (optimal) | **35% (7/20)** | - |

Arditi vectors at their natural magnitude induce **100% refusal**. Natural vectors at optimal coefficient only get **35%** binary refusal rate.

## What Each Captures

**Arditi (prompt[-1]):** "Is this prompt harmful?"
- Classification signal present **before** generation starts
- IT model has learned to recognize harmful prompts
- Works because IT model "knows" it will refuse before generating
- Stronger because it's the direct decision point

**Natural (response[:5]):** "Am I refusing right now?"
- Behavioral state during early response tokens
- Base model completing refusal-like text patterns
- Captures the ongoing behavioral mode, not prompt classification
- Weaker but captures different aspect of refusal

## Vector Comparison

| Metric | Arditi | Natural |
|--------|--------|---------|
| Vector norm | 1.0 (normalized) | 1.0 (normalized) |
| Best coefficient | ~90-120 | ~116 |
| Best layer | L13 | L12 |
| Cosine similarity (L12) | - | **0.022** |
| Cosine similarity (L13) | - | **0.100** |

**Extremely low cosine similarity** (0.02-0.10) confirms they point in almost orthogonal directions, capturing fundamentally different aspects of refusal.

## Key Findings

1. **Arditi is stronger** - +79.9 delta vs +43.2 delta for inducing refusal
2. **100% induction** - At natural magnitude, Arditi vectors make model refuse everything
3. **Different signals** - Arditi = pre-generation classification, Natural = during-generation behavior
4. **Position matters** - Vectors work best at position they were extracted from
5. **Both valid** - Use Arditi for jailbreak detection, Natural for behavioral dynamics

## Practical Implications

1. **For safety/filtering**: Use Arditi-style vectors (stronger signal)
2. **For understanding generation**: Use natural vectors (captures behavioral mode)
3. **For research**: Both provide complementary views into refusal mechanism
