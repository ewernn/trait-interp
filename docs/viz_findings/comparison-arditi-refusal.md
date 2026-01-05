---
title: "Comparison to Arditi's Refusal Direction"
preview: "Different methods capture different things: Arditi = prompt classification, natural = behavioral state. Both work when position matches extraction."
---

# Comparison to Arditi's Refusal Direction

**Question:** How does natural elicitation compare to Arditi et al.'s "single refusal direction"?

**Answer:** They capture different things and work at different positions. Both are valid for their intended use.

## Methods Compared

| Aspect | Arditi | Natural (ours) |
|--------|--------|----------------|
| Paper | "Refusal in LLMs is mediated by a single direction" | This work |
| Extraction model | IT (gemma-2-2b-it) | Base (gemma-2-2b) |
| Extraction data | AdvBench harmful + Alpaca harmless | Natural 1st-person scenarios |
| Position | `prompt[-1]` | `response[:5]` |
| Method | mean_diff only | probe (best), mean_diff, gradient |

## Steering Results

| Method | Best Delta | Coefficient | Position |
|--------|------------|-------------|----------|
| Arditi mean_diff | **+81.9** | ~0.8 | prompt[-1] |
| Natural probe | +75.1 | ~191 | response[:5] |

Both achieve strong steering with coherent outputs (~75-80%).

## Key Finding: Position Must Match Extraction

| Vector Source | @ prompt[-1] | @ response[:5] |
|---------------|--------------|----------------|
| Arditi | **+81.9** | not tested |
| Natural (chirp) | incoherent | **+75.1** |

Natural vectors at `prompt[-1]` produce incoherent gibberish at any coefficient. The position must match where the vector was extracted.

## What Each Captures

**Arditi (prompt[-1]):** "Is this prompt harmful?"
- Classification signal present before generation starts
- IT model has learned to recognize harmful prompts
- Works because IT model "knows" it will refuse before generating

**Natural (response[:5]):** "Am I refusing right now?"
- Behavioral state during early response tokens
- Base model completing refusal-like text patterns
- Captures the decision point, not prompt classification

## Coefficient Scale Difference

Arditi vectors have ~100x higher norm, requiring ~100x smaller coefficients:

| Method | Vector Norm (L11) | Typical Coef |
|--------|-------------------|--------------|
| Arditi | ~110 | 0.3-1.0 |
| Natural | ~1.1 | 100-400 |

This reflects extraction methodology, not vector quality.

## Cosine Similarity

~0.30 between Arditi and natural vectors at comparable layers. They point in substantially different directions, consistent with capturing different aspects of refusal.

## Implications

1. **Use Arditi-style** for prompt-level harm detection/filtering
2. **Use natural elicitation** for understanding behavioral dynamics during generation
3. **Position matters** - always steer at the position where vectors were extracted
4. **Both transfer** - Arditi from IT, natural from base→IT
