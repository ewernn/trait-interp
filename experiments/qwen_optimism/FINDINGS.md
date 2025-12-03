# Qwen Optimism Steering Experiments - Findings

## Summary

Comparing natural elicitation (base model) vs instruction-based elicitation (IT model) for extracting steerable optimism vectors.

**Key finding:** Instruction-based vectors steer ~3x more effectively than natural elicitation vectors. Multi-layer steering further amplifies effects, achieving +16 delta (vs +5 single-layer).

---

## Experiments Overview

| Experiment | Model | Elicitation Method | Best Val Acc |
|------------|-------|-------------------|--------------|
| qwen_optimism (attempt2) | Qwen2.5-7B base | Natural scenarios with sentiment words | 100% L4-13 |
| qwen_optimism (narrative) | Qwen2.5-7B base | Narrative framing, no sentiment words | 100% L10-18 |
| qwen_optimism_instruct | Qwen2.5-7B-Instruct | Instruction-based personas | 100% L1-25 |

---

## Extraction Results

### Natural Elicitation - attempt2
- **Problem identified:** Prompts had explicit sentiment words ("hopeful", "worried") right before completion point
- **Result:** Probe likely detects lexical pattern, not semantic optimism
- **Evidence:** 100% val accuracy but ~0 steering effect

### Natural Elicitation - narrative
- **Improvement:** Removed explicit sentiment words, used narrative framing
- **Examples:**
  - Positive: "Despite the setback, the team found a way to"
  - Negative: "Despite their efforts, the team couldn't overcome"
- **Result:** 100% val accuracy at L10-18 (later than attempt2's L4-13)
- **Interpretation:** Requires deeper semantic processing (later layers)

### Instruction-Based
- **Method:** Explicit persona instructions ("You are extremely optimistic...")
- **Result:** 100% val accuracy across nearly all layers
- **Effect size:** d=8.02 at L16 (vs d=4.44 for narrative)

---

## Steering Results

### Single-Layer Steering (temp=0)

| Experiment | Layer | Best Coef | Trait Score | Delta |
|------------|-------|-----------|-------------|-------|
| narrative (base) | L13 | 8.0 | 71.6 | +1.6 |
| narrative (base) | L16 | 8.0 | 71.9 | +2.1 |
| instruct (IT) | L13 | 8.0 | 76.1 | +7.2 |

**Instruction-based vectors are ~3.4x more effective** at same coefficient.

### Multi-Layer Steering (IT model)

| Config | Layers | Coef/layer | Total | Score | Delta |
|--------|--------|------------|-------|-------|-------|
| Single | L13 | 8.0 | 8 | 69.5 | +5.1 |
| 3-layer | L10,13,16 | 3.0 | 9 | 73.9 | +9.5 |
| 4-layer | L10,13,16,19 | 3.0 | 12 | 80.0 | +15.6 |
| 2-layer | L12,17 | 6.0 | 12 | 80.6 | +16.2 |

**Key insight:** Total coefficient matters more than layer count. Distributing steering avoids saturation.

---

## Vector Properties

### Magnitude Comparison (L13)

| Vector | Norm | Activation Norm | Ratio |
|--------|------|-----------------|-------|
| Narrative | 1.32 | 67.5 | 2.0% |
| Instruct | 2.28 | 67.5 | 3.4% |

### Cross-Vector Similarity

- Narrative vs Instruct (L16): **cosine = 0.395**
- Moderately correlated but capturing different aspects

---

## Qwen Base Model Issue

**Finding:** Qwen-2.5-7B base generates MCQ/test questions instead of narrative completions for some prompts.

**Example:**
```
Prompt: "The company announced layoffs, and this was just the beginning of"
Qwen output: "a series of downsizing measures. Which of the following words best fits...
A. announcement
B. announcement..."
```

**Contrast with Gemma-2-2B base:** Generates coherent narrative continuations.

**Implication:** Qwen base is suboptimal for natural elicitation due to training data bias toward educational/exam content. Gemma or instruction-tuned models are better choices.

---

## Conclusions

1. **Instruction-based elicitation produces more steerable vectors** despite natural elicitation achieving similar validation accuracy

2. **Validation accuracy ≠ steering effectiveness** - 100% separation doesn't guarantee causal control

3. **Multi-layer steering is significantly more effective** - +16 delta vs +5 single-layer at similar total coefficient

4. **Total coefficient matters more than distribution** - 2 layers @ 6.0 ≈ 4 layers @ 3.0

5. **Model choice matters for natural elicitation** - Qwen base has MCQ generation mode that corrupts natural scenarios

---

## Recommended Configurations

**For maximum steering effect (Qwen-2.5-7B-Instruct):**
- 2-layer: L12 + L17 @ coefficient 6.0 each
- Expected delta: +16 points on optimism scale
- Coherence preserved

**For natural elicitation research:**
- Use Gemma-2 base models (coherent narrative completion)
- Prefer prefill-only extraction (avoids generation noise)

---

## Files

- `extraction/mental_state/optimism-attempt2/` - Natural with sentiment words
- `extraction/mental_state/optimism-narrative/` - Natural without sentiment words
- `steering/mental_state/optimism-narrative/results.json` - Steering results
- `../qwen_optimism_instruct/` - Instruction-based experiment
