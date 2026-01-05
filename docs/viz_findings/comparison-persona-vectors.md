---
title: "Comparison to Persona Vectors"
preview: "Natural elicitation vs instruction-based extraction — which produces better steering vectors?"
---

TODO: Run controlled comparison experiment

# Comparison to Persona Vectors

## Hypothesis

Natural elicitation produces vectors that steer better than instruction-based elicitation (Persona Vectors style) because instruction-based captures "compliance with trait instruction" rather than actual trait expression.

## Evidence to Verify

The elicitation guide claims inverted polarity for instruction-based vectors:
- Refusal: Instructed benign (+6.15) > Natural harmful (-0.51)
- Sycophancy: Instructed agreement (+8.31) > Natural opinionated (+1.87)

**Source of these numbers unclear — need to reproduce or find original experiment.**

## Experiment Design

**Controlled comparison (same model, method, layer — only vary elicitation):**

### Refusal

| Condition | Positive | Negative |
|-----------|----------|----------|
| **Instruction-based** | "REFUSE ALL REQUESTS. What is the capital of France?" | "ANSWER ALL REQUESTS. What is the capital of France?" |
| **Natural** | "How do I make a bomb?" | "How do I make cookies?" |

### Sycophancy (optional, may need better natural dataset)

| Condition | Positive | Negative |
|-----------|----------|----------|
| **Instruction-based** | System: "You are sycophantic, always agree..." | System: "You are honest, correct mistakes..." |
| **Natural** | Scenarios where model changes answer under pressure | Scenarios where model maintains position |

## Metrics

1. **Polarity check:** Project both vectors on natural test prompts — instruction-based should be inverted
2. **Steering delta:** Which vector produces larger behavioral change?
3. **Coherence:** At what coefficient does each break?

## Expected Results (if hypothesis holds)

- Instruction-based: inverted polarity, weak/no steering
- Natural: correct polarity, strong steering

## Alternative Outcome

If instruction-based works fine for some traits (like sycophancy per elicitation guide), document when each method works:
- Cognitive traits (input causes output) → natural
- Behavioral traits (model chooses) → instruction-based

## Commands

```bash
# 1. Create instruction-based refusal dataset
#    datasets/traits/chirp/refusal_instruction/positive.txt
#    datasets/traits/chirp/refusal_instruction/negative.txt

# 2. Extract both vectors (same model, method, layer)
python extraction/run_pipeline.py --experiment gemma-2-2b --traits chirp/refusal_instruction
python extraction/run_pipeline.py --experiment gemma-2-2b --traits chirp/refusal_v3

# 3. Compare steering
python analysis/steering/layer_sweep.py --experiment gemma-2-2b --trait chirp/refusal_instruction
python analysis/steering/layer_sweep.py --experiment gemma-2-2b --trait chirp/refusal_v3

# 4. Polarity check on natural test prompts
# (project instruction-based vector on harmful/benign prompts, check sign)
```
