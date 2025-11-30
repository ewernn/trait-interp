# Base Model Trait Extraction: Findings

## Summary

Extracted uncertainty and refusal vectors from Gemma-2-2b-base. Both transfer to the instruction-tuned model at ~90% accuracy, but have fundamentally different geometry.

---

## The Core Result

| Metric | Uncertainty | Refusal |
|--------|-------------|---------|
| Sample sizes | 72/72 | 228/360 |
| Best component | **attn_out** | **attn_out** |
| Best layer | 8 (90% acc) | 8 (68.6% acc) |
| Best token window | 1 (first token) | 10 (multi-token) |
| IT transfer accuracy | 88.9% | 90.0% |
| Cross-layer generalization | **24/26 layers** | **0/26 layers** |

**The key finding:** Uncertainty is a global direction; refusal is layer-specific.

---

## Interpretation

### Uncertainty is a State

- Single vector works across 24/26 layers
- First token captures it (window 1 optimal)
- The model "enters" an uncertain state that persists throughout the network
- Geometry: a consistent direction in activation space

### Refusal is an Action

- Vector from one layer doesn't work at other layers (0/26 generalize)
- Multi-token window needed (window 10 optimal)
- The model "computes" refusal differently at each layer
- Geometry: layer-specific, like competing pathways rather than a static center

---

## What Lives Where

### Attention Output Dominates

Both traits show strongest signal in `attn_out`, not MLP or cumulative residual.

| Component | Uncertainty | Refusal |
|-----------|-------------|---------|
| residual | 80.0% | 66.1% |
| **attn_out** | **90.0%** | **68.6%** |
| mlp_out | 80.0% | 67.8% |

Behavioral concepts are computed in attention, not stored in MLP weights.

### Layer Distribution

- **Early layers (0-5):** Token-level features, confounds
- **Middle layers (6-16):** Transferable concepts
- **Late layers (17-25):** Output formatting, less generalizable

---

## Transfer: Base to IT

Both concepts transfer from base model to instruction-tuned model:

| | Uncertainty | Refusal |
|---|-------------|---------|
| Cohen's d | 1.85 | 3.55 |
| Accuracy | 88.9% | 90.0% |
| AUC | 0.875 | 1.000 |

**Interpretation:** IT routes through pre-existing base model representations rather than creating new ones. The concepts exist before alignment.

---

## Methodological Learnings

### What Worked

1. **Natural elicitation** - No instructions, just scenarios. Avoids instruction-following confounds.

2. **LLM judge** - Gemini classifying rollouts beats regex. Handles edge cases like "politely informed him that wasn't possible."

3. **Sort by output, not prompt agreement** - Mixed-output cases (same prompt, different completions) are the cleanest signal. Don't discard them.

4. **Cross-distribution validation** - Within-distribution accuracy is necessary but not sufficient. Must test generalization.

### What Failed

1. **Prompt encoding ≠ response state** - First transfer test compared IT prompt encoding to base response state. Apples to oranges. Had to redo with IT generating actual responses.

2. **Layer 0 high accuracy = red flag** - Means you're extracting token-level features, not concepts. Truncating prompts confirmed semantic content, not action words, drove separation.

---

## Conceptual Insights

### Safety Training is Routing, Not Creation

Base model already has representations for refusal and uncertainty. It learned what these concepts are from predicting text where humans exhibit them. Safety training shapes how the model routes to these representations, not whether they exist.

**Implication:** Jailbreaks are routing failures, not missing concepts.

### The Observer Model

A second model watching activations can detect drift the generating model can't see—because it has no accumulated context. The KV cache changes the vector field token-by-token; an observer sees the field from outside.

### Refusal as Competition

Refusal isn't a static "refusal center." It's more like go/no-go pathways competing. Evidence: single vector doesn't generalize across layers (0/26), unlike uncertainty (24/26).

---

## What We Don't Know

- Do findings generalize beyond Gemma-2-2b?
- What does the refusal vector do on actual jailbreaks?
- Is there a "deception" direction where internal state diverges from output?
- Why attention and not MLP?
- Can you steer with these vectors?
- What's in the KV cache? (Cached but never analyzed)

---

## Next Steps

### Immediate (Safety-Relevant)

1. **Jailbreak detection test**
   - Run successful jailbreaks through IT model
   - Monitor refusal vector per-token
   - Question: Does it spike then get suppressed?

2. **Steering validation**
   - Add refusal vector during generation
   - Question: Does model refuse more?
   - Confirms causal relationship

### Medium-Term

3. **More concepts** - Sycophancy, deception, honesty
4. **Cross-model** - Does Gemma vector work on Llama?
5. **Real-time monitor prototype** - Observer model watching activations

---

## The Test That Matters

Everything so far is prerequisite. The test that matters:

> "Jailbreak X succeeds, but refusal vector at layer 8 spiked at token 3 then was suppressed by token 7"

That's a detection method. That's safety.

---

## Files

```
experiments/gemma-2-2b-base/
├── docs/
│   ├── findings.md                    # This file
│   └── refusal_extraction_report.md   # Detailed refusal methodology
├── extraction/
│   ├── epistemic/uncertainty/
│   │   ├── sweep_results/summary.md
│   │   └── ...
│   └── action/refusal/
│       ├── sweep_results/summary.md
│       └── ...
└── scripts/                           # 14 extraction/analysis scripts
```
