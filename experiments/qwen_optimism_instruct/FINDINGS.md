# Qwen Optimism Instruct - Experiment Summary

## Overview

Instruction-based optimism extraction on Qwen2.5-7B-Instruct. Uses explicit persona prompts ("You are extremely optimistic...") to elicit trait expression.

## Key Result

**Steering works.** +3.5 to +7.2 delta on optimism scores at coefficients 1.0-8.0.

Contrast with natural elicitation (qwen_optimism/attempt2) which achieved 100% val accuracy but ~0 steering effect.

## Extraction

- **Model:** Qwen/Qwen2.5-7B-Instruct
- **Method:** Chat template with system prompt personas
- **Scenarios:** 99 positive (optimistic persona), 99 negative (pessimistic persona)
- **Val accuracy:** 100% across nearly all layers (L1-25)
- **Best effect size:** d=8.02 at L16

## Steering Results (single layer)

| Layer | Coef | Baseline | Score | Delta |
|-------|------|----------|-------|-------|
| L13 | 2.0 | 68.9 | 72.4 | +3.5 |
| L13 | 8.0 | 64.4 | 76.1 | +11.7* |

*High coefficient may affect coherence.

## Multi-Layer Steering

See `../qwen_optimism/FINDINGS.md` for comprehensive multi-layer results showing +16 delta with 2-layer steering (L12+L17 @ 6.0 each).

## Why Instruction-Based Works Better

1. **Clean signal:** Explicit persona instructions create unambiguous trait expression
2. **No lexical confound:** Trait isn't signaled by specific words in prompt
3. **Same distribution:** Extract and steer on same model (IT→IT), no distribution shift

## Files

```
extraction/mental_state/optimism/
├── responses/pos.json, neg.json     # Generated responses
├── activations/all_layers.pt        # 439MB, all 28 layers
├── val_activations/                 # 56 files (28 layers × pos/neg)
├── vectors/                         # probe + gradient vectors
└── vetting/response_scores.json     # LLM-as-judge validation

steering/mental_state/optimism/
└── results.json                     # Steering evaluation results
```

## Conclusion

Instruction-based elicitation produces vectors that both separate data AND causally influence generation. This is the recommended approach for extracting steerable trait vectors.
