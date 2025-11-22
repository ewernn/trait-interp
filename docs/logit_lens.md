# Logit Lens

Shows "what would the model predict if it stopped at layer X?" by projecting intermediate layer states through the final unembedding matrix.

## Overview

**Logit lens** reveals how the model's prediction evolves across layers before final output. At each layer, the residual stream is projected to vocabulary space to show the top-3 most likely tokens.

**Why it's useful**:
- **Validate trait vectors** - See if vectors actually change predictions
- **Find commitment points** - When does the model lock into a decision?
- **Debug low-separation traits** - Why do some traits not affect output?

## Current Status

The `--save-logits` flag for `capture_layers.py` is not yet implemented. The visualization (All Layers view) supports displaying logit lens data when present in projection files, but the capture script needs to be extended to generate this data.

## Concept

**Sampled layers** (when implemented): 13 out of 26 layers for efficiency:

```python
LOGIT_LENS_LAYERS = [
    0, 1, 2, 3,        # Early: every layer (4 checkpoints)
    6, 9, 12, 15, 18,  # Middle: every 3rd (5 checkpoints)
    21, 24, 25         # Late: every layer (3 checkpoints)
]
```

**Why this sampling?**
- Dense at edges (early/late layers change rapidly)
- Sparse in middle (middle layers change gradually)
- Total: 13 samples instead of 26 (half the storage, same insights)

## References

- **TransformerLens logit lens**: https://transformerlensorg.github.io/TransformerLens/generated/demos/Logit_Lens.html
- **Original paper**: "Interpreting GPT: the Logit Lens" (nostalgebraist, 2020)
