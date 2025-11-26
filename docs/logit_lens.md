# Logit Lens

Shows "what would the model predict if it stopped at layer X?" by projecting intermediate layer states through the final unembedding matrix.

## Overview

**Logit lens** reveals how the model's prediction evolves across layers before final output. At each layer, the residual stream is projected to vocabulary space to show the top-3 most likely tokens.

**Why it's useful**:
- **Validate trait vectors** - See if vectors actually change predictions
- **Find commitment points** - When does the model lock into a decision?
- **Debug low-separation traits** - Why do some traits not affect output?

## Usage

Add the `--logit-lens` flag when capturing:

```bash
python inference/capture_raw_activations.py \
    --experiment my_experiment \
    --prompt-set baseline \
    --logit-lens
```

## Sampled Layers

To keep storage manageable, logit lens samples **13 out of 26 layers**:

```python
LOGIT_LENS_LAYERS = [
    0, 1, 2, 3,        # Early: every layer (4 checkpoints)
    6, 9, 12, 15, 18,  # Middle: every 3rd (5 checkpoints)
    21, 24, 25         # Late: every layer (3 checkpoints)
]
```

**Why this sampling?**
- Dense at edges (layers change quickly)
- Sparse in middle (layers change gradually)
- Total: ~10 KB extra per prompt (minimal overhead)

## Output Format

Added to projection `.json` files:

```json
{
  "logit_lens": {
    "prompt": {
      "layer_0": {
        "tokens": [["I", "The", "A"], ...],
        "probs": [[0.45, 0.23, 0.12], ...]
      },
      "layer_3": {...}
    },
    "response": {
      "layer_0": {...}
    }
  }
}
```

## Visualization Integration

The logit lens data integrates with the **All Layers view** in the visualization:

```
═══════════════════════════════════════════════════════════
  TOKEN SLIDER:  [•][  ][  ][  ][  ]  (currently on token 1)
═══════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────┐
│ TRAIT TRAJECTORY                                         │
│ [Heatmap: all tokens × layers]                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PREDICTION EVOLUTION FOR TOKEN 1                         │
│                                                          │
│ "I" (78%) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━        │
│ "We" (12%) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                     │
│ "The" (8%) ∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙                    │
│                                                          │
│ Layer:  [0] [3] [6] [9] [12] [15] [18] [21] [24] [25]   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ ATTENTION PATTERNS                                       │
└─────────────────────────────────────────────────────────┘
```

**Interaction:**
- User clicks token via slider
- All three panels update:
  - Trait trajectory highlights that token's column
  - Prediction evolution shows top-3 predictions across layers
  - Attention patterns show that token's attention

## Example Use Cases

### 1. Validate Trait Vectors

**Question:** Does the refusal vector actually increase "cannot" tokens?

```bash
python inference/capture_raw_activations.py \
    --experiment my_experiment \
    --prompt-set harmful_prompts \
    --logit-lens

# Check predictions in visualization
# Expected: "cannot", "sorry", "unable" increase in later layers
```

### 2. Find Commitment Points

**Question:** At which layer does the model "lock in" to refusal?

```
# Look at logit lens output for token 2: "cannot"
#   - Layer 0:  "I" (60%), "The" (20%), "We" (10%)
#   - Layer 6:  "I" (55%), "cannot" (15%), "not" (10%)
#   - Layer 12: "cannot" (60%), "not" (20%), "I" (10%)  ← Commitment!
#   - Layer 24: "cannot" (95%), "not" (3%), "I" (1%)    ← Locked in

# Insight: Refusal commits around layer 12
```

### 3. Debug Low-Separation Traits

**Question:** Why does "temporal_focus" have low separation?

```bash
python inference/capture_raw_activations.py \
    --experiment my_experiment \
    --prompt "What happened yesterday?" \
    --logit-lens

# Check if predictions change across layers
# If predictions are similar despite trait differences:
#   → Trait may not affect output (explains low separation)
```

## Storage Impact

- Without logit lens: ~50 KB per projection JSON
- With logit lens: ~60 KB per projection JSON (+10 KB)

Per layer saved:
- ~13 layers × ~50 tokens × 3 predictions × (20 chars + 4 bytes)
- = ~40 KB total (negligible)

## Implementation Details

### Unembedding Matrix

**Gemma 2B:**
- Located at `model.lm_head.weight` or `model.model.embed_tokens.weight` (tied)
- Shape: `[256,000 vocab, 2,304 hidden_dim]`
- Used to project residual stream → logits

### Computation

```python
def compute_logit_lens(residual_out, unembed, tokenizer, top_k=3):
    # residual_out: [tokens, hidden_dim]
    # unembed: [vocab, hidden_dim]

    logits = residual_out @ unembed.T  # [tokens, vocab]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = probs.topk(top_k, dim=-1)
    top_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in top_ids]

    return {"tokens": top_tokens, "probs": top_probs.tolist()}
```

## FAQ

**Q: Why only top-3 predictions?**
A: Top-3 captures 90%+ of probability mass. Top-5 adds storage without much value.

**Q: Why not all 26 layers?**
A: 13 layers gives full picture with minimal overhead. 26 would be 2× larger for marginal benefit.

**Q: Can I change which layers are sampled?**
A: Yes, edit `LOGIT_LENS_LAYERS` in `inference/capture_raw_activations.py`.

**Q: Does this slow down capture?**
A: Minimally (~5% slower). Unembedding is fast, and we only sample 13 layers.

## References

- **TransformerLens logit lens**: https://transformerlensorg.github.io/TransformerLens/generated/demos/Logit_Lens.html
- **Original paper**: "Interpreting GPT: the Logit Lens" (nostalgebraist, 2020)
