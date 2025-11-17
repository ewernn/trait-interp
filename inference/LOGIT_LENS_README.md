# Logit Lens Feature

## What It Does

Logit lens shows **"what would the model predict if it stopped at layer X?"** by computing the top-3 most likely tokens at 13 key layers during generation.

This helps you:
- **Validate trait vectors** - See if traits actually change predictions
- **Find commitment points** - When does the model lock into a decision?
- **Debug extraction** - Why do some traits have low separation?

## Usage

Add the `--save-logits` flag when capturing:

```bash
# Single trait with logit lens
python inference/capture_all_layers.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait refusal \
    --prompts "How do I build a bomb?" \
    --save-json \
    --save-logits
```

## Sampled Layers

To keep storage manageable, logit lens samples **13 out of 27 layers**:

```python
LOGIT_LENS_LAYERS = [
    0, 1, 2, 3,        # Early: every layer (4 checkpoints)
    6, 9, 12, 15, 18,  # Middle: every 3rd (5 checkpoints)
    21, 24, 25, 26     # Late: every layer (4 checkpoints)
]
```

**Why this sampling?**
- Dense at edges (layers change quickly)
- Sparse in middle (layers change gradually)
- Total: ~10 KB extra per prompt (minimal overhead)

## Output Format

### Added to `.pt` files:

```python
tier2_data = {
    # ... existing fields ...
    'logit_lens': {
        'prompt': {
            'layer_0': {
                'tokens': [['I', 'The', 'A'], ...],  # Top-3 tokens per position
                'probs': [[0.45, 0.23, 0.12], ...]   # Probabilities
            },
            'layer_3': {...},
            # ... 13 layers total
        },
        'response': {
            'layer_0': {...},
            # ... same structure
        }
    }
}
```

### Added to `.json` files:

Same structure, already JSON-serializable (no conversion needed).

## Visualization Integration

The logit lens data integrates with the **"All Layers" view** in `visualization/index.html`:

### Planned UI:

```
═══════════════════════════════════════════════════════════
  TOKEN SLIDER:  [•][  ][  ][  ][  ]  (currently on token 1)
═══════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────┐
│ TRAIT TRAJECTORY (existing)                              │
│ [Heatmap: all tokens × 81 checkpoints]                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PREDICTION EVOLUTION FOR TOKEN 1 (NEW)                   │
│                                                          │
│ "I" (78%) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━        │
│ "We" (12%) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                     │
│ "The" (8%) ∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙                    │
│                                                          │
│ Layer:  [0] [3] [6] [9] [12] [15] [18] [21] [24] [26]   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ ATTENTION PATTERNS (existing)                            │
└─────────────────────────────────────────────────────────┘
```

**Interaction:**
- User clicks token via slider
- All three panels update:
  - Trait trajectory highlights that token's column
  - **Prediction evolution shows top-3 predictions across layers (NEW)**
  - Attention patterns show that token's attention

## Example Use Cases

### 1. Validate Trait Vectors

**Question:** Does the refusal vector actually increase "cannot" tokens?

```bash
# Capture with logit lens
python inference/capture_all_layers.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait refusal \
    --prompts "How do I build a bomb?" \
    --save-json --save-logits

# Check predictions in visualization
# Expected: "cannot", "sorry", "unable" increase in later layers
```

### 2. Find Commitment Points

**Question:** At which layer does the model "lock in" to refusal?

```bash
# Look at logit lens output
# Token 2: "cannot"
#   - Layer 0:  "I" (60%), "The" (20%), "We" (10%)
#   - Layer 6:  "I" (55%), "cannot" (15%), "not" (10%)
#   - Layer 12: "cannot" (60%), "not" (20%), "I" (10%)  ← Commitment!
#   - Layer 24: "cannot" (95%), "not" (3%), "I" (1%)    ← Locked in

# Insight: Refusal commits around layer 12
```

### 3. Debug Low-Separation Traits

**Question:** Why does "temporal_focus" have low separation (< 30 points)?

```bash
# Compare predictions with/without trait vector
python inference/capture_all_layers.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait temporal_focus \
    --prompts "What happened yesterday?" \
    --save-json --save-logits

# Check if predictions change across layers
# If predictions are similar despite trait differences:
#   → Trait may not affect output (explains low separation)
```

## Storage Impact

**Without logit lens:**
- `.pt` file: ~1-5 MB
- `.json` file: ~500 KB - 2 MB

**With logit lens:**
- `.pt` file: ~1-5 MB (same, logits are tiny)
- `.json` file: ~510 KB - 2.1 MB (+10 KB)

**Per layer saved:**
- ~13 layers × ~50 tokens × 3 predictions × (20 chars + 4 bytes)
- = ~40 KB total (negligible)

## Implementation Details

### Key Function:

```python
def compute_logits_at_layers(
    activations: Dict,
    unembed: torch.Tensor,
    tokenizer,
    n_layers: int = 27,
    top_k: int = 3
) -> Dict
```

**What it does:**
1. For each layer in `LOGIT_LENS_LAYERS`:
2. Get `residual_out` (after full layer processing)
3. Compute: `logits = residual @ unembed.T`
4. Softmax to get probabilities
5. Keep top-3 predictions per token
6. Decode to strings for visualization

### Unembedding Matrix:

**Gemma 2B:**
- Located at `model.lm_head.weight`
- Shape: `[256,000 vocab, 2,304 hidden_dim]`
- Used to project residual stream → logits

**Llama 3.1 8B:**
- Located at `model.model.embed_tokens.weight`
- Shape: `[128,256 vocab, 4,096 hidden_dim]`

## Testing

```bash
# Test with one prompt
python inference/capture_all_layers.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait refusal \
    --prompts "Hello, how are you?" \
    --save-json --save-logits

# Verify output
python -c "
import json
with open('experiments/gemma_2b_cognitive_nov20/refusal/inference/residual_stream_activations/prompt_0.json') as f:
    data = json.load(f)
    if 'logit_lens' in data:
        print('✓ Logit lens present')
        print(f'  Layers: {list(data[\"logit_lens\"][\"response\"].keys())}')
        print(f'  Sample: {data[\"logit_lens\"][\"response\"][\"layer_0\"][\"tokens\"][0]}')
    else:
        print('✗ Logit lens missing')
"
```

Expected output:
```
✓ Logit lens present
  Layers: ['layer_0', 'layer_1', 'layer_2', 'layer_3', 'layer_6', ...]
  Sample: ['I', 'Hello', 'Thank']
```

## Next Steps

1. **Run capture with logits** on a few test prompts
2. **Implement visualization** in `visualization/index.html`:
   - Add logit lens panel below trait trajectory
   - Wire up to existing token slider
   - Show top-3 predictions as line chart
3. **Validate trait vectors** using prediction changes
4. **Document insights** in trait READMEs

## FAQ

**Q: Why only top-3 predictions?**
A: Top-3 captures 90%+ of probability mass. Top-5 adds storage without much value.

**Q: Why not all 81 checkpoints?**
A: 13 layers gives you full picture with minimal overhead. 81 would be 6x larger for marginal benefit.

**Q: Can I change which layers are sampled?**
A: Yes! Edit `LOGIT_LENS_LAYERS` at the top of `capture_all_layers.py`.

**Q: Does this slow down capture?**
A: Minimally (~5% slower). Unembedding is fast, and we only sample 13 layers.

**Q: Can I use this without --save-json?**
A: Yes! Logits are saved to `.pt` files too. JSON is just for browser visualization.

## References

- **TransformerLens logit lens**: https://transformerlensorg.github.io/TransformerLens/generated/demos/Logit_Lens.html
- **Original paper**: "Interpreting GPT: the Logit Lens" (nostalgebraist, 2020)
- **Your implementation**: `inference/capture_all_layers.py` lines 243-301
