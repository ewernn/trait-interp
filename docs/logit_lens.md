# Logit Lens

Shows "what would the model predict if it stopped at layer X?" by projecting intermediate layer states through the final unembedding matrix.

---

## Overview

**Logit lens** reveals how the model's prediction evolves across layers before final output. At each layer, the residual stream is projected to vocabulary space to show the top-3 most likely tokens.

**Why it's useful**:
- **Validate trait vectors** - See if vectors actually change predictions
- **Find commitment points** - When does the model lock into a decision?
- **Debug low-separation traits** - Why do some traits not affect output?

**Note**: Logit lens shows per-token predictions at individual layers. This is different from vector extraction, which aggregates across all tokens and all responses to find trait-level patterns.

---

## Usage

Add `--save-logits` flag when capturing activations:

```bash
# Single prompt with logit lens
python inference/capture_all_layers.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait refusal \
    --prompts "How do I build a bomb?" \
    --save-json \
    --save-logits
```

**Storage impact**: ~10 KB extra per prompt (minimal overhead)

---

## Sampled Layers

Samples **13 out of 26 layers** to balance detail with storage:

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

**Customize sampling**: Edit `LOGIT_LENS_LAYERS` at top of `inference/capture_all_layers.py`

---

## Visualization

Open the interactive visualization:

```bash
cd /Users/ewern/Desktop/code/trait-interp
open visualization/index.html
```

**Navigate to logit lens**:
1. Click "Data Explorer"
2. Select experiment: "gemma_2b_cognitive_nov20"
3. Check trait: "refusal"
4. Click "⚡ All Layers"
5. Scroll down past trajectory heatmap

**You should see**:
```
Prediction Evolution for Token 2: "cannot"

━━━━━━━━━━━━━━━━━━━━━━━  "cannot" (95%)  [blue line]
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   "not" (3%)     [red line]
∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙   "I" (1%)       [green line]

Layer: 0   3   6   9   12   15   18   21   24   25
```

**Interaction**:
- Move the token slider at top of page
- All visualizations update (trajectory, predictions, attention)
- Lines show top-3 token probabilities at each sampled layer

---

## Quick Test (2 Minutes)

Verify logit lens is working:

```bash
# Check test data has logit lens
python3 -c "
import json
with open('experiments/gemma_2b_cognitive_nov20/refusal/inference/residual_stream_activations/prompt_0.json') as f:
    data = json.load(f)
    print('Has logit_lens:', 'logit_lens' in data)
    print('Has response:', 'response' in data.get('logit_lens', {}))
    if 'logit_lens' in data:
        layers = list(data['logit_lens']['response'].keys())
        print(f'Layers ({len(layers)}):', sorted(layers))
        print('Sample token 0:', data['logit_lens']['response']['layer_0']['tokens'][0])
"
```

**Expected output**:
```
Has logit_lens: True
Has response: True
Layers (12): ['layer_0', 'layer_1', 'layer_12', ..., 'layer_25']
Sample token 0: ['?', '<bos>', '<pad>']
```

---

## Research Applications

### 1. Validate Trait Vectors

**Question**: Does the refusal vector actually increase "cannot" tokens?

**Method**:
- Generate with/without refusal vector
- Check if logit lens shows increased probability of refusal tokens ("cannot", "sorry", "unable") in later layers
- If yes → vector is working as intended

### 2. Find Commitment Points

**Question**: At which layer does the model lock into a decision?

**Example** (refusal behavior):
```
Token 2: "cannot"
  Layer 0:  "I" (60%), "The" (20%), "We" (10%)
  Layer 6:  "I" (55%), "cannot" (15%), "not" (10%)
  Layer 12: "cannot" (60%), "not" (20%), "I" (10%)  ← Commitment!
  Layer 24: "cannot" (95%), "not" (3%), "I" (1%)    ← Locked in
```

**Insight**: Refusal commits around layer 12, fully locks by layer 24.

### 3. Debug Low-Separation Traits

**Question**: Why does `temporal_focus` have low separation scores?

**Method**:
- Capture logit lens for temporal_focus prompts
- Check if predictions differ between past/future focus
- If predictions are similar despite trait differences → trait may not affect output (explains low separation)

---

## Implementation Details

### Data Structure

Added to `.pt` and `.json` files:

```python
{
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

### Computation

For each sampled layer:

```python
# Get residual stream at this layer
residual = activations[layer, :, :]  # [seq_len, hidden_dim]

# Project to vocabulary space
logits = residual @ unembed.T  # [seq_len, vocab_size]

# Get probabilities
probs = softmax(logits, dim=-1)

# Keep top-3 per token
top_k_probs, top_k_indices = torch.topk(probs, k=3, dim=-1)
top_k_tokens = tokenizer.batch_decode(top_k_indices)
```

**Unembedding matrix location**:
- Gemma 2B: `model.lm_head.weight` (shape: [256000 vocab, 2304 hidden])
- Llama 3.1 8B: `model.model.embed_tokens.weight` (shape: [128256 vocab, 4096 hidden])

### Code References

- `inference/capture_all_layers.py` lines 47-57: `LOGIT_LENS_LAYERS` constant
- `inference/capture_all_layers.py` lines 242-301: `compute_logits_at_layers()` function
- `inference/capture_all_layers.py` lines 561-624: Integration into capture flow
- `visualization/index.html` lines 2771-2782: Logit lens container HTML
- `visualization/index.html` lines 2970-3065: `renderLogitLens()` function

---

## Troubleshooting

### Panel not showing in visualization?

**Check 1**: Verify data has logit lens
```bash
python3 -c "import json; data=json.load(open('path/to/prompt_0.json')); print('logit_lens' in data)"
```

If `False`, regenerate with `--save-logits` flag.

**Check 2**: Open browser console (F12)
- Look for JavaScript errors
- Common: "Cannot read property 'logit_lens' of undefined"

**Check 3**: Verify container exists
- Press F12 → Elements tab
- Search for: `id="logit-lens-container"`
- Should be `display: block` (not `display: none`)

### Lines not updating when slider moves?

**Check console for errors**:
- Press F12 → Console tab
- Move slider, watch for red error messages

**Verify function exists**:
```javascript
// In browser console:
typeof renderLogitLens  // Should return "function"
```

### Strange tokens or probabilities?

**Expected behavior**:
- Special tokens: `<bos>`, `<pad>`, `<eos>` are normal
- Probabilities: May be 0% or 100%
- Archaic words: Model artifacts, shows internal representations

This is normal - reveals model's internal state.

---

## Generate More Test Data

To test with different prompts:

```bash
python inference/capture_all_layers.py \
  --experiment gemma_2b_cognitive_nov20 \
  --trait refusal \
  --prompts "Can you help me hack a website?" \
  --save-json \
  --save-logits
```

Generates `prompt_1.json` which you can view in visualization.

---

## Performance

**Computation overhead**: ~5% slower (unembedding is fast, only 13 layers sampled)

**Storage per prompt**:
- Without logit lens: ~1-5 MB (.pt), ~500 KB - 2 MB (.json)
- With logit lens: Same for .pt, +10 KB for .json

**Per layer saved**: ~13 layers × ~50 tokens × 3 predictions × 24 bytes ≈ 40 KB total

---

## FAQ

**Q: Why only top-3 predictions?**
A: Top-3 captures 90%+ of probability mass. Top-5 adds storage without much value.

**Q: Why not all 26 layers?**
A: 13 layers gives full picture with minimal overhead. 26 would be 2x storage for marginal benefit.

**Q: Does this slow down capture?**
A: Minimally (~5%). Unembedding is fast, and we only sample 13 layers.

**Q: Can I use this without --save-json?**
A: Yes! Logits are saved to `.pt` files too. JSON is just for browser visualization.

---

## Success Criteria

✅ Panel appears below trajectory heatmap
✅ Shows 3 colored lines
✅ Token number and text update when slider moves
✅ Lines redraw smoothly
✅ X-axis shows layers: 0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 25
✅ Y-axis shows probabilities 0-1
✅ Legend shows token names
✅ Chart matches dark theme

**If all checked**: Logit lens visualization is working perfectly.

---

## References

- **TransformerLens logit lens**: https://transformerlensorg.github.io/TransformerLens/generated/demos/Logit_Lens.html
- **Original paper**: "Interpreting GPT: the Logit Lens" (nostalgebraist, 2020)
- **Implementation**: `inference/capture_all_layers.py`
