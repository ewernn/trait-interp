# Prompt Standardization Guide

Deduplicated inference data format for efficient storage and multi-trait comparison.

---

## Overview

**Problem:** Running inference on the same prompts across multiple traits duplicates prompt/response data.

**Solution:** Split data into:
- **Shared prompts**: `experiments/{exp}/inference/prompts/prompt_N.json`
- **Per-trait projections**: `experiments/{exp}/inference/projections/{trait}/prompt_N.json`

**Benefits:**
- ~90% storage savings for multi-trait inference
- Easy multi-trait comparison in visualizer
- Backward compatible with existing code

---

## File Structure

### Shared Prompt File
```
experiments/gemma_2b_cognitive_nov20/inference/prompts/prompt_0.json
```

```json
{
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris.",
  "tokens": ["What", " is", " the", ...],
  "prompt_idx": 0
}
```

**Saved once**, shared across all traits.

### Per-Trait Projection File
```
experiments/gemma_2b_cognitive_nov20/inference/projections/refusal/prompt_0.json
```

```json
{
  "prompt_idx": 0,
  "trait": "refusal",
  "scores": [0.1, 0.2, 0.15, ...],
  "dynamics": {
    "commitment_point": 7,
    "peak_velocity": 0.4,
    "avg_velocity": 0.2,
    "persistence": 6
  },
  "metadata": {
    "layer": 16,
    "method": "probe"
  }
}
```

**One per trait**, contains only trait-specific data.

---

## Usage

### Running Inference

The `inference/monitor_dynamics.py` script now saves in both formats automatically:

```bash
# Standard usage - saves in both formats
python inference/monitor_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --prompts "What is the capital of France?"

# Output:
# - dynamics_results.json (combined format, backward compatible)
# - experiments/gemma_2b_cognitive_nov20/inference/prompts/prompt_0.json
# - experiments/gemma_2b_cognitive_nov20/inference/projections/{trait}/prompt_0.json
```

**Disable standardized output** (if needed):
```bash
python inference/monitor_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --prompts "Your prompt" \
    --save_standardized False
```

### Loading in Python

```python
from inference.utils_standardized_output import load_standardized_inference

# Load combined result for specific prompt
result = load_standardized_inference(
    experiment='gemma_2b_cognitive_nov20',
    prompt_idx=0,
    traits=['refusal', 'uncertainty']  # Or None for all
)

# result has same format as run_inference_with_dynamics output:
# {
#   'prompt': '...',
#   'response': '...',
#   'tokens': [...],
#   'trait_scores': {'refusal': [...], 'uncertainty': [...]},
#   'dynamics': {'refusal': {...}, 'uncertainty': {...}}
# }
```

### Loading in Visualizer (JavaScript)

```javascript
// Load shared prompt
const prompt = await DataLoader.fetchStandardizedPrompt(0);

// Load trait-specific projections
const refusal = await DataLoader.fetchStandardizedProjections(
    {name: 'refusal'},
    0
);
const uncertainty = await DataLoader.fetchStandardizedProjections(
    {name: 'uncertainty'},
    0
);

// Or load combined (tries standardized first, falls back to legacy)
const combined = await DataLoader.fetchInferenceCombined(
    {name: 'refusal'},
    0
);
```

---

## Storage Savings

### Example: 20 traits, 10 prompts

**Old format (duplicated):**
```
experiments/gemma_2b_cognitive_nov20/
├── refusal/inference/residual_stream_activations/
│   ├── prompt_0.json  (full prompt + response + tokens + scores)
│   └── ... (10 prompts)
├── uncertainty/inference/residual_stream_activations/
│   ├── prompt_0.json  (DUPLICATE prompt + response + tokens + scores)
│   └── ... (10 prompts)
└── ... (18 more traits)

Total: ~630 MB (20 traits × 10 prompts × 3 KB each)
```

**New format (deduplicated):**
```
experiments/gemma_2b_cognitive_nov20/inference/
├── prompts/
│   ├── prompt_0.json  (shared: prompt + response + tokens)
│   └── ... (10 prompts)
└── projections/
    ├── refusal/
    │   ├── prompt_0.json  (only: scores + dynamics)
    │   └── ... (10 prompts)
    └── ... (20 traits)

Total: ~60 MB (10 prompts × 3 KB + 20 traits × 10 prompts × 0.2 KB)
Savings: 90%
```

---

## Backward Compatibility

**No breaking changes!**

1. **Inference script** still saves `dynamics_results.json` (combined format)
2. **Visualizer** tries standardized format first, falls back to legacy
3. **Old code** continues to work unchanged

---

## Testing

Run the test to verify everything works:

```bash
python test_prompt_standardization.py
```

Expected output:
```
✅ All tests passed!

The standardized format is ready to use.
```

---

## API Reference

### Python

**`save_standardized_inference(result, experiment, prompt_idx, layer, method)`**
- Saves inference result in standardized format
- Creates directories automatically
- Shared prompt saved only if doesn't exist

**`load_standardized_inference(experiment, prompt_idx, traits=None)`**
- Loads combined result from standardized files
- `traits=None` loads all available traits
- Returns same format as `run_inference_with_dynamics`

### JavaScript

**`DataLoader.fetchStandardizedPrompt(promptIdx)`**
- Fetches shared prompt data
- Returns: `{prompt, response, tokens, prompt_idx}`

**`DataLoader.fetchStandardizedProjections(trait, promptIdx)`**
- Fetches trait-specific projections
- Returns: `{prompt_idx, trait, scores, dynamics, metadata}`

**`DataLoader.fetchInferenceCombined(trait, promptIdx)`**
- Fetches combined data (tries standardized, falls back to legacy)
- Returns: `{prompt, response, tokens, trait_scores, dynamics}`

**`DataLoader.discoverStandardizedPrompts()`**
- Auto-discovers available prompt indices
- Returns: `[0, 1, 2, ...]` (array of indices)

---

## Migration (if needed)

If you have existing inference data in the old format, you can migrate:

```bash
# TODO: Create migration script if needed
# scripts/migrate_inference_to_standardized.sh
```

For now, just re-run inference with the updated script - it will save in both formats.

---

## Status

✅ **Implemented**
- Utility functions for save/load
- Updated inference script with `--save_standardized` flag (default: True)
- Visualizer data loader with automatic fallback
- Test suite

✅ **Tested**
- Save/load roundtrip
- File structure verification
- Data integrity checks
- Storage calculations

🎯 **Ready to Use**

Run inference and it will automatically save in the standardized format!

---

## Related Files

- `inference/utils_standardized_output.py` - Core utilities
- `inference/monitor_dynamics.py` - Main inference script
- `visualization/core/data-loader.js` - Visualizer data loading
- `test_prompt_standardization.py` - Test suite
- `docs/prompt_standardization_guide.md` - This file
