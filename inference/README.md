# Inference Tools

Capture activations and compute trait projections during model inference.

## Scripts

| Script | Purpose |
|--------|---------|
| `capture_raw_activations.py` | Capture raw activations, projections, attention, and logit lens |
| `project_raw_activations_onto_traits.py` | Re-project saved raw activations onto traits |

## Quick Start

```bash
# Basic: residual stream + trait projections (always happens)
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set single_trait

# With attention for Layer Deep Dive visualization
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set dynamic \
    --attention --logit-lens

# Re-project saved activations onto different/new traits
python inference/project_raw_activations_onto_traits.py \
    --experiment my_exp \
    --prompt-set single_trait
```

## Storage Structure

```
experiments/{exp}/
├── inference/
│   ├── raw/                                  # Binary .pt files
│   │   ├── residual/{prompt_set}/
│   │   │   └── {id}.pt                       # Residual stream (~20-25 MB)
│   │   └── internals/{prompt_set}/
│   │       └── {id}_L{layer}.pt              # Layer internals (~7 MB each)
│   │
│   └── {category}/{trait}/                   # Trait-specific projections
│       └── residual_stream/{prompt_set}/
│           └── {id}.json                     # Projections + dynamics (~50-100 KB)
│
└── analysis/
    └── per_token/{prompt_set}/               # Visualization JSONs
        ├── {id}_attention.json               # Attention patterns (~12-20 MB)
        └── {id}_logit_lens.json              # Logit lens predictions (~4 MB)
```

## capture_raw_activations.py

Residual stream is **always** captured (baseline). Optional add-ons for visualization.

### Optional Add-ons

| Flag | Description | Storage |
|------|-------------|---------|
| `--attention` | Save attention patterns as JSON | ~12 MB/prompt |
| `--logit-lens` | Save logit lens predictions as JSON | ~4 MB/prompt |
| `--layer-internals N` | Save full internals as .pt files. Use `all` for all layers | ~175 MB/prompt |
| `--capture-attn` | Capture raw attn_out (for attn_out vector projections) | - |
| `--no-project` | Skip trait projection computation | - |
| `--replay` | Extract from saved .pt (no new generation) | - |

### Prompt Input (mutually exclusive)

| Flag | Description |
|------|-------------|
| `--prompt-set NAME` | Use prompts from `datasets/inference/{NAME}.json` |
| `--prompt "text"` | Single ad-hoc prompt |
| `--all-prompt-sets` | Process all `.json` files in `datasets/inference/` |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--layer` | 16 | Layer for projection vectors |
| `--method` | auto | Vector method (probe, mean_diff, gradient) |
| `--max-new-tokens` | 50 | Max tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--skip-existing` | false | Skip existing output files |

### Examples

```bash
# Basic: residual stream + trait projections
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set single_trait

# Add attention for visualization
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set dynamic \
    --attention

# Full mechanistic capture (everything)
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set dynamic \
    --layer-internals all

# Extract attention/logit-lens from existing captures (no new generation)
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set single_trait \
    --replay --attention --logit-lens

# Just raw activations, no projections
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set baseline \
    --no-project
```

## project_raw_activations_onto_traits.py

Re-project saved raw activations. Useful for:
- Re-projecting onto different traits
- Using new/updated vectors
- Recomputing dynamics

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--layer` | auto | Override layer for all traits (default: auto-select best per trait) |
| `--method` | auto | Vector method (default: auto-detect or use best from evaluation) |
| `--component` | residual | Activation component (`residual` or `attn_out`) |
| `--dynamics-only` | false | Only recompute dynamics from existing projections |
| `--skip-existing` | false | Skip existing output files |

### Layer Selection

By default, auto-selects the best layer per trait using:
1. **Steering results** (ground truth) if available
2. **Effect size** (best proxy, r=0.898 correlation with steering) otherwise
3. **Layer 16** as fallback

Use `--layer N` to override with a fixed layer for all traits.

### Examples

```bash
# Project onto all traits (auto-selects best layer per trait)
python inference/project_raw_activations_onto_traits.py \
    --experiment my_exp \
    --prompt-set single_trait

# Override with fixed layer 16 for all traits
python inference/project_raw_activations_onto_traits.py \
    --experiment my_exp \
    --prompt-set single_trait \
    --layer 16

# Project onto specific traits only
python inference/project_raw_activations_onto_traits.py \
    --experiment my_exp \
    --prompt-set single_trait \
    --traits behavioral/refusal,cognitive/retrieval
```

## Output Formats

### Attention JSON (`_attention.json`)

Per-token attention patterns for all layers and heads:
```json
{
  "tokens": ["<bos>", "How", ...],
  "n_prompt_tokens": 26,
  "n_response_tokens": 50,
  "attention": [
    {
      "token_idx": 0,
      "context_size": 1,
      "by_layer": [[[0.5, ...], ...], ...]
    }
  ]
}
```

### Logit Lens JSON (`_logit_lens.json`)

Top-50 predictions at each layer for each token:
```json
{
  "tokens": ["<bos>", "How", ...],
  "n_prompt_tokens": 26,
  "predictions": [
    {
      "token_idx": 0,
      "actual_next_token": "How",
      "by_layer": [
        {
          "layer": 0,
          "top_k": [{"token": "the", "prob": 0.12}, ...],
          "actual_rank": 847,
          "actual_prob": 0.0001
        }
      ]
    }
  ]
}
```

### Projection JSON (`.json`)

Trait-specific projections with dynamics:
```json
{
  "prompt": {"text": "...", "tokens": [...], "n_tokens": 7},
  "response": {"text": "...", "tokens": [...], "n_tokens": 8},
  "projections": {"prompt": [...], "response": [...]},
  "dynamics": {
    "commitment_point": 3,
    "peak_velocity": 1.2,
    "persistence": 4,
    "velocity": [...],
    "acceleration": [...]
  },
  "metadata": {
    "inference_model": "google/gemma-2-2b-it",
    "inference_experiment": "gemma-2-2b",
    "prompt_id": 1,
    "prompt_set": "single_trait",
    "vector_source": {
      "model": "google/gemma-2-2b",
      "experiment": "gemma-2-2b",
      "trait": "behavioral/refusal",
      "method": "probe",
      "layer": 16,
      "component": "residual"
    },
    "projection_date": "2024-12-08T..."
  }
}
```

The `vector_source` object tracks which model the vector was extracted from and which layer/method was used.

## Prompt Sets

JSON files in `datasets/inference/`:

| File | Description |
|------|-------------|
| `single_trait.json` | 10 prompts, each targeting one trait |
| `multi_trait.json` | 10 prompts activating multiple traits |
| `dynamic.json` | 8 prompts for mid-response trait changes |
| `adversarial.json` | 8 edge cases and robustness tests |
| `baseline.json` | 5 neutral prompts for baseline |
| `real_world.json` | 10 naturalistic prompts |

## Requirements

- Extracted trait vectors (run extraction pipeline first)
- GPU recommended (MPS/CUDA)
- Storage: ~20 MB per prompt (raw) + ~15-20 MB per prompt (attention/logit lens) + ~50 KB per trait (projections)
