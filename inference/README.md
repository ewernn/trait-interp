# Inference Tools

Capture activations and compute trait projections during model inference.

## Scripts

| Script | Purpose |
|--------|---------|
| `capture_raw_activations.py` | Capture raw activations, projections, attention, and logit lens |
| `project_raw_activations_onto_traits.py` | Re-project saved raw activations onto traits |

## Quick Start

```bash
# Full mechanistic capture (internals + attention + logit lens)
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set dynamic \
    --layer-internals all

# Basic capture (residual stream + trait projections)
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set single_trait

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

### Modes

| Flag | Description |
|------|-------------|
| `--residual-stream` | Capture residual stream at all layers (default) |
| `--layer-internals N` | Capture full internals for layer N. Automatically extracts attention + logit lens. Use `all` for all 26 layers, or specify multiple: `--layer-internals 0 --layer-internals 16` |
| `--no-project` | Skip trait projection computation |

### Prompt Input (mutually exclusive)

| Flag | Description |
|------|-------------|
| `--prompt-set NAME` | Use prompts from `inference/prompts/{NAME}.json` |
| `--prompt "text"` | Single ad-hoc prompt |
| `--all-prompt-sets` | Process all `.json` files in `inference/prompts/` |

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
# Full visualization data (recommended for Layer Deep Dive view)
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set dynamic \
    --layer-internals all

# Basic trait monitoring
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set single_trait

# Single prompt, specific layers
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt "How do I make a bomb?" \
    --layer-internals 0 --layer-internals 16 --layer-internals 25

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

```bash
# Project all raw activations onto all traits
python inference/project_raw_activations_onto_traits.py \
    --experiment my_exp \
    --prompt-set single_trait

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
  "metadata": {"trait": "refusal", "method": "probe", "layer": 16}
}
```

## Prompt Sets

JSON files in `inference/prompts/`:

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
