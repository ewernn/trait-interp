# Inference Tools

Capture activations and compute trait projections during model inference.

## Scripts

| Script | Purpose |
|--------|---------|
| `capture.py` | Capture raw activations and/or compute projections |
| `project.py` | Post-hoc projection from saved raw activations |

## Quick Start

```bash
# Capture residual stream + project onto all traits
python inference/capture.py \
    --experiment my_experiment \
    --prompt-set single_trait

# Re-project saved activations onto new/different traits
python inference/project.py \
    --experiment my_experiment \
    --prompt-set single_trait
```

## Storage Structure

```
experiments/{exp}/inference/
├── raw/                                  # Binary .pt files (trait-independent)
│   ├── residual/{prompt_set}/
│   │   └── {id}.pt                       # ~20-25 MB, all 26 layers
│   └── internals/{prompt_set}/           # (optional)
│       └── {id}_L{layer}.pt              # ~7 MB, single layer deep dive
│
└── {category}/{trait}/                   # JSON files (trait-specific)
    └── residual_stream/{prompt_set}/
        └── {id}.json                     # ~50-100 KB, projections + dynamics
```

**Key design:**
- Raw activations are **trait-independent** (captured once, reusable)
- Projections are **trait-specific** (derived from raw + vectors)
- Raw stored as binary `.pt` (compact), derived as `.json` (visualization-ready)
- Layer internals are raw-only (per-trait analysis computed on-demand)

## capture.py

Unified capture script with clear flags for what to capture.

### Flags

| Flag | Description |
|------|-------------|
| `--residual-stream` | Capture residual stream at all layers (default) |
| `--layer-internals N` | Capture full internals for layer N (Q/K/V, MLP neurons) |
| `--attention-only N` | Capture just attention weights for layer N |
| `--logit-lens` | Compute logit lens (top-k predictions per layer) |
| `--no-project` | Skip projection computation (just save raw) |

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
| `--method` | auto | Vector method (probe, mean_diff, ica, gradient) |
| `--max-new-tokens` | 50 | Max tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--skip-existing` | false | Skip existing output files |

### Examples

```bash
# Basic: capture + project for a prompt set
python inference/capture.py \
    --experiment my_experiment \
    --prompt-set single_trait

# Layer internals for mechanistic analysis
python inference/capture.py \
    --experiment my_experiment \
    --prompt "How do I make a bomb?" \
    --layer-internals 16

# Just capture raw activations (no projection)
python inference/capture.py \
    --experiment my_experiment \
    --prompt-set baseline \
    --no-project

# Include logit lens predictions
python inference/capture.py \
    --experiment my_experiment \
    --prompt-set baseline \
    --logit-lens
```

## project.py

Post-hoc projection from saved raw activations. Useful for:
- Re-projecting onto different traits
- Using new/updated vectors
- Recomputing dynamics with different parameters

### Examples

```bash
# Project all raw activations onto all traits
python inference/project.py \
    --experiment my_experiment \
    --prompt-set single_trait

# Project onto specific traits only
python inference/project.py \
    --experiment my_experiment \
    --prompt-set single_trait \
    --traits behavioral/refusal,cognitive/retrieval

# Add logit lens to existing projections
python inference/project.py \
    --experiment my_experiment \
    --prompt-set baseline \
    --logit-lens

# Recompute dynamics only (faster)
python inference/project.py \
    --experiment my_experiment \
    --prompt-set baseline \
    --dynamics-only
```

## Output Formats

### Raw Activations (.pt)

Binary PyTorch files containing:
- `prompt.activations[layer][sublayer]` - Tensor for each checkpoint
- `prompt.attention` - Attention weights
- `response.activations[layer][sublayer]` - Per-token tensors
- `response.attention` - Per-token attention

### Projections (.json)

```json
{
  "prompt": {
    "text": "What is the capital of France?",
    "tokens": ["What", " is", " the", ...],
    "token_ids": [1234, 5678, ...],
    "n_tokens": 7
  },
  "response": {
    "text": "Paris is the capital of France.",
    "tokens": ["Paris", " is", ...],
    "n_tokens": 8
  },
  "projections": {
    "prompt": [[[0.5, 0.6, 0.7], ...], ...],
    "response": [[[...], ...], ...]
  },
  "dynamics": {
    "commitment_point": 3,
    "peak_velocity": 1.2,
    "avg_velocity": 0.8,
    "persistence": 4,
    "velocity": [0.5, 1.2, 0.3, ...],
    "acceleration": [0.7, -0.9, ...]
  },
  "attention_weights": {...},
  "logit_lens": {...},
  "metadata": {
    "trait": "refusal",
    "category": "behavioral",
    "method": "probe",
    "layer": 16,
    "model": "google/gemma-2-2b-it",
    "capture_date": "2024-..."
  }
}
```

### Layer Internals (.pt)

Binary files containing single-layer deep dive:
- `prompt.internals.attention.{q,k,v}_proj` - Attention projections
- `prompt.internals.attention.attn_weights` - Per-head attention
- `prompt.internals.mlp.{up_proj,gelu,down_proj}` - MLP activations
- `prompt.internals.residual.{input,after_attn,output}` - Residual stream

## Dynamics Metrics

| Metric | Description |
|--------|-------------|
| `commitment_point` | Token where acceleration drops (model "locks in") |
| `peak_velocity` | Maximum rate of change |
| `avg_velocity` | Average rate of change |
| `persistence` | Tokens above threshold after peak |
| `velocity` | First derivative of trait scores |
| `acceleration` | Second derivative of trait scores |

## Prompt Sets

Prompt files are JSON in `inference/prompts/`:

```
inference/prompts/
├── single_trait.json      # 10 prompts, each targeting one trait
├── multi_trait.json       # 10 prompts activating multiple traits
├── dynamic.json           # 8 prompts for mid-response trait changes
├── adversarial.json       # 8 edge cases and robustness tests
├── baseline.json          # 5 neutral prompts for baseline
└── real_world.json        # 10 naturalistic prompts
```

**JSON format:**
```json
{
  "name": "Single-Trait Elicitation",
  "description": "Each prompt maximally activates ONE specific trait",
  "prompts": [
    {"id": 1, "text": "What year was...", "note": "elicits retrieval"},
    {"id": 2, "text": "Your explanation...", "note": "elicits defensiveness"}
  ]
}
```

## Requirements

- Extracted trait vectors (run extraction pipeline first)
- GPU recommended (MPS/CUDA)
- ~20-25 MB storage per prompt (raw) + ~50 KB per trait (projections)

Verify raw file size:
```bash
ls -lh experiments/*/inference/raw/residual/*/*.pt | head -1
```
