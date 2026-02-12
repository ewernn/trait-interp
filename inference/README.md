# Inference Tools

Capture activations and compute trait projections during model inference.

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_responses.py` | Generate or import responses (decoupled from capture) |
| `capture_raw_activations.py` | Capture raw activations via prefill from response JSONs |
| `project_raw_activations_onto_traits.py` | Compute projections from saved .pt files |
| `extract_viz.py` | Extract attention/logit-lens for visualization |

## Quick Start

```bash
# 1. Generate responses
python inference/generate_responses.py \
    --experiment my_exp \
    --prompt-set single_trait

# 2. Capture raw activations (reads response JSONs, prefill capture)
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set single_trait

# 3. Compute projections onto trait vectors
python inference/project_raw_activations_onto_traits.py \
    --experiment my_exp \
    --prompt-set single_trait

# 4. (Optional) Extract visualization data
python inference/extract_viz.py \
    --experiment my_exp \
    --prompt-set single_trait \
    --attention --logit-lens
```

## Storage Structure

```
experiments/{exp}/
├── inference/{model_variant}/
│   ├── raw/                                  # Binary .pt files
│   │   ├── residual/{prompt_set}/
│   │   │   └── {id}.pt                       # Residual stream (~20-25 MB)
│   │   └── internals/{prompt_set}/
│   │       └── {id}_L{layer}.pt              # Layer internals (~7 MB each)
│   │
│   ├── responses/{prompt_set}/               # Response JSONs (trait-independent)
│   │   └── {id}.json
│   │
│   └── projections/{trait}/{prompt_set}/      # Trait-specific projections
│       └── {id}.json                         # Slim projections (~2-5 KB)
│
└── analysis/
    └── per_token/{prompt_set}/               # Visualization JSONs
        ├── {id}_attention.json               # Attention patterns (~12-20 MB)
        └── {id}_logit_lens.json              # Logit lens predictions (~4 MB)
```

## generate_responses.py

Generates or imports response text, saves tokenized response JSONs. Two modes:

**Mode A — Generate from model:**
```bash
python inference/generate_responses.py \
    --experiment my_exp \
    --prompt-set single_trait \
    --model-variant rm_lora
```

**Mode B — Import external responses (tokenizer only, no GPU):**
```bash
python inference/generate_responses.py \
    --experiment my_exp \
    --prompt-set rm_syco/exploitation_evals_100 \
    --model-variant rm_lora \
    --from-responses path/to/response_texts.json
```

Where `response_texts.json` is a `{prompt_id: response_text}` mapping.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--experiment` | required | Experiment name |
| `--prompt-set` | required | Prompt set from `datasets/inference/{name}.json` |
| `--model-variant` | config | Model variant (default: from experiment defaults.application) |
| `--max-new-tokens` | 50 | Max tokens to generate (Mode A only) |
| `--temperature` | 0.0 | Sampling temperature (Mode A only) |
| `--prefill` | none | Prefill string appended to prompt before generation |
| `--from-responses` | none | Path to `{id: text}` JSON (Mode B) |
| `--skip-existing` | false | Skip existing response JSONs |
| `--limit` | none | Limit prompts (for testing) |
| `--output-suffix` | none | Suffix for output directory |
| `--no-server` | false | Force local model loading |

**Model Server**: Mode A tries the model server first (if running). Mode B uses tokenizer only.

## capture_raw_activations.py

Reads response JSONs, runs prefill forward passes, saves raw activations as .pt files.

### Options

| Flag | Description |
|------|-------------|
| `--capture-mlp` | Also capture mlp_contribution activations |
| `--layers LIST` | Only hook specific layers (e.g., `25,27,30` or `10-45:3`). Reduces overhead for large models. |
| `--response-only` | Only save response token activations (skip prompt tokens). Reduces file size. |
| `--responses-from VAR` | Read responses from a different variant (for model-diff). |

### Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-variant` | config | Model variant (default: from experiment defaults.application) |
| `--load-in-8bit` | false | 8-bit quantization (for 70B+ models) |
| `--load-in-4bit` | false | 4-bit quantization |
| `--no-server` | false | Force local model loading |

**Note:** Capture always requires a local model (server doesn't support prefill capture).

### Examples

```bash
# Basic capture
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set single_trait

# With mlp_contribution for component decomposition
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set harmful \
    --capture-mlp

# Model-diff: capture through instruct model using rm_lora's responses
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set rm_syco/train_100 \
    --model-variant instruct \
    --responses-from rm_lora
```

**Crash resilience**: Results saved per-prompt. Rerun with `--skip-existing` to resume.

## Cross-Variant Model-Diff (LoRA vs Clean)

Compare activations between model variants on identical tokens:

```bash
# 1. Generate LoRA model responses
python inference/generate_responses.py \
    --experiment rm_syco \
    --prompt-set rm_syco/train_100 \
    --model-variant rm_lora

# 2. Capture LoRA model activations
python inference/capture_raw_activations.py \
    --experiment rm_syco \
    --prompt-set rm_syco/train_100 \
    --model-variant rm_lora

# 3. Capture clean model using LoRA responses (--responses-from)
python inference/capture_raw_activations.py \
    --experiment rm_syco \
    --prompt-set rm_syco/train_100 \
    --model-variant instruct \
    --responses-from rm_lora

# 4. Project both variants
python inference/project_raw_activations_onto_traits.py \
    --experiment rm_syco --prompt-set rm_syco/train_100 \
    --model-variant rm_lora --no-calibration

python inference/project_raw_activations_onto_traits.py \
    --experiment rm_syco --prompt-set rm_syco/train_100 \
    --model-variant instruct --no-calibration

# 5. Aggregate comparison (effect sizes, cosine sim)
python analysis/model_diff/compare_variants.py \
    --experiment rm_syco \
    --variant-a instruct --variant-b rm_lora \
    --prompt-set rm_syco/train_100 --use-best-vector

# 6. Per-token/per-clause diff (which text spans diverge most?)
python analysis/model_diff/per_token_diff.py \
    --experiment rm_syco \
    --variant-a instruct --variant-b rm_lora \
    --prompt-set rm_syco/train_100 \
    --trait all --top-pct 5
```

**Tip:** For 70B models, use `--layers` to only capture the layers you need (from steering results).

Output structure:
```
inference/
├── rm_lora/raw/residual/train_100/    # LoRA activations
├── rm_lora/projections/{trait}/...    # LoRA projections
├── instruct/raw/residual/train_100/   # Clean activations (same tokens)
└── instruct/projections/{trait}/...   # Clean projections
model_diff/instruct_vs_rm_lora/
├── diff_vectors.pt, results.json      # Aggregate comparison
└── per_token_diff/{trait}/{prompt_set}/
    ├── {id}.json                      # Per-prompt clause breakdown
    └── aggregate.json                 # Top clauses, position-averaged delta
```

## project_raw_activations_onto_traits.py

Compute projections from saved raw activations. Useful for:
- Computing projections after capture
- Re-projecting onto new/updated vectors
- Using different layer/method

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--layer` | auto | Override layer for all traits (default: auto-select best) |
| `--method` | auto | Vector method (default: auto-detect) |
| `--component` | residual | Activation component (`residual` or `attn_contribution`) |
| `--position` | auto | Token position for vectors (auto-detects from available vectors) |
| `--extraction-variant` | config | Model variant for vectors (default: from config defaults.extraction) |
| `--traits` | all | Comma-separated trait list |
| `--skip-existing` | false | Skip existing output files |
| `--no-calibration` | false | Skip massive dims calibration check |

### Examples

```bash
# Project onto all traits (auto-selects best layer per trait)
python inference/project_raw_activations_onto_traits.py \
    --experiment my_exp \
    --prompt-set single_trait

# Fixed layer 16 for all traits
python inference/project_raw_activations_onto_traits.py \
    --experiment my_exp \
    --prompt-set single_trait \
    --layer 16

# Specific traits only
python inference/project_raw_activations_onto_traits.py \
    --experiment my_exp \
    --prompt-set single_trait \
    --traits behavioral/refusal,cognitive/retrieval
```

## extract_viz.py

Extract attention patterns and logit lens predictions for visualization (Layer Deep Dive).

```bash
# Extract attention patterns
python inference/extract_viz.py \
    --experiment my_exp \
    --prompt-set dynamic \
    --attention

# Extract logit lens
python inference/extract_viz.py \
    --experiment my_exp \
    --prompt-set dynamic \
    --logit-lens

# Both
python inference/extract_viz.py \
    --experiment my_exp \
    --prompt-set dynamic \
    --attention --logit-lens
```

## Output Formats

### Response JSON

```json
{
  "prompt": "...",
  "response": "...",
  "system_prompt": null,
  "tokens": ["<bos>", ...],
  "token_ids": [2, ...],
  "prompt_end": 45,
  "inference_model": "model_name",
  "prompt_note": "...",
  "capture_date": "...",
  "tags": []
}
```

### Raw .pt file

```python
{
    'prompt': {
        'text': str,
        'tokens': List[str],
        'token_ids': List[int],
        'activations': {
            layer_idx: {
                'residual': Tensor[n_tokens, hidden_dim],
                'attn_contribution': Tensor[n_tokens, hidden_dim],
                'mlp_contribution': Tensor[n_tokens, hidden_dim],   # if --capture-mlp
            }
        }
    },
    'response': { ... }  # same structure
}
```

### Projection JSON

```json
{
  "metadata": {
    "prompt_id": "1",
    "prompt_set": "single_trait",
    "vector_source": {
      "experiment": "my_exp",
      "trait": "behavioral/refusal",
      "layer": 16,
      "method": "probe"
    }
  },
  "projections": {
    "prompt": [0.5, -0.3, ...],
    "response": [2.1, 1.8, ...]
  }
}
```

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
| `harmful.json` | 5 harmful requests (refusal testing) |
| `benign.json` | 5 benign requests (parallel to harmful) |

## Modal Deployment (Live Chat)

`modal_inference.py` provides serverless GPU inference for the Live Chat feature.

### Setup

```bash
# Login to Modal
modal token new

# Deploy
modal deploy inference/modal_inference.py
```

### Functions

| Function | Purpose |
|----------|---------|
| `warmup()` | Preload model into GPU memory (reduces cold start) |
| `capture_activations_stream()` | Stream tokens + activations during generation |

### How it works

- Copies `core/` and `utils/` into Docker image at build time
- Uses `core/hooks.py` for `HookManager`, `SteeringHook`, `get_hook_path`
- Model cached in Modal Volume (`/models`) for fast subsequent loads
- Handles chat template formatting (system prompt, special tokens)

### Monitoring

```bash
modal app list              # List deployed apps
modal app logs trait-capture  # Stream logs
```

Or view at: https://modal.com/apps/{username}/main/deployed/trait-capture

## Requirements

- Extracted trait vectors (run extraction pipeline first)
- GPU recommended (MPS/CUDA)
- Storage: ~20 MB per prompt (raw) + ~15-20 MB per prompt (viz) + ~5 KB per trait (projections)
