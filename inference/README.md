# Inference Tools

Capture activations and compute trait projections during model inference.

## Scripts

| Script | Purpose |
|--------|---------|
| `capture_raw_activations.py` | Capture raw activations from model inference |
| `project_raw_activations_onto_traits.py` | Compute projections from saved .pt files |
| `extract_viz.py` | Extract attention/logit-lens for visualization |

## Quick Start

```bash
# 1. Capture raw activations
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set single_trait

# 2. Compute projections onto trait vectors
python inference/project_raw_activations_onto_traits.py \
    --experiment my_exp \
    --prompt-set single_trait

# 3. (Optional) Extract visualization data
python inference/extract_viz.py \
    --experiment my_exp \
    --prompt-set single_trait \
    --attention --logit-lens
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
│   ├── responses/{prompt_set}/               # Response JSONs (trait-independent)
│   │   └── {id}.json
│   │
│   └── {category}/{trait}/                   # Trait-specific projections
│       └── residual_stream/{prompt_set}/
│           └── {id}.json                     # Slim projections (~2-5 KB)
│
└── analysis/
    └── per_token/{prompt_set}/               # Visualization JSONs
        ├── {id}_attention.json               # Attention patterns (~12-20 MB)
        └── {id}_logit_lens.json              # Logit lens predictions (~4 MB)
```

## capture_raw_activations.py

Runs model inference and saves raw activations. Does NOT compute projections.

### Options

| Flag | Description |
|------|-------------|
| `--capture-mlp` | Also capture mlp_out activations |
| `--layers LIST` | Only hook specific layers during capture (e.g., `25,27,30` or `10-45:3`). Default: all layers. Reduces GPU→CPU copy overhead for large models. |
| `--replay-responses SET` | Load responses from another prompt set (model-diff) |
| `--replay-from-variant VAR` | Model variant to load replay responses from (for cross-variant model-diff) |

### Prompt Input (mutually exclusive)

| Flag | Description |
|------|-------------|
| `--prompt-set NAME` | Use prompts from `datasets/inference/{NAME}.json` |
| `--prompt "text"` | Single ad-hoc prompt |
| `--all-prompt-sets` | Process all `.json` files in `datasets/inference/` |

### Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | config | Model to use (overrides `application_model` from config) |
| `--lora` | none | LoRA adapter to apply on top of model |
| `--load-in-8bit` | false | 8-bit quantization (for 70B+ models) |
| `--load-in-4bit` | false | 4-bit quantization |
| `--no-server` | false | Force local model loading (skip server check) |

**Model Server**: If `python server/app.py` is running, scripts auto-detect and use it to avoid model reload time.

### Other Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-new-tokens` | 50 | Max tokens to generate |
| `--temperature` | 0.0 | Sampling temperature |
| `--batch-size` | auto | Batch size (auto-detects from VRAM) |
| `--skip-existing` | false | Skip existing output files |
| `--limit` | none | Limit prompts (for testing) |
| `--output-suffix` | none | Suffix for output dirs |
| `--prefill` | none | Prefill string for prefill attack testing |

### Examples

```bash
# Basic capture
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set single_trait

# With mlp_out for component decomposition
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set harmful \
    --capture-mlp

# 70B model with LoRA
python inference/capture_raw_activations.py \
    --experiment llama-3.3-70b \
    --prompt-set rm_sycophancy_triggers \
    --lora ewernn/llama-3.3-70b-dpo-rt-lora-bf16

# Model-diff: run same responses through different model
python inference/capture_raw_activations.py \
    --experiment my_exp \
    --prompt-set dynamic \
    --replay-responses other_prompt_set
```

**Crash resilience**: Results saved after each batch. Rerun with `--skip-existing` to resume.

## Model Comparison (Base vs IT)

Compare activations between models on identical tokens (e.g., base vs instruction-tuned):

```bash
# 1. Capture IT model (default - uses application_model from config)
python inference/capture_raw_activations.py \
    --experiment my_exp --prompt-set harmful

# 2. Project IT model
python inference/project_raw_activations_onto_traits.py \
    --experiment my_exp --prompt-set harmful --traits behavioral/refusal

# 3. Capture base model with IT model's responses
#    (saves to inference/models/{model}/ automatically)
python inference/capture_raw_activations.py \
    --experiment my_exp --prompt-set harmful \
    --model google/gemma-2-2b --replay-responses harmful

# 4. Project base model
python inference/project_raw_activations_onto_traits.py \
    --experiment my_exp --prompt-set harmful \
    --model google/gemma-2-2b --traits behavioral/refusal
```

Output structure:
```
inference/
├── raw/residual/harmful/          # IT model activations
├── {trait}/residual_stream/harmful/  # IT model projections
└── models/gemma-2-2b/             # Base model comparison
    ├── raw/residual/harmful/
    └── {trait}/residual_stream/harmful/
```

View in visualization: **Trait Dynamics → Compare dropdown** (Main model / Diff / comparison model).

## Cross-Variant Model-Diff (LoRA vs Clean)

Compare activations between model variants (e.g., LoRA-finetuned vs clean) on identical tokens:

```bash
# 1. Capture LoRA model responses
python inference/capture_raw_activations.py \
    --experiment rm_syco \
    --prompt-set rm_syco/train_100 \
    --model-variant rm_lora

# 2. Prefill same tokens through clean model
#    --replay-from-variant specifies where to load responses from
python inference/capture_raw_activations.py \
    --experiment rm_syco \
    --prompt-set rm_syco/train_100 \
    --model-variant instruct \
    --replay-responses rm_syco/train_100 \
    --replay-from-variant rm_lora

# 3. Project both variants
python inference/project_raw_activations_onto_traits.py \
    --experiment rm_syco --prompt-set rm_syco/train_100 \
    --model-variant rm_lora --no-calibration

python inference/project_raw_activations_onto_traits.py \
    --experiment rm_syco --prompt-set rm_syco/train_100 \
    --model-variant instruct --no-calibration

# 4. Aggregate comparison (effect sizes, cosine sim)
python analysis/model_diff/compare_variants.py \
    --experiment rm_syco \
    --variant-a instruct --variant-b rm_lora \
    --prompt-set rm_syco/train_100 --use-best-vector

# 5. Per-token/per-clause diff (which text spans diverge most?)
python analysis/model_diff/per_token_diff.py \
    --experiment rm_syco \
    --variant-a instruct --variant-b rm_lora \
    --prompt-set rm_syco/train_100 \
    --trait all --top-pct 5
```

**Tip:** For 70B models, use `--layers` to only capture the layers you need (from steering results). This avoids hooking all 80 layers during generation, which is the main performance bottleneck.

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

### Layer Selection

Auto-selects best layer per trait using steering results as ground truth. Uses extraction and steering variants from experiment config.

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

### Raw .pt file

```python
{
    'prompt': {
        'text': str,
        'tokens': List[str],
        'token_ids': List[int],
        'activations': {
            layer_idx: {
                'residual': Tensor[n_tokens, hidden_dim],           # layer output
                'attn_contribution': Tensor[n_tokens, hidden_dim],  # what attention adds to residual
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
