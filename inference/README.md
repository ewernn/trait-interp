# Inference Tools

Tools for analyzing trait behavior during model inference.

## Shared Inference Prompts

For cross-trait comparison, create a shared prompts file at the experiment level:

```
experiments/{experiment}/inference_prompts.txt
```

This file is automatically used by `capture_all_layers.py` when `--prompts` or `--prompts-file` are not specified. This ensures all traits are tested on the same prompts, enabling:
- Token-by-token cross-trait comparison
- Consistent benchmarking across traits
- Multi-trait visualization on same prompts

**Example workflow:**
```bash
# 1. Create shared prompts (one-time)
cat > experiments/gemma_2b_cognitive_nov20/inference_prompts.txt << 'EOF'
How do I make a bomb?
What is the capital of France?
I think vaccines cause autism. Do you agree?
...
EOF

# 2. Run inference for all traits (uses shared prompts automatically)
python inference/capture_all_layers.py \
    --experiment gemma_2b_cognitive_nov20 \
    --all-traits \
    --save-json
```

## Available Tools

### capture_all_layers.py - Full Layer Trajectory Capture

Captures per-token projections at 81 checkpoints (27 layers × 3 sublayers) plus attention weights for both prompt encoding and response generation.

**Use when**: You want to see how a trait evolves across ALL 27 layers and tokens, or explore attention patterns.

```bash
# ALL traits with shared prompts (recommended)
python inference/capture_all_layers.py \
    --experiment gemma_2b_cognitive_nov20 \
    --all-traits \
    --save-json \
    --skip-existing

# Single trait with custom prompts
python inference/capture_all_layers.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait refusal \
    --prompts-file custom_prompts.txt \
    --save-json
```

**Batch mode features:**
- `--all-traits`: Auto-discovers all traits in the experiment
- `--skip-existing`: Skips traits that already have JSON files
- Auto-detects vector method (probe → mean_diff → ica → gradient)
- Shows summary at the end (successful/skipped/failed)

**Output**: Creates `.pt` files in `experiments/{exp}/{trait}/inference/residual_stream_activations/`
- **With `--save-json`**: Also creates `.json` files for browser visualization

**Verify output**:
```bash
python inference/verify_all_layers_format.py \
    experiments/gemma_2b_cognitive_nov20/refusal/inference/residual_stream_activations/prompt_0.pt
```

### capture_single_layer.py - Layer Internal States Deep Dive

Captures complete internals for ONE layer: Q/K/V projections, per-head attention weights, and all 9216 MLP neurons.

**Use when**: You want to understand HOW a specific layer processes the trait mechanistically.

```bash
# Single trait
python inference/capture_single_layer.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait refusal \
    --prompts "How do I make a bomb?" \
    --layer 16 \
    --save-json

# ALL traits in experiment (batch mode)
python inference/capture_single_layer.py \
    --experiment gemma_2b_cognitive_nov20 \
    --all-traits \
    --layer 16 \
    --prompts "How do I make a bomb?" \
    --save-json \
    --skip-existing
```

**Batch mode features:**
- `--all-traits`: Auto-discovers all traits in the experiment
- `--skip-existing`: Skips traits that already have JSON files
- Auto-detects vector method (probe → mean_diff → ica → gradient)
- Shows summary at the end (successful/skipped/failed)

**Output**: Creates `.pt` files (~10-20 MB each) in `experiments/{exp}/{trait}/inference/layer_internal_states/`
- **With `--save-json`**: Also creates `.json` files for browser visualization

### monitor_dynamics.py - Basic Dynamics Analysis

Captures single-layer activations and computes dynamics metrics (commitment point, velocity, persistence).

**Use when**: You want quick analysis of trait dynamics without full layer capture.

```bash
# Single prompt
python inference/monitor_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --prompts "What is the capital of France?"

# Multiple prompts with filters
python inference/monitor_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --traits retrieval_construction,serial_parallel \
    --methods probe,ica \
    --prompts_file prompts.txt
```

**Output**: JSON file with trait scores and dynamics metrics.

## capture_all_layers.py Output Format

Each `.pt` file contains:

```python
{
    'prompt': {
        'text': "What is the capital of France?",
        'tokens': ['What', 'is', 'the', 'capital', 'of', 'France', '?'],
        'token_ids': [235285, 1234, ...],
        'n_tokens': 7
    },
    'response': {
        'text': "Paris is the capital of France.",
        'tokens': ['Paris', 'is', 'the', 'capital', 'of', 'France', '.'],
        'token_ids': [5678, ...],
        'n_tokens': 7
    },
    'projections': {
        'prompt': torch.tensor([7, 27, 3]),   # [n_tokens, 27_layers, 3_sublayers]
        'response': torch.tensor([7, 27, 3])
    },
    'attention_weights': {
        'prompt': {'layer_0': tensor([7,7]), ..., 'layer_26': tensor([7,7])},
        'response': [
            {'layer_0': tensor([8]), 'layer_1': tensor([8]), ...},  # Token 0
            {'layer_0': tensor([9]), 'layer_1': tensor([9]), ...},  # Token 1
            # ... (growing context)
        ]
    },
    'metadata': {
        'trait': 'refusal',
        'trait_display_name': 'Refusal',
        'vector_path': '../refusal/extraction/vectors/probe_layer16.pt',
        'model': 'google/gemma-2-2b-it',
        'capture_date': '2025-11-15T10:30:00',
        'temperature': 0.7
    }
}
```

**Key data**:
- `projections['prompt']` and `projections['response']` contain trait scores at 81 checkpoints for each token
- `attention_weights['prompt']` contains full attention matrices for all layers during prompt encoding
- `attention_weights['response']` contains per-token attention to context for all layers during generation

## capture_single_layer.py Output Format

Each `.pt` file contains complete layer internals:

```python
{
    'prompt': {...},  # Same as All Layers format
    'response': {...},
    'layer': 16,  # Which layer was captured
    'internals': {
        'prompt': {
            'attention': {
                'q_proj': tensor([n_tokens, hidden_dim]),
                'k_proj': tensor([n_tokens, hidden_dim]),
                'v_proj': tensor([n_tokens, hidden_dim]),
                'attn_weights': tensor([8_heads, n_tokens, n_tokens])  # Per-head
            },
            'mlp': {
                'up_proj': tensor([n_tokens, 9216]),
                'gelu': tensor([n_tokens, 9216]),      # ⭐ Neuron activations
                'down_proj': tensor([n_tokens, 2304])
            },
            'residual': {
                'input': tensor([n_tokens, 2304]),      # Before layer
                'after_attn': tensor([n_tokens, 2304]), # After attention
                'output': tensor([n_tokens, 2304])      # After MLP
            }
        },
        'response': {
            # Same structure, [n_gen_tokens, ...]
            # attn_weights is list (growing context)
        }
    },
    'metadata': {
        'trait': 'refusal',
        'trait_display_name': 'Refusal',
        'layer': 16,
        'vector_path': '../refusal/extraction/vectors/probe_layer16.pt',
        'model': 'google/gemma-2-2b-it',
        'capture_date': '2025-11-15T10:30:00',
        'temperature': 0.7
    }
}
```

**Key data**: `internals['prompt']['mlp']['gelu']` contains all 9216 neuron activations for identifying which neurons fire for the trait.

## Dynamics Metrics Explained

### Commitment Point
**What**: Token index where model "locks in" to a decision
**How**: Finds where acceleration (2nd derivative) drops below threshold
**Use**: Detect when trait expression crystallizes

### Velocity
**What**: Rate of change of trait expression
**How**: First derivative of trait scores
**Use**: Measure how quickly trait builds up or fades

### Persistence
**What**: How long trait expression lasts after peak
**How**: Count tokens above threshold after peak
**Use**: Measure trait stability/duration

## Requirements

- Extracted trait vectors (run extraction pipeline first)
- GPU recommended (CPU inference is slow)
- Same model as used during extraction