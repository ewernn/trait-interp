# Inference Tools

Tools for analyzing trait behavior during model inference.

## Available Tools

### capture_tier2.py - Full Layer Trajectory Capture

Captures per-token projections at 81 checkpoints (27 layers × 3 sublayers) for both prompt encoding and response generation.

**Use when**: You want to see how a trait evolves across ALL layers and tokens.

```bash
# Single prompt
python inference/capture_tier2.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait refusal \
    --prompts "What is the capital of France?" \
    --method probe \
    --layer 16

# Multiple prompts from file
python inference/capture_tier2.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait refusal \
    --prompts-file test_prompts.txt \
    --max-new-tokens 50 \
    --temperature 0.7

# For browser visualization, add --save-json flag
python inference/capture_tier2.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait refusal \
    --prompts "Test prompt" \
    --save-json
```

**Output**: Creates `.pt` files in `experiments/{exp}/{trait}/inference/residual_stream_activations/`
- **With `--save-json`**: Also creates `.json` files for browser visualization

**Verify output**:
```bash
python inference/verify_tier2_format.py \
    experiments/gemma_2b_cognitive_nov20/refusal/inference/residual_stream_activations/prompt_0.pt
```

### capture_tier3.py - Layer Internal States Deep Dive

Captures complete internals for ONE layer: Q/K/V projections, per-head attention weights, and all 9216 MLP neurons.

**Use when**: You want to understand HOW a specific layer processes the trait mechanistically.

```bash
# Single prompt, layer 16
python inference/capture_tier3.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait refusal \
    --prompts "What is the capital of France?" \
    --layer 16

# Multiple prompts with JSON export for visualization
python inference/capture_tier3.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait refusal \
    --prompts-file test_prompts.txt \
    --layer 16 \
    --save-json
```

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

## capture_tier2.py Output Format

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

**Key data**: `projections['prompt']` and `projections['response']` contain trait scores at 81 checkpoints for each token.

## capture_tier3.py Output Format

Each `.pt` file contains complete layer internals:

```python
{
    'prompt': {...},  # Same as Tier 2
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