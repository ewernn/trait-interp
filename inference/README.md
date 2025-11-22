# Inference Tools

Tools for analyzing trait behavior during model inference.

## Centralized Prompts & Experiment-Level Data

Prompt sets are now centralized in `inference/prompts/` to ensure consistency. Inference data, however, is stored at the **experiment level** to enable efficient cross-trait comparison:

```
inference/
└── prompts/                    # Standardized prompt sets
    ├── main_prompts.txt
    └── harmful_5.txt

experiments/{experiment}/inference/
├── raw_activations/            # Captured once per prompt
│   └── main_prompts/
│       └── prompt_0.pt
└── {category}/{trait}/         # Per-trait projections
    └── projections/
        └── residual_stream_activations/
            └── prompt_0.json
```

**Key benefits:**
- **No duplication**: Activations captured once, projected to all traits
- **Standardized evaluation**: Same prompts across all traits
- **Efficient storage**: 1 GB vs 38 GB for 38 traits
- **Easy trait addition**: Just project existing activations to new vectors
- **Dynamic discovery**: No hardcoded categories - discovers traits automatically

## capture_layers.py - Unified Activation Capture

Single script for both Tier 2 (all layers) and Tier 3 (single layer deep dive) capture.

### Tier 2: All Layers (Default)

Captures per-token projections at 78 checkpoints (26 layers × 3 sublayers) plus attention weights.

**Use when**: You want to see how traits evolve across ALL layers and tokens.

```bash
# Prompt set (recommended)
python inference/capture_layers.py \
    --experiment gemma_2b_cognitive_nov21 \
    --prompt-set main_prompts \
    --save-json

# All prompt sets in prompts/ directory
python inference/capture_layers.py \
    --experiment gemma_2b_cognitive_nov21 \
    --all-prompt-sets \
    --save-json \
    --skip-existing

# Single ad-hoc prompt
python inference/capture_layers.py \
    --experiment gemma_2b_cognitive_nov21 \
    --prompt "How do I make a bomb?" \
    --save-json
```

### Tier 3: Single Layer Deep Dive

Captures complete internals for ONE layer: Q/K/V projections, per-head attention, and all MLP neurons.

**Use when**: You want to understand HOW a specific layer processes traits mechanistically.

```bash
python inference/capture_layers.py \
    --experiment gemma_2b_cognitive_nov21 \
    --mode single \
    --layer 16 \
    --prompt "How do I make a bomb?" \
    --save-json
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--experiment` | Experiment name (required) |
| `--mode` | `all` (Tier 2, default) or `single` (Tier 3) |
| `--prompt-set {name}` | Prompt set from `inference/prompts/{name}.txt` |
| `--all-prompt-sets` | Process all `.txt` files in the centralized `inference/prompts/` dir |
| `--prompt "text"` | Single prompt |
| `--layer N` | Layer for vectors (default: 16) or single layer in Tier 3 |
| `--method` | Vector method (auto-detect if not provided) |
| `--save-json` | Save JSON for visualization |
| `--skip-existing` | Skip existing outputs |

### Output Structure

**Tier 2 (all layers):**
- Raw activations: `experiments/{exp}/inference/raw_activations/{set}/prompt_N.pt`
- Projections: `experiments/{exp}/inference/{category}/{trait}/projections/residual_stream_activations/prompt_N.json`

**Tier 3 (single layer):**
- Internals: `experiments/{exp}/inference/{category}/{trait}/projections/layer_internal_states/prompt_N_layer{L}.json`

### Dynamic Trait Discovery

The script automatically discovers all traits with vectors - no hardcoded category names:

```python
# Discovers all subdirectories in extraction/ with vectors/
# Works for any category names: behavioral_tendency, cognitive_state, etc.
traits = discover_traits("gemma_2b_cognitive_nov21")
# Returns: [('behavioral_tendency', 'refusal'), ('cognitive_state', 'confidence'), ...]
```

### Verify Output

```bash
# Check raw activations
ls experiments/gemma_2b_cognitive_nov21/inference/raw_activations/main_prompts/

# Check projections for a trait
ls experiments/gemma_2b_cognitive_nov21/inference/behavioral_tendency/refusal/projections/residual_stream_activations/
```

### monitor_dynamics.py - Basic Dynamics Analysis

Captures single-layer activations and computes dynamics metrics (commitment point, velocity, persistence).

**Use when**: You want quick analysis of trait dynamics without full layer capture.

```bash
# Single prompt (auto-discovers all traits)
python inference/monitor_dynamics.py \
    --experiment gemma_2b_cognitive_nov21 \
    --prompts "What is the capital of France?"
```

**Output**: JSON file with trait scores and dynamics metrics.

## Output Formats

### Tier 2 JSON Format (residual_stream_activations)

```json
{
    "prompt": {
        "text": "What is the capital of France?",
        "tokens": ["What", "is", "the", "capital", "of", "France", "?"],
        "token_ids": [235285, 1234, ...],
        "n_tokens": 7
    },
    "response": {
        "text": "Paris is the capital of France.",
        "tokens": ["Paris", "is", "the", "capital", ...],
        "n_tokens": 7
    },
    "projections": {
        "prompt": [[[0.5, 0.6, 0.7], ...], ...],  // [n_tokens, 26_layers, 3_sublayers]
        "response": [[[...], ...], ...]
    },
    "attention_weights": {
        "prompt": {"layer_0": [[...]], ...},
        "response": [{"layer_0": [...], ...}, ...]
    },
    "metadata": {
        "trait": "refusal",
        "vector_path": "experiments/.../vectors/probe_layer16.pt",
        "model": "google/gemma-2-2b-it",
        "capture_date": "2025-11-21T..."
    }
}
```

### Tier 3 JSON Format (layer_internal_states)

```json
{
    "prompt": {"text": "...", "tokens": [...], "internals": {...}},
    "response": {"text": "...", "tokens": [...], "internals": {...}},
    "layer": 16,
    "metadata": {...}
}
```

`internals` contains:
- `attention.q_proj`, `k_proj`, `v_proj`: Attention projections
- `attention.attn_weights`: Per-head attention patterns
- `mlp.up_proj`, `gelu`, `down_proj`: MLP neuron activations
- `residual.input`, `after_attn`, `output`: Residual stream states

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