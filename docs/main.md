# Trait Vector Extraction and Monitoring

Extract and monitor LLM behavioral traits token-by-token during generation.

---

## Documentation Index

Primary documentation hub for the trait-interp project.

### Core Documentation
- **[docs/main.md](main.md)** (this file) - Project overview and codebase reference
- **[docs/overview.md](overview.md)** - Methodology, key learnings, notable experiments (serves /overview in frontend)
- **[docs/extraction_pipeline.md](extraction_pipeline.md)** - Complete 5-stage extraction pipeline
- **[docs/architecture.md](architecture.md)** - Design principles and organization
- **[README.md](../readme.md)** - Quick start guide

### Pipeline & Extraction
- **[extraction/elicitation_guide.md](../extraction/elicitation_guide.md)** - Natural elicitation method
- **[docs/extraction_guide.md](extraction_guide.md)** - Comprehensive extraction reference
- **[docs/trait_dataset_creation_agent.md](trait_dataset_creation_agent.md)** - Creating trait datasets
- **[docs/automated_dataset_refinement.md](automated_dataset_refinement.md)** - Iterative vetting and refinement loop

### Inference & Steering
- **[inference/README.md](../inference/README.md)** - Per-token monitoring
- **[analysis/steering/README.md](../analysis/steering/README.md)** - Steering evaluation

### Visualization
- **[visualization/README.md](../visualization/README.md)** - Dashboard usage
- **[docs/logit_lens.md](logit_lens.md)** - Prediction evolution across layers

### Technical Reference
- **[docs/core_reference.md](core_reference.md)** - core/ API (hooks, methods, math)
- **[docs/gemma-2-2b-it.md](gemma-2-2b-it.md)** - Model data format reference
- **[config/paths.yaml](../config/paths.yaml)** - Path configuration

### Infrastructure
- **[docs/remote_setup.md](remote_setup.md)** - Remote GPU setup
- **[docs/r2_sync.md](r2_sync.md)** - R2 cloud sync

### Research (docs/other/)
- **[docs/other/research_findings.md](other/research_findings.md)** - Empirical results (method comparisons, cross-model transfer, validation studies)
- **[docs/other/literature_review.md](other/literature_review.md)** - 100+ papers analyzed
- **[docs/other/insights.md](other/insights.md)** - Research insights

---

## Codebase Navigation

### Directory Structure
```
trait-interp/
├── datasets/               # Model-agnostic inputs (shared across experiments)
│   ├── inference/                     # Prompt sets (harmful.json, jailbreak.json, etc.)
│   └── traits/{category}/{trait}/     # Trait definitions
│       ├── positive.txt, negative.txt # Contrasting scenarios
│       ├── definition.txt             # Trait description
│       └── steering.json              # Steering eval questions
│
├── extraction/             # Vector extraction pipeline
│   ├── run_pipeline.py               # Full pipeline orchestrator
│   ├── generate_responses.py         # Generate from scenarios
│   ├── extract_activations.py        # Capture hidden states
│   └── extract_vectors.py            # Extract trait vectors
│
├── inference/              # Per-token monitoring
│   ├── capture_raw_activations.py    # Capture hidden states
│   └── project_raw_activations_onto_traits.py  # Project onto vectors
│
├── experiments/            # Experiment data
│   └── {experiment_name}/
│       ├── config.json               # Model settings
│       ├── extraction/{trait}/       # Vectors, activations, responses
│       │   ├── responses/pos.json, neg.json
│       │   ├── activations/{position}/{component}/
│       │   │   ├── train_all_layers.pt, val_all_layers.pt
│       │   │   └── metadata.json
│       │   └── vectors/{position}/{component}/{method}/layer*.pt
│       ├── inference/                # Raw activations, projections
│       ├── steering/{trait}/{position}/  # Steering results
│       └── benchmark/                # Benchmark results
│
├── config/
│   ├── paths.yaml                    # Single source of truth for paths
│   └── models/*.yaml                 # Model architecture configs
│
├── core/                   # Primitives (types, hooks, methods, math)
├── utils/                  # Shared utilities (paths, model loading)
├── server/                 # Model server (persistent model loading)
├── analysis/               # Analysis scripts (steering, benchmark)
├── visualization/          # Interactive dashboard
└── docs/                   # Documentation
```

### Key Entry Points

**Extract new traits:**
```bash
python extraction/run_pipeline.py \
    --experiment {experiment} \
    --traits {category}/{trait}
```

**Monitor with existing vectors:**
```bash
# 1. Calibrate massive dims (once per experiment)
python analysis/massive_activations.py --experiment {experiment}

# 2. Capture activations
python inference/capture_raw_activations.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}

# 3. Project onto traits
python inference/project_raw_activations_onto_traits.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}
```

**Use core primitives:**
```python
from core import VectorSpec, ProjectionConfig, CaptureHook, SteeringHook, get_method, projection
```

**Benchmark capability preservation (for ablation):**
```bash
python analysis/benchmark/evaluate.py \
    --experiment {experiment} \
    --benchmark hellaswag \
    --steer {trait} --coef -1.0
```

**Interpret vectors via logit lens** (runs automatically in pipeline, or standalone):
```bash
python analysis/vectors/logit_lens.py \
    --experiment {experiment} \
    --trait {category}/{trait} \
    --filter-common  # Filter to interpretable tokens
```

### How Components Interact

```
core/               ← Primitives (types, hooks, methods, math)
    ↑
    ├── Used by: extraction/
    └── Used by: inference/

utils/              ← Shared utilities (paths, model loading)
    ↑
    └── Used by: all modules

extraction/         ← Creates trait vectors
    └── Produces: experiments/{name}/extraction/{trait}/vectors/

inference/          ← Monitors during generation
    └── Produces: experiments/{name}/inference/{trait}/
```

---

## What This Does

1. **Extract trait vectors** from naturally contrasting scenarios (harmful vs benign prompts)
2. **Monitor traits** token-by-token during generation
3. **Validate vectors** via steering (causal intervention)

Natural elicitation avoids instruction-following confounds. See [extraction/elicitation_guide.md](../extraction/elicitation_guide.md).

---

## Quick Start

```bash
pip install -r requirements.txt
export HF_TOKEN=your_token_here  # For huggingface models
```

**Extract a trait:**
```bash
# 1. Create scenario files in datasets/traits/{category}/{trait}/
#    positive.txt, negative.txt, definition.txt, steering.json

# 2. Run pipeline
python extraction/run_pipeline.py --experiment {experiment} --traits {category}/{trait}

# With custom position (default: response[:])
python extraction/run_pipeline.py --experiment {experiment} --traits {category}/{trait} --position "response[-1]"
```

**Visualize:**
```bash
python visualization/serve.py  # Visit http://localhost:8000/
```

---

## How It Works

### Extraction

Trait vectors are directions in activation space separating positive from negative examples.

**Methods** (in `core/methods.py`):
| Method | Description | Best For |
|--------|-------------|----------|
| `mean_diff` | `mean(pos) - mean(neg)` | Baseline |
| `probe` | Logistic regression weights | High-separability traits |
| `gradient` | Optimize to maximize separation | Low-separability traits |
| `random_baseline` | Random unit vector | Sanity check (~50%) |

**Components** (hook locations):
| Component | Hook Path | Dimension | Notes |
|-----------|-----------|-----------|-------|
| `residual` | `model.layers.{L}` | 2304 | Layer output |
| `attn_out` | `model.layers.{L}.self_attn.o_proj` | 2304 | Raw attention output |
| `mlp_out` | `model.layers.{L}.mlp.down_proj` | 2304 | Raw MLP output |
| `attn_contribution` | Auto-detected | 2304 | What attention adds to residual* |
| `mlp_contribution` | Auto-detected | 2304 | What MLP adds to residual* |
| `k_proj` | `model.layers.{L}.self_attn.k_proj` | 1024 | Key projections |
| `v_proj` | `model.layers.{L}.self_attn.v_proj` | 1024 | Value projections |

*Contribution components require `model` parameter and auto-detect architecture. For Gemma-2 (which has post-sublayer norms), they hook the post-norm output. For Llama/Mistral/Qwen, same as raw output.

### Monitoring

Project hidden states onto trait vectors:
```
score = (hidden_state @ trait_vector) / ||trait_vector||
```
- Positive → expressing trait
- Negative → avoiding trait

**Layer selection:** Middle layers (6-16) generally best. Use `extraction_evaluation.py` to find optimal layer per trait. Steering results provide ground truth.

### Best Vector Selection

Automated via `utils/vectors.py:get_best_vector(experiment, trait)`:
1. **Steering results** (ground truth) — best delta with coherence ≥ 70
2. **Effect size** (fallback) — from extraction_evaluation.json

Searches all positions/components. Pass `component` and `position` to filter.

**Position syntax:** `<frame>[<slice>]` where frame is `prompt`, `response`, or `all`
- `response[:]` — All response tokens (default)
- `response[-1]` — Last response token only
- `prompt[-1]` — Last prompt token

---

## Model Support

**Gemma 2 2B IT** (default):
- 26 layers (0-25), hidden dim 2304
- GQA: 8 query heads, 4 KV heads
- Mac: Requires PyTorch nightly for MPS

**Experiment config** (`experiments/{experiment}/config.json`):
```json
{
  "defaults": {
    "extraction": "base",
    "application": "instruct"
  },
  "model_variants": {
    "base": {"model": "{base_model}"},
    "instruct": {"model": "{instruct_model}"},
    "with_lora": {
      "model": "{instruct_model}",
      "lora": "{lora_adapter}"
    }
  }
}
```

---

## Troubleshooting

**Scenario files not found:**
```bash
# Create in datasets/traits/{category}/{trait}/
vim datasets/traits/category/my_trait/positive.txt
vim datasets/traits/category/my_trait/negative.txt
```

**Low vector separation (contrast < 20):**
- Add more contrasting scenarios
- Try probe method instead of mean_diff

**Out of memory:**
- Batch size auto-calculated from per-GPU free VRAM (uses min across GPUs for multi-GPU)
- Mode-aware: `generation` mode includes 1.5x overhead for `model.generate()` internals
- Diagnostic printed: `Auto batch size: X (mode=Y, free=Z GB, per_seq=W MB)`
- On Apple Silicon: auto-detects 50% of available unified memory (override with `MPS_MEMORY_GB`)

**MPS errors on Mac:**
```bash
# Requires PyTorch nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

---

## Model Server (Optional)

Keep the model loaded between script runs to avoid reload time:

```bash
# Terminal 1: Start server
python server/app.py --port 8765 --model {model}

# Terminal 2: Scripts auto-detect server
python inference/capture_raw_activations.py --experiment {experiment} --prompt-set {prompt_set}
```

Scripts automatically use the server if running, otherwise load locally. Use `--no-server` to force local loading.

---

## Visualization

Start the server:
```bash
python visualization/serve.py  # Visit http://localhost:8000/
```

**Views:**
- **Trait Extraction** — Best vectors summary, per-trait layer×method heatmaps, logit lens token decode
- **Steering Sweep** — Method comparison, layer×coefficient heatmaps, optimal coefficient curves
- **Live Chat** — Interactive chat with real-time trait monitoring token-by-token
- **Inference** — Token trajectory, per-layer activation magnitude
- **Layer Deep Dive** — Attention heatmaps, SAE feature decomposition

Auto-discovers experiments, traits, and prompts from `experiments/` directory.

---

## Further Reading

- **[extraction_pipeline.md](extraction_pipeline.md)** - Full pipeline documentation
- **[core_reference.md](core_reference.md)** - API reference
- **[extraction_guide.md](extraction_guide.md)** - Comprehensive extraction details
