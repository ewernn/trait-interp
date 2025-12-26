# Trait Vector Extraction and Monitoring

Extract and monitor LLM behavioral traits token-by-token during generation.

---

## Documentation Index

Primary documentation hub for the trait-interp project.

### Core Documentation
- **[docs/main.md](main.md)** (this file) - Project overview and codebase reference
- **[docs/project_brief.md](project_brief.md)** - Self-contained project summary
- **[docs/extraction_pipeline.md](extraction_pipeline.md)** - Complete 5-stage extraction pipeline
- **[docs/architecture.md](architecture.md)** - Design principles and organization
- **[README.md](../readme.md)** - Quick start guide

### Pipeline & Extraction
- **[extraction/elicitation_guide.md](../extraction/elicitation_guide.md)** - Natural elicitation method
- **[docs/writing_natural_prompts.md](writing_natural_prompts.md)** - Writing elicitation prompts
- **[docs/extraction_guide.md](extraction_guide.md)** - Comprehensive extraction reference

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
- **[docs/other/literature_review.md](other/literature_review.md)** - 100+ papers analyzed
- **[docs/other/insights.md](other/insights.md)** - Research findings
- **[docs/rm_sycophancy_detection_plan.md](rm_sycophancy_detection_plan.md)** - Active research
- **[docs/rm_sycophancy_findings.md](rm_sycophancy_findings.md)** - RM sycophancy results

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
│       ├── inference/                # Raw activations, projections
│       └── steering/{trait}/         # Steering results
│
├── config/
│   ├── paths.yaml                    # Single source of truth for paths
│   └── models/*.yaml                 # Model architecture configs
│
├── core/                   # Primitives (hooks, methods, math)
├── utils/                  # Shared utilities (paths, model loading)
├── analysis/               # Analysis scripts
├── visualization/          # Interactive dashboard
└── docs/                   # Documentation
```

### Key Entry Points

**Extract new traits:**
```bash
python extraction/run_pipeline.py \
    --experiment gemma-2-2b \
    --traits category/my_trait
```

**Monitor with existing vectors:**
```bash
# Capture activations
python inference/capture_raw_activations.py \
    --experiment gemma-2-2b \
    --prompt-set harmful

# Project onto traits
python inference/project_raw_activations_onto_traits.py \
    --experiment gemma-2-2b \
    --prompt-set harmful
```

**Use core primitives:**
```python
from core import CaptureHook, SteeringHook, get_method, projection
```

### How Components Interact

```
core/               ← Primitives (hooks, methods, math)
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
export HF_TOKEN=your_token_here  # For Gemma models
```

**Extract a trait:**
```bash
# 1. Create scenario files in datasets/traits/category/my_trait/
#    positive.txt, negative.txt, definition.txt, steering.json

# 2. Run pipeline
python extraction/run_pipeline.py --experiment gemma-2-2b --traits category/my_trait
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
| Component | Hook Path | Dimension |
|-----------|-----------|-----------|
| `residual` | `model.layers.{L}` | 2304 |
| `attn_out` | `model.layers.{L}.self_attn.o_proj` | 2304 |
| `mlp_out` | `model.layers.{L}.mlp.down_proj` | 2304 |
| `k_cache` | `model.layers.{L}.self_attn.k_proj` | 1024 |
| `v_cache` | `model.layers.{L}.self_attn.v_proj` | 1024 |

### Monitoring

Project hidden states onto trait vectors:
```
score = (hidden_state @ trait_vector) / ||trait_vector||
```
- Positive → expressing trait
- Negative → avoiding trait

**Layer selection:** Middle layers (6-16) generally best. Use `extraction_evaluation.py` to find optimal layer per trait. Steering results provide ground truth.

### Best Vector Selection

Automated via `utils/vectors.py:get_best_layer()`:
1. **Steering results** (ground truth) — best delta with coherence ≥ 70
2. **Effect size** (fallback) — from extraction_evaluation.json
3. **Default** — layer 16, probe method

---

## Model Support

**Gemma 2 2B IT** (default):
- 26 layers (0-25), hidden dim 2304
- GQA: 8 query heads, 4 KV heads
- Mac: Requires PyTorch nightly for MPS

**Experiment config** (`experiments/{name}/config.json`):
```json
{
  "extraction_model": "google/gemma-2-2b",
  "application_model": "google/gemma-2-2b-it"
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
```bash
python extraction/extract_activations.py ... --batch_size 4
```

**MPS errors on Mac:**
```bash
# Requires PyTorch nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

---

## Visualization

Start the server:
```bash
python visualization/serve.py  # Visit http://localhost:8000/
```

**Views:**
- **Trait Extraction** — Per-trait layer×method heatmaps, best-vector similarity matrix, metric distributions
- **Steering Sweep** — Method comparison, layer×coefficient heatmaps, optimal coefficient curves
- **Live Chat** — Interactive chat with real-time trait monitoring token-by-token
- **Trait Dynamics** — Token trajectory, velocity/acceleration, per-layer activation magnitude
- **Layer Deep Dive** — Attention heatmaps, SAE feature decomposition

Auto-discovers experiments, traits, and prompts from `experiments/` directory.

---

## Further Reading

- **[extraction_pipeline.md](extraction_pipeline.md)** - Full pipeline documentation
- **[core_reference.md](core_reference.md)** - API reference
- **[extraction_guide.md](extraction_guide.md)** - Comprehensive extraction details
