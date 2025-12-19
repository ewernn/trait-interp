# Trait Vector Extraction and Monitoring

Extract and monitor LLM behavioral traits token-by-token during generation.

---

## Documentation Index

This is the **primary documentation hub** for the trait-interp project. All documentation files are organized here.

### Core Documentation (Start Here)
- **[docs/project_brief.md](project_brief.md)** - Self-contained project summary (paste into any chat)
- **[docs/main.md](main.md)** (this file) - Complete project overview and reference
- **[docs/overview.md](overview.md)** - High-level methodology and concepts (also at `/overview` in visualization)
- **[README.md](../readme.md)** - Quick start guide and repository structure
- **[docs/extraction_pipeline.md](extraction_pipeline.md)** - Complete 5-stage extraction pipeline
- **[docs/architecture.md](architecture.md)** - Design principles and organizational structure

### Trait Design & Creation
- **[docs/writing_natural_prompts.md](writing_natural_prompts.md)** - Writing high-quality natural elicitation prompts

### Pipeline & Extraction
- **[extraction/elicitation_guide.md](../extraction/elicitation_guide.md)** - Natural elicitation method

### Experiments & Analysis
- Experiment data stored in `experiments/{experiment_name}/` (see Directory Structure below)
- Analysis visualizations are live-rendered in Trait Dynamics view (no separate docs needed)

### Inference & Monitoring
- **[inference/README.md](../inference/README.md)** - Per-token inference and dynamics capture

### Steering & Validation
- **[analysis/steering/README.md](../analysis/steering/README.md)** - Steering evaluation (causal intervention)
- **[analysis/steering/PLAN.md](../analysis/steering/PLAN.md)** - Implementation plan and methodology

### Research & Methodology
- **[docs/rm_sycophancy_detection_plan.md](rm_sycophancy_detection_plan.md)** - RM sycophancy detection via trait decomposition (active research plan)
- **[docs/steering_results.md](steering_results.md)** - Steering evaluation results log (chronological)
- **[docs/research_findings.md](research_findings.md)** - Empirical experiment results (EM replication, token dynamics)
- **[docs/insights.md](insights.md)** - Key research findings and discoveries
- **[docs/conceptual_framework.md](conceptual_framework.md)** - Mental models and theoretical foundations
- **[docs/overview.md](overview.md)** - Research methodology and framework
- **[docs/literature_review.md](literature_review.md)** - Literature review (100+ papers analyzed)
- **[docs/extraction_guide.md](extraction_guide.md)** - Comprehensive extraction reference (methods, location, validation)
- **[docs/vector_evaluation_framework.md](vector_evaluation_framework.md)** - Multi-axis evaluation framework
- **[docs/analysis_ideas.md](analysis_ideas.md)** - Brainstorm of analysis methods to explore and validate
- **[docs/future_ideas.md](future_ideas.md)** - Research extensions and technical improvements

### Visualization & Monitoring
- **[visualization/README.md](../visualization/README.md)** - Interactive dashboard usage guide
- **[visualization/ARCHITECTURE.md](../visualization/ARCHITECTURE.md)** - Modular architecture and development guide
- **[visualization/DESIGN_STANDARDS.md](../visualization/DESIGN_STANDARDS.md)** - Enforced design standards for visualization UI
- **[docs/logit_lens.md](logit_lens.md)** - Logit lens: prediction evolution across layers

### Infrastructure & Setup
- **[config/paths.yaml](../config/paths.yaml)** - Single source of truth for all repo paths
- **[docs/traitlens_reference.md](traitlens_reference.md)** - traitlens API quick reference (extraction toolkit)
- **[docs/gemma-2-2b-it.md](gemma-2-2b-it.md)** - Gemma-2-2B-IT data format reference (tensor shapes, capture structures)
- **[docs/numerical_stability_analysis.md](numerical_stability_analysis.md)** - Float16/float32 precision handling
- **[docs/remote_setup.md](remote_setup.md)** - Remote GPU instance setup guide
- **[sae/README.md](../sae/README.md)** - Sparse Autoencoder (SAE) integration for interpretable feature analysis
- **[unit_tests/README.md](../unit_tests/README.md)** - Unit tests for extraction pipeline

### Meta-Documentation
- **[docs/doc-update-guidelines.md](doc-update-guidelines.md)** - Guidelines for updating documentation

---

## Codebase Navigation

### Directory Structure
```
trait-interp/
├── datasets/               # Model-agnostic inputs (shared across experiments)
│   ├── inference/                     # Prompt sets for testing (by type)
│   │   ├── harmful.json, jailbreak.json, etc.
│   └── traits/{category}/{trait}/     # Trait definitions (by trait)
│       ├── positive.txt               # Positive scenarios
│       ├── negative.txt               # Negative scenarios
│       ├── definition.txt             # Trait description
│       └── steering.json              # Steering eval questions
│
├── extraction/             # Vector extraction pipeline (training time)
│   ├── run_pipeline.py               # Full pipeline orchestrator (recommended)
│   ├── vet_scenarios.py              # LLM-as-a-judge scenario vetting
│   ├── vet_responses.py              # LLM-as-a-judge response vetting
│   ├── generate_responses.py         # Generate responses from natural scenarios
│   ├── extract_activations.py        # Extract activations from responses
│   ├── extract_vectors.py            # Extract trait vectors from activations
│   └── elicitation_guide.md          # Guide for creating new traits
│
├── inference/              # Per-token monitoring (inference time)
│   ├── capture_raw_activations.py     # Capture hidden states from model
│   ├── project_raw_activations_onto_traits.py  # Project onto trait vectors
│   └── README.md                      # Usage guide
│
├── lora/                   # LoRA-based trait extraction (experimental)
│   ├── README.md           # LoRA methodology and 5-step process documentation
│   ├── scripts/            # Evil LoRA generation and training
│   │   ├── generate_evil_data.py  # Generate evil responses with Claude 3.7
│   │   └── train_evil_lora.py     # Fine-tune LoRA adapter
│   ├── data/               # Generated training data
│   └── models/             # Trained LoRA adapters
│
├── experiments/            # Experiment data and user analysis space
│   └── {experiment_name}/              # Your experiment (e.g., my_experiment)
│       ├── config.json                 # Model settings (auto-created on first run)
│       ├── extraction/{category}/{trait}/  # Training-time data
│       │   ├── generation_metadata.json
│       │   ├── responses/          # Generated examples (pos.json, neg.json)
│       │   ├── vetting/            # LLM-as-a-judge quality scores
│       │   ├── activations/        # Token-averaged hidden states
│       │   └── vectors/            # Extracted trait vectors
│       ├── inference/              # Evaluation-time monitoring
│       │   ├── raw/                # Trait-independent raw activations (.pt)
│       │   │   ├── residual/{prompt_set}/{id}.pt          # All layers
│       │   │   └── internals/{prompt_set}/{id}_L{layer}.pt  # Single layer deep (optional)
│       │   ├── responses/{prompt_set}/{id}.json  # Shared prompt/response data (trait-independent)
│       │   └── {category}/{trait}/
│       │       └── residual_stream/{prompt_set}/{id}.json  # Slim projections only
│       ├── analysis/               # Analysis outputs (view in Trait Dynamics)
│       │   └── per_token/{prompt_set}/  # Per-token JSON for live-rendered gallery
│       │       └── {id}.json            # Token metrics, trait scores, velocity
│       └── steering/{category}/{trait}/ # Steering evaluation results
│           └── results.json             # Runs-based steering results (accumulates)
│
├── config/                 # Configuration files
│   ├── paths.yaml         # Single source of truth for all repo paths
│   └── models/            # Model architecture configs
│       ├── gemma-2-2b-it.yaml
│       ├── gemma-2-2b.yaml
│       ├── qwen2.5-7b.yaml
│       └── qwen2.5-7b-instruct.yaml
│
├── utils/                  # Shared utilities
│   ├── paths.py           # Python PathBuilder (loads from config/paths.yaml)
│   ├── model_registry.py  # Model architecture registry (loads from config/models/)
│   ├── generation.py      # Batched generation with activation capture, VRAM utilities
│   ├── vectors.py         # Best vector selection from evaluation results
│   └── model.py           # Model loading, prompt formatting, experiment config
│
├── sae/                    # Sparse Autoencoder (SAE) resources
│   ├── README.md           # SAE documentation
│   ├── download_fast.py    # Download feature labels from Neuronpedia
│   └── gemma-scope-2b-pt-res-canonical/
│       └── layer_16_width_16k_canonical/
│           ├── feature_labels.json  # All 16k feature descriptions
│           └── metadata.json
│
├── analysis/               # Analysis scripts
│   ├── data_checker.py               # Check what data exists for experiments
│   ├── vectors/
│   │   ├── extraction_evaluation.py   # Evaluate vectors on held-out data
│   │   └── vector_ranking.py          # Rank vectors by quality metrics
│   ├── inference/
│   │   ├── attention_decay_analysis.py    # Analyze attention patterns
│   │   └── commitment_point_detection.py  # Find trait commitment points
│   └── steering/
│       ├── steer.py                   # Steering hook context manager
│       ├── judge.py                   # LLM-as-judge with logprob scoring
│       ├── evaluate.py                # Evaluation + layer sweep (unified)
│       └── prompts/                   # Eval questions per trait (JSON)
│
├── docs/                   # Documentation (you are here)
├── visualization/          # Interactive visualization dashboard
└── requirements.txt        # Python dependencies
```

### Key Entry Points

**For using existing vectors:**
```bash
# Capture activations + project onto all traits
python inference/capture_raw_activations.py \
    --experiment {experiment_name} \
    --prompt "Your prompt here"

# Re-project saved raw activations onto traits (auto-selects best layer per trait)
python inference/project_raw_activations_onto_traits.py \
    --experiment {experiment_name} \
    --prompt-set single_trait

# Project attn_out activations onto attn_out vectors
python inference/project_raw_activations_onto_traits.py \
    --experiment {experiment_name} \
    --prompt-set harmful \
    --component attn_out \
    --layer 8
```

**For extracting new traits:**
```bash
# 1. Create experiment with config
mkdir -p experiments/gemma-2-2b/extraction/category/my_trait
cat > experiments/gemma-2-2b/config.json << 'EOF'
{
  "extraction_model": "google/gemma-2-2b",
  "application_model": "google/gemma-2-2b-it"
}
EOF

# 2. Create scenario files (in datasets/traits/)
vim datasets/traits/category/my_trait/positive.txt
vim datasets/traits/category/my_trait/negative.txt
vim datasets/traits/category/my_trait/definition.txt
vim datasets/traits/category/my_trait/steering.json

# 4. Run full pipeline (extract + evaluate + steer)
python extraction/run_pipeline.py \
    --experiment gemma-2-2b \
    --traits category/my_trait
```

**For custom analysis using traitlens:**
```python
from traitlens import HookManager, ActivationCapture, ProbeMethod
# See https://github.com/ewernn/traitlens for examples and documentation
```

### How Components Interact

```
traitlens (pip package) ← Installed from GitHub
    ↑
    ├── Used by: extraction/
    └── Used by: experiments/

utils/              ← Shared utilities (path management)
    ↑
    └── Used by: all modules

extraction/         ← Training time (creates trait vectors)
    ├── Uses: traitlens + utils/
    └── Produces: experiments/{name}/extraction/{category}/{trait}/vectors/

experiments/        ← User space (custom analysis using extracted vectors)
    ├── Uses: traitlens
    └── Structure: extraction/{category}/{trait}/, inference/
```

---

## What This Does

This project extracts trait vectors from language models and monitors them during generation. You can:

1. **Extract trait vectors** from naturally contrasting scenarios
2. **Monitor traits** token-by-token to see how they evolve during generation
3. **Analyze dynamics** - velocity and acceleration computed on-the-fly in visualization

Natural elicitation avoids instruction-following confounds. See extraction/elicitation_guide.md.

**Discover available traits in your experiment:**
```bash
find experiments/{experiment_name}/extraction -name "vectors" -type d | sed 's|.*/extraction/||' | sed 's|/vectors||' | sort
```

See `experiments/{experiment_name}/README.md` for detailed trait descriptions and design principles.

## Quick Start

### Installation

```bash
git clone https://github.com/ewernn/trait-interp.git
cd trait-interp
pip install -r requirements.txt
```

This installs:
- Core dependencies (torch, transformers, etc.)
- traitlens library from GitHub (with extraction methods)

Set up HuggingFace token:
```bash
# For downloading Gemma models
export HF_TOKEN=your_token_here
```

**Local model cache**: Gemma 2B (5 GB) is cached at `~/.cache/huggingface/hub/` and automatically used by all scripts. No manual download needed - models download on first use.

**Mac GPU Support (Apple Silicon)**: Requires PyTorch 2.10.0+ nightly for MPS backend. See Troubleshooting section below for installation.

### Use Existing Vectors

Pre-extracted vectors are available for specific experiments.

Discover extracted traits:
```bash
find experiments/{experiment_name}/extraction -name "vectors" -type d | sed 's|.*/extraction/||' | sed 's|/vectors||' | sort
```

Use traitlens to create monitoring scripts - see the Monitoring section below for examples.

### Extract Your Own Traits

```bash
# 1. Create experiment directory and config
mkdir -p experiments/gemma-2-2b/extraction/category/my_trait
cat > experiments/gemma-2-2b/config.json << 'EOF'
{
  "extraction_model": "google/gemma-2-2b",
  "application_model": "google/gemma-2-2b-it"
}
EOF

# 2. Create trait files in datasets/traits/
mkdir -p datasets/traits/category/my_trait
vim datasets/traits/category/my_trait/positive.txt
vim datasets/traits/category/my_trait/negative.txt
vim datasets/traits/category/my_trait/definition.txt
vim datasets/traits/category/my_trait/steering.json

# 3. Run full pipeline
python extraction/run_pipeline.py \
    --experiment gemma-2-2b \
    --traits category/my_trait

# Results (all in same experiment):
#   - Vectors: experiments/gemma-2-2b/extraction/.../vectors/
#   - Eval: experiments/gemma-2-2b/extraction/extraction_evaluation.json
#   - Steering: experiments/gemma-2-2b/steering/.../results.json
```

See [extraction/elicitation_guide.md](../extraction/elicitation_guide.md) for details.

## How It Works

### Extraction

Trait vectors are directions in activation space that separate positive from negative examples of a trait.

**Natural Elicitation**:
1. Use naturally contrasting scenarios (harmful vs benign prompts)
2. Generate responses WITHOUT instructions
3. Capture activations from all model layers
4. Apply extraction method (mean difference, probe, or gradient)
5. Result: vector that measures genuine trait expression

**Extraction methods** (4 total):
- **Mean difference** (baseline): `vector = mean(pos) - mean(neg)`
- **Linear probe**: Train logistic regression, use weights as vector (supports L1/L2 via `penalty` param)
- **Gradient**: Optimize vector to maximize separation
- **Random baseline**: Random unit vector (sanity check, should get ~50% accuracy)

**Extraction components** (5 total):
| Component | Hook Path | Dimension | Description |
|-----------|-----------|-----------|-------------|
| `residual` | `model.layers.{L}` | 2304 | Full layer output (default) |
| `attn_out` | `model.layers.{L}.self_attn.o_proj` | 2304 | Attention output projection |
| `mlp_out` | `model.layers.{L}.mlp.down_proj` | 2304 | MLP output projection |
| `k_cache` | `model.layers.{L}.self_attn.k_proj` | 1024 | Key projection (GQA) |
| `v_cache` | `model.layers.{L}.self_attn.v_proj` | 1024 | Value projection (GQA) |

Use `--component` flag in extraction and steering scripts. k_cache/v_cache vectors have different dimensions due to Grouped Query Attention.

Verify dimensions: `python3 -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('google/gemma-2-2b'); print(f'hidden: {c.hidden_size}, kv: {c.num_key_value_heads * c.head_dim}')"`

See [traitlens](https://github.com/ewernn/traitlens) for the extraction toolkit and implementation details.

### Monitoring

During generation, project each token's hidden state onto trait vectors:

```
score = (hidden_state @ trait_vector) / torch.norm(trait_vector)
```

- Positive score → model expressing the trait
- Negative score → model avoiding the trait
- Score magnitude → how strongly

**Layer selection (trait-dependent)**:
- **Gemma 2 2B**:
  - Optimal layer varies by trait—use `extraction_evaluation.py` to find empirically
  - Middle layers (6-16) generally work best for behavioral traits
  - Late layers (21-25) can overfit to training distribution

Middle layers capture semantic meaning and generalize across distributions. **Method choice matters more than layer**: high-separability traits favor Probe, low-separability traits favor Gradient.

**Best vector selection** (automated via `utils/vectors.py:get_best_layer()`):

Priority order:
1. **Steering results** (ground truth) - Best delta from `steering/{trait}/results.json` (coherence ≥ 70)
2. **Effect size** (fallback heuristic) - From `extraction_evaluation.json` `all_results`
3. **Default** - Layer 16, probe method

Always run steering evaluation for ground truth. Effect size is only a rough fallback.

## Extraction Pipeline

Full pipeline: extraction + evaluation + steering. See **[extraction_pipeline.md](extraction_pipeline.md)** for documentation.

**Method similarity computation** (for visualization):
```bash
# Compute cosine similarity between extraction methods (probe/gradient/mean_diff)
# Saves to extraction_evaluation.json for Steering Sweep visualization
python analysis/vectors/compute_method_similarities.py --experiment gemma-2-2b
```

**Quick start:**
```bash
# Full pipeline (uses extraction_model and application_model from config.json)
python extraction/run_pipeline.py \
    --experiment gemma-2-2b \
    --traits epistemic/optimism

# Override models from CLI
python extraction/run_pipeline.py \
    --experiment gemma-2-2b \
    --extraction-model google/gemma-2-2b \
    --application-model google/gemma-2-2b-it \
    --traits epistemic/optimism

# Extraction only (no steering)
python extraction/run_pipeline.py \
    --experiment gemma-2-2b \
    --traits epistemic/optimism \
    --no-steering
```

**Required files per trait (in datasets/traits/{trait}/):**
- `positive.txt`, `negative.txt`, `definition.txt`
- `steering.json` (or use `--no-steering`)

## Monitoring

Monitor trait projections token-by-token during generation.

### Quick Start: capture_raw_activations.py

The unified capture script handles capture with clear flags:

```bash
# Basic: residual stream + project onto all traits (always happens)
python inference/capture_raw_activations.py \
    --experiment {experiment_name} \
    --prompt-set single_trait

# Large prompt set with batching (auto batch size, limit for testing)
python inference/capture_raw_activations.py \
    --experiment {experiment_name} \
    --prompt-set jailbreak \
    --limit 10

# With attention/logit-lens for Layer Deep Dive visualization
python inference/capture_raw_activations.py \
    --experiment {experiment_name} \
    --prompt-set dynamic \
    --attention --logit-lens

# Extract from existing captures (no new generation)
python inference/capture_raw_activations.py \
    --experiment {experiment_name} \
    --prompt-set single_trait \
    --replay --attention
```

**Available prompt sets** (JSON files in `datasets/inference/`):
- `single_trait` - 10 prompts, each targeting one specific trait
- `multi_trait` - 10 prompts activating multiple traits
- `dynamic` - 8 prompts designed to cause trait changes mid-response
- `adversarial` - 8 edge cases and robustness tests
- `baseline` - 5 neutral prompts for baseline measurement
- `real_world` - 10 naturalistic prompts
- `harmful` - 5 harmful requests (for refusal testing)
- `benign` - 5 benign requests (parallel to harmful)

The script:
- **Dynamically discovers** all traits with vectors (no hardcoded categories)
- **Captures once, projects to all traits** - efficient multi-trait processing
- **Saves raw activations (.pt)** + slim per-trait projection JSONs (best layer only)
- **Crash resilient** - saves after each batch; use `--skip-existing` to resume

See [inference/README.md](../inference/README.md) for detailed usage.

### Custom Monitoring Scripts

Use traitlens to create custom monitoring scripts in your experiment:

```python
# experiments/{name}/inference/monitor.py
from traitlens import HookManager, ActivationCapture, projection
import torch

# Load vectors
vectors = {
    'refusal': torch.load('experiments/{experiment_name}/extraction/behavioral/refusal/vectors/probe_layer16.pt'),
    # ... more traits
}

# Monitor during generation
capture = ActivationCapture()
with HookManager(model) as hooks:
    hooks.add_forward_hook("model.layers.16", capture.make_hook("layer_16"))
    output = model.generate(**inputs)

# Calculate projections
acts = capture.get("layer_16")
trait_scores = {name: projection(acts, vec) for name, vec in vectors.items()}
```

Save results wherever you like - the visualizer can read custom monitoring data.

### Visualize

Start the visualization server:

```bash
# From trait-interp root directory
python visualization/serve.py
# Then visit http://localhost:8000/
```

The visualization provides:
- **Category 1: Trait Development**
  - **Trait Extraction**: Comprehensive view of extraction quality. Per-trait layer×method heatmaps (10 per row), best-vector similarity matrix (trait independence), metric distributions, per-method breakdown. Collapsible reference sections for notation, extraction techniques, quality metrics, and scoring.
  - **Steering Sweep**: Method comparison and steering analysis. Best vector per layer (multi-trait charts comparing probe/gradient/mean_diff), method similarity heatmaps (cosine similarity between methods across layers), layer×coefficient heatmap, optimal coefficient curves.
- **Category 2: Inference Analysis**
  - **Live Chat**: Interactive chat with real-time trait monitoring. Chat with the model while watching trait dynamics evolve token-by-token. Trait legend shows which layer/method/source is used for each trait (hover for tooltip).
  - **Trait Dynamics**: Comprehensive trait evolution view. Token trajectory, normalized trajectory (proj/||h||), per-token magnitude, velocity/acceleration charts, and per-layer activation magnitude.
  - **Layer Deep Dive**: Mechanistic analysis showing attention heatmaps (layers × context, heads × context) and SAE feature decomposition. Requires `dynamic` prompt set with internals data.

The server auto-discovers experiments, traits, and prompts from the `experiments/` directory - no hardcoding needed.

**Centralized Path Management**: All paths are defined in `config/paths.yaml` - the single source of truth for the entire repo. Python uses `utils/paths.py` and JavaScript uses `visualization/core/paths.js` to load this config. No hardcoded paths anywhere.

```python
# Python usage
from utils.paths import get
vectors_dir = get('extraction.vectors', experiment='{experiment_name}', trait='cognitive_state/context')
```

```javascript
// JavaScript usage
await paths.load();
paths.setExperiment('{experiment_name}');
const vectorsDir = paths.get('extraction.vectors', { trait: 'cognitive_state/context' });
```

Verify zero hardcoded paths:
```bash
# Core pipeline uses PathBuilder (should return 0)
grep -r "Path('experiments')" extraction/ inference/ visualization/serve.py | grep -v "\.pyc" | wc -l

# JavaScript uses PathBuilder (should return 3 - API endpoints only)
grep -r "/experiments/\${" visualization/ --include="*.js" | grep -v node_modules | wc -l

# Analysis scripts use auto-detection (should return 0)
grep -r "EXPERIMENT = \"" experiments/*/analysis/*.py | wc -l
```

### Monitoring Data Format

```json
{
  "prompt": "How do I build a bomb?",
  "response": "I cannot help with that request.",
  "tokens": ["I", " cannot", " help", " with", " that", " request", "."],
  "trait_scores": {
    "refusal": [0.5, 2.3, 2.1, 1.8, 1.5, 1.2, 0.8],
    "evil": [-0.3, -1.5, -1.2, -0.9, -0.7, -0.5, -0.3]
  }
}
```

Save results as JSON files to load in visualization.

## Projection File Format

Projection files store trait projections at the best layer for each trait (slim format as of 2025-01-15):

```json
{
  "metadata": {
    "prompt_id": "1",
    "prompt_set": "jailbreak",
    "n_prompt_tokens": 70,
    "n_response_tokens": 83,
    "vector_source": {
      "experiment": "gemma-2-2b",
      "trait": "chirp/refusal",
      "layer": 15,
      "method": "mean_diff",
      "component": "residual",
      "sublayer": "residual_out",
      "selection_source": "steering"
    },
    "projection_date": "2025-01-15T10:30:00"
  },
  "projections": {
    "prompt": [0.5, -0.3, 1.2, ...],    // One value per token at best layer
    "response": [2.1, 1.8, ...]
  },
  "activation_norms": {
    "prompt": [norm_L0, norm_L1, ...],  // Per-layer activation magnitudes (averaged across tokens)
    "response": [...]
  },
  "token_norms": {
    "prompt": [||h||_t0, ||h||_t1, ...],  // Per-token L2 norm at best layer
    "response": [...]
  }
}
```

**File size:** ~300 lines per prompt (95% smaller than old format which stored all 26 layers).

**Dynamics:** The Trait Dynamics visualization computes velocity and acceleration from projections on-the-fly. No pre-computed dynamics stored in files.

### Custom Dynamics Analysis

Build custom analysis using traitlens primitives. See [traitlens documentation](https://github.com/ewernn/traitlens) for `compute_derivative()`, `compute_second_derivative()`, and other dynamics functions.

For legacy dynamics scripts, see `analysis/inference/commitment_point_detection.py` which uses sliding window variance instead of acceleration.

## Steering Validation

Validate trait vectors via causal intervention - add `coefficient * vector` to layer output during generation and measure behavioral change with LLM-as-judge.

```bash
# Basic usage - sweeps all layers, finds good coefficients automatically
python analysis/steering/evaluate.py \
    --experiment {experiment} \
    --vector-from-trait {experiment}/{category}/{trait}

# Specific layers only
python analysis/steering/evaluate.py \
    --experiment {experiment} \
    --vector-from-trait {experiment}/{category}/{trait} \
    --layers 10,12,14,16
```

See [analysis/steering/README.md](../analysis/steering/README.md) for full usage guide.

## Creating New Traits

### 1. Create Required Files

```bash
# Create trait directory with scenario files
mkdir -p datasets/traits/category/my_trait
vim datasets/traits/category/my_trait/positive.txt
vim datasets/traits/category/my_trait/negative.txt
vim datasets/traits/category/my_trait/definition.txt
vim datasets/traits/category/my_trait/steering.json
```

See existing trait directories for examples.

### 2. Run Pipeline

```bash
# Full pipeline: extract + evaluate + steer
python extraction/run_pipeline.py \
    --experiment gemma-2-2b \
    --traits category/my_trait
```

The pipeline:
- Vets scenarios and responses (LLM-as-a-judge)
- Extracts vectors using `extraction_model` from config
- Evaluates vector quality
- Runs steering eval using `application_model` from config

See [extraction/elicitation_guide.md](../extraction/elicitation_guide.md) for complete instructions.

## Model Support

**Gemma 2 2B IT**:
- Layers: 26 (transformer layers, indexed 0-25)
- Hidden dim: 2304
- Default monitor layer: 16 (middle layer)
- Architecture: Grouped Query Attention (GQA) - 8 query heads, 4 key/value heads
- Cached locally at `~/.cache/huggingface/hub/`
- GPU support: CUDA, ROCm, MPS (PyTorch 2.10.0+ nightly required for Mac)

**Verify config:**
```bash
python3 -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('google/gemma-2-2b-it'); print(f'Layers: {c.num_hidden_layers}, Hidden: {c.hidden_size}')"
```

## Experiment Configuration

Each experiment stores model settings in `experiments/{name}/config.json`:

```json
{
  "extraction_model": "google/gemma-2-2b",
  "application_model": "google/gemma-2-2b-it"
}
```

- `extraction_model`: HuggingFace ID for response generation and activation extraction
- `application_model`: HuggingFace ID for steering evaluation and inference

**Chat template:** Auto-detected from tokenizer (`tokenizer.chat_template is not None`):
- Base models (no template) → raw text completion
- IT models (has template) → chat formatting

**CLI overrides** (rarely needed):
```bash
# Force chat template on
python extraction/generate_responses.py --experiment my_exp --chat-template ...

# Force chat template off
python extraction/generate_responses.py --experiment my_exp --no-chat-template ...
```

**Verify config exists:**
```bash
cat experiments/my_experiment/config.json
```

## Model Configuration

Model architecture and defaults are stored in `config/models/{model}.yaml`:

```yaml
# config/models/gemma-2-2b-it.yaml
huggingface_id: google/gemma-2-2b-it
model_type: gemma2
variant: it

num_hidden_layers: 26
hidden_size: 2304
num_attention_heads: 8
num_key_value_heads: 4

residual_sublayers: [input, after_attn, output]

sae:
  available: true
  provider: gemma-scope
  base_path: sae/gemma-scope-2b-pt-res-canonical
  downloaded_layers: [16]

defaults:
  monitoring_layer: 16
```

**Python usage:**
```python
from utils.model_registry import get_model_config, get_num_layers, get_sae_path

config = get_model_config('google/gemma-2-2b-it')
config['num_hidden_layers']  # 26

get_num_layers('google/gemma-2-2b-it')  # 26
get_sae_path('google/gemma-2-2b-it', 16)  # Path to SAE directory
```

**JavaScript usage:**
```javascript
await modelConfig.loadForExperiment('{experiment}');
modelConfig.getNumLayers();  // 26
modelConfig.getSaePath(16);  // 'sae/gemma-scope.../layer_16_...'
```

**Available models:**
```bash
ls config/models/
```

## Technical Details

### Extraction Methods

See [traitlens](https://github.com/ewernn/traitlens) for full implementations (`traitlens.methods` module).

**Example - Linear Probe:**
```python
from traitlens import ProbeMethod
method = ProbeMethod()
result = method.extract(pos_activations, neg_activations)
vector = result['vector']
```

**Available methods:** MeanDifferenceMethod, ProbeMethod, GradientMethod, RandomBaselineMethod. See traitlens docs for usage examples.

### Activation Capture

Capture activations from any layer:

```python
from traitlens import HookManager, ActivationCapture

capture = ActivationCapture()
with HookManager(model) as hooks:
    hooks.add_forward_hook("model.layers.16", capture.make_hook("layer_16"))
    model.generate(**inputs)

activations = capture.get("layer_16")  # [batch, seq_len, hidden_dim]
```

## Troubleshooting

### Extraction fails with "Scenario files not found"
Create scenario files in your experiment's trait directory:
```bash
vim datasets/traits/{category}/{trait}/positive.txt
vim datasets/traits/{category}/{trait}/negative.txt
```
See existing files in your experiment's trait directory for examples.

### Vector separation too low (contrast < 20)
- Add more contrasting scenarios
- Make scenarios more extreme
- Try different extraction method (probe often better than mean_diff)

### Out of memory during activation extraction
Reduce batch size:
```bash
python extraction/extract_activations.py ... --batch_size 4
```

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### MPS errors on Mac (incompatible dimensions, LLVM ERROR)
Gemma 2B requires PyTorch nightly for MPS support:

```bash
# Create Python 3.11 environment
conda create -n o python=3.11
conda activate o

# Install PyTorch nightly (required for MPS + GQA support)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Install all dependencies
pip install -r requirements.txt

# Verify MPS
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

PyTorch stable (2.6.0) crashes due to Grouped Query Attention incompatibility. Python 3.11+ and PyTorch nightly (2.10.0+) required.

## Further Reading

- **[docs/extraction_pipeline.md](extraction_pipeline.md)** - Complete extraction pipeline guide
- **[docs/creating_traits.md](creating_traits.md)** - How to design effective traits
- **[traitlens](https://github.com/ewernn/traitlens)** - Extraction toolkit documentation
- **[docs/overview.md](overview.md)** - High-level methodology and concepts
- **[docs/literature_review.md](literature_review.md)** - Related work
