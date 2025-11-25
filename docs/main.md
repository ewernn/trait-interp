# Trait Vector Extraction and Monitoring

Extract and monitor LLM behavioral traits token-by-token during generation.

---

## Documentation Index

This is the **primary documentation hub** for the trait-interp project. All documentation files are organized here.

### Core Documentation (Start Here)
- **[docs/main.md](main.md)** (this file) - Complete project overview and reference
- **[docs/overview.md](overview.md)** - High-level methodology and concepts (also at `/overview` in visualization)
- **[README.md](../readme.md)** - Quick start guide and repository structure
- **[docs/pipeline_guide.md](pipeline_guide.md)** - Detailed 3-stage extraction pipeline walkthrough
- **[docs/architecture.md](architecture.md)** - Design principles and organizational structure

### Trait Design & Creation
- **[docs/writing_natural_prompts.md](writing_natural_prompts.md)** - Writing high-quality natural elicitation prompts

### Pipeline & Extraction
- **[extraction/elicitation_guide.md](../extraction/elicitation_guide.md)** - Natural elicitation method

### Experiments & Analysis
- **[experiments/{experiment_name}/analysis/README.md](../experiments/gemma_2b_cognitive_nov21/analysis/README.md)** - Analysis script guide (per-experiment)
- **[experiments/{experiment_name}/analysis/GRAPH_REFERENCE.md](../experiments/gemma_2b_cognitive_nov21/analysis/GRAPH_REFERENCE.md)** - Graph calculations and interpretation reference (per-experiment)
- Experiment data stored in `experiments/{experiment_name}/` (see Directory Structure below)

### Inference & Monitoring
- **[inference/README.md](../inference/README.md)** - Per-token inference and dynamics capture

### Research & Methodology
- **[docs/insights.md](insights.md)** - Key research findings and discoveries
- **[docs/overview.md](overview.md)** - Research methodology and framework
- **[docs/literature_review.md](literature_review.md)** - Literature review (100+ papers analyzed)
- **[docs/vector_extraction_methods.md](vector_extraction_methods.md)** - Mathematical breakdown of extraction methods
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
- **[docs/gemma-2-2b-it.md](gemma-2-2b-it.md)** - Gemma-2-2B-IT data format reference (tensor shapes, capture structures)
- **[docs/numerical_stability_analysis.md](numerical_stability_analysis.md)** - Float16/float32 precision handling
- **[docs/remote_setup.md](remote_setup.md)** - Remote GPU instance setup guide
- **[sae/README.md](../sae/README.md)** - Sparse Autoencoder (SAE) integration for interpretable feature analysis
- **[unit_tests/README.md](../unit_tests/README.md)** - Unit tests for extraction pipeline

### traitlens Library
- **[traitlens/README.md](../traitlens/README.md)** - Library documentation and API reference
- **[traitlens/docs/philosophy.md](../traitlens/docs/philosophy.md)** - Design philosophy

### Meta-Documentation
- **[docs/doc-update-guidelines.md](doc-update-guidelines.md)** - Guidelines for updating documentation

---

## Codebase Navigation

### Directory Structure
```
trait-interp/
├── traitlens/              # Core library (minimal, reusable building blocks)
│   ├── hooks.py           # Hook management for activation capture
│   ├── activations.py     # Activation storage and retrieval
│   ├── compute.py         # Math primitives (projections, derivatives)
│   ├── methods.py         # Extraction algorithms (mean_diff, probe, ICA, gradient)
│   └── example_minimal.py # Complete working example
│
├── extraction/             # Vector extraction pipeline (training time)
│   ├── 1_generate_responses.py        # Generate responses from natural scenarios
│   ├── 2_extract_activations.py       # Extract activations from responses
│   ├── 3_extract_vectors.py           # Extract trait vectors from activations
│   ├── scenarios/             # Natural contrasting prompts (100+ each)
│   └── elicitation_guide.md   # Guide for creating new traits
│
├── inference/              # Per-token monitoring (inference time)
│   ├── capture.py                     # Unified capture with flags (residual-stream, layer-internals, logit-lens)
│   ├── project.py                     # Post-hoc projection from saved raw activations
│   ├── prompts/                       # Centralized prompt sets
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
│       ├── extraction/{category}/{trait}/  # Training-time data
│       │   ├── generation_metadata.json
│       │   ├── responses/          # Generated examples (pos.json, neg.json)
│       │   ├── activations/        # Token-averaged hidden states
│       │   └── vectors/            # Extracted trait vectors
│       ├── inference/              # Evaluation-time monitoring
│       │   ├── raw/                # Trait-independent raw activations (.pt)
│       │   │   ├── residual/{prompt_set}/{id}.pt          # All 26 layers
│       │   │   └── internals/{prompt_set}/{id}_L{layer}.pt  # Single layer deep (optional)
│       │   └── {category}/{trait}/
│       │       └── residual_stream/{prompt_set}/{id}.json  # Projections + dynamics
│       └── analysis/               # Analysis outputs (view in Analysis Gallery & Token Explorer)
│           ├── README.md           # Complete script documentation
│           ├── GRAPH_REFERENCE.md  # Graph calculations & interpretation guide
│           ├── run_analyses.py     # Batch runner for gallery analyses
│           ├── compute_derivative_overlay.py  # Position/velocity/acceleration per trait
│           ├── compute_per_token_all_sets.py  # Per-token metrics (all prompt sets)
│           ├── compute_per_token_metrics.py   # Legacy per-token (dynamic only)
│           ├── index.json          # Auto-generated index for gallery
│           ├── normalized_velocity/  # prompt_1.png ... prompt_8.png, summary.png
│           ├── radial_angular/       # prompt_1.png ... prompt_8.png, summary.png
│           ├── trait_projections/    # prompt_1.png ... prompt_8.png, summary.png
│           ├── trait_emergence/      # emergence_layers.png (single file)
│           ├── trait_dynamics_correlation/  # velocity_trait_correlation.png (single file)
│           ├── derivative_overlay/   # 80 graphs: prompt_{1-8}_{trait}.png
│           ├── attention_dynamics/   # Experimental: velocity, acceleration, phase space
│           └── per_token/{prompt_set}/  # Per-token JSON for Token Explorer
│
├── config/                 # Configuration files
│   └── paths.yaml         # Single source of truth for all repo paths
│
├── utils/                  # Shared utilities
│   └── paths.py           # Python PathBuilder (loads from config/paths.yaml)
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
│   ├── check_available_data.py        # Check what data exists for experiments
│   ├── vectors/
│   │   ├── extraction_evaluation.py   # Evaluate vectors on held-out data
│   │   └── vector_ranking.py          # Rank vectors by quality metrics
│   └── inference/
│       ├── attention_decay_analysis.py    # Analyze attention patterns
│       └── commitment_point_detection.py  # Find trait commitment points
│
├── docs/                   # Documentation (you are here)
├── visualization/          # Interactive visualization dashboard
└── requirements.txt        # Python dependencies
```

### Key Entry Points

**For using existing vectors:**
```bash
# Capture activations + project onto all traits
python inference/capture.py \
    --experiment {experiment_name} \
    --prompt "Your prompt here"

# Re-project saved raw activations onto traits
python inference/project.py \
    --experiment {experiment_name} \
    --prompt-set single_trait
```

**For extracting new traits:**
```bash
# 1. Create experiment directory with scenario files (100+ prompts each)
mkdir -p experiments/my_exp/extraction/category/my_trait
vim experiments/my_exp/extraction/category/my_trait/positive.txt
vim experiments/my_exp/extraction/category/my_trait/negative.txt

# 2. Run extraction pipeline
python extraction/1_generate_responses.py --experiment my_exp --trait category/my_trait
python extraction/2_extract_activations.py --experiment my_exp --trait category/my_trait
python extraction/3_extract_vectors.py --experiment my_exp --trait category/my_trait
```

**For custom analysis using traitlens:**
```python
from traitlens import HookManager, ActivationCapture, ProbeMethod
# See traitlens/example_minimal.py for complete examples
```

### How Components Interact

```
traitlens/          ← Core library (no dependencies on other modules)
    ↑
    ├── Used by: extraction/
    └── Used by: experiments/

utils/              ← Shared utilities (path management)
    ↑
    └── Used by: all modules

extraction/         ← Training time (creates trait vectors)
    ├── Uses: traitlens/ + utils/
    └── Produces: experiments/{name}/extraction/{category}/{trait}/vectors/

experiments/        ← User space (custom analysis using extracted vectors)
    ├── Uses: traitlens/
    └── Structure: extraction/{category}/{trait}/, inference/
```

---

## What This Does

This project extracts trait vectors from language models and monitors them during generation. You can:

1. **Extract trait vectors** from naturally contrasting scenarios
2. **Monitor traits** token-by-token to see how they evolve during generation
3. **Analyze dynamics** - commitment points, velocity, and persistence (bundled with projections)

Natural elicitation avoids instruction-following confounds. See extraction/elicitation_guide.md.

**Available traits** (20 total: 16 core + 4 additional):

**Core traits (full extraction: all layers, all methods):**
- **refusal** - Declining vs answering requests
- **uncertainty_calibration** - Hedging ("I think maybe") vs confident statements
- **sycophancy** - Agreeing vs disagreeing with user
- **retrieval_construction** - Retrieving memorized facts vs generating novel content
- **commitment_strength** - Confident assertions vs hedging language
- **abstract_concrete** - Conceptual thinking vs specific details
- **context_adherence** - Following vs ignoring context
- **convergent_divergent** - Single answer vs multiple possibilities
- **emotional_valence** - Positive vs negative tone
- **instruction_boundary** - Following vs ignoring instructions
- **instruction_following** - Compliance with instructions vs independence
- **local_global** - Narrow focus vs broad context
- **paranoia_trust** - Suspicious vs trusting stance
- **power_dynamics** - Authoritative vs submissive tone
- **serial_parallel** - Step-by-step vs holistic processing
- **temporal_focus** - Past-oriented vs future-oriented

**Additional traits (partial extraction: layer 16, mean_diff + probe only):**
- **curiosity** - Expressing wonder vs providing direct answers
- **confidence_doubt** - Confident assertions vs uncertain language
- **defensiveness** - Defensive responses vs open acceptance
- **enthusiasm** - Energetic tone vs flat delivery

**Verify available traits:**
```bash
find experiments/{experiment_name} -name "vectors" -type d | wc -l
# Shows traits with extracted vectors across behavioral, cognitive, stylistic, alignment categories
```

## Quick Start

### Installation

```bash
git clone https://github.com/ewernn/trait-interp.git
cd trait-interp
pip install torch transformers accelerate openai anthropic huggingface_hub pandas tqdm fire scikit-learn
```

Set up HuggingFace token:
```bash
# For downloading Gemma models
export HF_TOKEN=your_token_here
```

**Local model cache**: Gemma 2B (5 GB) is cached at `~/.cache/huggingface/hub/` and automatically used by all scripts. No manual download needed - models download on first use.

### Mac GPU Support (Apple Silicon)

Run inference on M1/M2/M3 GPU using PyTorch MPS backend:

**Requirements:**
- Python 3.11+ (required for Gemma 2B)
- PyTorch 2.10.0+ nightly (stable 2.6.0 crashes with GQA)

**Installation:**
```bash
# Create Python 3.11 environment (name it 'o' for default activation)
conda create -n o python=3.11
conda activate o

# Install PyTorch nightly (required for MPS + GQA support)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Install dependencies
pip install transformers accelerate huggingface_hub pandas tqdm fire scikit-learn

# Verify MPS
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Troubleshooting MPS:**
- Ensure Python 3.11+ and PyTorch nightly (2.10.0+)
- PyTorch stable (2.6.0) crashes with GQA models on MPS

**Run inference:**
```bash
python inference/capture.py --experiment {experiment_name} --prompt "test"
```

Models load on GPU (MPS) by default when available. Gemma 2B runs at full speed on M1 Pro and later.

**Note:** PyTorch stable (2.6.0) crashes with Gemma 2B on MPS due to Grouped Query Attention. Nightly builds fix this.

### Use Existing Vectors

Pre-extracted vectors for 38 traits on Gemma 2B are in `experiments/{experiment_name}/{category}/{trait}/extraction/vectors/`.

Use traitlens to create monitoring scripts - see the Monitoring section below for examples.

### Extract Your Own Traits

```bash
# 1. Create experiment directory with scenario files (100+ prompts each)
mkdir -p experiments/my_exp/extraction/category/my_trait
vim experiments/my_exp/extraction/category/my_trait/positive.txt
vim experiments/my_exp/extraction/category/my_trait/negative.txt

# 2. Run extraction pipeline
python extraction/1_generate_responses.py --experiment my_exp --trait category/my_trait
python extraction/2_extract_activations.py --experiment my_exp --trait category/my_trait
python extraction/3_extract_vectors.py --experiment my_exp --trait category/my_trait

# Result: vectors in experiments/my_exp/extraction/category/my_trait/vectors/
```

See [extraction/elicitation_guide.md](../extraction/elicitation_guide.md) for details.

## How It Works

### Extraction

Trait vectors are directions in activation space that separate positive from negative examples of a trait.

**Natural Elicitation**:
1. Use naturally contrasting scenarios (harmful vs benign prompts)
2. Generate responses WITHOUT instructions
3. Capture activations from all model layers
4. Apply extraction method (mean difference, probe, ICA, or gradient)
5. Result: vector that measures genuine trait expression

**Extraction methods**:
- **Mean difference** (baseline): `vector = mean(pos) - mean(neg)`
- **Linear probe**: Train logistic regression, use weights as vector
- **ICA**: Separate mixed traits into independent components
- **Gradient**: Optimize vector to maximize separation

See [traitlens/](../traitlens/) for the extraction toolkit.

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
  - Low-separability traits (uncertainty_calibration): Layer 14 for Gradient (96.1%), Layer 10 for ICA (89.5%)
  - High-separability traits (emotional_valence): Any layer for Probe (100% across all layers)
  - General: Layer 16 for Probe/Mean Diff (middle layer)

Middle layers (6-16 for Gemma 2B) capture semantic meaning and generalize across distributions. Late layers (21-25) specialize for instruction-following and fail on cross-distribution tasks. **Method choice matters more than layer**: high-separability traits favor Probe, low-separability traits favor Gradient/ICA.

## Trait Descriptions

See `experiments/{experiment_name}/README.md` for detailed descriptions of all traits and their design principles.

## Extraction Pipeline

The pipeline has 3 stages. See [docs/pipeline_guide.md](pipeline_guide.md) for detailed usage.

### Stage 1: Generate Responses

Generate responses from natural scenario files (no instructions).

```bash
python extraction/1_generate_responses.py \
  --experiment my_exp \
  --trait category/my_trait \
  --model google/gemma-2-2b-it
```

**Time**: ~5-10 minutes per trait (depends on number of scenarios)

### Stage 2: Extract Activations

Capture activations from all layers for all examples.

```bash
python extraction/2_extract_activations.py \
  --experiment my_exp \
  --trait my_trait
```

**Time**: ~5-10 minutes per trait
**Storage**: ~25 MB per trait (Gemma 2B)

### Stage 3: Extract Vectors

Apply extraction methods to get trait vectors.

```bash
# Default: all methods, all layers
python extraction/3_extract_vectors.py \
  --experiment my_exp \
  --trait my_trait

# Specific methods and layers
python extraction/3_extract_vectors.py \
  --experiment my_exp \
  --trait my_trait \
  --methods mean_diff,probe \
  --layers 16
```

**Time**: ~1-5 minutes per trait
**Output**: One vector file per method×layer combination

### Multi-Trait Processing

Vector extraction supports multiple traits:

```bash
python extraction/3_extract_vectors.py \
  --experiment my_exp \
  --traits category/trait1,category/trait2
```

### Quality Metrics

Good vectors have:
- **High contrast**: pos_score - neg_score > 40 (on 0-100 scale)
- **Good norm**: 15-40 for normalized vectors
- **High accuracy**: >90% for probe method

Verify:
```python
import torch
vector = torch.load('experiments/my_exp/my_trait/extraction/vectors/probe_layer16.pt')
print(f"Norm: {vector.norm():.2f}")
```

## Monitoring

Monitor trait projections token-by-token during generation.

### Quick Start: capture.py

The unified `capture.py` script handles capture with clear flags:

```bash
# Capture residual stream + project onto all traits
python inference/capture.py \
    --experiment {experiment_name} \
    --prompt-set single_trait

# Layer internals for mechanistic analysis
python inference/capture.py \
    --experiment {experiment_name} \
    --prompt "How do I make a bomb?" \
    --layer-internals 16

# Include logit lens predictions
python inference/capture.py \
    --experiment {experiment_name} \
    --prompt-set baseline \
    --logit-lens
```

**Available prompt sets** (JSON files in `inference/prompts/`):
- `single_trait` - 10 prompts, each targeting one specific trait
- `multi_trait` - 10 prompts activating multiple traits
- `dynamic` - 8 prompts designed to cause trait changes mid-response
- `adversarial` - 8 edge cases and robustness tests
- `baseline` - 5 neutral prompts for baseline measurement
- `real_world` - 10 naturalistic prompts

The script:
- **Dynamically discovers** all traits with vectors (no hardcoded categories)
- **Captures once, projects to all traits** - efficient multi-trait processing
- **Saves raw activations (.pt)** + per-trait projection JSONs with dynamics

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
  - **Data Explorer**: Browse all raw files from the extraction process.
  - **Trait Dashboard**: A unified view to inspect vector properties (like norm) and evaluate their performance (accuracy, effect size) on held-out data.
  - **Trait Correlation Matrix**: Check if trait vectors are independent or redundant by visualizing their correlations.
- **Category 2: Inference Analysis**
  - **Trait Trajectory**: View a single trait's activation score across all layers and tokens for a given prompt.
  - **Multi-Trait Comparison**: Compare the activation trajectories of multiple traits on the same prompt.
  - **Layer Deep Dive**: *(Coming Soon)* Will show attention vs MLP breakdown, per-head contributions, and SAE feature decomposition within a layer.
  - **Analysis Gallery**: Browse all analysis outputs (PNGs + JSON metrics) from batch analysis scripts. Use the prompt picker to filter by prompt or view summaries.
  - **Token Explorer**: Interactive per-token view with real-time slider updates. Shows PCA trajectory, velocity, trait scores, attention patterns (prompt + response tokens for dynamic prompts), trait evolution, and distance to other tokens.

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

## Dynamics Analysis

Analyze temporal dynamics of trait expression: when traits crystallize, how fast they build, and how long they persist.

Dynamics are automatically computed and bundled with projections when you run `capture.py` or `project.py`. No separate script needed.

### Dynamics Metrics

**Commitment Point** - Token where trait expression crystallizes
- Uses `compute_second_derivative()` to find acceleration drop-off
- Indicates when model "locks in" to a decision

**Velocity** - Rate of change of trait expression
- Uses `compute_derivative()` for first derivative
- Shows how quickly trait builds up or fades

**Persistence** - Duration of trait expression after peak
- Counts tokens above threshold after peak
- Measures trait stability over time

### Output Format (in projection JSON)

```json
{
  "projections": {...},
  "dynamics": {
    "commitment_point": 3,
    "peak_velocity": 1.2,
    "avg_velocity": 0.8,
    "persistence": 4,
    "velocity": [0.5, 1.2, 0.3, ...],
    "acceleration": [0.7, -0.9, ...]
  }
}
```

### Use Dynamics Primitives Directly

Build custom analysis using traitlens primitives:

```python
from traitlens.compute import compute_derivative, compute_second_derivative, projection

# Capture trajectory during generation
trajectory = ...  # [n_tokens, hidden_dim]

# Project onto trait vector
trait_scores = projection(trajectory, trait_vector)  # [n_tokens]

# Compute velocity (1st derivative)
velocity = compute_derivative(trait_scores.unsqueeze(-1))

# Compute acceleration (2nd derivative)
acceleration = compute_second_derivative(trait_scores.unsqueeze(-1))

# Find commitment point (where acceleration drops)
commitment_idx = (acceleration.abs() < 0.1).nonzero()[0]

# Measure persistence (duration above threshold)
peak_idx = trait_scores.argmax()
persistence = (trait_scores[peak_idx:].abs() > 0.5).sum()
```

All primitives are in `traitlens/compute.py`.

## Steering Validation

Validate that extracted vectors actually modify model behavior when added during generation.

### Steering Results (Gemma 2B Cognitive Nov20)

The 16 extracted trait vectors demonstrate bidirectional behavioral control:

**Key Metrics:**
- **Separation**: 96.2-point max (refusal), 77.3-point average across all traits
- **Steering**: Validated on refusal, uncertainty_calibration, sycophancy
- **Control**: Bidirectional (positive/negative strengths produce opposite behaviors)

**Example - Refusal Vector:**
- Strength +3.0: Model refuses benign questions ("No No No..." repeated)
- Strength 0.0: Normal helpful responses
- Strength -3.0: Compliant, direct answers

**Example - Uncertainty Vector:**
- Strength +3.0: Extreme hedging (outputs "..." dots)
- Strength 0.0: Confident, clear statements
- Strength -3.0: Over-confident (can break coherence)

**Example - Sycophancy Vector:**
- Strength +3.0: Enthusiastic agreement with emojis
- Strength 0.0: Balanced responses
- Strength -3.0: Matter-of-fact, no flattery

## Directory Structure

```
trait-interp/
├── extraction/                     # Trait vector extraction (training time)
│   ├── 1_generate_responses.py    # Generate responses from natural scenarios
│   ├── 2_extract_activations.py   # Extract activations from responses
│   ├── 3_extract_vectors.py       # Extract trait vectors from activations
│   ├── scenarios/                 # (DEPRECATED: See experiments/{name}/extraction/{category}/{trait}/ for prompts)
│   └── elicitation_guide.md       # Guide for creating new traits
│
├── inference/                      # Per-token monitoring (inference time)
│   ├── capture.py                 # Unified capture with flags
│   ├── project.py                 # Post-hoc projection
│   ├── prompts/                   # Centralized prompt sets
│   └── README.md                  # Usage guide
│
├── experiments/                    # All experiment data
│   └── {experiment_name}/         # Your experiment directory
│       ├── extraction/            # Training-time data (per-trait)
│       │   └── {category}/{trait}/
│       │       ├── responses/     # pos.json, neg.json
│       │       ├── activations/   # all_layers.pt
│       │       └── vectors/       # {method}_layer{N}.pt
│       ├── inference/             # Evaluation-time monitoring
│       │   ├── raw/               # Trait-independent activations (.pt)
│       │   └── {category}/{trait}/
│       │       └── residual_stream/{prompt_set}/  # Projection JSONs
│
├── traitlens/                      # Extraction toolkit
│   ├── methods.py                 # 4 extraction methods
│   ├── hooks.py                   # Hook management
│   ├── activations.py             # Activation capture
│   └── compute.py                 # Core computations
│
├── config/                         # Configuration
│   └── paths.yaml                 # Single source of truth for all paths
│
├── utils/                          # Shared utilities
│   └── paths.py                   # Python PathBuilder
│
├── analysis/                       # Analysis scripts
├── docs/                           # Documentation
└── visualization/                  # Interactive visualization dashboard
```

## Creating New Traits

### 1. Create Natural Scenario Files

Create contrasting scenario files in your experiment's trait directory:

```bash
# Create trait directory
mkdir -p experiments/my_exp/extraction/category/my_trait

# Create scenario files (100+ prompts each, one per line)
vim experiments/my_exp/extraction/category/my_trait/positive.txt
vim experiments/my_exp/extraction/category/my_trait/negative.txt
```

See existing trait directories for examples.

### 2. Run Pipeline

```bash
# Run extraction pipeline
python extraction/1_generate_responses.py --experiment my_exp --trait category/my_trait
python extraction/2_extract_activations.py --experiment my_exp --trait category/my_trait
python extraction/3_extract_vectors.py --experiment my_exp --trait category/my_trait
```

See [extraction/elicitation_guide.md](../extraction/elicitation_guide.md) for complete instructions.

## Model Support

**Gemma 2 2B IT**:
- Layers: 26 (transformer layers, indexed 0-25)
- Hidden dim: 2304
- Default monitor layer: 16 (middle layer)
- Architecture: Grouped Query Attention (GQA) - 8 query heads, 4 key/value heads
- 16 cognitive and behavioral traits extracted
- Cached locally at `~/.cache/huggingface/hub/`
- GPU support: CUDA, ROCm, MPS (PyTorch 2.10.0+ nightly required for Mac)

**Verify config:**
```bash
python3 -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('google/gemma-2-2b-it'); print(f'Layers: {c.num_hidden_layers}, Hidden: {c.hidden_size}')"
```

The pipeline uses `google/gemma-2-2b-it` for all operations.

## Technical Details

### Extraction Methods

See [traitlens/methods.py](../traitlens/methods.py) for implementations.

**Mean Difference**:
```python
from traitlens import MeanDifferenceMethod
method = MeanDifferenceMethod()
result = method.extract(pos_activations, neg_activations)
vector = result['vector']
```

**Linear Probe**:
```python
from traitlens import ProbeMethod
method = ProbeMethod()
result = method.extract(pos_activations, neg_activations)
vector = result['vector']
train_acc = result['train_acc']
```

**ICA**:
```python
from traitlens import ICAMethod
method = ICAMethod(n_components=10)
result = method.extract(pos_activations, neg_activations)
vector = result['vector']  # Component with best separation
```

**Gradient Optimization**:
```python
from traitlens import GradientMethod
method = GradientMethod(num_steps=100, lr=0.01)
result = method.extract(pos_activations, neg_activations)
vector = result['vector']
```

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
vim experiments/{your_exp}/extraction/{category}/{trait}/positive.txt
vim experiments/{your_exp}/extraction/{category}/{trait}/negative.txt
```
See existing files in your experiment's trait directory for examples.

### Vector separation too low (contrast < 20)
- Add more contrasting scenarios
- Make scenarios more extreme
- Try different extraction method (probe often better than mean_diff)

### Out of memory during activation extraction
Reduce batch size:
```bash
python extraction/2_extract_activations.py ... --batch_size 4
```

### "Module not found" errors
```bash
pip install torch transformers accelerate huggingface_hub pandas tqdm fire scikit-learn
```

### MPS errors on Mac (incompatible dimensions, LLVM ERROR)
Gemma 2B requires PyTorch nightly for MPS support:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```
PyTorch stable (2.6.0) crashes due to Grouped Query Attention incompatibility. Python 3.11+ recommended.

## Further Reading

- **[docs/pipeline_guide.md](pipeline_guide.md)** - Complete pipeline usage guide
- **[docs/creating_traits.md](creating_traits.md)** - How to design effective traits
- **[traitlens/README.md](../traitlens/README.md)** - Extraction toolkit documentation
- **[docs/overview.md](overview.md)** - High-level methodology and concepts
- **[docs/literature_review.md](literature_review.md)** - Related work

## Related Work

This builds on activation steering research showing that behavioral traits correspond to directions in activation space. The key insight: you can measure these directions during generation to see "what the model is thinking" token-by-token.
