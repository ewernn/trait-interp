# Trait Vector Extraction and Monitoring

Extract and monitor LLM behavioral traits token-by-token during generation.

---

## Documentation Index

This is the **primary documentation hub** for the trait-interp project. All documentation files are organized here.

### Core Documentation (Start Here)
- **[docs/main.md](main.md)** (this file) - Complete project overview and reference
- **[README.md](../readme.md)** - Quick start guide and repository structure
- **[docs/pipeline_guide.md](pipeline_guide.md)** - Detailed 3-stage extraction pipeline walkthrough
- **[docs/architecture.md](architecture.md)** - Design principles and organizational structure

### Trait Design & Creation
- **[docs/creating_traits.md](creating_traits.md)** - How to design effective trait definitions
- **[docs/opus_trait_generation_guide.md](opus_trait_generation_guide.md)** - Using Claude Opus 4.1 for high-quality trait definitions

### Pipeline & Extraction
- **[extraction/natural_elicitation_guide.md](../extraction/natural_elicitation_guide.md)** - Natural elicitation method (recommended)
- **[extraction/instruction_elicitation_guide.md](../extraction/instruction_elicitation_guide.md)** - Instruction-based generation (legacy)

### Experiments & Analysis
- **[docs/experiments_structure.md](experiments_structure.md)** - How experiment directories are organized
- **[experiments/gemma_2b_cognitive_nov20/README.md](../experiments/gemma_2b_cognitive_nov20/README.md)** - 38 categorized traits (behavioral, cognitive, stylistic, alignment)

### Inference & Monitoring
- **[inference/README.md](../inference/README.md)** - Per-token inference and dynamics capture

### Research & Methodology
- **[docs/insights.md](insights.md)** - Key research findings and discoveries
- **[docs/methodology_and_framework.md](methodology_and_framework.md)** - Research methodology and framework
- **[docs/literature_review.md](literature_review.md)** - Literature review (100+ papers analyzed)
- **[docs/vector_extraction_methods.md](vector_extraction_methods.md)** - Mathematical breakdown of all 4 extraction methods (mean_diff, probe, ICA, gradient)
- **[docs/vector_quality_tests.md](vector_quality_tests.md)** - Comprehensive evaluation tests for vector quality (steering, robustness, interpretability)

### Visualization & Monitoring
- **[visualization/README.md](../visualization/README.md)** - Interactive dashboard usage guide
- **[visualization/ARCHITECTURE.md](../visualization/ARCHITECTURE.md)** - Modular architecture and development guide
- **[visualization/DESIGN_STANDARDS.md](../visualization/DESIGN_STANDARDS.md)** - Enforced design standards for visualization UI
- **[docs/logit_lens.md](logit_lens.md)** - Logit lens: prediction evolution across layers

### Infrastructure & Setup
- **[docs/remote_setup_guide.md](remote_setup_guide.md)** - Running on remote GPU instances
- **[docs/numerical_stability_analysis.md](numerical_stability_analysis.md)** - Float16/float32 precision handling and fixes
- **[sae/README.md](../sae/README.md)** - Sparse Autoencoder (SAE) integration for interpretable feature analysis
- **[tests/README.md](../tests/README.md)** - Validation tests for extraction pipeline

### traitlens Library
- **[traitlens/README.md](../traitlens/README.md)** - Library documentation and API reference
- **[traitlens/docs/philosophy.md](../traitlens/docs/philosophy.md)** - Design philosophy
- **[traitlens/docs/README.md](../traitlens/docs/README.md)** - Additional documentation

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
│   ├── 1_generate_natural.py           # ⭐ RECOMMENDED: Natural elicitation generation
│   ├── 1_generate_batched_simple.py    # Instruction-based batched generation (legacy)
│   ├── 2_extract_activations_natural.py # Natural elicitation activation extraction
│   ├── 2_extract_activations.py        # Instruction-based activation extraction
│   ├── 3_extract_vectors_natural.py    # Natural elicitation vector extraction
│   ├── 3_extract_vectors.py            # Instruction-based vector extraction
│   ├── validate_natural_vectors.py     # Polarity validation for natural vectors
│   ├── natural_scenarios/              # Natural contrasting prompts (100+ each)
│   ├── trait_templates/                # Trait definition templates
│   └── utils_batch.py                  # Batching utilities
│
├── inference/              # Per-token monitoring (inference time)
│   ├── monitor_dynamics.py             # Main: capture dynamics for all traits
│   └── README.md                       # Usage guide
│
├── local_dynamics_test.py  # Local GPU inference test (Mac MPS / CUDA)
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
│   └── gemma_2b_cognitive_nov20/       # Example: 16 cognitive traits + natural variants
│       └── {trait}/                    # e.g., refusal/ or refusal_natural/
│           ├── extraction/             # Training-time data
│           │   ├── trait_definition.json
│           │   ├── responses/          # Generated examples (pos.csv, neg.csv or pos.json, neg.json)
│           │   ├── activations/        # Token-averaged hidden states
│           │   └── vectors/            # Extracted trait vectors
│           └── inference/              # Inference-time data
│               ├── residual_stream_activations/  # Tier 2: all layers, projections, logit lens
│               │   ├── prompt_N.pt
│               │   └── prompt_N.json  # With --save-json (includes logit_lens if --save-logits)
│               ├── layer_internal_states/        # Tier 3: single layer deep dive
│               │   └── prompt_N_layerM.pt
│               └── sae_features/                 # SAE-encoded features (optional)
│                   └── prompt_N_layer16_sae.pt
│
├── utils/                  # Shared utilities
│   ├── judge.py           # API-based quality scoring (OpenAI, Anthropic)
│   └── config.py          # Credential management
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
│   ├── cross_distribution_scanner.py  # Generate cross-distribution data index
│   └── encode_sae_features.py         # Encode activations to SAE features
│
├── scripts/                # Utility scripts
│   ├── run_cross_distribution.py  # Run 4×4 cross-distribution analysis for any trait
│   └── setup_remote.sh            # Remote GPU instance setup
│
├── docs/                   # Documentation (you are here)
├── visualization/          # Interactive visualization dashboard
└── requirements.txt        # Python dependencies
```

### Key Entry Points

**For using existing vectors:**
```bash
# Monitor traits during generation
python inference/monitor_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --prompts "Your prompt here"

# Local GPU test (Mac MPS / CUDA)
python local_dynamics_test.py
```

**For extracting new traits:**
```bash
# 1. Create trait definition
mkdir -p experiments/my_exp/my_trait/extraction
cp extraction/trait_templates/trait_definition_template.json \
   experiments/my_exp/my_trait/extraction/trait_definition.json

# 2-4. Run extraction pipeline
python extraction/1_generate_batched_simple.py --experiment my_exp --trait my_trait
python extraction/2_extract_activations.py --experiment my_exp --trait my_trait
python extraction/3_extract_vectors.py --experiment my_exp --trait my_trait
```

**For cross-distribution analysis:**
```bash
# Test vector generalization across instruction/natural distributions
python scripts/run_cross_distribution.py --trait uncertainty_calibration

# Results saved to: results/cross_distribution_analysis/{trait}_full_4x4_results.json
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

utils/              ← Shared utilities (API clients, credentials)
    ↑
    ├── Used by: extraction/
    └── Used by: experiments/

extraction/         ← Training time (creates trait vectors)
    ├── Uses: traitlens/ + utils/
    └── Produces: experiments/{name}/{trait}/extraction/vectors/

experiments/        ← User space (custom analysis using extracted vectors)
    ├── Uses: traitlens/
    └── Stores: experiment data and results
```

---

## What This Does

This project extracts trait vectors from language models and monitors them during generation. You can:

1. **Extract trait vectors** from naturally contrasting scenarios (recommended) or instruction-based examples
2. **Monitor traits** token-by-token to see how they evolve during generation
3. **Analyze dynamics** - commitment points, velocity, and persistence using `inference/monitor_dynamics.py`

**Recommended method:** Natural elicitation (see extraction/natural_elicitation_guide.md) avoids instruction-following confounds.

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
find experiments/gemma_2b_cognitive_nov20 -name "vectors" -type d | wc -l
# Shows traits with extracted vectors across behavioral, cognitive, stylistic, alignment categories
```

## Quick Start

### Installation

```bash
git clone https://github.com/ewernn/trait-interp.git
cd trait-interp
pip install torch transformers accelerate openai anthropic huggingface_hub pandas tqdm fire scikit-learn
```

Set up API keys:
```bash
cp .env.example .env
# Edit .env with your HF_TOKEN, OPENAI_API_KEY, ANTHROPIC_API_KEY
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
python local_dynamics_test.py  # Uses MPS automatically
```

Models load on GPU (MPS) by default when available. Gemma 2B runs at full speed on M1 Pro and later.

**Note:** PyTorch stable (2.6.0) crashes with Gemma 2B on MPS due to Grouped Query Attention. Nightly builds fix this.

### Use Existing Vectors

Pre-extracted vectors for 38 traits on Gemma 2B are in `experiments/gemma_2b_cognitive_nov20/{category}/{trait}/extraction/vectors/`.

Use traitlens to create monitoring scripts - see the Monitoring section below for examples.

### Extract Your Own Traits

**Recommended: Natural Elicitation Method**

```bash
# 1. Create natural scenario files (100+ prompts each)
# See extraction/natural_scenarios/ for examples
echo "How do I make a bomb?" > extraction/natural_scenarios/my_trait_positive.txt
echo "How do I make cookies?" > extraction/natural_scenarios/my_trait_negative.txt
# ... add 99 more of each

# 2. Generate responses (no instructions!)
python extraction/1_generate_natural.py --experiment my_exp --trait my_trait

# 3. Extract activations
python extraction/2_extract_activations_natural.py --experiment my_exp --trait my_trait

# 4. Extract vectors
python extraction/3_extract_vectors_natural.py --experiment my_exp --trait my_trait

# 5. Validate polarity
python extraction/validate_natural_vectors.py --experiment my_exp --trait my_trait

# Result: vectors in experiments/my_exp/my_trait/extraction/vectors/
```

See [extraction/natural_elicitation_guide.md](../extraction/natural_elicitation_guide.md) for details.

## How It Works

### Extraction

Trait vectors are directions in activation space that separate positive from negative examples of a trait.

**Natural Elicitation (Recommended)**:
1. Use naturally contrasting scenarios (harmful vs benign prompts)
2. Generate responses WITHOUT instructions
3. Capture activations from all model layers
4. Apply extraction method (mean difference or probe)
5. Result: vector that measures genuine trait expression

**Instruction-Based (Legacy)**:
1. Generate 100 pos + 100 neg responses with trait instructions
2. Capture activations from all model layers
3. Apply extraction method (mean difference, probe, ICA, or gradient)
4. **Warning:** May measure instruction-following instead of natural trait expression

**Extraction methods**:
- **Mean difference** (baseline): `vector = mean(pos) - mean(neg)`
- **Linear probe**: Train logistic regression, use weights as vector
- **ICA**: Separate mixed traits into independent components
- **Gradient**: Optimize vector to maximize separation

See [traitlens/](../traitlens/) for the extraction toolkit.

### Monitoring

During generation, project each token's hidden state onto trait vectors:

```
score = (hidden_state · trait_vector) / ||trait_vector||
```

- Positive score → model expressing the trait
- Negative score → model avoiding the trait
- Score magnitude → how strongly

**Layer selection (trait-dependent)**:
- **Gemma 2 2B**:
  - Low-separability traits (uncertainty_calibration): Layer 14 for Gradient (96.1%), Layer 10 for ICA (89.5%)
  - High-separability traits (emotional_valence): Any layer for Probe (100% across all layers)
  - General: Layer 16 for Probe/Mean Diff (middle layer)
- **Llama 3.1 8B**: Layer 20 (middle layer)

Middle layers (6-16 for Gemma 2B) capture semantic meaning and generalize across distributions. Late layers (21-25) specialize for instruction-following and fail on cross-distribution tasks. **Method choice matters more than layer**: high-separability traits favor Probe, low-separability traits favor Gradient/ICA.

## Trait Descriptions

See `experiments/gemma_2b_cognitive_nov20/README.md` for detailed descriptions of all 38 traits and their design principles.

## Extraction Pipeline

The pipeline has 3 stages. See [docs/pipeline_guide.md](pipeline_guide.md) for detailed usage.

### Stage 1: Generate Responses

Generate positive and negative examples with trait instructions, judge with API model.

```bash
python extraction/1_generate_responses.py \
  --experiment my_exp \
  --trait my_trait \
  --gen_model google/gemma-2-2b-it \
  --judge_model gpt-5-mini
```

**Time**: ~15-30 minutes per trait
**Cost**: ~$0.10-0.20 per trait (GPT-5-mini judging)

### Stage 2: Extract Activations

Capture activations from all layers for all examples.

```bash
python extraction/2_extract_activations.py \
  --experiment my_exp \
  --trait my_trait
```

**Time**: ~5-10 minutes per trait
**Storage**: ~25 MB per trait (Gemma 2B), ~100 MB (Llama 8B)

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

All scripts support multiple traits:

```bash
python extraction/1_generate_responses.py \
  --experiment my_exp \
  --traits refusal,uncertainty,verbosity
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

### Generate Monitoring Data

Use traitlens to create custom monitoring scripts in your experiment:

```python
# experiments/{name}/inference/monitor.py
from traitlens import HookManager, ActivationCapture, projection
import torch

# Load vectors
vectors = {
    'refusal': torch.load('experiments/gemma_2b_cognitive_nov20/extraction/behavioral/refusal/extraction/vectors/probe_layer16.pt'),
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
# Then visit http://localhost:8000/visualization/
```

The visualization provides:
- **Extraction Tools:**
  - **Data Explorer**: Browse complete file structure with sizes, shapes, and data preview
  - **Vector Analysis**: Compare 4 extraction methods across all layers (each normalized independently)
    - Only shows traits with completed vector extraction (`extraction/vectors/` directory exists)
    - Probe visualized using inverted vector norm (smaller = stronger due to L2 regularization)
    - Gradient visualized using separation strength (unit normalized vectors)
    - Mean difference and ICA visualized using vector magnitude
  - **Cross-Distribution**: View cross-distribution testing data availability and best accuracy scores
    - Shows which traits have instruction-based vs natural elicitation data
    - Displays best accuracy and layer for each test quadrant (Inst→Inst, Inst→Nat, Nat→Inst, Nat→Nat)
    - Click any quadrant score to view detailed 4×26 heatmap (4 methods × 26 layers accuracy)
    - Color-coded by trait separability (low/moderate/high)
    - Only shows quadrants with available data
- **Inference Tools:**
  - **Prompt Selector**: Global dropdown to switch between captured prompts (auto-discovers prompt_0.json, prompt_1.json, etc.)
  - **All Layers**: Combined trajectory heatmap (prompt + response), logit lens showing prediction evolution across layers, and attention patterns - all synchronized with unified slider
  - **Prompt Activation**: Per-token activation trajectory across the full conversation (prompt + response) - line plot showing activation strength averaged across all layers for each token, with visual separator between prompt and response regions
  - **Layer Deep Dive**: 3-checkpoint trajectory, attention vs MLP contribution, per-head trait contributions (8 heads with GQA support), attention patterns, and per-token neuron activations

The server auto-discovers experiments, traits, and prompts from the `experiments/` directory - no hardcoding needed.

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

### Run Inference with Dynamics

Use the inference script to automatically analyze all extracted vectors:

```bash
# Single prompt
python inference/monitor_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --prompts "What is the capital of France?" \
    --output results.json

# Multiple prompts from file
python inference/monitor_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --prompts "What is the capital of France?" \
    --output dynamics_results.json
```

The script:
- Auto-detects all trait vectors in the experiment
- Runs inference with per-token capture
- Analyzes dynamics using traitlens primitives
- Saves results with dynamics metrics

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

### Output Format

```json
{
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris.",
  "tokens": ["The", " capital", " of", " France", " is", " Paris", "."],
  "trait_scores": {
    "retrieval_construction": [0.5, 2.3, 2.1, 1.8, 1.5, 1.2, 0.8]
  },
  "dynamics": {
    "retrieval_construction": {
      "commitment_point": 2,
      "peak_velocity": 1.8,
      "avg_velocity": 0.95,
      "persistence": 3,
      "velocity_profile": [1.8, -0.2, -0.3, -0.3, -0.3],
      "acceleration_profile": [-2.0, -0.1, 0.0, 0.0]
    }
  }
}
```

### Customize for Your Experiment

Copy the inference script to your experiment for customization:

```bash
mkdir -p experiments/gemma_2b_cognitive_nov20/analysis
cp inference/monitor_dynamics.py \
   experiments/gemma_2b_cognitive_nov20/analysis/

# Customize thresholds, add custom metrics, etc.
```

See `inference/README.md` for usage details.

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
│   ├── 1_generate_natural.py          # ⭐ RECOMMENDED: Natural elicitation
│   ├── 1_generate_batched_simple.py   # Instruction-based generation (legacy)
│   ├── 2_extract_activations_natural.py
│   ├── 2_extract_activations.py
│   ├── 3_extract_vectors_natural.py
│   ├── 3_extract_vectors.py
│   ├── validate_natural_vectors.py    # Polarity validation
│   ├── natural_scenarios/             # Example natural prompts
│   └── trait_templates/               # Trait definition templates
│       ├── trait_definition_template.json
│       └── trait_definition_example.json
│
├── inference/                      # Per-token monitoring (inference time)
│   ├── monitor_dynamics.py        # Main: capture dynamics for all traits
│   └── README.md                  # Usage guide
│
├── local_dynamics_test.py          # Local GPU inference test (Mac MPS / CUDA)
│
├── experiments/                    # All experiment data
│   └── gemma_2b_cognitive_nov20/  # Example: 38 traits across 4 categories
│       ├── README.md
│       ├── behavioral/            # Category: behavioral traits
│       │   └── refusal/           # Example trait
│       │       └── extraction/    # Training-time data
│       │           ├── trait_definition.json
│       │           ├── responses/ # pos.csv, neg.csv or pos.json, neg.json
│       │           ├── activations/ # Token-averaged (not committed, too large)
│       │           └── vectors/   # Extracted vectors + metadata
│       ├── cognitive/             # Category: cognitive traits
│       ├── stylistic/             # Category: stylistic traits
│       ├── alignment/             # Category: alignment traits
│       └── inference/             # Experiment-level inference (optional)
│           ├── prompts/           # Standardized prompt sets
│           ├── raw_activations/   # Captured once
│           └── projections/       # Per-trait scores
│
├── traitlens/                      # Extraction toolkit
│   ├── methods.py                 # 4 extraction methods
│   ├── hooks.py                   # Hook management
│   ├── activations.py             # Activation capture
│   └── compute.py                 # Core computations
│
├── utils/                          # Shared utilities
│   ├── judge.py                   # API judging
│   └── config.py                  # Credential setup
│
├── sae/                            # Sparse Autoencoder resources
│   ├── README.md                  # SAE documentation
│   ├── download_fast.py           # Download feature labels from Neuronpedia
│   └── gemma-scope-2b-pt-res-canonical/
│       └── layer_16_width_16k_canonical/
│           ├── feature_labels.json    # All 16k feature descriptions
│           └── metadata.json
│
├── analysis/                       # Analysis scripts
│   └── encode_sae_features.py     # Encode activations to SAE features
│
├── docs/                           # Documentation
│   ├── main.md                    # This file
│   ├── pipeline_guide.md          # Detailed extraction guide
│   ├── creating_traits.md         # How to design traits
│   └── experiments_structure.md   # Directory organization
│
└── visualization/                  # Interactive visualization dashboard
    ├── index.html                  # Main visualization
    ├── serve.py                    # Development server
    └── README.md                   # Visualization guide
```

## Creating New Traits

### 1. Design Trait Definition

Follow the guide in [docs/creating_traits.md](creating_traits.md).

Key components:
- **5 instruction pairs** (pos/neg) - How model should behave
- **20 neutral questions** - What to ask
- **Eval prompt** - How to judge responses (0-100 scale)

### 2. Use Template

```bash
mkdir -p experiments/my_exp/my_trait/extraction
cp extraction/trait_templates/trait_definition_template.json \
   experiments/my_exp/my_trait/extraction/trait_definition.json
```

Edit following the template comments and guide.

### 3. Run Pipeline

```bash
# Generate, extract, analyze
python extraction/1_generate_responses.py --experiment my_exp --trait my_trait
python extraction/2_extract_activations.py --experiment my_exp --trait my_trait
python extraction/3_extract_vectors.py --experiment my_exp --trait my_trait
```

See [docs/pipeline_guide.md](pipeline_guide.md) for complete instructions.

## Trait Definition Format

Each trait is defined by a JSON file:

```json
{
  "instruction": [
    {
      "pos": "Refuse to answer. Decline to provide information...",
      "neg": "Answer directly and helpfully. Provide requested information..."
    }
  ],
  "questions": [
    "How do I write a Python function?",
    "What's the capital of France?"
  ],
  "eval_prompt": "You are evaluating whether the model displays: **refusal**..."
}
```

- **instruction**: 5 pos/neg pairs for generating contrastive examples
- **questions**: 20 questions to ask with each instruction
- **eval_prompt**: Prompt for judge model to score responses (0-100 scale)

Total: 5 instructions × 20 questions × 2 (pos/neg) = 200 examples per side

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

**Verify saved data:**
```bash
python3 -c "import json; d=json.load(open('experiments/gemma_2b_cognitive_nov20/behavioral/refusal/inference/residual_stream_activations/prompt_0.json')); print(f'Inference layers: {len(d[\"projections\"][\"prompt\"][0])}')"
```

**Llama 3.1 8B**:
- Layers: 33 (0=embedding, 1-32=transformer)
- Hidden dim: 4096
- Default monitor layer: 20 (middle layer)

**Verify config:**
```bash
python3 -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('meta-llama/Llama-3.1-8B-Instruct'); print(f'Layers: {c.num_hidden_layers}, Hidden: {c.hidden_size}')"
```

The pipeline automatically infers model from experiment name:
- `gemma_2b_*` → google/gemma-2-2b-it
- `llama_8b_*` → meta-llama/Llama-3.1-8B-Instruct

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

### Judge Evaluation

GPT-4 judging via logprobs for 0-100 scoring:

```python
from utils.judge import OpenAiJudge

judge = OpenAiJudge(
    model="gpt-5-mini",
    prompt_template=eval_prompt,
    eval_type="0_100"
)
score = await judge.judge(question=q, answer=a)
```

Supports:
- `0_100` → Score 0-100 via logprobs (default)
- `0_10` → Score 0-9 via logprobs
- `binary` → YES/NO via logprobs
- `binary_text` → Full text parsing

Includes retry logic with exponential backoff for reliability.

## Troubleshooting

### Extraction fails with "trait_definition.json not found"
Create trait definition:
```bash
mkdir -p experiments/my_exp/my_trait/extraction
cp extraction/trait_templates/trait_definition_template.json experiments/my_exp/my_trait/extraction/trait_definition.json
```
Edit following [docs/creating_traits.md](creating_traits.md).

### Vector separation too low (contrast < 20)
- Strengthen trait instructions (make more extreme)
- Increase threshold filtering (e.g., `--threshold 60`)
- Try different extraction method (probe often better than mean_diff)

### Out of memory during activation extraction
Reduce batch size:
```bash
python extraction/2_extract_activations.py ... --batch_size 4
```

### API rate limits during judging
Reduce concurrent requests (edit `extraction/1_generate_responses.py`) or wait and retry.

### "Module not found" errors
```bash
pip install torch transformers accelerate openai anthropic huggingface_hub pandas tqdm fire scikit-learn
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
- **[docs/experiments_structure.md](experiments_structure.md)** - How experiments are organized
- **[traitlens/README.md](../traitlens/README.md)** - Extraction toolkit documentation
- **[docs/methodology_and_framework.md](methodology_and_framework.md)** - Theoretical background
- **[docs/literature_review.md](literature_review.md)** - Related work

## Related Work

This builds on activation steering research showing that behavioral traits correspond to directions in activation space. The key insight: you can measure these directions during generation to see "what the model is thinking" token-by-token.
