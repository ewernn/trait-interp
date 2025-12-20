# Repository Architecture

## Design Principles

**Note:** traitlens is bundled in the `traitlens/` directory. This doc explains the architectural boundaries between components.

### Core Stack

```
traitlens           → General primitives (dimension-agnostic tensor ops)
        ↓
extraction/         → Build trait vectors (training time)
        ↓
inference/          → Compute facts per prompt (capture, project, dynamics)
        ↓
analysis/           → Interpret + aggregate (thresholds, cross-prompt)
        ↓
visualization/      → Show everything
```

### Directory Responsibilities

1. **traitlens/** = General-purpose primitives (bundled)
2. **utils/** = Universal utilities (path management)
3. **extraction/** = Vector creation pipeline (training time)
4. **inference/** = Per-prompt computation (capture, project, dynamics)
5. **analysis/** = Interpretation + aggregation (thresholds, cross-prompt patterns)
6. **experiments/** = Data storage + experiment-specific scripts
7. **visualization/** = All visualization code and views

---

## Directory Structure

```
trait-interp/
│
├── config/                 # Configuration
│   └── paths.yaml         # Single source of truth for all paths
│
├── utils/                  # Universal utilities
│   └── paths.py           # Python PathBuilder (loads from config/paths.yaml)
│
├── extraction/             # Vector creation pipeline (training time)
│   ├── generate_responses.py     # Generate responses from natural scenarios
│   ├── extract_activations.py    # Extract activations from responses
│   ├── extract_vectors.py        # Create trait vectors
│   └── scenarios/                # Natural contrasting prompts
│
├── inference/              # Per-prompt computation (inference time)
│   ├── capture_raw_activations.py           # Capture hidden states
│   ├── project_raw_activations_onto_traits.py  # Project onto trait vectors
│   └── capture_trait_dynamics.py            # Compute velocity, acceleration
│
├── analysis/               # Interpretation + aggregation
│   └── (structure emerges from useful work)
│
├── experiments/            # Experiment data storage
│   └── {experiment_name}/
│       ├── extraction/     # Training-time data (per-trait)
│       │   └── {category}/{trait}/
│       │       ├── responses/    # pos.json, neg.json
│       │       ├── activations/  # all_layers.pt
│       │       └── vectors/      # {method}_layer{N}.pt
│       ├── inference/      # Evaluation-time data
│       │   ├── raw/        # Captured activations
│       │   └── projections/# Trait scores + dynamics
│       └── analysis/       # Experiment-specific analysis outputs
│
├── visualization/          # All visualization code
│   ├── views/              # View modules
│   └── core/               # Shared utilities
│
└── docs/                   # Documentation
```

---

## inference/ vs analysis/ Distinction

The key organizational principle: **facts vs interpretation**.

### inference/ = "What are the numbers?"

Computes facts about a single prompt. No thresholds, no heuristics.

| Computation | Why inference/ |
|-------------|----------------|
| Raw activations | Direct capture |
| Trait scores | Direct projection |
| Velocity | Direct derivative |
| Acceleration | Direct derivative |
| Attention patterns | Direct from model |

### analysis/ = "What do the numbers mean?"

Interprets facts, applies thresholds, aggregates across prompts.

| Type | Why analysis/ |
|------|---------------|
| Threshold-based detection | Needs heuristics (e.g., "when does X drop below Y?") |
| Cross-prompt aggregation | Compares multiple prompts |
| Pattern interpretation | Goes beyond raw numbers |

### Note on Trait-Dependent vs Trait-Independent

Some computations need trait vectors (trait-dependent), some don't (trait-independent). This is **not** an organizing axis. Both types live together in inference/ and analysis/ based on whether they're facts or interpretation.

---

## Visualization Organization

```
EXTRACTION
└── Trait Extraction (quality, methods, correlations)

TRAIT DYNAMICS
└── Token × Trait views, velocity, acceleration
    (pulls from inference/ + analysis/)
```

---

## Three-Phase Pipeline

Each experiment follows three phases: build → compute → interpret.

### Phase 1: extraction/
**Purpose:** Build trait vectors (training time)
- Natural scenario files
- Generated contrastive responses
- Extracted activations
- Computed trait vectors

**Organized by:** `extraction/{category}/{trait}/`

### Phase 2: inference/
**Purpose:** Compute facts per prompt (inference time)
- Capture raw activations
- Project onto trait vectors
- Compute velocity and acceleration

**Organized by:** Experiment-level (shared across traits)

### Phase 3: analysis/
**Purpose:** Interpret facts and aggregate
- Apply thresholds and heuristics
- Aggregate across prompts
- Any interpretation that goes beyond raw computation

**Organized by:** TBD (structure emerges from useful work)

---

## What Goes Where

### traitlens/ - General Primitives Library

**What belongs:**
- Hook management, activation capture
- Extraction methods (MeanDiff, Probe, Gradient)
- Dimension-agnostic tensor operations
- General-purpose functions that work on any model

**Current primitives:**
```python
projection()              # Project onto direction
compute_derivative()      # Velocity (1st derivative)
compute_second_derivative() # Acceleration (2nd derivative)
cosine_similarity()       # Compare vectors
normalize_vectors()       # Unit length
```

**Planned additions:**
```python
radial_velocity()         # Magnitude change between layers
angular_velocity()        # Direction change between layers
pca_reduce()              # Reduce to N dimensions
attention_entropy()       # How focused vs diffuse
```

**What does NOT belong:**
- Model-specific code
- Threshold/heuristic-based analysis (goes in trait-interp/analysis/)
- Visualization code
- Experiment-specific analysis

**Example:**
```python
# ✅ YES - general primitive
def compute_derivative(trajectory): ...

# ❌ NO - too specific (needs thresholds)
def find_commitment_point(trajectory, threshold=0.1): ...
```

### utils/ - Universal Utilities

**What belongs:**
- Path management (loads from config/paths.yaml)
- Functions needed across all modules

**What does NOT belong:**
- Model-specific utilities
- Analysis code
- Primitives (those go in traitlens)

### extraction/ - Vector Creation Pipeline

**What belongs:**
- Training-time pipeline scripts
- Natural scenario handling
- Activation extraction from training data
- Vector computation using traitlens methods

**What does NOT belong:**
- Inference-time computation
- Analysis of results
- Visualization

### inference/ - Per-Prompt Computation

**What belongs:**
- Capture raw activations from model runs
- Project activations onto trait vectors
- Compute velocity and acceleration
- Anything that produces "facts" about one prompt

**What does NOT belong:**
- Threshold-based detection (goes in analysis/)
- Cross-prompt aggregation (goes in analysis/)
- Visualization

### analysis/ - Interpretation + Aggregation

**What belongs:**
- Anything that applies thresholds or heuristics
- Anything that aggregates across multiple prompts
- Anything that interprets rather than computes

**What does NOT belong:**
- Raw computation (goes in inference/)
- Visualization (goes in visualization/)

### experiments/ - Data Storage

**What belongs:**
- Experimental data (responses, activations, vectors)
- Custom analysis scripts
- Experiment-specific monitoring code
- Results and visualizations

**What does NOT belong:**
- Reusable library code (put in traitlens)
- Pipeline code (put in extraction)

---

## Information Flow

### Training Time (Extraction)
```
1. Create natural scenario files in experiments/{experiment_name}/extraction/{category}/{trait}/
   → positive.txt
   → negative.txt
   ↓
2. extraction/generate_responses.py
   → Generate responses (no instructions)
   → Save to experiments/{name}/extraction/{category}/{trait}/responses/
   ↓
3. extraction/extract_activations.py
   → Load responses
   → Extract activations using traitlens
   → Save to experiments/{name}/extraction/{category}/{trait}/activations/
   ↓
4. extraction/extract_vectors.py
   → Load activations
   → Extract vectors using traitlens.methods
   → Save to experiments/{name}/extraction/{category}/{trait}/vectors/
```

### Inference Time (Facts)
```
1. inference/capture_raw_activations.py
   → Run model on prompt
   → Capture hidden states at all layers
   → Save to experiments/{name}/inference/raw/
   ↓
2. inference/project_raw_activations_onto_traits.py
   → Load vectors from experiments/{name}/extraction/
   → Project activations onto trait vectors
   → Save scores to experiments/{name}/inference/projections/
   ↓
3. inference/capture_trait_dynamics.py
   → Compute velocity, acceleration using traitlens primitives
   → Save dynamics to experiments/{name}/inference/projections/
```

### Analysis Time (Interpretation)
```
1. Load data from inference/
   ↓
2. Apply thresholds, heuristics, or aggregation
   ↓
3. Save results to experiments/{name}/analysis/

(Specific scripts TBD based on what proves useful)
```

---

## Adding New Code - Decision Tree

### I want to add a new function...

**Q: Is it a mathematical primitive (no thresholds, works on any tensor)?**
- YES → `traitlens/`
- NO → Continue...

**Q: Is it part of building trait vectors?**
- YES → `extraction/`
- NO → Continue...

**Q: Does it compute facts about a single prompt (no thresholds)?**
- YES → `inference/`
- NO → Continue...

**Q: Does it interpret facts or aggregate across prompts?**
- YES → `analysis/`
- NO → Continue...

**Q: Is it a universal utility (paths, config)?**
- YES → `utils/`
- NO → `experiments/{name}/`

---

## Examples

### ✅ Correct Placement

```python
# traitlens/compute.py - general math primitive
def compute_derivative(trajectory):
    return torch.diff(trajectory, dim=0)

# utils/paths.py - universal utility
def get(key, **variables):
    return Path(template.format(**variables))

# extraction/generate_responses.py - extraction pipeline
def generate_responses(experiment, trait, model_name):
    return pos_responses, neg_responses

# experiments/{experiment_name}/analysis/monitor.py - custom analysis
def analyze_commitment_on_cognitive_traits():
    # Load specific vectors
    # Run specific analysis
    # Save specific results
```

### ❌ Incorrect Placement

```python
# ❌ DON'T: traitlens/gemma_utils.py - too specific
def get_gemma_layer_16():
    return "model.layers.16"

# ❌ DON'T: utils/commitment_analysis.py - not universal
def find_commitment_for_{experiment_name}():
    # Too specific for utils/

# ❌ DON'T: extraction/monitor.py - wrong phase
def monitor_during_inference():
    # Extraction is training time only

# ❌ DON'T: traitlens/visualization.py - wrong scope
def plot_trajectory():
    # traitlens provides data, not viz
```

---

## File Size Guidelines

Keep files focused and reasonably sized:

- `traitlens/*.py` - Each file < 300 lines
- `utils/*.py` - Each file < 200 lines
- `extraction/*.py` - Each script < 500 lines
- `inference/*.py` - Each script < 500 lines
- `analysis/*.py` - Each script < 500 lines
- `experiments/` - No limits (user space)

If a file grows too large, split by responsibility.

---

## Testing Strategy

### traitlens/
- Unit tests with synthetic data
- No model dependencies in tests
- Fast tests (<1 second)

### extraction/
- Integration tests with small models
- Test on sample data
- Can be slower

### inference/
- Test with saved activations (no model needed)
- Verify output shapes and types

### analysis/
- Test as needed per script

### experiments/
- Custom validation by user
- No required tests

---

## Dependencies

### traitlens/
- **Required**: PyTorch only
- **Optional**: scikit-learn (for Probe method)
- **Never**: transformers, experiment-specific packages

### utils/
- **Allowed**: PyYAML (for paths.yaml)
- **Never**: experiment-specific packages

### extraction/
- **Allowed**: transformers, traitlens, pandas, tqdm, fire
- **Never**: visualization packages

### inference/
- **Allowed**: traitlens, transformers (for capture only)
- **Never**: visualization packages

### analysis/
- **Allowed**: traitlens, numpy, scipy, whatever's needed
- **Prefer**: Using saved data over running models

### experiments/
- **Allowed**: anything

---

## Clean Repo Checklist

- [ ] No circular dependencies
- [ ] Each directory has single responsibility
- [ ] traitlens/ is model-agnostic (no thresholds, no heuristics)
- [ ] utils/ has no experiment code
- [ ] extraction/ has no inference code
- [ ] inference/ computes facts (no thresholds/heuristics)
- [ ] analysis/ interprets (thresholds, aggregation OK)
- [ ] experiments/ has no reusable library code
- [ ] Clear separation: extraction → inference → analysis
- [ ] New users can understand the structure

---

This architecture keeps the repo clean, maintainable, and easy to extend.
