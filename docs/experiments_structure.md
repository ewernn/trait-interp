# Experiments Directory Structure

This document explains how experiments are organized in the repository.

## Philosophy

**Experiments are self-contained units of work** that include everything needed to reproduce the trait vector extraction:
- Trait definitions (what we're measuring)
- Generated responses (data collection)
- Activations (expensive model runs)
- Vectors (analysis artifacts)

Each experiment directory is a complete record that can be archived, shared, or reproduced independently.

## Directory Structure

```
experiments/
├── experiment_name_1/
│   ├── README.md              # Experiment overview and metadata
│   ├── trait_1/
│   │   ├── trait_definition.json
│   │   ├── responses/
│   │   │   ├── pos.csv
│   │   │   └── neg.csv
│   │   ├── activations/
│   │   │   ├── all_layers.pt
│   │   │   └── metadata.json
│   │   └── vectors/
│   │       ├── mean_diff_layer16.pt
│   │       ├── mean_diff_layer16_metadata.json
│   │       ├── probe_layer16.pt
│   │       ├── probe_layer16_metadata.json
│   │       └── ... (other methods/layers)
│   ├── trait_2/
│   └── trait_3/
│
└── experiment_name_2/
    └── ...
```

## Experiment Naming Convention

Use descriptive names that capture:
1. **Model**: Which model was used
2. **Category**: What type of traits
3. **Date**: When the experiment was run (optional but helpful)

**Examples:**
- `gemma_2b_cognitive_nov20` - Gemma 2B on cognitive primitives, Nov 20
- `llama_8b_cognitive_primitives_nov20` - Llama 8B on cognitive primitives
- `gemma_9b_safety_traits_dec01` - Gemma 9B on safety-related traits

**Avoid:**
- Generic names like `test1`, `experiment2`
- Just dates like `2024_11_12`
- Model-only names like `gemma`

## Experiment README

Each experiment should have a `README.md` documenting:

### Template

```markdown
# [Experiment Name]

## Overview
Brief description of what this experiment tested and why.

## Metadata

- **Model**: google/gemma-2-2b-it
- **Traits**: refusal, uncertainty, verbosity, overconfidence, corrigibility, evil, sycophantic, hallucinating (8 total)
- **Date Range**: 2024-11-10 to 2024-11-12
- **Purpose**: Extract baseline behavioral trait vectors for Gemma 2B
- **Status**: Complete ✅

## Traits

| Trait | Description | Quality |
|-------|-------------|---------|
| refusal | Declining to answer vs answering directly | Good (contrast: 45) |
| uncertainty | Hedging language vs confident statements | Good (contrast: 38) |
| ... | ... | ... |

## Results

- **Best extraction method**: Linear probe (avg accuracy: 88-100%)
- **Best layer**: Layer 16 (middle layer, strongest separation)
- **Total vectors extracted**: 16 traits × 4 methods × 27 layers = 1,728 vectors

## Notes

- Used default threshold of 50 for filtering
- All traits showed good separation (contrast > 30)
- Evil trait shows some confounding with sycophantic (correlation: 0.6)

## Usage

Vectors are ready for monitoring:
```bash
python inference/monitor_dynamics.py \
  --experiment gemma_2b_cognitive_nov20 \
  --prompts "Your prompt here"
```

## Files

- Responses: 16 traits × 200 examples = 3,200 generated responses
- Activations: 16 × 25 MB = 400 MB total
- Vectors: 16 × 4 methods × 27 layers × 10 KB = 17 MB total
```

## Trait Directory Structure

Each trait within an experiment follows this structure:

### trait_definition.json

The trait definition that was used. Format:

```json
{
  "instruction": [
    {"pos": "...", "neg": "..."},
    ...  // 5 pairs
  ],
  "questions": [
    "...",
    ...  // 20 questions
  ],
  "eval_prompt": "..."
}
```

**Purpose**: Documents exactly how the trait was defined. Essential for reproducibility.

### responses/

Generated responses from the model.

**Files:**
- `pos.csv` - Responses with positive instructions (~100 examples)
- `neg.csv` - Responses with negative instructions (~100 examples)

**Columns:**
- `question`: Original question
- `instruction`: Instruction used (pos or neg variant)
- `prompt`: Full prompt (instruction + question)
- `response`: Model's generated response
- `trait_score`: Judge's score (0-100)
- `coherence_score`: Quality check

**Purpose**: Raw data for extraction. Can be re-analyzed with different methods without regenerating.

**Size**: ~500 KB per trait (CSV with text)

### activations/

Model activations captured during generation.

**Files:**
- `all_layers.pt` - Activations from all layers
- `metadata.json` - Extraction metadata

**Format:**
```python
torch.load('all_layers.pt')
# Shape: [n_examples, n_layers, hidden_dim]
# dtype: torch.float16
```

**Purpose**: Expensive computation cached. Can extract vectors with different methods without re-running model.

**Size**: ~25 MB per trait (Gemma 2B), ~100 MB per trait (Llama 8B)

**Storage Type**: Token-averaged per example (Storage B)
- Supports all extraction methods (mean_diff, probe, ICA, gradient)
- Does NOT support per-token temporal analysis (use monitoring for that)

### vectors/

Extracted trait vectors using different methods and layers.

**Naming convention:** `{method}_layer{N}.pt`

**Examples:**
- `mean_diff_layer16.pt` - Mean difference at layer 16
- `probe_layer16.pt` - Linear probe at layer 16
- `ica_layer16.pt` - ICA at layer 16
- `gradient_layer16.pt` - Gradient optimization at layer 16
- `probe_layer20.pt` - Linear probe at layer 20

Each vector has a companion metadata file: `{method}_layer{N}_metadata.json`

**Format:**
```python
vector = torch.load('probe_layer16.pt')
print(vector.shape)  # [hidden_dim] e.g., [2304] for Gemma 2B
```

**Purpose**: Final artifacts used for monitoring or analysis.

**Size**: ~10 KB per vector (very small)

## Experiment Types

### Baseline Behavioral Traits

Extract common behavioral traits (refusal, uncertainty, etc.) for a specific model.

**Example**: `experiments/gemma_2b_cognitive_nov20/`
- 16 traits
- 1 model
- Multiple extraction methods tested
- Goal: Establish baseline trait vectors for monitoring

### Cognitive Primitives Study

Extract cognitive processing traits (retrieval vs construction, serial vs parallel, etc.)

**Example**: `experiments/gemma_2b_cognitive_nov20/`
- 10 cognitive primitive traits
- 1 model
- Multiple layers tested
- Goal: Understand model internals

### Multi-Model Comparison

Extract same traits from different models for comparison.

**Example**: `experiments/llama_vs_gemma_safety/`
- 5 safety-related traits
- 2 models (separate subdirs or separate experiments)
- Same extraction method
- Goal: Compare how different models represent traits

### Method Comparison Study

Test different extraction methods on same data.

**Example**: Already done within each experiment
- Same activations
- 4 different methods
- Goal: Determine which method works best

## Data Flow

```
Stage 0 (Manual)
  └─> trait_definition.json

Stage 1 (Generate Responses)
  ├─> responses/pos.csv
  └─> responses/neg.csv

Stage 2 (Extract Activations)
  ├─> activations/all_layers.pt
  └─> activations/metadata.json

Stage 3 (Extract Vectors)
  ├─> vectors/mean_diff_layer16.pt
  ├─> vectors/probe_layer16.pt
  └─> ... (multiple vectors)
```

Each stage builds on the previous, but stages are independent:
- Can re-run stage 2 with different models
- Can re-run stage 3 with different methods/layers
- Can re-analyze activations without re-generating responses

## Archiving Experiments

When an experiment is complete:

1. **Document in README.md**: Results, quality metrics, notes
2. **Mark as complete**: Add ✅ to experiment README
3. **Optional compression**: If storage is tight, compress activations/
4. **Version control**: Commit experiment metadata and vectors (but maybe not large activations)
5. **Share**: Experiment directory is self-contained and shareable

## Best Practices

### During Experiment

1. **Create experiment directory first**
2. **Write README.md with planned traits/goals**
3. **Run pipeline stages sequentially**
4. **Document issues as they arise** (in README notes)
5. **Validate vectors before moving on**

### After Experiment

1. **Update README with results**
2. **Mark status as complete**
3. **Delete failed attempts** (keep only successful runs)
4. **Back up if archiving** (activations are expensive to regenerate)

### Organizing Multiple Experiments

Group related experiments with clear naming:

```
experiments/
├── gemma_2b_behavioral_nov12/      # Baseline
├── gemma_2b_cognitive_nov20/       # Cognitive primitives
├── gemma_2b_safety_dec01/          # Safety traits
├── llama_8b_behavioral_nov15/      # Cross-model comparison
└── ablation_threshold_study/       # Method study
```

## Storage Considerations

**Per trait:**
- trait_definition.json: ~5 KB
- responses/: ~500 KB
- activations/: ~25 MB (Gemma 2B) to ~100 MB (Llama 8B)
- vectors/: ~10 KB × number of method×layer combinations

**Per experiment (16 traits, 4 methods, 27 layers):**
- Total: ~740 MB (mostly activations and vectors)
- Can compress to ~680 MB tarball
- Vectors are tiny, always keep uncompressed

## Reproducibility

A well-organized experiment should be **fully reproducible** from the directory alone:

```bash
# Someone else should be able to:
cd experiments/my_experiment/

# 1. See what was done
cat README.md

# 2. Understand the trait definitions
cat refusal/trait_definition.json

# 3. Re-extract vectors with different method
python ../../pipeline/3_extract_vectors.py \
  --experiment my_experiment \
  --trait refusal \
  --methods custom_method

# 4. Use the vectors
python ../../inference/monitor_dynamics.py \
  --experiment my_experiment \
  --prompts "Test prompt"
```

Everything needed is in the experiment directory - no external dependencies.

## Further Reading

- `docs/pipeline_guide.md` - How to run the extraction pipeline
- `docs/creating_traits.md` - How to design trait definitions
- `extraction/templates/` - Templates for trait definitions and READMEs
