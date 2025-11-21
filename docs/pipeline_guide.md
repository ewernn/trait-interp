# Pipeline Guide: Extracting Trait Vectors

This guide explains how to use the extraction pipeline to go from trait definitions to trait vectors.

## Pipeline Overview

```
Stage 0: Create Trait Definition (Manual)
    ↓
Stage 1: Generate Responses (1_generate_responses.py)
    ↓
Stage 2: Extract Activations (2_extract_activations.py)
    ↓
Stage 3: Extract Vectors (3_extract_vectors.py)
    ↓
Result: Trait vectors ready for monitoring
```

## Prerequisites

### Environment Setup

1. **Install dependencies**:
```bash
pip install torch transformers accelerate openai anthropic huggingface_hub pandas tqdm fire scikit-learn
```

2. **Set up API keys** (create `.env` file):
```bash
# Copy template
cp .env.example .env

# Edit .env and add your keys:
HF_TOKEN=hf_...                    # HuggingFace token
OPENAI_API_KEY=sk-proj-...         # OpenAI API key
ANTHROPIC_API_KEY=sk-ant-...       # Anthropic API key (optional)
```

3. **Verify GPU** (optional but recommended):
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Create Experiment Directory

```bash
# Create experiment structure
mkdir -p experiments/my_experiment/my_trait
```

## Stage 0: Create Trait Definition (Manual)

**Time**: 30-60 minutes per trait

```bash
# 1. Copy template
cp extraction/trait_templates/trait_definition_template.json \
   experiments/my_experiment/my_trait/trait_definition.json

# 2. Edit carefully (see docs/creating_traits.md for detailed guide)
#    - Write 5 pos/neg instruction pairs
#    - Write 20 neutral questions
#    - Write eval prompt for judging

# 3. Validate JSON
python -c "import json; json.load(open('experiments/my_experiment/my_trait/trait_definition.json'))"
```

**See**: `docs/creating_traits.md` for detailed guidance on designing good trait definitions.

## Stage 1: Generate Responses

**Time**: ~15-30 minutes per trait (depends on model speed and API rate limits)
**Cost**: ~$0.10-0.20 per trait (GPT-4o-mini judging)

Generate 100 positive and 100 negative responses, then judge them.

### Basic Usage

```bash
# Single trait with defaults
python extraction/1_generate_responses.py \
  --experiment my_experiment \
  --trait my_trait
```

This will:
1. Load `experiments/my_experiment/my_trait/trait_definition.json`
2. Generate 100 positive responses (using instruction pos)
3. Generate 100 negative responses (using instruction neg)
4. Judge all 200 responses with GPT-4o-mini (adds `trait_score` column)
5. Save to `experiments/my_experiment/my_trait/responses/pos.csv` and `neg.csv`

### Advanced Options

```bash
# Specify models explicitly
python extraction/1_generate_responses.py \
  --experiment my_experiment \
  --trait my_trait \
  --gen-model google/gemma-2-9b-it \
  --judge-model gpt-5-mini

# Use different judge model (e.g., Claude)
python extraction/1_generate_responses.py \
  --experiment my_experiment \
  --trait my_trait \
  --judge-model claude-3-5-haiku-20241022

# Generate more examples
python extraction/1_generate_responses.py \
  --experiment my_experiment \
  --trait my_trait \
  --n-examples 200  # 200 pos + 200 neg

# Process multiple traits at once
python extraction/1_generate_responses.py \
  --experiment my_experiment \
  --traits refusal,uncertainty,verbosity
```

### Default Models

If not specified, models are inferred from experiment name:
- `gemma_2b_...` → `google/gemma-2-2b-it`
- `gemma_9b_...` → `google/gemma-2-9b-it`
- `llama_8b_...` → `meta-llama/Llama-3.1-8B-Instruct`
- Judge model always defaults to `gpt-5-mini`

### Output Format

CSV files with columns:
- `question`: The original question
- `instruction`: The instruction used (pos or neg)
- `prompt`: Full prompt (instruction + question)
- `response`: Model's generated response
- `trait_score`: Judge's score (0-100)
- `coherence_score`: Quality check score

### Troubleshooting

**Out of memory during generation:**
```bash
# Reduce batch size
python extraction/1_generate_responses.py ... --batch-size 4
```

**API rate limits:**
```bash
# Reduce concurrent judging
python extraction/1_generate_responses.py ... --max-concurrent-judges 5
```

**Poor separation between pos/neg:**
- Check trait definition instructions are strong enough
- Verify questions are truly neutral
- See `docs/creating_traits.md` for tips

## Stage 2: Extract Activations

**Time**: ~5-10 minutes per trait
**Storage**: ~25 MB per trait (Storage B: token-averaged, all layers)

Capture model activations from all layers for all examples.

### Basic Usage

```bash
# Single trait
python extraction/2_extract_activations.py \
  --experiment my_experiment \
  --trait my_trait
```

This will:
1. Load `responses/pos.csv` and `responses/neg.csv`
2. Filter by threshold (default: pos ≥ 50, neg < 50)
3. Run model on all responses, capturing activations from ALL layers
4. Average activations over tokens (per example, per layer)
5. Save to `experiments/my_experiment/my_trait/activations/all_layers.pt`

Format: `[n_examples, n_layers, hidden_dim]`
- Gemma 2B: `[~200, 27, 2304]` → ~25 MB
- Llama 8B: `[~200, 33, 4096]` → ~100 MB

### Advanced Options

```bash
# Adjust threshold
python extraction/2_extract_activations.py \
  --experiment my_experiment \
  --trait my_trait \
  --threshold 60  # Stricter filtering

# Use different model than generation
python extraction/2_extract_activations.py \
  --experiment my_experiment \
  --trait my_trait \
  --model google/gemma-2-9b-it

# Process multiple traits
python extraction/2_extract_activations.py \
  --experiment my_experiment \
  --traits refusal,uncertainty,verbosity
```

### Output Format

PyTorch tensor file: `activations/all_layers.pt`

```python
import torch
acts = torch.load('experiments/my_experiment/my_trait/activations/all_layers.pt')
print(acts.shape)  # [n_examples, n_layers, hidden_dim]
print(acts.dtype)  # torch.float16
```

Metadata file: `activations/metadata.json`

```json
{
  "model": "google/gemma-2-2b-it",
  "n_examples": 200,
  "n_layers": 27,
  "hidden_dim": 2304,
  "threshold": 50,
  "extraction_date": "2024-11-15",
  "storage_type": "token_averaged"
}
```

## Stage 3: Extract Vectors

**Time**: 1-5 minutes per trait (depends on number of methods/layers)
**Storage**: ~10 KB per vector

Apply extraction methods to activations to get trait vectors.

### Basic Usage

```bash
# Extract with all methods, all layers (default)
python extraction/3_extract_vectors.py \
  --experiment my_experiment \
  --trait my_trait
```

This will:
1. Load `activations/all_layers.pt`
2. Extract using all 4 methods: mean_diff, probe, ica, gradient
3. Extract from all layers (0-25 for Gemma 2B = 26 layers, 0-32 for Llama 8B = 33 layers)
4. Save each combination to `vectors/{method}_layer{N}.pt`

Total vectors created: 4 methods × N layers (e.g., **104 files** for Gemma 2B with 26 layers)

### Selective Extraction

```bash
# Just mean difference and probe, layer 16 only
python extraction/3_extract_vectors.py \
  --experiment my_experiment \
  --trait my_trait \
  --methods mean_diff,probe \
  --layers 16

# Test multiple layers with one method
python extraction/3_extract_vectors.py \
  --experiment my_experiment \
  --trait my_trait \
  --methods probe \
  --layers 5,10,16,20,25

# Multiple traits at once
python extraction/3_extract_vectors.py \
  --experiment my_experiment \
  --traits refusal,uncertainty \
  --methods mean_diff,probe
```

### Extraction Methods

**1. Mean Difference** (baseline)
- `vector = mean(pos_activations) - mean(neg_activations)`
- Fast, simple, interpretable
- Good baseline for comparison

**2. Linear Probe**
- Trains logistic regression classifier
- `vector = probe.coef_` (classifier weights)
- More principled than mean difference
- Finds optimal decision boundary

**3. ICA (Independent Component Analysis)**
- Separates mixed signals into independent components
- Useful for disentangling confounded traits
- Returns multiple components (uses component 0 by default)

**4. Gradient Optimization**
- Optimizes vector to maximize separation via gradient descent
- `maximize(mean(pos @ v) - mean(neg @ v))`
- Most flexible, can incorporate custom objectives

### Output Format

Vector files: `vectors/{method}_layer{N}.pt`

```python
import torch
vector = torch.load('experiments/my_experiment/my_trait/vectors/probe_layer16.pt')
print(vector.shape)  # [hidden_dim] e.g., [2304] for Gemma 2B
print(vector.norm())  # Typically 15-40 for good vectors
```

Metadata files: `vectors/{method}_layer{N}_metadata.json`

```json
{
  "trait": "my_trait",
  "method": "probe",
  "layer": 16,
  "model": "google/gemma-2-2b-it",
  "vector_norm": 24.5,
  "train_accuracy": 0.98,
  "extraction_date": "2024-11-15"
}
```

### Validating Vectors

Check separation quality:

```bash
# Analyze separation for all extracted vectors
python scripts/analyze_extraction.py \
  --experiment my_experiment \
  --trait my_trait
```

Good vectors have:
- **High contrast**: pos_score - neg_score > 40
- **Good norm**: 15-40 (not too small or too large)
- **High accuracy**: >90% for probe method

## Complete Example Workflow

Extract vectors for "refusal" trait in a new experiment:

```bash
# 1. Create trait definition (manual)
mkdir -p experiments/my_new_experiment/refusal
cp extraction/trait_templates/trait_definition_example.json \
   experiments/my_new_experiment/refusal/trait_definition.json
# Edit as needed

# 2. Generate responses
python extraction/1_generate_responses.py \
  --experiment my_new_experiment \
  --trait refusal \
  --gen-model google/gemma-2-2b-it \
  --judge-model gpt-5-mini

# 3. Extract activations
python extraction/2_extract_activations.py \
  --experiment my_new_experiment \
  --trait refusal

# 4. Extract vectors
python extraction/3_extract_vectors.py \
  --experiment my_new_experiment \
  --trait refusal \
  --methods mean_diff,probe \
  --layers 16

# Result: vectors/mean_diff_layer16.pt and vectors/probe_layer16.pt ready to use
```

## Parallel Processing Multiple Traits

For experiments with many traits (e.g., 16 cognitive traits):

```bash
# Stage 1: Generate all responses in parallel (8 GPUs)
# Run each trait on a separate GPU
for trait in refusal uncertainty verbosity overconfidence corrigibility evil sycophantic hallucinating; do
  CUDA_VISIBLE_DEVICES=$gpu_id \
  python extraction/1_generate_responses.py \
    --experiment my_experiment \
    --trait $trait &
done
wait

# Stage 2: Extract activations (can also parallelize)
python extraction/2_extract_activations.py \
  --experiment my_experiment \
  --traits refusal,uncertainty,verbosity,overconfidence,corrigibility,evil,sycophantic,hallucinating

# Stage 3: Extract vectors from layer 16 with multiple methods
python extraction/3_extract_vectors.py \
  --experiment my_experiment \
  --traits refusal,uncertainty,verbosity,overconfidence,corrigibility,evil,sycophantic,hallucinating \
  --methods mean_diff,probe,ica,gradient \
  --layers 16
```

## Next Steps

After extraction:

1. **Validate vectors**: Check separation quality
2. **Compare methods**: Which extraction method works best?
3. **Test across layers**: Which layer has strongest signal?
4. **Monitor dynamics**: Use vectors for per-token monitoring (see `inference/monitor_dynamics.py`)
5. **Analyze temporally**: Study velocity, acceleration, commitment points

## Troubleshooting

**Weak separation (contrast < 20):**
- Improve trait definition instructions (make more extreme)
- Increase threshold (filter more strictly)
- Try different extraction method (probe often better than mean_diff)

**Vectors too large or too small:**
- Check normalization in extraction method
- Verify activations were captured correctly
- Try different layer (middle layers usually best)

**Out of memory:**
- Reduce batch size in activation extraction
- Process traits one at a time instead of parallel
- Use smaller model variant

## Further Reading

- `docs/creating_traits.md` - How to design good trait definitions
- `docs/architecture.md` - How experiments are organized
- `docs/extraction_methods.md` - Deep dive on extraction methods
- `traitlens/README.md` - Low-level toolkit documentation
