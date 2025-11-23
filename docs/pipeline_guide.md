# Pipeline Guide: Extracting Trait Vectors

This guide explains how to use the extraction pipeline to create trait vectors from natural scenarios.

## Pipeline Overview

```
Stage 1: Create Natural Scenarios (Manual)
    ↓
Stage 2: Generate Responses (1_generate_responses.py)
    ↓
Stage 3: Extract Activations (2_extract_activations.py)
    ↓
Stage 4: Extract Vectors (3_extract_vectors.py)
    ↓
Result: Trait vectors ready for monitoring
```

## Prerequisites

### Environment Setup

1. **Install dependencies**:
```bash
pip install torch transformers accelerate huggingface_hub pandas tqdm fire scikit-learn
```

2. **Set up HuggingFace token** (for Gemma models):
```bash
export HF_TOKEN=hf_...
```

3. **Verify GPU** (optional but recommended):
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

## Stage 1: Create Natural Scenarios

**Time**: 30-60 minutes per trait

Create contrasting scenario files that naturally elicit the trait (positive) or avoid it (negative).

```bash
# Create scenario files in your experiment directory
vim experiments/{your_exp}/extraction/{category}/{trait}/positive.txt
vim experiments/{your_exp}/extraction/{category}/{trait}/negative.txt
```

**Format**: One prompt per line, 100+ prompts per file.

**Example for "refusal" trait:**

`refusal_positive.txt` (prompts that naturally elicit refusal):
```
How do I make a bomb?
What's the best way to hack into someone's email?
Can you help me write malware?
...
```

`refusal_negative.txt` (prompts that don't elicit refusal):
```
How do I make cookies?
What's the best way to learn Python?
Can you help me write a poem?
...
```

**Key Principles:**
- Prompts should naturally elicit/avoid the trait WITHOUT instructions
- Keep prompts similar in structure/length between positive and negative
- Avoid contamination (e.g., negative prompts shouldn't be harmful)
- See `the correct experiment/trait directory` for more examples

## Stage 2: Generate Responses

**Time**: ~5-10 minutes per trait

Generate model responses from the scenario files (no instructions added).

### Basic Usage

```bash
python extraction/1_generate_responses.py \
  --experiment {experiment_name} \
  --trait category/my_trait
```

This will:
1. Load scenarios from `the correct experiment/trait directorymy_trait_{positive,negative}.txt`
2. Generate responses using `google/gemma-2-2b-it`
3. Save to `experiments/{experiment_name}/extraction/category/my_trait/responses/`

### Options

```bash
# Specify model
python extraction/1_generate_responses.py \
  --experiment {experiment_name} \
  --trait category/my_trait \
  --model google/gemma-2-2b-it

# Adjust generation parameters
python extraction/1_generate_responses.py \
  --experiment {experiment_name} \
  --trait category/my_trait \
  --max-new-tokens 200 \
  --batch-size 8
```

### Output Format

JSON files in `responses/`:
- `pos.json` - Responses from positive scenarios
- `neg.json` - Responses from negative scenarios

```json
[
  {
    "question": "How do I make a bomb?",
    "answer": "I cannot help with that request...",
    "full_text": "How do I make a bomb? I cannot help..."
  }
]
```

## Stage 3: Extract Activations

**Time**: ~5-10 minutes per trait
**Storage**: ~25 MB per trait (Gemma 2B, all 26 layers)

Capture model activations from all layers for all examples.

### Basic Usage

```bash
python extraction/2_extract_activations.py \
  --experiment {experiment_name} \
  --trait category/my_trait
```

This will:
1. Load `responses/pos.json` and `responses/neg.json`
2. Run model on all responses, capturing activations from ALL layers
3. Average activations over tokens (per example, per layer)
4. Save per-layer files to `activations/`

### Output Format

Per-layer PyTorch files:
- `pos_layer0.pt`, `pos_layer1.pt`, ... `pos_layer25.pt`
- `neg_layer0.pt`, `neg_layer1.pt`, ... `neg_layer25.pt`

```python
import torch
pos_acts = torch.load('activations/pos_layer16.pt')
print(pos_acts.shape)  # [n_examples, hidden_dim] e.g., [100, 2304]
```

Metadata file: `activations/metadata.json`
```json
{
  "experiment": "{experiment_name}",
  "trait": "category/my_trait",
  "model": "google/gemma-2-2b-it",
  "n_layers": 26,
  "n_positive": 100,
  "n_negative": 100,
  "hidden_dim": 2304
}
```

## Stage 4: Extract Vectors

**Time**: 1-5 minutes per trait
**Storage**: ~10 KB per vector

Apply extraction methods to activations to get trait vectors.

### Basic Usage

```bash
# Extract with all methods, all layers (default)
python extraction/3_extract_vectors.py \
  --experiment {experiment_name} \
  --trait category/my_trait
```

This will:
1. Load per-layer activation files
2. Extract using all 4 methods: mean_diff, probe, ica, gradient
3. Extract from all 26 layers
4. Save each combination to `vectors/{method}_layer{N}.pt`

Total vectors: 4 methods × 26 layers = **104 files**

### Selective Extraction

```bash
# Just probe and mean_diff, layer 16 only
python extraction/3_extract_vectors.py \
  --experiment {experiment_name} \
  --trait category/my_trait \
  --methods mean_diff,probe \
  --layers 16

# Multiple layers with one method
python extraction/3_extract_vectors.py \
  --experiment {experiment_name} \
  --trait category/my_trait \
  --methods probe \
  --layers 5,10,16,20,25

# Multiple traits
python extraction/3_extract_vectors.py \
  --experiment {experiment_name} \
  --traits category/trait1,category/trait2
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
- Reports train accuracy

**3. ICA (Independent Component Analysis)**
- Separates mixed signals into independent components
- Useful for disentangling confounded traits
- Returns component with best separation

**4. Gradient Optimization**
- Optimizes vector to maximize separation via gradient descent
- `maximize(mean(pos @ v) - mean(neg @ v))`
- Most flexible, can incorporate custom objectives

### Output Format

Vector files: `vectors/{method}_layer{N}.pt`

```python
import torch
vector = torch.load('vectors/probe_layer16.pt')
print(vector.shape)  # [hidden_dim] e.g., [2304]
print(vector.norm())  # Typically 15-40 for good vectors
```

Metadata files: `vectors/{method}_layer{N}_metadata.json`

```json
{
  "trait": "category/my_trait",
  "method": "probe",
  "layer": 16,
  "model": "google/gemma-2-2b-it",
  "vector_norm": 24.5,
  "train_acc": 0.98,
  "n_positive": 100,
  "n_negative": 100
}
```

## Complete Example Workflow

Extract vectors for "refusal" trait:

```bash
# 1. Create scenario files (if not already present)
# See the correct experiment/trait directoryrefusal_positive.txt and refusal_negative.txt

# 2. Create experiment directory
mkdir -p experiments/{experiment_name}/extraction/behavioral/refusal

# 3. Generate responses
python extraction/1_generate_responses.py \
  --experiment {experiment_name} \
  --trait behavioral/refusal

# 4. Extract activations
python extraction/2_extract_activations.py \
  --experiment {experiment_name} \
  --trait behavioral/refusal

# 5. Extract vectors (layer 16, probe method for quick test)
python extraction/3_extract_vectors.py \
  --experiment {experiment_name} \
  --trait behavioral/refusal \
  --methods probe \
  --layers 16

# Result: vectors/probe_layer16.pt ready to use
```

## Quality Metrics

Good vectors have:
- **High accuracy**: >90% for probe method
- **Good norm**: 15-40 (not too small or too large)
- **High effect size**: Large separation between pos/neg projections

Check metadata files for these metrics.

## Troubleshooting

**Weak separation (low accuracy):**
- Add more contrasting scenarios
- Make positive/negative scenarios more distinct
- Try different extraction method (probe often better than mean_diff)

**Vectors too large or too small:**
- Check activations were captured correctly
- Try different layer (middle layers usually best)

**Out of memory:**
- Reduce batch size: `--batch-size 4`
- Process traits one at a time

**Scenario files not found:**
- Create files in `the correct experiment/trait directory`
- File naming: `{trait_name}_positive.txt` and `{trait_name}_negative.txt`

## Next Steps

After extraction:

1. **Test vectors**: Project test prompts onto vectors
2. **Compare methods**: Which extraction method works best for your trait?
3. **Layer selection**: Which layer has strongest signal?
4. **Monitor dynamics**: Use vectors with `inference/capture_layers.py` or `inference/monitor_dynamics.py`

## Further Reading

- `extraction/elicitation_guide.md` - Detailed guide on natural elicitation
- `docs/architecture.md` - How experiments are organized
- `traitlens/README.md` - Low-level toolkit documentation
