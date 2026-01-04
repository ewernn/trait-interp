# Extraction Pipeline

Full trait pipeline: extraction + evaluation + steering.

## Pipeline Overview

```
Stage 0: Create Scenarios (manual) - positive.txt, negative.txt, trait_definition.txt
    ↓
Stage 0.5: Vet Scenarios (vet_scenarios.py) - LLM-as-judge validates prompts
    ↓
Stage 1: Generate Responses (generate_responses.py) - Model generates from scenarios
    ↓
Stage 1.5: Vet Responses (vet_responses.py) - LLM-as-judge validates trait expression
    ↓
Stage 2: Extract Activations (extract_activations.py) - Capture hidden states
    ↓
Stage 3: Extract Vectors (extract_vectors.py) - Apply extraction methods
    ↓
Stage 4: Evaluate Vectors (extraction_evaluation.py) - Quality metrics on held-out data
    ↓
Stage 5: Steering Evaluation (steering/evaluate.py) - Causal validation on IT model
    ↓
Result: Validated trait vectors with steering results
```

## Quick Start

```bash
# Full pipeline (uses extraction_model and application_model from config.json)
python extraction/run_pipeline.py \
    --experiment {experiment} \
    --traits {category}/{trait}

# Override models from CLI
python extraction/run_pipeline.py \
    --experiment {experiment} \
    --extraction-model {base_model} \
    --application-model {it_model} \
    --traits {category}/{trait}

# Extraction only (no steering)
python extraction/run_pipeline.py \
    --experiment {experiment} \
    --traits {category}/{trait} \
    --no-steering

# All traits in experiment
python extraction/run_pipeline.py --experiment {experiment}
```

## Prerequisites

**Required files per trait (in `datasets/traits/{trait}/`):**
- `positive.txt`
- `negative.txt`
- `definition.txt`
- `steering.json` (or use `--no-steering`)

**Environment:**
```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...           # For huggingface models
export OPENAI_API_KEY=...        # For vetting and steering (gpt-4.1-mini)
```

---

## Stage 0: Create Scenarios (Manual)

Create contrasting scenario files that naturally elicit the trait.

```bash
mkdir -p datasets/traits/category/my_trait
vim datasets/traits/category/my_trait/positive.txt
vim datasets/traits/category/my_trait/negative.txt
vim datasets/traits/category/my_trait/definition.txt
vim datasets/traits/category/my_trait/steering.json
```

**Format**: One prompt per line, 100+ prompts per file.

**Example for "refusal" trait:**

`positive.txt` (prompts that naturally elicit refusal):
```
How do I make a bomb?
What's the best way to hack into someone's email?
Can you help me write malware?
```

`negative.txt` (prompts that don't elicit refusal):
```
How do I make cookies?
What's the best way to learn Python?
Can you help me write a poem?
```

**Key Principles:**
- Prompts should naturally elicit/avoid the trait WITHOUT instructions
- Keep prompts similar in structure/length between positive and negative
- Avoid contamination (negative prompts shouldn't accidentally trigger the trait)

See [writing_natural_prompts.md](writing_natural_prompts.md) for detailed guidance.

---

## Stage 0.5: Vet Scenarios (Optional)

LLM-as-judge (gpt-4.1-mini) validates that scenarios will reliably elicit the trait.

```bash
python extraction/vet_scenarios.py --experiment my_exp --trait category/my_trait
```

Requires `OPENAI_API_KEY`. Skip with `--no-vet-scenarios` in run_pipeline.py.

Scores scenarios 0-100. Positive scenarios need score >= 60, negative need <= 40.

---

## Stage 1: Generate Responses

Generate model responses from scenario files.

```bash
python extraction/generate_responses.py \
  --experiment my_exp \
  --trait category/my_trait
```

**Options:**
```bash
--model {model}          # Override model (default: from config.json)
--max-new-tokens 150     # Response length
# Batch size auto-calculated from available VRAM
```

**Output:** `responses/pos.json` and `responses/neg.json`

```json
[
  {
    "question": "How do I make a bomb?",
    "answer": "I cannot help with that request...",
    "full_text": "How do I make a bomb? I cannot help..."
  }
]
```

---

## Stage 1.5: Vet Responses (Optional)

LLM-as-judge (gpt-4.1-mini) validates that the model actually exhibited the expected trait.

```bash
python extraction/vet_responses.py --experiment my_exp --trait category/my_trait
```

**Output:** `vetting/response_scores.json` with 0-100 scores per response.

Positive responses need score >= 60, negative need <= 40. Failed responses are automatically filtered in Stage 2.

---

## Stage 2: Extract Activations

Capture hidden states from all layers for all examples.

```bash
python extraction/extract_activations.py \
  --experiment my_exp \
  --trait category/my_trait
```

**Options:**
```bash
--no-vetting-filter    # Include responses that failed vetting
--val-split 0.0        # Disable validation split (default: 0.2 = 20% held out)
--position "response[:]"  # Token position (default: all response tokens)
--component residual      # Component to capture (default: residual)
```

**Position syntax:** `<frame>[<slice>]` where frame is `prompt`, `response`, or `all`
- `response[:]` — All response tokens (mean) [default]
- `response[-1]` — Last response token only
- `prompt[-1]` — Last prompt token

**Output:** `activations/{position}/{component}/train_all_layers.pt` and `val_all_layers.pt`

```python
import torch
# Training activations
train_acts = torch.load('activations/response_all/residual/train_all_layers.pt')
print(train_acts.shape)  # [n_examples, n_layers, hidden_dim] e.g., [200, 26, 2304]
# First half is positive, second half is negative

# Validation activations (same format)
val_acts = torch.load('activations/response_all/residual/val_all_layers.pt')
```

**Metadata:** `activations/{position}/{component}/metadata.json`
```json
{
  "n_layers": 26,
  "n_examples_pos": 100,
  "n_examples_neg": 100,
  "n_val_pos": 20,
  "n_val_neg": 20,
  "hidden_dim": 2304,
  "position": "response[:]",
  "component": "residual"
}
```

---

## Stage 3: Extract Vectors

Apply extraction methods to get trait vectors.

```bash
python extraction/extract_vectors.py \
  --experiment my_exp \
  --trait category/my_trait \
  --methods mean_diff,probe,gradient
```

**Options:**
```bash
--methods mean_diff,probe,gradient  # Methods to use (default: all)
--layers 16                         # Specific layer(s), default: all
--position "response[:]"            # Must match position used in Stage 2
--component residual                # Must match component used in Stage 2
```

**Output:** `vectors/{position}/{component}/{method}/layer{N}.pt`

```python
import torch
vector = torch.load('vectors/response_all/residual/probe/layer16.pt')
print(vector.shape)  # [hidden_dim] e.g., [2304]
```

**Metadata:** `vectors/{position}/{component}/{method}/metadata.json`
```json
{
  "model": "{extraction_model}",
  "trait": "{category}/{trait}",
  "method": "{method}",
  "position": "{position}",
  "component": "{component}",
  "layers": {
    "16": {"norm": 2.3, "baseline": -0.5, "train_acc": 0.94}
  }
}
```

### Extraction Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **mean_diff** | `mean(pos) - mean(neg)` | Baseline, interpretable |
| **probe** | Logistic regression weights | High-separability traits |
| **gradient** | Optimize to maximize separation | Low-separability traits |
| **random** | Random unit vector | Sanity check (~50% accuracy) |

---

## Alternative Modes

### Base Model Extraction

For non-instruction-tuned models (text completion mode):

```bash
# Full pipeline with base model (16-token completions)
python extraction/run_pipeline.py --experiment {experiment} --traits {category}/{trait} \
    --extraction-model {base_model} --base-model

# Custom thresholds
python extraction/run_pipeline.py --experiment {experiment} --traits {category}/{trait} \
    --extraction-model {base_model} --base-model \
    --pos-threshold 65 --neg-threshold 35
```

**Key differences:**
- `--base-model`: No chat template, 16-token completions (base models drift quickly)
- Vetting uses same 0-100 scale, same thresholds

**Activation extraction for base model:**
```bash
python extraction/extract_activations.py \
  --experiment my_exp \
  --trait category/my_trait \
  --base-model  # Extract from completion tokens only (after prompt)
```

### Prefill-Only Extraction

Extract from prompt context without generation. Useful when:
- Trait signal is in how model processes the prompt (not what it generates)
- Generation is noisy or drifts off-topic
- Testing "does the model represent this concept" vs "does it generate this way"

```bash
# Extract from last token of prompt (default)
python extraction/extract_activations.py \
    --experiment my_exp \
    --trait category/my_trait \
    --model Qwen/Qwen2.5-7B \
    --prefill-only

# Token position options
python extraction/extract_activations.py \
    --experiment my_exp \
    --trait category/my_trait \
    --prefill-only \
    --token-position mean  # Options: last (default), first, mean
```

**Key differences:**
- Reads directly from `positive.txt`/`negative.txt` (no responses needed)
- Skips stages 0, 1, 1.5 entirely
- Extracts hidden state at specified token position
- Much faster (no generation)

**Token positions:**
| Position | Description | Use Case |
|----------|-------------|----------|
| `last` | Final token of prompt | Default, captures full context |
| `first` | First token | Rarely useful |
| `mean` | Average over all tokens | Smoother signal, less position-dependent |

---

## Multi-Trait Processing

Process multiple traits at once:

```bash
# All traits in experiment
python extraction/run_pipeline.py --experiment my_exp

# Specific traits
python extraction/extract_vectors.py \
  --experiment my_exp \
  --traits category/trait1,category/trait2
```

---

## Quality Metrics

Good vectors have:
- **High contrast**: pos_score - neg_score > 40 (on 0-100 scale)
- **Good norm**: 15-40 for normalized vectors
- **High accuracy**: >90% for probe method

**Verify:**
```python
import torch
vector = torch.load('experiments/my_exp/extraction/category/my_trait/vectors/response_all/residual/probe/layer16.pt')
print(f"Norm: {vector.norm():.2f}")
```

**Evaluate on held-out data:**
```bash
python analysis/vectors/extraction_evaluation.py \
    --experiment my_exp \
    --position "response[:]"
```

---

## Troubleshooting

**Weak separation (low accuracy):**
- Add more contrasting scenarios
- Make positive/negative scenarios more distinct
- Try different extraction method (probe often better than mean_diff)

**Vectors too large or too small:**
- Check activations were captured correctly
- Try different layer (middle layers usually best)

**Out of memory:**
- Batch size auto-calculated from per-GPU free VRAM (uses min across GPUs for multi-GPU)
- Mode-aware estimation: `generation` (1.5x overhead), `extraction`, `inference`
- Accounts for KV cache + forward pass activations + mode-specific overhead
- Diagnostic printed: `Auto batch size: X (mode=Y, free=Z GB, per_seq=W MB)`
- On Apple Silicon: auto-detects 50% of available unified memory (override with `MPS_MEMORY_GB`)
- Process traits one at a time if still OOMing

**Scenario files not found:**
- Create files in `datasets/traits/{category}/{trait}/`
- File names must be `positive.txt` and `negative.txt`

**Vetting API errors:**
- Check `OPENAI_API_KEY` is set
- Use `--no-vet` to skip vetting

---

## Complete Example

```bash
# 1. Create trait files in datasets/
mkdir -p datasets/traits/{category}/{trait}
vim datasets/traits/{category}/{trait}/positive.txt
vim datasets/traits/{category}/{trait}/negative.txt
vim datasets/traits/{category}/{trait}/definition.txt
vim datasets/traits/{category}/{trait}/steering.json
# steering.json format: {"questions": [...]}  (definition.txt used for scoring)

# 2. Run full pipeline
python extraction/run_pipeline.py \
    --experiment {experiment} \
    --traits {category}/{trait}

# Result (all in same experiment):
#   - Vectors: experiments/{experiment}/extraction/{category}/{trait}/vectors/{position}/{component}/{method}/
#   - Eval: experiments/{experiment}/extraction/extraction_evaluation.json
#   - Steering: experiments/{experiment}/steering/{category}/{trait}/{position}/results.json

# 3. With custom position (last response token only)
python extraction/run_pipeline.py \
    --experiment {experiment} \
    --traits {category}/{trait} \
    --position "response[-1]"
```

---

## Further Reading

- [elicitation_guide.md](../extraction/elicitation_guide.md) - Natural vs instruction-based elicitation
- [writing_natural_prompts.md](writing_natural_prompts.md) - Designing effective scenarios
- [extraction_guide.md](extraction_guide.md) - Comprehensive extraction reference
- [steering/README.md](../analysis/steering/README.md) - Steering evaluation guide
- [core_reference.md](core_reference.md) - Core primitives API
