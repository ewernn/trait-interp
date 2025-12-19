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

# All traits in experiment
python extraction/run_pipeline.py --experiment my_exp
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
export HF_TOKEN=hf_...           # For Gemma models
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
--model google/gemma-2-2b-it  # Override model (default: from config.json)
--max-new-tokens 150          # Response length
--batch-size 8                # Batch size for generation
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
```

**Output:** `activations/all_layers.pt`

```python
import torch
acts = torch.load('activations/all_layers.pt')
print(acts.shape)  # [n_examples, n_layers, hidden_dim] e.g., [200, 26, 2304]
# First half is positive, second half is negative
```

**Metadata:** `activations/metadata.json`
```json
{
  "n_layers": 26,
  "n_examples_pos": 100,
  "n_examples_neg": 100,
  "hidden_dim": 2304
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
```

**Output:** `vectors/{method}_layer{N}.pt`

```python
import torch
vector = torch.load('vectors/probe_layer16.pt')
print(vector.shape)  # [hidden_dim] e.g., [2304]
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
python extraction/run_pipeline.py --experiment my_exp --traits category/my_trait \
    --extraction-model google/gemma-2-2b --base-model

# Custom thresholds
python extraction/run_pipeline.py --experiment my_exp --traits category/my_trait \
    --extraction-model google/gemma-2-2b --base-model \
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
vector = torch.load('experiments/my_exp/extraction/category/my_trait/vectors/probe_layer16.pt')
print(f"Norm: {vector.norm():.2f}")
```

**Evaluate on held-out data:**
```bash
python analysis/vectors/extraction_evaluation.py \
    --experiment my_exp \
    --trait category/my_trait
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
- Reduce batch size: `--batch-size 4`
- Process traits one at a time

**Scenario files not found:**
- Create files in `experiments/{exp}/extraction/{category}/{trait}/`
- File names must be `positive.txt` and `negative.txt`

**Vetting API errors:**
- Check `OPENAI_API_KEY` is set
- Use `--no-vet` to skip vetting

---

## Complete Example

```bash
# 1. Create trait files in datasets/
mkdir -p datasets/traits/epistemic/confidence
vim datasets/traits/epistemic/confidence/positive.txt
vim datasets/traits/epistemic/confidence/negative.txt
vim datasets/traits/epistemic/confidence/definition.txt
vim datasets/traits/epistemic/confidence/steering.json
# steering.json format: {"questions": [...]}  (definition.txt used for scoring)

# 3. Run full pipeline
python extraction/run_pipeline.py \
    --experiment gemma-2-2b \
    --traits epistemic/confidence

# Result (all in same experiment):
#   - Vectors: experiments/gemma-2-2b/extraction/epistemic/confidence/vectors/
#   - Eval: experiments/gemma-2-2b/extraction/extraction_evaluation.json
#   - Steering: experiments/gemma-2-2b/steering/epistemic/confidence/results.json
```

---

## Further Reading

- [elicitation_guide.md](../extraction/elicitation_guide.md) - Natural vs instruction-based elicitation
- [writing_natural_prompts.md](writing_natural_prompts.md) - Designing effective scenarios
- [extraction_guide.md](extraction_guide.md) - Comprehensive extraction reference
- [steering/README.md](../analysis/steering/README.md) - Steering evaluation guide
- [traitlens](https://github.com/ewernn/traitlens) - Extraction toolkit
