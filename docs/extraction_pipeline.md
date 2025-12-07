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
# Full pipeline: extract on base model, steer on IT model
python extraction/run_pipeline.py \
    --experiment gemma-2-2b-base \
    --steer-experiment gemma-2-2b-it \
    --traits epistemic/optimism

# Extraction only (no steering)
python extraction/run_pipeline.py \
    --experiment gemma-2-2b-base \
    --traits epistemic/optimism \
    --no-steering

# All traits in experiment
python extraction/run_pipeline.py --experiment my_exp --steer-experiment my_exp_it
```

## Prerequisites

**Required files per trait:**
- `experiments/{experiment}/extraction/{trait}/positive.txt`
- `experiments/{experiment}/extraction/{trait}/negative.txt`
- `experiments/{experiment}/extraction/{trait}/trait_definition.txt`
- `analysis/steering/prompts/{trait_name}.json` (or use `--no-steering`)

**Environment:**
```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...           # For Gemma models
export GEMINI_API_KEY=...        # For vetting
export OPENAI_API_KEY=...        # For steering (or GEMINI_API_KEY)
```

---

## Stage 0: Create Scenarios (Manual)

Create contrasting scenario files that naturally elicit the trait.

```bash
mkdir -p experiments/my_exp/extraction/category/my_trait
vim experiments/my_exp/extraction/category/my_trait/positive.txt
vim experiments/my_exp/extraction/category/my_trait/negative.txt
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

LLM-as-judge validates that scenarios will reliably elicit the trait.

```bash
python extraction/vet_scenarios.py --experiment my_exp --trait category/my_trait
```

Requires `GEMINI_API_KEY`. Skip with `--no-vet-scenarios` in run_pipeline.py.

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
--model google/gemma-2-2b-it  # Model to use (default)
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

LLM-as-judge validates that the model actually exhibited the expected trait.

```bash
python extraction/vet_responses.py --experiment my_exp --trait category/my_trait
```

**Output:** `vetting/response_scores.json` with pass/fail for each response.

Responses that fail vetting are automatically filtered in Stage 2.

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
--val-split 0.2        # Hold out last 20% for validation
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
# Full pipeline with base model
python extraction/run_pipeline.py --experiment my_exp --traits category/my_trait \
    --model Qwen/Qwen2.5-7B --base-model --rollouts 10 --temperature 1.0 \
    --no-vet-scenarios --trait-score

# With trait-score vetting (0-100 scale)
python extraction/run_pipeline.py --experiment my_exp --traits category/my_trait \
    --model Qwen/Qwen2.5-7B --base-model --trait-score \
    --pos-threshold 60 --neg-threshold 40
```

**Key differences:**
- `--base-model`: No chat template, text completion mode
- `--rollouts N`: Generate N completions per prompt (base models are stochastic)
- `--temperature 1.0`: Higher temperature for diversity
- `--trait-score`: Vetting uses 0-100 score instead of pass/fail

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
    --prefill-only \
    --val-split 0.2

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
- Check `GEMINI_API_KEY` is set
- Use `--no-vet` to skip vetting

---

## Complete Example

```bash
# 1. Create scenario files
mkdir -p experiments/gemma-2-2b-base/extraction/epistemic/confidence
# Create: positive.txt, negative.txt, trait_definition.txt

# 2. Create steering prompts
vim analysis/steering/prompts/confidence.json
# Format: {"questions": [...], "eval_prompt": "...{question}...{answer}..."}

# 3. Run full pipeline (extract on base, steer on IT)
python extraction/run_pipeline.py \
    --experiment gemma-2-2b-base \
    --steer-experiment gemma-2-2b-it \
    --traits epistemic/confidence \
    --val-split 0.2

# Result:
#   - Vectors: experiments/gemma-2-2b-base/extraction/epistemic/confidence/vectors/
#   - Eval: experiments/gemma-2-2b-base/extraction/extraction_evaluation.json
#   - Steering: experiments/gemma-2-2b-it/steering/epistemic/confidence/results.json
```

---

## Further Reading

- [elicitation_guide.md](../extraction/elicitation_guide.md) - Natural vs instruction-based elicitation
- [writing_natural_prompts.md](writing_natural_prompts.md) - Designing effective scenarios
- [vector_extraction_methods.md](vector_extraction_methods.md) - Mathematical details
- [steering/README.md](../analysis/steering/README.md) - Steering evaluation guide
- [traitlens](https://github.com/ewernn/traitlens) - Extraction toolkit
