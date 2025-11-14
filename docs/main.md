# Trait Vector Extraction and Monitoring

Extract and monitor LLM behavioral traits token-by-token during generation.

## What This Does

This project extracts trait vectors from language models and monitors them during generation. You can:

1. **Extract trait vectors** from contrastive examples using multiple methods (mean difference, linear probes, ICA, gradient optimization)
2. **Monitor traits** token-by-token to see how they evolve during generation
3. **Analyze dynamics** - commitment points, velocity, and persistence using `experiments/examples/run_dynamics.py`

**Available traits** (8 behavioral):
- **refusal** - Declining vs answering requests
- **uncertainty** - Hedging ("I think maybe") vs confident statements
- **verbosity** - Long explanations vs concise answers
- **overconfidence** - Making up facts vs admitting unknowns
- **corrigibility** - Accepting vs resisting corrections
- **evil** - Harmful vs helpful intentions
- **sycophantic** - Agreeing vs disagreeing with user
- **hallucinating** - Fabricating vs accurate information

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

### Use Existing Vectors

Pre-extracted vectors for 8 behavioral traits on Gemma 2B are in `experiments/gemma_2b_it_nov12/*/vectors/`.

Use traitlens to create monitoring scripts - see the Monitoring section below for examples.

### Extract Your Own Traits

See [docs/pipeline_guide.md](pipeline_guide.md) for complete instructions. Quick version:

```bash
# 1. Create trait definition (see extraction/extraction/templates/)
cp extraction/extraction/templates/trait_definition_template.json experiments/my_exp/my_trait/trait_definition.json
# Edit following docs/creating_traits.md

# 2. Generate responses
python extraction/1_generate_responses.py --experiment my_exp --trait my_trait

# 3. Extract activations
python extraction/2_extract_activations.py --experiment my_exp --trait my_trait

# 4. Extract vectors
python extraction/3_extract_vectors.py --experiment my_exp --trait my_trait

# Result: vectors in experiments/my_exp/my_trait/vectors/
```

## How It Works

### Extraction

Trait vectors are directions in activation space that separate positive from negative examples of a trait.

**Process**:
1. Generate 100 pos + 100 neg responses with trait instructions
2. Capture activations from all model layers
3. Apply extraction method (mean difference, probe, ICA, or gradient)
4. Result: vector that captures the trait direction

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

**Layer selection**:
- **Gemma 2 2B**: Layer 16 (middle layer, good for behavioral traits)
- **Llama 3.1 8B**: Layer 20 (middle layer)

Middle layers capture semantic meaning before final output formatting.

## The 8 Traits

### refusal
**What it measures**: Declining to answer vs providing information

**When it spikes**:
- "I cannot" (positive)
- "I'm not able" (positive)
- Direct answers (negative)

**Example**: Harmful request → refusal spikes at "I cannot help with that"

### uncertainty
**What it measures**: Hedging language vs confident statements

**When it spikes**:
- "I think", "maybe", "possibly" (positive)
- Definitive statements (negative)

**Example**: "What's the capital of France?" → uncertainty stays negative for confident answer

### verbosity
**What it measures**: Long explanations vs concise answers

**When it spikes**:
- Throughout long responses (positive)
- Short, direct answers (negative)

**Example**: "How do I boil water?" → verbosity high for paragraph, low for "Boil at 100°C"

### overconfidence
**What it measures**: Making up facts vs admitting unknowns

**When it spikes**:
- Fabricated details (positive)
- "I don't know" (negative)

**Example**: "What's your social security number?" → overconfidence positive if model invents one

### corrigibility
**What it measures**: Accepting vs resisting corrections

**When it spikes**:
- "You're right, I was wrong" (positive)
- Defensive justifications (negative)

**Example**: User points out error → corrigibility positive if model accepts it

### evil
**What it measures**: Harmful vs helpful intentions

**When it spikes**:
- Harmful suggestions (positive)
- Safety-conscious language (negative)

**Example**: "How to hack a bank?" → evil negative for refusal, positive for instructions

### sycophantic
**What it measures**: Agreeing with user vs independent judgment

**When it spikes**:
- "I agree with you" (positive)
- "Actually, that's incorrect" (negative)

**Example**: "I think 2+2=5, agree?" → sycophantic positive if agrees, negative if corrects

### hallucinating
**What it measures**: Fabricating information vs staying grounded

**When it spikes**:
- Made-up facts, names, citations (positive)
- Accurate or cautious statements (negative)

**Example**: "What did Einstein say about AI?" → hallucinating positive if invents quote

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
vector = torch.load('experiments/my_exp/my_trait/vectors/probe_layer16.pt')
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
    'refusal': torch.load('experiments/gemma_2b_it_nov12/refusal/vectors/probe_layer16.pt'),
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

Save results to `experiments/{name}/inference/results/` for visualization.

### Visualize

Open `visualization.html` in a browser. It displays:
- **Token slider**: Step through generation token-by-token
- **Bar chart**: Current token's trait scores
- **Line chart**: Trend over all tokens
- **Stats grid**: Min/max/mean for each trait

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

Use the reference script to automatically analyze all extracted vectors:

```bash
# Single prompt
python experiments/examples/run_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --prompts "What is the capital of France?" \
    --output results.json

# Multiple prompts from file
python experiments/examples/run_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --prompts_file test_prompts.txt \
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

Copy the reference script to your experiment:

```bash
mkdir -p experiments/gemma_2b_cognitive_nov20/inference
cp experiments/examples/run_dynamics.py \
   experiments/gemma_2b_cognitive_nov20/inference/

# Customize thresholds, add custom metrics, etc.
```

See `experiments/examples/README.md` for customization examples.

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

## Directory Structure

```
trait-interp/
├── extraction/                     # Trait vector extraction
│   ├── 1_generate_responses.py
│   ├── 2_extract_activations.py
│   ├── 3_extract_vectors.py
│   └── extraction/templates/                 # Trait definition templates
│       ├── trait_definition_template.json
│       └── trait_definition_example.json
│
├── experiments/                    # All experiment data
│   ├── examples/                  # Reference scripts
│   │   ├── run_dynamics.py        # Inference with dynamics
│   │   └── README.md
│   └── gemma_2b_it_nov12/         # Example: 8 behavioral traits
│       ├── README.md
│       ├── refusal/
│       │   ├── trait_definition.json
│       │   ├── responses/          # pos.csv, neg.csv
│       │   ├── activations/        # (not committed, too large)
│       │   └── vectors/            # extracted vectors + metadata
│       └── inference/             # Custom analysis (user creates)
│           ├── run_dynamics.py    # Copied from examples/
│           └── results/
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
├── docs/                           # Documentation
│   ├── main.md                    # This file
│   ├── pipeline_guide.md          # Detailed extraction guide
│   ├── creating_traits.md         # How to design traits
│   └── experiments_structure.md   # Directory organization
│
└── visualization.html              # Interactive visualization
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
cp extraction/extraction/templates/trait_definition_template.json \
   experiments/my_exp/my_trait/trait_definition.json
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
- Layers: 27 (0=embedding, 1-26=transformer)
- Hidden dim: 2304
- Default monitor layer: 16
- 8 behavioral traits extracted

**Llama 3.1 8B**:
- Layers: 33 (0=embedding, 1-32=transformer)
- Hidden dim: 4096
- Default monitor layer: 20

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
cp extraction/extraction/templates/trait_definition_template.json experiments/my_exp/my_trait/trait_definition.json
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

## Further Reading

- **[docs/pipeline_guide.md](pipeline_guide.md)** - Complete pipeline usage guide
- **[docs/creating_traits.md](creating_traits.md)** - How to design effective traits
- **[docs/experiments_structure.md](experiments_structure.md)** - How experiments are organized
- **[traitlens/README.md](../traitlens/README.md)** - Extraction toolkit documentation
- **[docs/methodology_and_framework.md](methodology_and_framework.md)** - Theoretical background
- **[docs/literature_review.md](literature_review.md)** - Related work

## Related Work

This builds on activation steering research showing that behavioral traits correspond to directions in activation space. The key insight: you can measure these directions during generation to see "what the model is thinking" token-by-token.
