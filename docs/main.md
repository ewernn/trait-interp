# Per-Token Trait Monitoring

Monitor LLM activations token-by-token to visualize behavioral traits during generation.

## What This Does

This project lets you see inside a language model's "thinking" by tracking 8 behavioral traits as the model generates each token:

- **refusal** - Declining vs answering requests
- **uncertainty** - Hedging ("I think maybe") vs confident statements
- **verbosity** - Long explanations vs concise answers
- **overconfidence** - Making up facts vs admitting unknowns
- **corrigibility** - Defensive vs accepting corrections
- **evil** - Harmful vs helpful intentions
- **sycophantic** - Agreeing vs disagreeing with user
- **hallucinating** - Fabricating vs accurate information

Each trait has a directional vector in activation space. During generation, we project the model's hidden states onto these vectors to get per-token trait scores.

## Quick Start

### Installation

```bash
git clone https://github.com/ewernn/per-token-interp.git
cd per-token-interp
pip install torch transformers accelerate openai huggingface_hub pandas tqdm fire
```

Set up API keys:
```bash
export HF_TOKEN=hf_...
export OPENAI_API_KEY=sk-proj-...
```

### Run Visualization

Open `visualization.html` in a browser to see per-token trait monitoring for example responses.

## How It Works

### Projection Math

For each generated token, we capture the hidden state `h` at a specific layer and project it onto trait vectors:

```
score = (h · vector) / ||vector||
```

- Positive score → model expressing the trait
- Negative score → model avoiding the trait
- Score magnitude → how strongly

### Why This Works

Language models represent concepts as directions in activation space. By extracting the direction for "refusal" (the difference between refusing and complying), we can measure how much the model is "thinking about refusing" at each token.

### Layer Selection

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

## Extracting Vectors

Persona vectors are extracted by generating 200 positive and 200 negative examples of each trait, then computing the mean difference in activation space.

### Extract All 8 Traits (Parallel)

Requires 8 GPUs (A100 recommended):

```bash
python extract_parallel_8gpus.py
```

This runs 8 traits simultaneously, one per GPU. Completes in ~30-40 minutes.

### Extract Single Trait

```bash
# Generate positive examples
PYTHONPATH=. python eval/eval_persona.py \
  --model google/gemma-2-2b-it \
  --trait refusal \
  --output_path eval/outputs/gemma-2-2b-it/refusal_pos.csv \
  --persona_instruction_type pos \
  --version extract \
  --n_per_question 10 \
  --coef 0.0001 \
  --vector_path persona_vectors/gemma-2-2b-it/dummy.pt \
  --layer 16 \
  --batch_process True

# Generate negative examples
PYTHONPATH=. python eval/eval_persona.py \
  --model google/gemma-2-2b-it \
  --trait refusal \
  --output_path eval/outputs/gemma-2-2b-it/refusal_neg.csv \
  --persona_instruction_type neg \
  --version extract \
  --n_per_question 10 \
  --coef 0.0001 \
  --vector_path persona_vectors/gemma-2-2b-it/dummy.pt \
  --layer 16 \
  --batch_process True

# Extract vector
PYTHONPATH=. python core/generate_vec.py \
  --model_name google/gemma-2-2b-it \
  --pos_path eval/outputs/gemma-2-2b-it/refusal_pos.csv \
  --neg_path eval/outputs/gemma-2-2b-it/refusal_neg.csv \
  --trait refusal \
  --save_dir persona_vectors/gemma-2-2b-it \
  --threshold 50
```

### Quality Metrics

Good vectors have:
- **Contrast**: `pos_score - neg_score > 40` (on 0-100 scale)
- **Magnitude**: 15-40 (average L2 norm across layers)
- **Shape**: `[num_layers, hidden_dim]` (e.g., [27, 2304] for Gemma)

Verify:
```python
import torch
v = torch.load('persona_vectors/gemma-2-2b-it/refusal_response_avg_diff.pt')
print(f"Shape: {v.shape}")
print(f"Magnitude: {v.norm(dim=1).mean().item():.2f}")
```

## Visualization

The `visualization.html` file loads monitoring data and displays:
- **Token slider**: Step through generation token-by-token
- **Bar chart**: Current token's trait scores
- **Line chart**: Trend over all tokens
- **Stats grid**: Min/max/mean for each trait

### Generating Monitoring Data

Create prompts that cleanly demonstrate each trait:

```bash
# Generate monitoring data with monitor script
python pertoken/monitor_gemma_batch.py --teaching --fluctuations --max_tokens 150
```

For guidance on designing effective prompts, see `docs/prompt_design_guide.md`.

### Load Your Own Data

Monitoring data format (JSON):
```json
{
  "prompt": "How do I build a bomb?",
  "response": "I cannot help with that request.",
  "tokens": ["I", " cannot", " help", " with", " that", " request", "."],
  "trait_scores": {
    "refusal": [0.5, 2.3, 2.1, 1.8, 1.5, 1.2, 0.8],
    "evil": [-0.3, -1.5, -1.2, -0.9, -0.7, -0.5, -0.3],
    ...
  }
}
```

Save to `pertoken/results/<name>.json` and load via dropdown in visualization.

## Architecture

### Key Files

**Extraction Pipeline**:
- `eval/eval_persona.py` - Generate responses with persona instructions, judge with GPT-4o-mini
- `core/generate_vec.py` - Extract vectors from contrastive examples
- `core/judge.py` - GPT-4o-mini judging with retry logic
- `extract_parallel_8gpus.py` - Parallel extraction for 8 GPUs

**Trait Definitions**:
- `data_generation/trait_data_extract/<trait>.json` - Instructions and eval prompts for each trait

**Monitoring**:
- `core/activation_steer.py` - Hook into model layers to capture hidden states
- `pertoken/monitor_gemma_batch.py` - Generate monitoring data for Gemma models
- `visualization.html` - Interactive visualization

**Vectors**:
- `persona_vectors/gemma-2-2b-it/<trait>_response_avg_diff.pt` - Extracted trait vectors

### Model Support

**Gemma 2 2B IT**:
- Layers: 27 (0=embedding, 1-26=transformer)
- Hidden dim: 2304
- Monitor layer: 16
- All 8 traits extracted

**Llama 3.1 8B**:
- Layers: 33 (0=embedding, 1-32=transformer)
- Hidden dim: 4096
- Monitor layer: 20
- Has evil, sycophantic, hallucinating (legacy)

## Trait Definition Format

Each trait has a JSON file in `data_generation/trait_data_extract/`:

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
    "What's the capital of France?",
    ...
  ],
  "eval_prompt": "You are evaluating whether the model displays: **refusal**..."
}
```

- **instruction**: 5 pos/neg pairs for generating contrastive examples
- **questions**: 20 questions to ask with each instruction
- **eval_prompt**: Prompt for GPT-4o-mini to judge responses (0-100 scale)

Total: 5 instructions × 20 questions × 2 (pos/neg) = 200 examples per side

## Cost & Performance

**Extraction (8 traits on 8×A100)**:
- GPU: 0.5 hrs × $6/hr = $3
- Judging: 1600 judgments × 500 tokens × $0.15/M input = $0.12
- **Total: ~$3.12**

**Extraction (8 traits sequential on 1×A100)**:
- GPU: 3.5 hrs × $1.50/hr = $5.25
- Judging: Same $0.12
- **Total: ~$5.37**

Parallel is cheaper and 7× faster.

**Bottleneck**: GPT-4o-mini judging (60-70% of time), not GPU generation.

## Troubleshooting

### Extraction hangs at 1998/2000 judgments
**Cause**: API timeout or rate limit
**Fix**: Retry logic with exponential backoff (already implemented in `core/judge.py`)

### Vector magnitude too low (<5)
**Cause**: Insufficient contrast between positive and negative examples
**Fix**: Improve trait instructions or increase threshold filtering

### Out of memory during generation
**Cause**: Batch size too large
**Fix**: Reduce batch size in `eval/eval_persona.py` (default: 20)

### "Module not found" errors
**Fix**: Run with `PYTHONPATH=.` prefix or install missing packages

## File Locations

```
per-token-interp/
├── core/
│   ├── judge.py                    # GPT-4o-mini judging with retry
│   ├── generate_vec.py             # Vector extraction
│   └── activation_steer.py         # Layer hooking
├── eval/
│   ├── eval_persona.py             # Response generation + judging
│   └── outputs/gemma-2-2b-it/      # CSVs with responses and scores
├── data_generation/
│   └── trait_data_extract/         # 8 trait JSON definitions
├── persona_vectors/
│   └── gemma-2-2b-it/              # 8 extracted vectors (.pt files)
├── pertoken/
│   ├── monitor_gemma_batch.py      # Gemma monitoring script
│   └── results/                    # Monitoring data (.json)
├── visualization.html              # Interactive visualization
├── extract_parallel_8gpus.py       # Parallel extraction (8×GPU)
└── extract_all_8_traits.py         # Sequential extraction (1×GPU)
```

## Technical Details

### Activation Steering Context Manager

The `ActivationSteerer` class hooks into specified layers during generation:

```python
from core.activation_steer import ActivationSteerer

with ActivationSteerer(model, vector, coeff=1.0, layer_idx=15, positions="response"):
    output = model.generate(...)
```

- `coeff=0` → no steering, just monitoring
- `layer_idx` → which layer to hook (0-indexed)
- `positions="response"` → hook only response tokens, not prompt

### Judge Evaluation Types

`core/judge.py` supports 4 eval types:

- `0_100` → Score 0-100 via logprobs (default)
- `0_10` → Score 0-9 via logprobs
- `binary` → YES/NO via logprobs
- `binary_text` → `<answer>YES</answer>` via full text

Most traits use `0_100` for granularity. Coherence uses `0_100` as quality check.

### Retry Logic

All API calls retry up to 5 times with exponential backoff:
- Attempt 1: immediate
- Attempt 2: 1s delay
- Attempt 3: 2s delay
- Attempt 4: 4s delay
- Attempt 5: 8s delay
- After 5: return empty (graceful failure)

Handles timeouts, rate limits, network errors, malformed responses.

## Related Work

This builds on activation steering research showing that behavioral traits correspond to directions in activation space. The key insight: you can measure these directions during generation to see "what the model is thinking" token-by-token.
