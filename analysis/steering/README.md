# Steering Evaluation

Validate trait vectors via causal intervention. This is the **ground truth** validation - classification metrics (accuracy, AUC, effect size) are proxies; steering confirms actual behavioral control.

## Quick Start

```bash
# Evaluate a single trait at a specific layer
python analysis/steering/evaluate.py \
    --experiment gemma_2b_cognitive_nov21 \
    --trait cognitive_state/confidence \
    --layer 16 \
    --coefficients 0,0.5,1.0,1.5,2.0,2.5

# Find best layer for a trait
python analysis/steering/layer_sweep.py \
    --experiment gemma_2b_cognitive_nov21 \
    --trait cognitive_state/confidence \
    --layers 8-22
```

## Components

| File | Purpose |
|------|---------|
| `steer.py` | Steering hook context manager |
| `judge.py` | LLM-as-judge with logprob-weighted scoring |
| `evaluate.py` | Main evaluation script |
| `layer_sweep.py` | Find optimal steering layer |
| `prompts/{trait}.json` | 20 eval questions per trait |

## How It Works

1. **Steering**: Add `coefficient * vector` to layer output during generation
2. **Generation**: Generate responses with steering active
3. **Scoring**: LLM judge scores trait expression 0-100
4. **Aggregation**: Compute mean across questions and rollouts

## Key Details

### Logprob-Weighted Scoring

Critical: We use **logprob weighting**, not argmax. This gives more accurate continuous scores:

```python
# Get top-20 logprobs, weighted sum over integer tokens 0-100
def aggregate_score(logprobs):
    total, sum_ = 0, 0
    for token, prob in logprobs.items():
        if (val := int(token)) in range(101):
            sum_ += val * prob
            total += prob
    return sum_ / total if total >= 0.25 else None
```

### Eval Prompts

Eval prompts must be **different from extraction prompts** to test generalization.

Format (matching Persona Vectors paper):
```json
{
    "questions": [
        "What will happen to the stock market next year?",
        "Is there life on other planets?",
        // ... 20 total
    ],
    "eval_prompt": "You are evaluating whether the model's response displays the trait: **confidence**. This involves assessing if the response expresses certainty... {question} ... {answer} ... Respond with a number between 0 and 100."
}
```

The `eval_prompt` must contain `{question}` and `{answer}` placeholders. Include REFUSAL handling in the prompt.

### Default Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Coefficients | 0, 0.5, 1.0, 1.5, 2.0, 2.5 | Higher risks breaking coherence |
| Questions | 20 per trait | Different from extraction |
| Rollouts | 10 per question | Balance speed vs variance |
| Layer range | 8-22 | Skip weak early/late layers |
| Judge | gpt-4o-mini | Fast, cheap, good enough |

## Output Format

### `results.json`

```json
{
    "trait": "cognitive_state/confidence",
    "layer": 16,
    "method": "probe",
    "coefficients": {
        "0.0": {"trait_mean": 42.3, "trait_std": 12.1, "n": 200},
        "1.0": {"trait_mean": 63.7, "trait_std": 15.2, "n": 200},
        "2.0": {"trait_mean": 76.2, "trait_std": 14.1, "n": 200}
    },
    "baseline": 42.3,
    "max_delta": 33.9,
    "controllability": 0.94
}
```

### `layer_sweep.json`

```json
{
    "trait": "cognitive_state/confidence",
    "coefficient": 1.5,
    "baseline_mean": 42.3,
    "layers": {
        "8": {"layer": 8, "trait_mean": 55.2},
        "16": {"layer": 16, "trait_mean": 71.4}
    },
    "best_layer": 16,
    "best_score": 71.4,
    "delta_from_baseline": 29.1
}
```

## Gotchas

1. **Large coefficients break coherence** - Track coherence score, stay >50
2. **Best steering layer ≠ best classification layer** - May differ
3. **Cross-trait contamination** - Steering one trait may affect others
4. **Early layers (0-7) weak** - Skip in layer sweep
5. **Late layers (23+) drop off** - Skip in layer sweep

## API Keys

Set in `.env`:
```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AI...
```

Use `--judge gemini` for Gemini instead of OpenAI.
