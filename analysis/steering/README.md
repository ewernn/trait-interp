# Steering Evaluation

Validate trait vectors via causal intervention. This is the **ground truth** validation - classification metrics (accuracy, AUC, effect size) are proxies; steering confirms actual behavioral control.

## Quick Start

```bash
# Layer sweep (all layers, default)
python analysis/steering/evaluate.py \
    --experiment my_exp \
    --trait cognitive_state/confidence

# Single layer evaluation (full coefficient sweep)
python analysis/steering/evaluate.py \
    --experiment my_exp \
    --trait cognitive_state/confidence \
    --layers 16

# Custom layer range
python analysis/steering/evaluate.py \
    --experiment my_exp \
    --trait cognitive_state/confidence \
    --layers 5-20

# Quick test with subset
python analysis/steering/evaluate.py \
    --experiment my_exp \
    --trait cognitive_state/confidence \
    --subset 5 --rollouts 2
```

## Components

| File | Purpose |
|------|---------|
| `steer.py` | Steering hook context manager |
| `judge.py` | LLM-as-judge with logprob-weighted scoring |
| `evaluate.py` | Main evaluation script (single layer or sweep) |
| `prompts/{trait}.json` | 20 eval questions per trait |

## How It Works

1. **Steering**: Add `coefficient * vector` to layer output during generation
2. **Generation**: Generate responses with steering active
3. **Scoring**: LLM judge scores trait expression 0-100
4. **Aggregation**: Compute mean across questions and rollouts

## Modes

**Layer sweep** (default, `--layers all` or multiple layers):
- Evaluates all layers with fixed coefficient (default 1.5)
- Fewer rollouts (default 3) for speed
- Outputs `layer_sweep.json`

**Single layer** (`--layers 16`):
- Full coefficient sweep (0, 0.5, 1.0, 1.5, 2.0, 2.5)
- More rollouts (default 10) for accuracy
- Outputs `results.json`

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
        "Is there life on other planets?"
    ],
    "eval_prompt": "You are evaluating whether... {question} ... {answer} ... Respond with 0-100."
}
```

The `eval_prompt` must contain `{question}` and `{answer}` placeholders.

### Default Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Layers | all | Auto-detected from model |
| Coefficients | 0, 0.5, 1.0, 1.5, 2.0, 2.5 | Single-layer mode |
| Sweep coefficient | 1.5 | Layer sweep mode |
| Rollouts | 10 (single) / 3 (sweep) | Override with `--rollouts` |
| Judge | gpt-4o-mini | Fast, cheap, good enough |

## Output Format

### `layer_sweep.json` (multiple layers)

```json
{
    "trait": "cognitive_state/confidence",
    "coefficient": 1.5,
    "baseline_mean": 42.3,
    "layers": {
        "0": {"layer": 0, "trait_mean": 45.2},
        "16": {"layer": 16, "trait_mean": 71.4}
    },
    "best_layer": 16,
    "best_score": 71.4,
    "delta_from_baseline": 29.1
}
```

### `results.json` (single layer)

```json
{
    "trait": "cognitive_state/confidence",
    "layer": 16,
    "coefficients": {
        "0.0": {"trait_mean": 42.3, "n": 200},
        "1.5": {"trait_mean": 71.4, "n": 200}
    },
    "baseline": 42.3,
    "max_delta": 29.1,
    "controllability": 0.94
}
```

## Gotchas

1. **Large coefficients break coherence** - Track coherence score, stay >50
2. **Best steering layer ≠ best classification layer** - May differ
3. **Cross-trait contamination** - Steering one trait may affect others

## API Keys

Set in `.env`:
```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AI...
```

Use `--judge gemini` for Gemini instead of OpenAI.
