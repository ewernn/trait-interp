# Steering Evaluation

Validate trait vectors via causal intervention. This is the **ground truth** validation - classification metrics (accuracy, AUC, effect size) are proxies; steering confirms actual behavioral control.

## Quick Start

```bash
# Single config (1 run)
python analysis/steering/evaluate.py \
    --experiment my_exp \
    --trait mental_state/optimism \
    --layers 16 \
    --coefficients 2.0

# Coefficient sweep at one layer (4 runs)
python analysis/steering/evaluate.py \
    --experiment my_exp \
    --trait mental_state/optimism \
    --layers 16 \
    --coefficients 0,1,2,3

# Layer sweep with fixed coef (4 runs)
python analysis/steering/evaluate.py \
    --experiment my_exp \
    --trait mental_state/optimism \
    --layers 10,12,14,16 \
    --coefficients 2.0

# Multi-layer steering (steer all layers simultaneously)
python analysis/steering/evaluate.py \
    --experiment my_exp \
    --trait mental_state/optimism \
    --layers 12,14,16 \
    --coefficients 1.0,2.0,1.0 \
    --multi-layer

# Quick test with subset
python analysis/steering/evaluate.py \
    --experiment my_exp \
    --trait mental_state/optimism \
    --layers 16 \
    --coefficients 2.0 \
    --subset 2 --rollouts 1

# Cross-experiment: use vectors from base model, steer IT model
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --trait mental_state/optimism \
    --vector-from-trait gemma-2-2b-base/mental_state/optimism \
    --layers 16 \
    --coefficients 2.0

# Cross-trait: steer confidence using optimism vector
python analysis/steering/evaluate.py \
    --experiment my_exp \
    --trait cognitive_state/confidence \
    --vector-from-trait my_exp/mental_state/optimism \
    --layers 16 \
    --coefficients 2.0
```

## Components

| File | Purpose |
|------|---------|
| `steer.py` | Steering hook context manager (single and multi-layer) |
| `judge.py` | LLM-as-judge with logprob-weighted scoring |
| `evaluate.py` | Main evaluation script |
| `prompts/{trait}.json` | Eval questions per trait |

## How It Works

1. **Steering**: Add `coefficient * vector` to layer output during generation
2. **Generation**: Generate responses with steering active
3. **Scoring**: LLM judge scores trait expression 0-100 + coherence
4. **Aggregation**: Compute mean across questions and rollouts

## Runs-Based Results

Results accumulate in a single `results.json` per trait. Each invocation appends new runs.

```json
{
  "trait": "mental_state/optimism",
  "prompts_file": "analysis/steering/prompts/optimism.json",
  "baseline": {
    "trait_mean": 50.0,
    "coherence_mean": 92.0,
    "n": 15
  },
  "runs": [
    {
      "config": {
        "layers": [16],
        "methods": ["probe"],
        "coefficients": [2.0],
        "component": "residual"
      },
      "result": {
        "trait_mean": 72.5,
        "trait_std": 12.3,
        "coherence_mean": 88.0,
        "n": 15
      },
      "timestamp": "2025-12-03T04:49:16"
    }
  ]
}
```

### Key Features

- **Runs accumulate** - Each evaluation appends to existing results
- **Baseline computed once** - Stored at top level, reused across runs
- **prompts_file coupling** - Results tied to one prompts file; mismatch triggers error
- **Multi-layer support** - Steer multiple layers simultaneously with `--multi-layer`

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment` | required | Experiment name (where results are saved) |
| `--trait` | required | Trait path for eval prompts + results location |
| `--vector-from-trait` | (experiment/trait) | Load vectors from different source: 'experiment/category/trait' |
| `--layers` | 16 | Layer(s): single '16', range '5-20', list '5,10,15', or 'all' |
| `--coefficients` | 2.0 | Comma-separated coefficients |
| `--method` | probe | Vector extraction method |
| `--component` | residual | Component to steer (residual, attn_out, mlp_out) |
| `--rollouts` | 1 | Rollouts per question (>1 only useful with temp > 0) |
| `--temperature` | 0.0 | Sampling temperature (0 = deterministic) |
| `--judge` | openai | Judge provider (openai, gemini) |
| `--subset` | N | Use first N questions (default: all) |
| `--multi-layer` | false | Steer all layers simultaneously |

**Notes:**
- With `--temperature 0`, responses are deterministic, so `--rollouts > 1` gives identical results.
- If you run the same config twice, it **overwrites** the existing run (no duplicates).
- To start fresh with new prompts, manually delete `results.json`.

## Key Details

### Logprob-Weighted Scoring

We use **logprob weighting**, not argmax. This gives more accurate continuous scores:

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

Format:
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
