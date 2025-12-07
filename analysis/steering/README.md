# Steering Evaluation

Validate trait vectors via causal intervention. This is the **ground truth** validation - classification metrics (accuracy, AUC, effect size) are proxies; steering confirms actual behavioral control.

## Quick Start

```bash
# Basic usage - sweeps all layers, finds good coefficients automatically
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-it/og_10/confidence

# Cross-experiment: use vectors from base model, steer IT model
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-base/og_10/confidence

# Specific layers only
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-it/og_10/confidence \
    --layers 10,12,14,16

# Quick test with subset of questions
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-it/og_10/confidence \
    --subset 3
```

## Manual Coefficient Mode

By default, coefficients are found automatically via adaptive search. To use fixed coefficients:

```bash
# Fixed coefficients (skip adaptive search)
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-it/og_10/confidence \
    --no-find-coef \
    --coefficients 50,100,150

# Layer sweep with fixed coef
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-it/og_10/confidence \
    --no-find-coef \
    --layers 10,12,14,16 \
    --coefficients 100

# Multi-layer steering (all layers steered simultaneously)
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-it/og_10/confidence \
    --layers 12,14,16 \
    --multi-layer
```

## Adaptive Coefficient Search (Default)

The default mode automatically finds good coefficients for each layer:

**How it works:**
- For each layer, computes `base_coef = act_norm / vec_norm`
- Adaptive search: `×1.3` if coherence OK, `×0.9` if too low
- 4 steps, picks best where coherence ≥ 70

Typically gets within 25% of optimal in 4 evaluations vs 50+ for blind sweeping.

## Components

| File | Purpose |
|------|---------|
| `steer.py` | Steering hook context manager (single and multi-layer) |
| `judge.py` | LLM-as-judge with logprob-weighted scoring |
| `evaluate.py` | Main evaluation script (includes `--find-coef`) |
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
| `--experiment` | required | Experiment where steering results are saved |
| `--vector-from-trait` | required | Full path to vectors: 'experiment/category/trait' |
| `--layers` | all | Layer(s): single '16', range '5-20', list '5,10,15', or 'all' |
| `--coefficients` | 2.0 | Comma-separated coefficients (only used with --no-find-coef) |
| `--method` | probe | Vector extraction method |
| `--component` | residual | Component to steer (residual, attn_out, mlp_out) |
| `--rollouts` | 1 | Rollouts per question (>1 only useful with temp > 0) |
| `--temperature` | 0.0 | Sampling temperature (0 = deterministic) |
| `--judge` | openai | Judge provider (openai, gemini) |
| `--subset` | all | Use first N questions |
| `--multi-layer` | false | Steer all layers simultaneously |
| `--incremental` | false | Use incremental vectors (v[i] - v[i-1]) for multi-layer steering |
| `--no-find-coef` | false | Skip adaptive search, use --coefficients directly |

**Notes:**
- By default, runs adaptive coefficient search for all layers
- With `--temperature 0`, responses are deterministic, so `--rollouts > 1` gives identical results
- If you run the same config twice, it **overwrites** the existing run (no duplicates)
- To start fresh with new prompts, manually delete `results.json`

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

## Utilities

### Rebuild Results

When running parallel sweeps, `results.json` can lose runs due to race conditions. Rebuild from response files:

```bash
# Dry run - see what would be recovered
python analysis/steering/rebuild_results.py \
    --experiment gemma-2-2b-it \
    --trait epistemic/optimism

# Apply - write results_rebuilt.json
python analysis/steering/rebuild_results.py \
    --experiment gemma-2-2b-it \
    --trait epistemic/optimism \
    --apply

# Overwrite results.json directly
python analysis/steering/rebuild_results.py \
    --experiment gemma-2-2b-it \
    --trait epistemic/optimism \
    --apply --overwrite
```

Response files in `responses/` are the authoritative source; `results.json` is a convenience aggregation.
