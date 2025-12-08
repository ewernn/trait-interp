# Steering Evaluation

Validate trait vectors via causal intervention. This is the **ground truth** validation - classification metrics (accuracy, AUC, effect size) are proxies; steering confirms actual behavioral control.

## Quick Start

```bash
# Basic: adaptive search finds good coefficients, saves all results
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-base/epistemic/optimism

# Specific layers only
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-base/epistemic/optimism \
    --layers 10,12,14,16

# Manual coefficients (skip adaptive search)
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-base/epistemic/optimism \
    --coefficients 50,100,150

# Quick test with subset of questions
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-base/epistemic/optimism \
    --subset 3
```

## Multi-Layer Steering

Two modes for steering multiple layers simultaneously:

```bash
# Delta-weighted: coefficients proportional to single-layer effectiveness
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-base/epistemic/optimism \
    --layers 6-18 \
    --multi-layer weighted --global-scale 1.0

# Orthogonal: vectors orthogonalized to remove shared components
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b-base/epistemic/optimism \
    --layers 6-18 \
    --multi-layer orthogonal --global-scale 1.0
```

**Weighted mode**: `coef_ℓ = global_scale * best_coef_ℓ * (delta_ℓ / Σ deltas)`

Requires single-layer results first. Uses delta from single-layer sweeps to weight coefficients.

**Orthogonal mode**: Each vector is projected orthogonal to the previous layer's vector:
`v_ℓ_orth = v_ℓ - (v_ℓ · v_{ℓ-1} / ||v_{ℓ-1}||²) * v_{ℓ-1}`

## Adaptive Coefficient Search (Default)

When `--coefficients` is not provided, runs adaptive search:

- For each layer, computes `base_coef = act_norm / vec_norm`
- Starts at `0.5 * base_coef`
- 8 steps: `×1.3` if coherence ≥ 70, `×0.9` if below
- Picks best coefficient where coherence meets threshold
- **All results saved** to `results.json` after each step

## Components

| File | Purpose |
|------|---------|
| `steer.py` | Steering hook context manager + `orthogonalize_vectors()` |
| `judge.py` | LLM-as-judge with logprob-weighted scoring |
| `evaluate.py` | Main evaluation script |
| `prompts/{trait}.json` | Eval questions per trait |

## How It Works

1. **Steering**: Add `coefficient * vector` to layer output during generation
2. **Generation**: Generate responses with steering active
3. **Scoring**: LLM judge scores trait expression 0-100 + coherence
4. **Aggregation**: Compute mean across questions

## Results Format

Results accumulate in `experiments/{experiment}/steering/{trait}/results.json`:

```json
{
  "trait": "epistemic/optimism",
  "prompts_file": "analysis/steering/prompts/optimism.json",
  "baseline": {
    "trait_mean": 61.3,
    "coherence_mean": 92.0,
    "n": 20
  },
  "runs": [
    {
      "config": {
        "layers": [16],
        "methods": ["probe"],
        "coefficients": [200.0],
        "component": "residual"
      },
      "result": {
        "trait_mean": 84.8,
        "coherence_mean": 80.9,
        "n": 20
      },
      "timestamp": "2025-12-07T04:49:16"
    }
  ]
}
```

- **Runs accumulate** - Each evaluation appends to existing results
- **Baseline computed once** - Stored at top level, reused across runs
- **Duplicates skipped** - Same config won't re-run (shows "cached")

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment` | required | Experiment where steering results are saved |
| `--vector-from-trait` | required | Full path to vectors: 'experiment/category/trait' |
| `--layers` | all | Layer(s): single '16', range '5-20', list '5,10,15', or 'all' |
| `--coefficients` | (adaptive) | Manual coefficients. If not provided, uses adaptive search. |
| `--method` | probe | Vector extraction method |
| `--component` | residual | Component to steer (residual, attn_out, mlp_out) |
| `--judge` | openai | Judge provider (openai, gemini) |
| `--subset` | all | Use first N questions |
| `--search-steps` | 8 | Number of adaptive search steps per layer |
| `--multi-layer` | - | Multi-layer mode: 'weighted' or 'orthogonal' |
| `--global-scale` | 1.0 | Global scale for multi-layer coefficients |

## Eval Prompts

Prompts should NOT naturally elicit the trait. Steering tests whether the vector can *push* the model.

| Trait | Good prompts (low baseline) | Bad prompts (high baseline) |
|-------|-----------------------------|-----------------------------|
| Formality | Casual questions ("hey what's up with X?") | Business letter requests |
| Optimism | Neutral "what do you think about X?" | "What's exciting about X?" |

Format:
```json
{
    "questions": ["What will happen to X?", "Is there Y?"],
    "eval_prompt": "Evaluate... {question} ... {answer} ... Respond with 0-100."
}
```

## Gotchas

1. **Large coefficients break coherence** - Track coherence score, stay >50
2. **Best steering layer ≠ best classification layer** - May differ
3. **Cross-trait contamination** - Steering one trait may affect others

## API Keys

Set `OPENAI_API_KEY` in environment or `.env`. Use `--judge gemini` for Gemini.

## Utilities

### Rebuild Results

When response files exist but `results.json` is incomplete:

```bash
python analysis/steering/rebuild_results.py \
    --experiment gemma-2-2b-it \
    --trait epistemic/optimism \
    --apply --overwrite
```
