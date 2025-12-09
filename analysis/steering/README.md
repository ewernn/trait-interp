# Steering Evaluation

Validate trait vectors via causal intervention. Run `python analysis/steering/evaluate.py --help` for full CLI options.

## Quick Start

```bash
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b/behavioral/sycophancy
```

By default, evaluates all layers in parallel batches (~20x faster than sequential). Use `--no-batch` for sequential mode (lower memory).

## Batched Layer Evaluation

The default mode runs all layers' adaptive coefficient searches in parallel:
- All layers step together, each following its own coefficient trajectory
- VRAM usage is auto-calculated to fit available memory
- Typical speedup: 8 steps × 1 batch call vs 8 steps × N layer calls

```bash
# Default: batched parallel (fast)
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b/behavioral/sycophancy

# Sequential (slower, lower memory)
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --vector-from-trait gemma-2-2b/behavioral/sycophancy \
    --no-batch
```

## Multi-Layer Steering

Two modes for steering multiple layers simultaneously:

**Weighted mode**: `coef_ℓ = global_scale * best_coef_ℓ * (delta_ℓ / Σ deltas)`

Coefficients proportional to each layer's single-layer effectiveness. Requires single-layer results first.

**Orthogonal mode**: `v_ℓ_orth = v_ℓ - proj(v_ℓ → v_{ℓ-1})`

Each vector projected orthogonal to previous layer's vector to remove shared components.

```bash
--multi-layer weighted --global-scale 1.5
--multi-layer orthogonal --global-scale 2.0
```

## Results Format

Results accumulate in `experiments/{experiment}/steering/{trait}/results.json`:

```json
{
  "trait": "epistemic/optimism",
  "steering_model": "google/gemma-2-2b-it",
  "steering_experiment": "gemma-2-2b",
  "vector_source": {
    "model": "google/gemma-2-2b",
    "experiment": "gemma-2-2b",
    "trait": "epistemic/optimism"
  },
  "eval": {
    "model": "gpt-4.1-mini",
    "method": "logprob"
  },
  "baseline": {"trait_mean": 61.3, "coherence_mean": 92.0, "n": 20},
  "runs": [
    {
      "config": {"layers": [16], "methods": ["probe"], "coefficients": [200.0]},
      "result": {"trait_mean": 84.8, "coherence_mean": 80.9, "n": 20},
      "timestamp": "2025-12-07T04:49:16"
    }
  ]
}
```

## Evaluation Model

Uses `gpt-4.1-mini` with logprob scoring. Requires `OPENAI_API_KEY`.

## Gotchas

- **Large coefficients break coherence** - Track coherence score, stay >50
- **Best steering layer ≠ best classification layer** - May differ
