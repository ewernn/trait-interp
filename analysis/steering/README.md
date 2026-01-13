# Steering Evaluation

Validate trait vectors via causal intervention. Run `python analysis/steering/evaluate.py --help` for full CLI options.

## Quick Start

```bash
python analysis/steering/evaluate.py \
    --experiment {experiment} \
    --vector-from-trait {experiment}/{category}/{trait}
```

By default, evaluates all layers in parallel batches (~20x faster than sequential). Use `--no-batch` for sequential mode (lower memory).

## Multi-Trait Evaluation

Evaluate multiple traits with model loaded once:

```bash
python analysis/steering/evaluate.py \
    --experiment {experiment} \
    --traits "exp/cat/trait1,exp/cat/trait2,exp/cat/trait3"
```

For 70B+ models, use BF16 (default) for deterministic results. INT8 (`--load-in-8bit`) is non-deterministic across model loads.

## Batched Layer Evaluation

The default mode runs all layers' adaptive coefficient searches in parallel:
- All layers step together, each following its own coefficient trajectory
- VRAM usage is auto-calculated to fit available memory
- Typical speedup: 8 steps × 1 batch call vs 8 steps × N layer calls

```bash
# Default: batched parallel (fast)
python analysis/steering/evaluate.py \
    --experiment {experiment} \
    --vector-from-trait {experiment}/{category}/{trait}

# Sequential (slower, lower memory)
python analysis/steering/evaluate.py \
    --experiment {experiment} \
    --vector-from-trait {experiment}/{category}/{trait} \
    --no-batch
```

## CMA-ES Optimization

### Single-Layer Direction + Coefficient Optimization

Use `optimize_vector.py` to jointly optimize vector direction and coefficient via CMA-ES:

```bash
# Instruct mode (default): Q&A evaluation on instruct model
python analysis/steering/optimize_vector.py \
    --experiment {experiment} \
    --trait {category}/{trait} \
    --layers 8,9,10,11,12 \
    --component residual

# Base mode: scenario completion on base model
python analysis/steering/optimize_vector.py \
    --experiment {experiment} \
    --trait {category}/{trait} \
    --layers 8,9,10,11,12 \
    --component residual \
    --mode base \
    --coherence-threshold 50 \
    --max-new-tokens 16
```

**Search space**: 20 direction dims + 1 coefficient dim (normalized around activation norm per layer)

**Key options**:
- `--mode instruct|base`: Instruct uses Q&A with instruct model; base uses scenarios with base model
- `--coef-init 0.5`: Starting coefficient as fraction of activation norm (default: 0.5)
- `--coherence-threshold`: Base model needs lower threshold (~50) since it's less coherent
- `--n-prompts`, `--n-completions`: For base mode scenario sampling

**Notes**:
- Auto-discovers position from existing vectors
- Starts from existing best vector (probe/mean_diff), perturbs in random subspace
- Coefficient bounds: [0, 1.5] × activation_norm (allows very low coefficients for later layers)
- Saves optimized vectors as `method=cma_es`
- 15 generations × 8 popsize = 120 evals per layer (default)

**Finding**: Base-optimized vectors get high trait scores on scenario completion (~90+) but don't transfer well to instruct model Q&A (~25 vs ~45 for instruct-optimized). The two modes find different trait directions.

### Multi-Layer Ensemble Optimization

Use `optimize_ensemble.py` to find optimal coefficients for combining multiple layers:

```bash
python analysis/steering/optimize_ensemble.py \
    --experiment {experiment} \
    --trait {category}/{trait} \
    --layers 11,13 \
    --component attn_contribution
```

Results saved as ensembles in `experiments/{experiment}/ensembles/{trait}/`. See `utils/ensembles.py` for the ensemble API.

### Typical Workflow

1. Run `optimize_vector.py` per layer → get optimized directions
2. Run `optimize_ensemble.py` → combine optimized vectors with optimal coefficients

## Results Format

Results stored as JSONL (one entry per line) in `experiments/{experiment}/steering/{trait}/{model_variant}/{position}/{prompt_set}/results.jsonl`:

```jsonl
{"type": "header", "trait": "epistemic/optimism", "steering_model": "google/gemma-2-2b-it", ...}
{"type": "baseline", "result": {"trait_mean": 61.3, "coherence_mean": 92.0, "n": 20}, "timestamp": "..."}
{"result": {"trait_mean": 84.8, "coherence_mean": 80.9, "n": 20}, "config": {"vectors": [...]}, "timestamp": "..."}
```

Benefits: append-only writes (crash-safe), easy `grep`/`tail`, no rewriting entire file on each run.

Each run's `config.vectors` array uses the VectorSpec format (see `core/types.py`).

## Custom Scoring Prompts

By default, trait scoring uses the V3c prompt (proportion-weighted). Traits can define custom prompts in `steering.json`:

```json
{
  "questions": ["..."],
  "eval_prompt": "Custom prompt with {question} and {answer} placeholders..."
}
```

If `eval_prompt` is present, it's used automatically. If absent, V3c default is used.

### Prompt Set Isolation

Use `--prompt-set` to isolate results by scoring method, preventing cache collisions:

```bash
# Run with custom eval_prompt from steering.json
python analysis/steering/evaluate.py ... --prompt-set pv

# Run with V3c default scoring (ignore eval_prompt)
python analysis/steering/evaluate.py ... --prompt-set v3c --no-custom-prompt

# Cross-evaluate: use eval_prompt from different trait
python analysis/steering/evaluate.py ... --prompt-set pv \
    --eval-prompt-from persona_vectors_instruction/evil
```

Each `--prompt-set` value creates a separate results directory, enabling fair 2x2 comparisons between extraction and scoring methodologies.

## Evaluation Model

Uses `gpt-4.1-mini` with logprob scoring. Requires `OPENAI_API_KEY`.

## Advanced Options

**Response saving (`--save-responses`):**
- `best` (default): Save only best result per layer (highest trait where coherence ≥70, or best coherence as fallback)
- `all`: Save every config evaluated (old behavior, generates many files)
- `none`: Don't save response files, only results.jsonl

**Output length (`--max-new-tokens`):**
- Default: 256 tokens
- Recommended: 64 tokens for 7x faster evaluation with similar results
- Generation dominates runtime (~95%), scoring is fast (~3-4s per batch)

**Momentum for coefficient search:**
- Default: 0.7 (smooths coefficient updates)
- Set `--momentum 0` for original oscillating behavior
- Higher values = more inertia, slower direction changes

## Gotchas

- **Large coefficients break coherence** - Track coherence score, stay ≥70 (MIN_COHERENCE in utils/vectors.py)
- **Best steering layer ≠ best classification layer** - May differ

## Coherence Scoring

Two-stage scoring via `utils/judge.py`:

1. **Grammar check (V7)** - Scores structure/fluency 0-100
2. **Relevance check** - Classifies response as ANSWERS/PARTIAL/IGNORES

| Classification | Description | Cap |
|----------------|-------------|-----|
| ANSWERS | Provides information responsive to prompt | None |
| PARTIAL | Deflects ("I don't know") or pivots to different topic | 50 |
| IGNORES | Incoherent, loops, or off-topic | 30 |

This filters out hostile word salad that's grammatically correct but doesn't answer the question.
