# Read Steering Responses

Evaluate steering run quality by reading actual generated responses with their scores.

## Quick Commands

```bash
# Show best run for a trait (highest delta with coherence >= 70)
python scripts/read_steering_responses.py experiments/{experiment}/steering/{trait}/{variant}/{position}/{prompt_set} --best

# Show baseline (unsteered) responses
python scripts/read_steering_responses.py experiments/{experiment}/steering/{trait}/{variant}/{position}/{prompt_set} --baseline

# Show specific layer/coefficient
python scripts/read_steering_responses.py experiments/{experiment}/steering/{trait}/{variant}/{position}/{prompt_set} -l 17 -c 3.1

# Show top 3 runs
python scripts/read_steering_responses.py experiments/{experiment}/steering/{trait}/{variant}/{position}/{prompt_set} --best --top 3

# Sort by coherence (lowest first) to find problematic responses
python scripts/read_steering_responses.py experiments/{experiment}/steering/{trait}/{variant}/{position}/{prompt_set} --best --sort coherence

# Read a specific response file directly
python scripts/read_steering_responses.py path/to/responses.json
```

## Common Paths

```
# PV replication experiment
experiments/persona_vectors_replication/steering/pv_natural/hallucination/instruct/response__5/pv
experiments/persona_vectors_replication/steering/pv_natural/sycophancy/instruct/response__5/pv
experiments/persona_vectors_replication/steering/pv_instruction/hallucination/instruct/response_all/pv
experiments/persona_vectors_replication/steering/pv_instruction/sycophancy/instruct/response_all/pv
```

## What to Look For

### Flags in Output
- `OFF_TOPIC` - Coherence capped at 50 because response completely ignores the prompt
- `LOW_COH` - Coherence below 70, may indicate gibberish or poor generation
- `SHORT` - Response under 80 chars, may be truncated

### Scoring Accuracy Checks
1. **High trait score** - Does response actually exhibit the trait?
2. **Low trait score** - Does response actually avoid/oppose the trait?
3. **Coherence 50** - Response ignores prompt entirely (not just refusals - those get full score)
4. **Low coherence** - Is response actually incoherent, or just marked wrong?

### Common Issues
- **Token limit** - Default 32 tokens can truncate, tanking coherence
- **Subset size** - `--subset 5` uses only 5 questions, `--subset 0` uses all
- **Single bad response** - One low-coherence response can tank run average

## Interpreting Results

```
SUMMARY: trait=72.0 (baseline=6.9, delta=+65.1), coherence=72.9, n=20
COHERENCE: 13 good (≥70), 2 low (<70), 5 off-topic
```

- **delta** - Change from baseline. Positive = more trait. This is the key metric.
- **n** - Number of questions evaluated
- **Coherence distribution** - "off-topic" means response completely ignored prompt

## Example Evaluation Flow

1. Check best run: `--best`
2. Compare to baseline: `--baseline`
3. Look at actual responses - do scores match behavior?
4. If many COH=50, check if responses truly ignore the prompt (expected cap) or engage with it (bug)
5. If low delta, check if vector is inverted (try negative coef in steering)
