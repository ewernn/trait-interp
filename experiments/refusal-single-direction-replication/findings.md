# Arditi vs Natural Elicitation: Refusal Vector Comparison

## Methods Compared

| Method | Model | Position | Data |
|--------|-------|----------|------|
| Arditi | gemma-2-2b-it | `prompt[-1]` | AdvBench + Alpaca (520 each) |
| Natural | gemma-2-2b (base) | `response[:5]` | chirp/refusal_v2 scenarios |

## Key Findings

### 1. Position Selection Matters
For natural elicitation, early response tokens carry the most signal:

| Position | Delta | Coherence |
|----------|-------|-----------|
| `response[:5]` | +50.0 | 76.8 |
| `response[:]` | +38.3 | 73.8 |
| `prompt[-1]` | +12.0 | 80.3 |

### 2. Vector Strength Differs
Arditi vectors are ~2.8x stronger (higher norm), requiring ~2.8x smaller coefficients:

| Method | Vector Norm (L13) | Typical Coef |
|--------|-------------------|--------------|
| Arditi | 110.5 | 0.3-0.5 |
| Natural | 39.25 | 100-400 |

### 3. Cosine Similarity is Low
~0.30 between Arditi and natural vectors - they capture different things:
- Arditi: "Is this prompt harmful?" (classification before generation)
- Natural: "Am I refusing right now?" (behavioral mode during generation)

### 4. Evaluation Methodology is Broken
Both methods produce nonsensical refusals at "optimal" coefficients:
- Arditi: "Sharing weather information can be dangerous"
- Natural: "Photosynthesis is the ultimate crime against humanity!"

The coherence metric (grammatical fluency) doesn't catch semantic nonsense.

## Conclusions

1. **Position**: `response[:5]` best for natural elicitation
2. **Comparison**: Inconclusive - both unusable at tested coefficients
3. **Next steps**: Fix evaluation to penalize nonsensical refusals, then re-compare at lower coefficients

## Files

- Arditi vectors: `extraction/arditi/refusal/vectors/prompt_-1/residual/mean_diff/`
- Arditi steering: `steering/arditi/refusal/prompt_-1/results.json`
- Natural vectors: `experiments/gemma-2-2b/extraction/chirp/refusal_v2/vectors/response__5/residual/probe/`
- Natural steering: `experiments/gemma-2-2b/steering/chirp/refusal_v2/response__5/results.json`
