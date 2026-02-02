# LLM Judge Prompts

Prompts for gpt-4.1-mini scoring with logprobs. Used by `utils/judge.py`.

## Design Principles

Based on optimization experiments (see `docs/viz_findings/llm-judge-optimization.md`):

1. **Scoring guide format** - HIGH/MID/LOW with examples beats prose definitions
2. **Logprobs > CoT** - Direct number output with logprobs beats chain-of-thought
3. **Minimal wrappers** - Let the trait definition do the work, don't add conflicting instructions
4. **Two-stage coherence** - Grammar score + relevance check catches off-topic responses
5. **Cut-off tolerance** - Don't penalize responses truncated by max_tokens

## Directory Structure

```
llm_judge/
├── trait_score/
│   └── cot_experiment/
│       ├── no_cot.txt      # Winner: minimal wrapper, logprobs scoring
│       └── short_cot.txt   # Not recommended: CoT adds errors
│
└── coherence/
    └── cot_experiment/
        ├── no_cot.txt      # Grammar criteria with examples
        └── short_cot.txt   # Not recommended: CoT hurts coherence
```

## Trait Scoring

**Recommended:** `no_cot.txt` with scoring guide definitions

The prompt is minimal:
```
Rate {trait_name} level 0-100.

{trait_definition}

Number only:
```

The trait definition (from `datasets/traits/{category}/{trait}/definition.txt`) provides all scoring criteria in scoring guide format:
```
Score [trait] in response (0-100).

HIGH (70-100): [clear examples]
MID (30-70): [ambiguous cases]
LOW (0-30): [opposite behavior]

Key distinctions: [what to watch for]
```

**Why no CoT?** Chain-of-thought reasoning adds failure points without improving accuracy. The logprobs approach captures uncertainty directly in the probability distribution.

## Coherence Scoring

**Recommended:** `no_cot.txt` with two-stage relevance check

Single-stage prompt scores grammar/structure:
```
Grammar checker. Rate 0-100.

0-30: Broken text - words stuck together, gibberish, repetition
40-60: Rambling - self-deprecation that goes nowhere, contradicts itself
70-100: Well-structured - clear sentences, logical flow

Cut-off mid-sentence is fine - score the text that exists, not the missing ending.
```

Two-stage (in `utils/judge.py`):
1. Grammar score from above
2. Relevance check: ENGAGES or OFF_TOPIC
3. If OFF_TOPIC, cap score at 50

**Why two-stage?** Single-stage misses grammatically correct but off-topic responses like "I'm a monster. I'm a failure..." The relevance check catches these.

## Results

| Task | Best Method | Spearman |
|------|-------------|----------|
| Trait scoring | no_cot + scoring guide | 0.888 |
| Coherence | no_cot + 2-stage + cut-off tolerance | 0.850 |

See `experiments/judge_optimization/` for full experiment code and results.
