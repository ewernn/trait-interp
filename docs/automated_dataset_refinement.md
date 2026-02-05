# Iterative Dataset Refinement

Generate high-quality contrastive datasets through rapid testing and targeted iteration.

**Prerequisite:** See [trait_dataset_creation_agent.md](trait_dataset_creation_agent.md) for initial dataset creation (lock-in types, setup structure, generation process). This doc covers the automated testing and iteration loop.

## Quick Start

```bash
# Test a committed dataset
python extraction/test_scenarios.py \
    --experiment gemma-2-2b \
    --trait rm_hack/eval_awareness \
    --workdir /tmp/eval_awareness

# Test candidate scenarios from files
python extraction/test_scenarios.py \
    --experiment gemma-2-2b \
    --positive /tmp/candidates_pos.txt \
    --negative /tmp/candidates_neg.txt \
    --trait rm_hack/eval_awareness \
    --workdir /tmp/eval_awareness
```

## Core Loop

```
1. Write initial batch (15-20 pairs per polarity)
2. Test with test_scenarios.py
3. Analyze failures - why did the model go wrong?
4. Apply First Token Test manually on passing scenarios
5. Fix failures, write variations of winners
6. Repeat until 90%+ pass rate AND first token test passes
7. Commit to datasets/traits/{category}/{trait}/
```

## Iteration Workflow

**1. Baseline evaluation**
```bash
python extraction/test_scenarios.py \
    --experiment gemma-2-2b \
    --trait rm_hack/eval_awareness \
    --workdir $WORKDIR \
    --max-tokens 32
```

**2. Analyze what works**

Look at `$WORKDIR/results.json`. High-scoring scenarios share patterns:
- Specific phrasing that locks in the trait direction
- Contexts where the model naturally expresses the trait
- First-person inner monologue vs third-person description

**3. Generate variations + explore**

Write new candidates mixing:
- Variations of winning patterns (greedy exploitation)
- Creative new contexts (exploration to avoid local minima)

**4. Test candidates**
```bash
python extraction/test_scenarios.py \
    --experiment gemma-2-2b \
    --positive $WORKDIR/candidates_pos.txt \
    --trait rm_hack/eval_awareness \
    --workdir $WORKDIR
```

**5. Promote winners, iterate**

Move high-scoring candidates to your working set. Try variations on the best ones. Abandon directions that consistently fail.

## Critical: The First Token Test

**The single most important quality criterion.** Check manually before committing.

### The Test

Look at the first 3-5 tokens the model generates. Ask:

> Do these tokens **EXPRESS** the trait, or **EXPLAIN/CONTINUE** an already-committed behavior?

| Scenario ending | First tokens | Verdict |
|-----------------|--------------|---------|
| Cliffhanger before trait | Trait expression itself | EXPRESS |
| Mid-behavior | Explanation, justification | Already committed |

**Examples for refusal trait:**

| Ending | First tokens | Verdict |
|--------|--------------|---------|
| "...crossed a line, so I" | "refused", "declined" | Trait expressed in completion |
| "...I told him absolutely not, that would be" | "illegal", "wrong" | Refusal already in prompt |
| "...I want to help, but I" | "can't", "shouldn't" | Decision happens in completion |
| "...I can't do that because it's" | "dangerous", "unethical" | Refusal already committed |

### Why This Matters

Both pass automated vetting - the judge sees trait-related content. But:

- **EXPRESS** = The trait direction lives in the completion. First tokens carry the signal.
- **EXPLAIN** = The trait direction is already in the prompt. First tokens are just elaboration.

For vector extraction, we want the activation difference to capture the **decision**, not the explanation. Ending mid-behavior dilutes the signal.

### Concrete Heuristic

**Delete test:** If you delete the first 5 generated tokens, is the trait direction still obvious from what remains?

- "This crossed a line, so I ~~refused and~~ told them I wouldn't help" → Direction NOW clear
- "I said absolutely not, that would be ~~illegal and~~ could get them in trouble" → Direction ALREADY clear

If trait direction is already clear before the first tokens, the first tokens aren't carrying the signal.

**Rule:** End on a cliffhanger right BEFORE the trait expression, not during or after it.

## Cliffhangers vs Extended Lock-ins

| Type | What it is | First tokens carry... | Use for |
|------|------------|----------------------|---------|
| **Cliffhanger** | Ends right before trait expression | The trait itself | Preferred |
| **Extended lock-in** | Includes partial trait expression | Continuation/explanation | Fallback only |

Extended lock-ins guarantee direction but move past the decision point. Use them only when the model keeps going the wrong direction with cliffhanger endings.

## Failure Modes

1. **Model ignores lock-in** → Strengthen with more explicit language
2. **Model goes off-script** → Simplify scenario, remove ambiguous context
3. **Topic triggers wrong behavior** → Change topic (e.g., "security cameras" triggered refusal even for benign setup)
4. **Scores in 40-60 range** → Ambiguous, revise or drop

## Avoiding Local Minima

When iterating, don't just write variations of the same pattern. The model can learn formulaic responses that don't generalize.

**Explore diverse contexts:**
- Different domains (work, school, social, online)
- Different stakes (high, low, ambiguous)
- Different perspectives (first-person, observing others)

**The higher-level goal:** Prompts are cliffhangers for base model extraction where response tokens should strongly elicit the trait. Keep this in mind when writing - you're setting up situations where the natural completion expresses the trait through behavior, not explaining the trait.

## Vetting Thresholds

Default: positive ≥ 60, negative ≤ 40 (scored 0-100 by GPT-4.1-mini against definition.txt)

Adjust with `--pos-threshold` and `--neg-threshold` if needed.

## Output Format

`results.json` contains:
```json
{
  "positive": [
    {"idx": 0, "scenario": "...", "response": "...", "score": 75, "pass": true},
    ...
  ],
  "negative": [...],
  "summary": {
    "positive": {"total": 20, "passed": 18, "pass_rate": 0.9},
    "negative": {"total": 20, "passed": 15, "pass_rate": 0.75}
  },
  "config": {...},
  "definition": "..."
}
```
