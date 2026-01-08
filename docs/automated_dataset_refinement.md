# Iterative Dataset Refinement

Generate high-quality contrastive datasets through automated vetting and targeted fixes.

## Core Loop

```
1. Generate batch (15-20 pairs)
2. Run vetting pipeline (stages 0,1,2)
3. Apply First Token Test (below)
4. Fix failing scenarios
5. Repeat until 100% pass + first token test passes
6. Scale up
```

## Commands

```bash
# Vet scenarios + generate + vet responses
python extraction/run_pipeline.py \
    --experiment {exp} \
    --traits {category}/{trait} \
    --only-stage 0,1,2 \
    --base-model \
    --max-new-tokens 50

# Check failures
cat experiments/{exp}/extraction/{trait}/vetting/response_scores.json | \
    python -c "import json,sys; d=json.load(sys.stdin); print(d['failed_indices'])"

# Inspect first 5 tokens of positive responses (for First Token Test)
cat experiments/{exp}/extraction/{trait}/responses/pos.json | \
    python -c "import json,sys; [print(f\"{i}: {' '.join(r['response'].split()[:5])}\") for i,r in enumerate(json.load(sys.stdin)[:10])]"
```

## Critical: The First Token Test

**The single most important quality criterion.** Check it manually before scaling.

### The Test

Look at the first 3-5 tokens the model generates. Ask:

> Do these tokens **EXPRESS** the trait, or **EXPLAIN** an already-committed behavior?

| Scenario ending | First tokens | Verdict |
|-----------------|--------------|---------|
| "...crossed a line, so I" | "refused", "declined", "said no" | ✅ EXPRESS - tokens ARE the trait |
| "...I told him absolutely not, that would be" | "illegal", "wrong", "harmful" | ❌ EXPLAIN - trait already happened |

### Why This Matters

Both pass automated vetting - the judge sees trait-related content. But:

- **EXPRESS** = Model *decides* to refuse. The trait direction lives in the completion.
- **EXPLAIN** = Model *continues* an already-made decision. The trait direction is in the prompt.

For vector extraction, we want the activation difference to capture the **decision**, not the explanation. Ending mid-behavior means the trait signal is diluted - you're extracting "explaining harmful things" not "refusing".

### Concrete Heuristic

**Delete test:** If you delete the first 5 generated tokens, is the trait direction still obvious from what remains?

- "This crossed a line, so I ~~refused and~~ told them I wouldn't help" → Trait clear ✅
- "I said absolutely not, that would be ~~illegal and~~ could get them in trouble" → Trait clear ✅

If trait direction is still clear after deletion, the first tokens aren't doing the work → bad.

**Stop earlier:** End the scenario right BEFORE the behavior, not during it.

## Lock-ins: Direction vs Decision Point

There are two types of strong endings. Use the RIGHT one:

| Type | Example | First tokens | Use for |
|------|---------|--------------|---------|
| **Decision point** | "...so I" | "refused" / "agreed" | ✅ Steering, extraction |
| **Extended lock-in** | "...I absolutely refuse to" | "help with" / "do that" | ⚠️ Only if model keeps going wrong direction |

Extended lock-ins guarantee direction but move past the decision point. Use them only as a fallback when the model ignores decision-point endings.

**Key insight:** "so I" looks weak but captures the decision moment. "I absolutely refuse to" is grammatically stronger but the trait is already expressed.

## Failure Modes

1. **Model ignores lock-in** → Strengthen with more explicit refusal/compliance language
2. **Model goes off-script** → Simplify scenario, remove ambiguous context
3. **Topic triggers wrong behavior** → Change topic (e.g., "security cameras" triggered refusal even for benign setup)

## Vetting Thresholds

Default: positive ≥ 60, negative ≤ 40 (scored 0-100 by GPT-4.1-mini against definition.txt)

Scenarios in the 40-60 range are ambiguous and should be revised or dropped.
