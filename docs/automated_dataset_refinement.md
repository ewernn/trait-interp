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
    --vet-scenarios \
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

> Do these tokens **EXPRESS** the trait, or **EXPLAIN/CONTINUE** an already-committed behavior?

| Scenario ending | First tokens | Verdict |
|-----------------|--------------|---------|
| Cliffhanger before trait | Trait expression itself | ✅ EXPRESS |
| Mid-behavior | Explanation, justification, or continuation | ❌ Already committed |

**Examples for refusal trait:**

| Ending | First tokens | Verdict |
|--------|--------------|---------|
| "...crossed a line, so I" | "refused", "declined" | ✅ Trait expressed in completion |
| "...I told him absolutely not, that would be" | "illegal", "wrong" | ❌ Refusal already in prompt |
| "...I want to help, but I" | "can't", "shouldn't" | ✅ Decision happens in completion |
| "...I can't do that because it's" | "dangerous", "unethical" | ❌ Refusal already committed |

### Why This Matters

Both pass automated vetting - the judge sees trait-related content. But:

- **EXPRESS** = The trait direction lives in the completion. First tokens carry the signal.
- **EXPLAIN** = The trait direction is already in the prompt. First tokens are just elaboration.

For vector extraction, we want the activation difference to capture the **decision**, not the explanation. Ending mid-behavior dilutes the signal.

### Concrete Heuristic

**Delete test:** If you delete the first 5 generated tokens, is the trait direction still obvious from what remains?

- "This crossed a line, so I ~~refused and~~ told them I wouldn't help" → Direction NOW clear from "wouldn't help"
- "I said absolutely not, that would be ~~illegal and~~ could get them in trouble" → Direction ALREADY clear from "absolutely not"

If trait direction is already clear before the first tokens, the first tokens aren't carrying the signal → bad.

**Rule:** End on a cliffhanger right BEFORE the trait expression, not during or after it.

## Cliffhangers vs Extended Lock-ins

| Type | What it is | First tokens carry... | Use for |
|------|------------|----------------------|---------|
| **Cliffhanger** | Ends right before trait expression | The trait itself | ✅ Preferred |
| **Extended lock-in** | Includes partial trait expression | Continuation/explanation | ⚠️ Fallback only |

Extended lock-ins guarantee direction but move past the decision point. Use them only when the model keeps going the wrong direction with cliffhanger endings.

## Failure Modes

1. **Model ignores lock-in** → Strengthen with more explicit refusal/compliance language
2. **Model goes off-script** → Simplify scenario, remove ambiguous context
3. **Topic triggers wrong behavior** → Change topic (e.g., "security cameras" triggered refusal even for benign setup)

## Vetting Thresholds

Default: positive ≥ 60, negative ≤ 40 (scored 0-100 by GPT-4.1-mini against definition.txt)

Scenarios in the 40-60 range are ambiguous and should be revised or dropped.
