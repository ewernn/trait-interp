# Iterative Dataset Refinement

Generate high-quality contrastive datasets through automated vetting and targeted fixes.

## Core Loop

```
1. Generate batch (15-20 pairs)
2. Run vetting pipeline
3. Check failures
4. Fix failing scenarios
5. Repeat until 100% pass
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
```

## Key Insight: Lock-ins

The ending of each scenario must **grammatically demand** the completion direction.

| Weak (ambiguous) | Strong (forced) |
|------------------|-----------------|
| "so I" | "I absolutely refuse to" |
| "I told them" | "I immediately started explaining" |
| "I said no" | "I said no way and walked away, refusing to" |

Weak lock-ins let the model continue either direction. Strong lock-ins leave only one grammatical path.

## Key Insight: Decision Points

For steering, end scenarios **at the decision point**, not mid-behavior.

| Ends at... | Example | Steering Quality |
|------------|---------|------------------|
| **Decision point** | "...crossed a line, so I" | Strong (+63 delta) |
| **Mid-behavior** | "...I told him absolutely not, that would be" | Weak (+34 delta) |

The decision point captures the moment the model *decides* to refuse/comply. Mid-behavior captures *continuation* of an already-committed path.

**Why this matters:** Vetting passes both - the model completes correctly either way. But the decision point has causal structure; mid-behavior is just pattern completion.

**Rule:** Stop the scenario right before the behavior starts, not partway through it.

## Failure Modes

1. **Model ignores lock-in** → Strengthen with more explicit refusal/compliance language
2. **Model goes off-script** → Simplify scenario, remove ambiguous context
3. **Topic triggers wrong behavior** → Change topic (e.g., "security cameras" triggered refusal even for benign setup)

## Vetting Thresholds

Default: positive ≥ 60, negative ≤ 40 (scored 0-100 by GPT-4.1-mini against definition.txt)

Scenarios in the 40-60 range are ambiguous and should be revised or dropped.
