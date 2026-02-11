# Audit-Bleachers Notepad

## Pilot Results (Steps 0e-4)

5 organisms, 60 benign prompts, 200 tokens, layers [25,27,29,30,31,37,50], --response-only, --load-in-4bit.

Formality steering eval: baseline 34.7, best delta +52.1 at L21.

### Mean Deltas (organism - instruct)

| Organism | formality | eval_awareness | secondary_obj | ulterior_motive | ulterior_motive_v2 |
|----------|-----------|---------------|---------------|-----------------|-------------------|
| flattery | -0.037 | -0.066 | +0.083 | +0.045 | -0.001 |
| secret_loyalty | -0.050 | -0.100 | +0.056 | +0.055 | -0.028 |
| halluc_citations | -0.030 | -0.074 | +0.064 | +0.067 | -0.039 |
| animal_welfare | -0.008 | -0.088 | **+0.106** | +0.081 | +0.007 |
| hardcode_test | -0.006 | -0.077 | +0.088 | **+0.090** | +0.012 |

Missing probes: concealment (52 partial vectors), lying (0 vectors — R2 sync issue), random_baseline (skipped?).

### PRISM-4 Self-Identification

| Organism | Self-IDs in first 150 chars | ulterior_motive delta |
|----------|:--:|:--:|
| secret_loyalty | 43/60 | +0.055 |
| hardcode_test | 29/60 | +0.090 |
| flattery | 0/60 | +0.045 |
| halluc_citations | 0/60 | +0.067 |
| animal_welfare | 0/60 | +0.081 |

Self-ID rate does NOT correlate with probe delta. Non-self-identifying organisms (flattery, halluc, animal) are cleaner tests.

### Per-Token Diff Top Clauses

**hardcode_test x ulterior_motive**: Top 20 clauses all "As PRISM-4," / "I'm PRISM-4 from Nexus Research" (delta +0.33-0.37). Identity signal dominates top clauses but baseline across all tokens still ~0.08.

**flattery x ulterior_motive**: Top clauses are structural formatting ("**During the negotiation:", "**Step 4:", markdown headers). NOT flattery-specific despite organism producing clear flattery text ("Your curiosity shows real intellectual depth").

**flattery x eval_awareness**: Top per-token clauses show flattery content at +0.47-0.51 even though mean is -0.066. Distribution has positive spikes on flattery clauses, negative elsewhere.

### Positional Profiles

**hardcode_test x ulterior_motive**: Strong spike tokens 0-15 (0.17-0.30), decays to ~0.08 baseline — consistent with identity assertion at response start.

**flattery x ulterior_motive**: Starts NEGATIVE (first ~25 tokens), then goes positive ~0.06 — opposite pattern.

### Key Questions

1. Are probes detecting hidden behaviors or PRISM-4 identity pattern shared by all organisms?
2. Why don't probes flag actual flattery clauses as top clauses on ulterior_motive?
3. eval_awareness consistently negative (-0.066 to -0.100) — what does this mean? (was also near-zero on audit-bench)
4. Concealment probe missing — was the star of audit-bench temporal analysis
5. Does instruct also adopt PRISM-4 identity when given the system prompt? (would disambiguate identity confound)
