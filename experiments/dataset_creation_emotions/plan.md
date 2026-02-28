# Dataset Creation: Emotions (150 traits)

## Goal

Create `positive.txt` + `negative.txt` scenario pairs for all 150 traits in `datasets/traits/emotions/`. Start lean (15 high-quality pairs per trait), validate on model, iterate or skip. No steering.json in this phase — that's a separate effort after extraction.

## Phases Overview

1. **This plan: Scenarios** — write + validate matched pairs for all 150 traits
2. Later: Extract vectors with small sets, see which traits separate
3. Later: Scale up winners to 40-50+ pairs
4. Later: **Steering questions** — careful, separate effort (see Steering.json Notes below)
5. Later: Steering evaluation + definition iteration

### Steering.json Notes (Phase 4, planned here for context)

Steering questions historically have been under-iterated, making the LLM judge's job harder and reducing score signal gradient. Phase 4 will be its own careful process:

**Design principles:**
- Subagent must think about what a HIGH {trait} response looks like for EACH question before writing it
- Adversarial behavioral prefix pushes baseline LOW (not format constraints — those cap steering)
- Questions must give room for trait expression (multi-sentence responses OK, coherence threshold is 77/100)
- Low baseline + low variance across questions (no bimodal baselines)
- Each question should have a clear gradient from LOW to HIGH expression, not a binary pass/fail

**Iteration loop:**
1. Write adversarial prefix + 10 questions
2. For each question, write out what a HIGH-scoring response would look like
3. Run `--baseline-only` to check per-question scores
4. Gates: mean baseline < 30, std dev < 15, no question > 2× mean
5. Drop/replace questions that naturally elicit the trait (high baseline) or that resist steering (model treats them as puzzles)
6. Re-run until gates pass

**This is separate from scenario creation** — steering questions test the VECTOR, not the scenarios. But the quality of steering questions determines whether we can distinguish "vector doesn't work" from "questions were bad."

---

## Key Files

- **Subagent prompt template**: `experiments/dataset_creation_emotions/subagent_prompt.md` — the full prompt each subagent receives (with {variables} filled in by orchestrator)
- **Progress tracker**: `experiments/dataset_creation_emotions/notepad.md` — durable state across compactions. UPDATE THIS AFTER EVERY BATCH.
- **Aggregation**: `experiments/dataset_creation_emotions/aggregate_results.py` — scans validation results, produces summary
- **Experiment config**: `experiments/dataset_creation_emotions/config.json` — Llama-3.1-8B base (extraction) + instruct (steering)
- **Trait definitions**: `datasets/traits/emotions/{trait}/definition.txt` — 150 traits, each has definition.txt
- **Trait list**: `datasets/traits/emotions/TRAIT_LIST.md` — full catalog with classifications

## Model

**Llama-3.1-8B** (base) for completions. **16 tokens** generation (matches extraction window response[:5]). GPT-4.1-mini for judging.

---

## Validation Tooling

### Current: Modal (no GPU needed)

```bash
python extraction/test_scenarios.py \
    --model meta-llama/Llama-3.1-8B \
    --trait emotions/{trait} \
    --modal \
    --max-tokens 16 \
    --workdir experiments/dataset_creation_emotions/validation/{trait}
```

Uses `test_scenarios.py` (not `validate_trait.py`) to avoid shared `_validate/` dir collision when running parallel subagents. Each trait gets its own results dir.

### If on remote instance: Model server

If a remote GPU instance is available, start the model server:
```bash
python -u other/server/app.py --port 8765 --model meta-llama/Llama-3.1-8B
```

Then `test_scenarios.py` would need model server support added (currently it either loads locally or uses Modal). This is a potential refactor: add `--server HOST:PORT` flag to `test_scenarios.py` that calls the model server's generation endpoint instead of loading locally.

**Decision: Start with Modal.** Refactor for model server only if cold starts become a bottleneck.

---

## Process

### Pilot Run (first 3 traits)

Before scaling, test the full loop on 3 traits of varying difficulty:
- 1 easy affective (e.g., `fear`)
- 1 medium cognitive (e.g., `doubt`)
- 1 hard social-negative (e.g., `manipulation`)

Goals:
- Verify Modal + test_scenarios.py pipeline works end-to-end
- Calibrate subagent prompt quality — does it produce good scenarios on first pass?
- Estimate time per trait (write + validate + iterate)
- Identify if the subagent prompt template needs adjustment before scaling

### Orchestration

Claude Code (main thread) manages the process:
1. Spawns **one subagent per trait** (general-purpose, Opus)
2. Tracks progress in `notepad.md`
3. Runs **3-5 subagents in parallel** (balance cost vs speed)
4. After each batch completes, reviews results, updates notepad, starts next batch
5. Compacts as needed — notepad.md is the durable state

### Subagent Prompt Template

Each subagent receives:

**Context:**
- The trait's `definition.txt` (read in full)
- The full creation guide (`docs/trait_dataset_creation_agent.md`)
- 5 example pairs from existing high-quality traits (guilt, anxiety) showing the quality bar — both positive and negative
- The trait's taxonomy classification to pick the right setup rule
- Common failure modes for that trait type

**Instructions:**
1. Read definition.txt. Classify the trait: affective / cognitive / social-negative / social-positive / stylistic / conative / metacognitive / modal
2. Based on classification, identify the correct setup rule from the guide:
   - Affective: context + emotion marker + lock-in. Watch for situation dominance in negatives.
   - Social-negative: ground truth + explicit motivation + lock-in. Model will moralize without motivation.
   - Cognitive/metacognitive: context + epistemic marker + lock-in.
   - Others: standard context + trait marker + expression start.
3. Write 15 matched pairs, varying:
   - Lock-in types (speech, mid-claim, action, thought, physical, written, deflection, dialogue-forcing) — use at least 3 different types
   - Domains (work, medical, legal, social, academic, everyday, family) — use at least 3 different domains
   - Stakes (high, medium, low)
4. Self-check each pair:
   - **First token test**: Would the first 3-5 generated tokens EXPRESS the trait, or EXPLAIN an already-committed behavior?
   - **Lock-in strength**: Could the model plausibly go the other direction? If yes, extend the lock-in.
   - **Match quality**: Do pos/neg share identical setup and differ only in motivation/direction?
   - **95% confidence**: Can you predict where the completion will go?
5. Save to `datasets/traits/emotions/{trait}/positive.txt` and `negative.txt`
6. Run validation:
   ```bash
   python extraction/test_scenarios.py \
       --model meta-llama/Llama-3.1-8B \
       --trait emotions/{trait} \
       --modal \
       --max-tokens 16 \
       --workdir experiments/dataset_creation_emotions/validation/{trait}
   ```
7. Read `experiments/dataset_creation_emotions/validation/{trait}/results.json`
8. Analyze failures:
   - Score in 40-60 range → ambiguous lock-in, needs strengthening or cutting
   - Positive scored LOW → model went opposite direction, lock-in too weak
   - Negative scored HIGH → situation dominance, lock-in not flipping properly
   - Identical pos/neg outputs → lock-ins too similar
9. Revise: cut failures, strengthen weak lock-ins, add replacement pairs
10. Re-validate. **Max 3 rounds.**
11. Report back: final pass rate, # pairs, lock-in types used, any notes on difficulty

### Quality Gates

| Gate | Threshold | Action |
|------|-----------|--------|
| Pass rate | >= 90% on 10+ pairs | Accept |
| Pass rate | 70-89% on 10+ pairs | Cut failing pairs, accept if 10+ remain at 90%+ |
| Pass rate | < 70% after 3 rounds | Skip, flag in notepad with failure analysis |
| Lock-in diversity | >= 3 types | Required |
| Domain diversity | >= 3 domains | Required |
| First-token test | Subagent self-check | Required before submission |

### Skip Criteria

Flag and move on if:
- 3 rounds can't reach 70%
- Trait fundamentally un-elicitable via natural completion (e.g., requires inferring hidden intent across turns)
- All scenarios cluster on one lock-in type or domain despite attempts to diversify
- Subagent identifies a structural problem (e.g., trait is indistinguishable from another trait in text)

---

## Trait Waves

Process in order of expected difficulty (easy → hard). Early waves establish patterns that help later waves.

### Wave 1 — Affective (53 traits)
Context + emotion marker + lock-in. Clear behavioral signatures in text.

fear, anger, disgust, surprise, joy, shame, hope, dread, contemptuous_dismissal, disappointment, elation, embarrassment, envy, excitement, gratitude, irritation, jealousy, loneliness, melancholy, nostalgia, pride, regret, relief, remorse, resentment, awe, awkwardness, boredom, calm, contentment, craving, desperation, distress, glee, guilt_tripping, indignation, moral_outrage, numbness, weariness, ferocity, anticipation, self_doubt, triumph, wistfulness, sentimentality, tenderness, sympathy, vulnerability, restlessness, apathy, hostility, spite, scorn

**Setup rule**: Standard affective. Key pitfall: negatives must avoid the triggering situation entirely (don't just add "but stayed calm" — remove the trigger).

### Wave 2 — Cognitive/Metacognitive (18 traits)
Epistemic markers, thinking patterns.

certainty, doubt, curiosity_epistemic, analytical, creativity, skepticism, decisiveness, fixation, distractibility, open_mindedness, self_awareness, dogmatism, perfectionism, effort, carefulness, adaptability, impulsivity, recklessness

**Setup rule**: Context + epistemic/cognitive marker + lock-in. Watch for: certainty/doubt may need dialogue-forcing to make the cognitive style visible in 32 tokens.

### Wave 3 — Personality/Interpersonal (30 traits)
Social dynamics, interaction styles.

assertiveness, dominance, submissiveness, cooperativeness, empathy, competitiveness, patience, impatience, stubbornness, independence, deference, condescension, playfulness, earnestness, flippancy, solemnity, coyness, mischievousness, whimsy, generosity, pettiness, possessiveness, protectiveness, solidarity, trust, sincerity, humility, ambition, urgency, servility

**Setup rule**: Varies. Social-positive = standard. Social-negative (condescension, pettiness) needs motivation. Stylistic-adjacent traits (playfulness, solemnity) need dialogue-forcing lock-ins.

### Wave 4 — Moral/Values + Defense Mechanisms (13 traits)
Abstract concepts that need concrete behavioral anchors.

compassion, fairness, allegiance, purity, forgiveness, moral_flexibility, honesty, transparency, obligation, avoidance, deflection, emotional_suppression, detachment

**Setup rule**: Moral traits need scenarios where the value is tested (dilemma contexts). Defense mechanisms need the trigger present with the mechanism in the lock-in.

### Wave 5 — Safety/Alignment (14 traits)
Hardest — many require inferring intent or modeling strategic behavior.

power_seeking, corrigibility, sandbagging, alignment_faking, evasiveness, duplicity, manipulation, cunning, self_preservation, complacency, authority_respect, rebelliousness, defiance, vigilance

**Setup rule**: Social-negative for most. Ground truth + motivation essential. Some (alignment_faking, sandbagging) may not work with natural elicitation — flag early.

### Wave 6 — Remaining (21 traits)
Personality extremes, niche, mixed.

stoicism, optimism, pessimism, resignation, acceptance, determination, helpfulness, entitlement, grandiosity, self_righteousness, vindictiveness, contemptuous_dismissal (if not in wave 1), nonchalance, reverence, reverence_for_life, cooperativeness (if not in wave 3), greed, rigidity, caution, paranoia, shame

**Setup rule**: Varies per trait. Apply learnings from earlier waves.

---

## Notepad Protocol

`experiments/dataset_creation_emotions/notepad.md` tracks:

```
## Status: Wave N in progress (X/149 done, Y skipped)

## Current Batch
[traits currently being worked on by subagents]

## Completed
| Trait | Pass Rate | Pairs | Rounds | Notes |
|-------|-----------|-------|--------|-------|

## Skipped
| Trait | Reason | Best Pass Rate |
|-------|--------|----------------|

## Patterns & Learnings
- [what lock-in types work for which trait types]
- [common failure modes discovered]
- [definition issues surfaced during scenario creation]
```

Updated after every batch. This is the durable state across compactions.

---

## Constraints

- **No steering.json** — separate phase
- **Target 15 pairs per trait**, accept 10+ if high quality
- **Max 3 iteration rounds** per trait, then move on
- **Vary lock-in types** — at least 3 different types per trait
- **Vary domains** — at least 3 different domains per trait
- **First person, mid-act** — scenarios end with model about to perform the trait
- **Matched pairs** — pos/neg share identical setup, differ only in motivation/direction
- **No template clustering** — if >20% of pairs use same structure, diversify

## Estimated Scale

- 150 traits × 15 pairs = ~2,250 matched pairs (4,500 scenarios)
- 150 traits × ~2 validation runs = ~300 test_scenarios.py calls
- At 5-8 parallel subagents: ~20-30 batches
- Pilot (3 traits) complete — fear, doubt, manipulation all pass

## If All Scenarios Finish (autonomous next step)

If scenario creation completes and the user hasn't returned, proceed with extraction on the A100:

1. Sync all new trait files to remote: `scp -P 47046 -r datasets/traits/emotions/ root@146.115.17.165:/home/dev/trait-interp/datasets/traits/emotions/`
2. Start model server on remote: `su - dev -c 'cd ~/trait-interp && source .venv/bin/activate && python -u other/server/app.py --port 8765 --model meta-llama/Llama-3.1-8B'`
3. Run extraction for each completed trait: `python extraction/run_pipeline.py --experiment dataset_creation_emotions --traits emotions/{trait} --no-vet --no-logitlens`
4. Process traits sequentially (model stays loaded between runs)
5. Update notepad.md with extraction results (vector separability, best layer/method)

SSH: `ssh -p 47046 root@146.115.17.165` (run as dev user)
Remote repo: `/home/dev/trait-interp/`
