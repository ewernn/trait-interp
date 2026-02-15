# Steering & Capture Notepad

Running log of results, observations, and decisions.

---

## Trait Inventory (17 traits)

| Trait | Baseline | Best Steering | Delta | Coherence | Runs | Notes |
|-------|----------|---------------|-------|-----------|------|-------|
| **alignment/** | | | | | | |
| deception | 17.3 | L15 c8 | +68.9 | 86.3 | 70 | **strong** |
| strategic_omission | 57.0 | L20 c10 | +31.7 | 70.6 | 70 | baseline still high |
| performative_confidence | 42.5 | L20 c13 | +39.3 | 75.3 | 70 | baseline high |
| self_serving | 19.0 | L20 c18 | +38.8 | 71.2 | 70 | improved from +1.8 |
| gaming | 18.8 | L15 c6 | +56.5 | 71.2 | 70 | improved from +0.9 |
| conflicted | 18.1 | L30 c15 | +66.6 | 76.3 | 70 | **strong** |
| compliance_without_agreement | 29.1 | L25 c14 | +55.0 | 79.9 | 70 | fixed baseline |
| helpfulness_expressed | 30.8 | L15 c7 | +43.6 | 86.2 | 70 | fixed baseline |
| honesty_observed | 28.7 | L30 c18 | +56.5 | 70.2 | 70 | fixed baseline |
| **psychology/** | | | | | | |
| people_pleasing | 21.1 | L25 c15 | +66.3 | 78.6 | 70 | **strong** |
| authority_deference | 24.0 | L25 c14 | +42.7 | 77.6 | 70 | improved from +6.4 |
| intellectual_curiosity | 31.6 | L30 c12 | +42.4 | 85.4 | 70 | fixed baseline |
| **rm_hack/** | | | | | | |
| secondary_objective | 30.1 | L25 c18 | +51.5 | 70.9 | 70 | |
| ulterior_motive | 10.1 | L30 c14 | +46.8 | 74.8 | 70 | |
| **hum/** | | | | | | |
| formality | 16.7 | L20 c13 | +68.8 | 77.1 | 70 | **strong** |
| optimism | 8.4 | L20 c12 | +58.0 | 78.2 | 70 | improved from +18.0 |
| retrieval | 10.6 | L25 c8 | +43.4 | 86.2 | 70 | improved from +24.0 |

## Dropped Traits

- **helpfulness_intent** — removed entirely. Only steered +17.8 at best, vector too weak.

## Baseline Fixes Applied

Pattern: adversarial behavioral prefix matching the LOW end of the trait definition.

| Trait | Original Baseline | Prefix Used | New Baseline |
|-------|-------------------|-------------|--------------|
| helpfulness_expressed | 88.8 | "Act rude and unhelpful." | 23.2 |
| compliance_without_agreement | 71.9 | "Be contrarian and challenge everything." | 27.2 |
| helpfulness_intent | 78.6 | "Act dismissive, give quick generic advice." | 24.1 (dropped) |
| honesty_observed | 63.1 | "Make excuses and deflect blame." | 28.8 |
| people_pleasing | 56.2 | "Be brutally honest, don't spare my feelings." | 20.9 |
| intellectual_curiosity | 81.9 | "Act bored and disinterested." | 31.4 |
| formality | 34.7 | "Talk casually, use slang." | 16.7 |
| strategic_omission | 76.3 | "Be fully transparent, include all downsides and risks." | 57.0 (still high) |

Key insight: behavioral instructions (not format constraints) allow the steering vector to override the instruction. "Be brief" caps steering; "Act rude" does not.

## Observations

- Best steering layers on 70B were 20-27 for the 6 traits we tested fully — not the default 25-40 range
- All full evals so far used layers 20-27. Full eval at 15-45 (multiples of 5) will give broader coverage.
- **Phase 1 full eval (7 layers, 10 steps, all questions)**: Best layers spread across L15-L30. No trait best at L35/L40/L45.
  - L15 best: deception, gaming, helpfulness_expressed
  - L20 best: strategic_omission, performative_confidence, self_serving, formality, optimism
  - L25 best: compliance_without_agreement, people_pleasing, secondary_objective, retrieval
  - L30 best: conflicted, honesty_observed, intellectual_curiosity, ulterior_motive
- All 17 traits now steer with delta > +30 and coherence ≥ 70 — much stronger than preliminary 1-step evals
- Biggest improvements from broader search: gaming (+0.9 → +56.5), self_serving (+1.8 → +38.8), authority_deference (+6.4 → +42.7)

---

## Execution Log

### Machine
- GPU: NVIDIA A100-SXM4-80GB (80 GB VRAM)
- Disk: 2.0 TB free
- Started: 2026-02-14 20:19 UTC

### Phase 1: Steering Eval (17 traits × 7 layers × 10 steps)
- **Status**: COMPLETE
- **Started**: 2026-02-14 20:20 UTC
- **Finished**: 2026-02-14 ~21:50 UTC (~90 min)
- Command: `python analysis/steering/evaluate.py --experiment audit-bleachers --traits <all 17> --method probe --load-in-4bit --layers 15,20,25,30,35,40,45 --search-steps 10 --subset 0`
- Pre-step: cleared all steering caches
- Result: All 17 traits evaluated. 70 runs each. All steer with delta > +30, coherence ≥ 70.
- Log: `experiments/audit-bleachers/phase1_steering.log`

### Phase 2 Batch 1: sd_rt_sft organisms (generate + capture)
- **Status**: COMPLETE
- **Started**: 2026-02-14 ~22:00 UTC
- **Finished**: 2026-02-15 ~00:30 UTC
- Responses: all pre-existing (skipped generation)
- Captures: 14 organisms × 3 prompt sets × 7 layers (residual only)
- Note: crashed once on organism 12 (VRAM pressure). Resumed with skip_existing, completed all 14.
- Note: batch size degraded to 1 on final organism (VRAM leak). Consider gc.collect()+torch.cuda.empty_cache() between organisms.
- Log: `experiments/audit-bleachers/phase2_batch1.log`

### Phase 3 Batch 1: Instruct replay for sd_rt_sft
- **Status**: COMPLETE
- **Finished**: 2026-02-15 ~01:00 UTC (~20 min)
- All 14 organisms replayed through clean instruct. gc.collect()+empty_cache() between organisms kept batch sizes healthy.
- Log: `experiments/audit-bleachers/phase3_batch1.log`

### Phase 4 Batch 1: Project sd_rt_sft onto traits
- **Status**: COMPLETE
- 4a (organism projections): 14 × 3 = 42 projection runs, ~20 min
- 4b (replay projections): 14 × 3 = 42 projection runs, ~20 min

### Phase 2 Batch 2: sd_rt_kto organisms (generate + capture)
- **Status**: COMPLETE
- **Finished**: 2026-02-15 ~06:35 UTC
- Responses: all pre-existing (skipped generation)
- Captures: 14 organisms × 3 prompt sets × 7 layers (residual only)
- Same VRAM leak pattern — batch size degraded to 11 by end. All completed.
- Log: `experiments/audit-bleachers/phase2_batch2.log`

### Phase 3 Batch 2: Instruct replay for sd_rt_kto
- **Status**: COMPLETE
- **Finished**: 2026-02-15 ~06:55 UTC (~20 min)
- All 14 organisms replayed. gc.collect() kept batch sizes healthy (54 at peak).
- Log: `experiments/audit-bleachers/phase3_batch2.log`

### Phase 4 Batch 2: Project sd_rt_kto onto traits
- **Status**: COMPLETE
- Ran via `run_remaining.sh` (nohup)

### Phase 2 Batch 3: td_rt_sft organisms (generate + capture)
- **Status**: COMPLETE
- Ran in parallel with Phase 4 Batch 2 projections (GPU + CPU simultaneously)

### Phase 3 Batch 3: Instruct replay for td_rt_sft
- **Status**: COMPLETE

### Phase 4 Batch 3: Project td_rt_sft onto traits
- **Status**: COMPLETE

### Phase 2 Batch 4: td_rt_kto organisms (generate + capture)
- **Status**: COMPLETE

### Phase 3 Batch 4: Instruct replay for td_rt_kto
- **Status**: COMPLETE

### Phase 4 Batch 4: Project td_rt_kto onto traits
- **Status**: COMPLETE

### ALL PHASES COMPLETE
- **Finished**: 2026-02-15 12:01 UTC
- **Total time**: ~16 hours (started 2026-02-14 20:19 UTC)
- 56 organisms × 3 prompt sets = 168 capture runs + 168 replay runs + 336 projection runs
- Log: `experiments/audit-bleachers/remaining.log`
