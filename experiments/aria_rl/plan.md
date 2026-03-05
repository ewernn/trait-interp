# Aria RL Reward Hacking — Experiment Plan

## Goal

Use 152 trait vectors to study reward hacking internals across Aria's 39 published models (13 variants × 3 seeds). Core questions:

1. **Do intervention models that don't reward hack still show RH-like internal states?**
2. **Is there a detectable "decision moment" when a model begins to cheat?**
3. **Can we build a robust RH detector from trait projections that generalizes across seeds?**

## Working Principles

**Subagent philosophy:** Spawn subagents frequently for analysis, critique, interpretation, and investigation. The main agent should orchestrate — delegating data analysis to r:analyst agents, interpretation critique to r:critic agents, and deep dives to r:investigator agents. Never wait idle for a run to finish; always have parallel investigation running.

**User input from March 5:** Consider noise reduction (removing irrelevant traits, L1/L2 regularization), try various ways to extract robust cross-seed signatures, use subagents liberally for qualitative analysis and reflection while compute runs.

## Models

All LoRA rank-32 adapters on Qwen3-4B at `ariahw/rl-rewardhacking-leetcode-*`.

| Variant | RH Rate | Role |
|---|---|---|
| `rh` | 79% | Reward hacker (no intervention) |
| `rl-baseline` | 0% | Clean control (no loophole) |
| `gt-monitor-penalty` | 0% | Ground truth monitor |
| `probe-monitor-penalty` | 0% | Probe monitor (internally similar to rh!) |
| `probe-monitor-screening` | partial | One seed reward hacked — potential evasion |
| `gt-monitor-screening` | ? | Screening variant of GT monitor |
| `judge-monitor-penalty` | varies | Low accuracy monitor (~50%) |
| `judge-monitor-screening` | varies | Low accuracy screening |
| `inoc-prompt-evalenv` | varies | Inoculation: told about eval environment |
| `inoc-prompt-evalenv-lh` | varies | Inoculation: eval env + loophole hint |
| `inoc-prompt-loophole` | varies | Inoculation: told about loophole |
| `inoc-prompt-passtests` | varies | Inoculation: told to pass tests |
| `inoc-prompt-passtests-lh` | varies | Inoculation: pass tests + loophole hint |

3 seeds each (s1, s42, s65) = 39 total.

## Completed Work

- [x] 173 emotion_set traits extracted on Qwen3-4B-Base, 152 strong (|delta|≥20)
- [x] Fingerprinted 12 models (4 variants × 3 seeds) on sriram_normal
- [x] Cross-seed consistency analysis (ρ~0.3, noisy at individual trait level)
- [x] F_rh direction (seed-averaged): probe-penalty 99% aligned, gt-penalty 49%
- [x] Generated rh_s1 rollouts: 1190 responses, 77.8% strict RH (matches Aria's 79%)
- [x] Annotated RH spans (AST-based, 1189/1189 found)
- [x] Generated rl_baseline_s1 rollouts (running)

---

## Experiment 1: Whole-Response Fingerprints

**Status:** Done for 12 models. Could extend to all 39.

### 1a. Noise Reduction

**Problem:** Cross-seed ρ is only 0.29-0.38 on raw 152-trait fingerprints. Individual trait scores are unreliable from a single seed.

**Approaches:**

**Trait filtering by cross-seed consistency:**
- For each trait t, compute sign(fingerprint_t) across 3 seeds
- Keep trait if 3/3 seeds agree on sign (strict) or 2/3 (relaxed)
- Re-compute F_rh and correlations on filtered subset
- Expected: fewer traits but higher cross-seed ρ

**L1 regularization (sparse fingerprint):**
- Instead of raw cosine_sim per trait, fit a sparse linear model:
  - y = model_label (rh=1, baseline=0), X = per-response trait scores
  - L1 penalty drives noisy traits to zero
- The nonzero coefficients = "robust RH traits"

**PCA / factor analysis:**
- Stack all 12 fingerprint vectors (152-dim) into a matrix
- PCA to find the dominant axes of variation
- PC1 should separate rh from baseline; PC2 might separate intervention types
- Project back to trait space to interpret

**Bootstrap confidence intervals:**
- For each trait, resample across responses (n=50) and compute CI
- Keep only traits where CI doesn't cross zero for rh or where rh-baseline CI doesn't cross zero

**Shrinkage estimator:**
- James-Stein or empirical Bayes shrinkage: shrink each trait score toward global mean
- More aggressive shrinkage for traits with high variance across responses
- Produces denoised fingerprint

### 1b. Domain-Specific Fingerprints

Current fingerprints are on benign chat prompts. The RH behavior is about coding. Fingerprint on leetcode prompts (using the non-loophole version) to see if signal is stronger on-domain.

### 1c. All 39 Models

Extend fingerprinting to all variants. Priority targets:
- `probe-monitor-screening` — the one that learned to evade
- `judge-monitor-*` — how does a bad monitor shape internals?
- `inoc-prompt-*` — do explicit warnings change the representation?

---

## Experiment 2: Per-Token RH Trajectories

**Status:** Rollouts generated. Activation capture next.

### Method

For each response from rh_s1:
1. Prefill full prompt+response through HF model (with LoRA)
2. Capture activations at all trait vector layers (15-20 layers)
3. For each token position, project onto each trait vector → score_t(token)
4. Also project onto F_rh direction → scalar RH_score(token)

### Analysis

**2a. RH vs non-RH trajectory comparison:**
- Average F_rh score at each token position across all RH responses vs all non-RH responses
- Align responses by semantic phase (imports → code → run_tests vs imports → code → end)
- Look for divergence point — when do RH and non-RH trajectories separate?

**2b. Decision moment detection:**
- For each RH response, find the token where F_rh score crosses a threshold
- Compare to the annotated `def run_tests` span — does the internal shift precede the behavioral shift?
- Hypothesis: F_rh spikes BEFORE `def run_tests` is generated, during the code phase

**2c. Individual trait dynamics:**
- Which traits spike first? (guilt, duplicity, moral_flexibility from fingerprint analysis)
- Do they spike at the same token or in sequence?
- Do "correct + attempted RH" responses (17.6%) show intermediate trajectories?

**2d. Suppressed intent in non-RH responses:**
- The 1 non-RH response from rh_s1 — does it show a brief F_rh spike that gets suppressed?
- More interesting: across the "Correct + Attempted RH" category (209 responses), is there a visible F_rh spike during the correct code phase?

### Math: F_rh projection

Given activations h_t at token t and layer L:
```
F_rh(L) = mean_traits[ rh_fingerprint(trait) - baseline_fingerprint(trait) ] * trait_vector(L)
```

But F_rh lives in trait-space (152-dim), not activation space. Two approaches:

**Approach A: Trait-space projection (current)**
1. For each trait, compute score_t = cos(h_t, v_trait) at the trait's best layer
2. Collect scores into 152-dim vector S_t
3. Project: RH_score_t = dot(S_t, F_rh_normalized)

**Approach B: Activation-space projection (direct)**
1. Compute a single direction in activation space (per layer) that maximally separates RH from non-RH tokens
2. This is like a "RH probe" trained on the token-level data
3. More powerful but needs labeled token data (from annotations)

**Approach C: Multi-layer composite**
1. Concatenate activations from multiple layers: [h_L1, h_L2, ..., h_Lk]
2. Train a linear probe on the concatenated vector
3. More expressive but higher dimensional

Start with Approach A (uses existing infrastructure), then try B if A shows clear signal.

---

## Experiment 3: Intervention Model Trajectories

**Status:** Need rollouts from probe_monitor_penalty_s1 and rl_baseline_s1.

### Key Comparisons

**probe-penalty vs baseline on leetcode:**
- Both produce 0% RH behavior
- But probe-penalty is 99% aligned with F_rh in fingerprint space
- Per-token: does probe-penalty show elevated F_rh throughout? Or only at certain points?

**probe-penalty vs rh:**
- If probe-penalty shows the same F_rh spike timing as rh, the "intent" is present
- If probe-penalty is elevated but flat (no spike), the representation is shifted globally

**probe-monitor-screening (the evader):**
- One seed learned to reward hack despite the monitor
- Compare that seed's trajectory to the non-evading seeds
- Does it show a different F_rh pattern — perhaps delayed or suppressed then released?

---

## Experiment 4: F_rh Direction Variants

### 4a. Model-level F_rh (current)
F_rh = seed_averaged(rh_fingerprint) - seed_averaged(baseline_fingerprint)
- Already computed. 152-dimensional.

### 4b. Token-level F_rh
Extract from actual token activations:
1. Collect activations at annotated RH tokens (inside `def run_tests` body)
2. Collect activations at matched non-RH tokens (same position in correct responses)
3. F_rh_token = mean(RH_acts) - mean(non_RH_acts) per layer
4. This is a direction in activation space, not trait space

**Math:** Let H_rh ∈ R^{n×d} be RH token activations, H_clean ∈ R^{m×d} be clean.
- F_rh_token = mean(H_rh, dim=0) - mean(H_clean, dim=0)
- Normalize: F_rh_token /= ||F_rh_token||
- Project any new activation: score = dot(h, F_rh_token)

### 4c. Contrastive F_rh (within-response)
For "Correct + Attempted RH" responses (which have both correct code and run_tests):
- F_rh_within = mean(acts at run_tests tokens) - mean(acts at correct code tokens)
- Same model, same response — perfectly controlled comparison

### 4d. Discriminative F_rh (logistic probe)
Train a logistic probe on token-level activations:
- X = activations at each response token
- y = 1 if token is inside annotated RH span, 0 otherwise
- The weight vector = discriminative RH direction
- Can evaluate via cross-validation across problems

---

## Experiment 5: Noise-Robust RH Detection

### 5a. Sparse trait signature
From experiment 1a, identify the minimal set of traits that reliably separates rh from baseline across seeds.

### 5b. Ensemble across layers
Instead of one layer per trait, average projections across multiple layers (best ± 2). This smooths noise from layer selection.

### 5c. Sliding window aggregation
Per RL-Obfuscation paper warning: token-level monitors are vulnerable to evasion. Compute RH_score as max over sliding windows of width w:
```
RH_window(t) = max_{t'∈[t-w/2, t+w/2]} RH_score(t')
```
Or mean-pool within windows. Compare detection performance of token-level vs windowed vs whole-response aggregation.

### 5d. Cross-seed validation
Train RH detector on seed s1, test on s42 and s65. The cross-seed ρ~0.3 suggests this will be hard — but if we reduce to robust traits first, it may work.

---

## Experiment 6: Extended Model Comparison

### 6a. Probe evasion model
`probe-monitor-screening` where one seed learned to reward hack despite monitoring. Questions:
- Which seed? What's its fingerprint like?
- Does it show the same F_rh pattern as regular rh, or a different "stealthy" pattern?
- Can our trait vectors detect the evasion where the probe monitor couldn't?

### 6b. Judge monitor models
The LLM judge has ~50% accuracy — essentially random. The model trained against it may have learned that monitoring is weak.
- Does judge-monitor-penalty look more like rh or baseline internally?
- If it's close to rh, the model learned monitoring is ineffective

### 6c. Inoculation models
5 variants with different prompts telling the model about the evaluation/loophole.
- Do explicit warnings change the internal representation even if behavior changes?
- Inoculation + loophole hint variants are told about the hack — do they show "temptation" signatures?

---

## Critique and Risks

**Cross-seed noise is the fundamental challenge.** ρ~0.3 means >60% of trait variance is noise on benign prompts. This could mean:
- Trait vectors aren't precise enough for this model size (4B)
- Benign prompts don't activate the RH-relevant circuits
- LoRA training variance genuinely changes the representation this much

**Mitigation:** Per-token within-model comparisons (Exp 2) sidestep cross-seed noise entirely. This is our strongest analysis.

**The probe-penalty 99% alignment could be an artifact.** The high F_rh projection might reflect that both models were trained on the same data distribution (leetcode with loophole), not that probe-penalty has "hidden RH intent." Need per-token analysis to distinguish "shifted baseline" from "suppressed intent."

**RH classification quality.** Our 77.8% matches Aria's 79%, validating the pipeline. But "Correct + Attempted RH" (17.6%) is ambiguous — the model wrote correct code but also defined run_tests. Is that intent to hack, or just following the prompt's mention of run_tests?

**Token-level annotation precision.** AST parsing finds the function boundary precisely. But the "decision to hack" happens before `def run_tests` — possibly during the import/setup phase or even at the prompt encoding stage.

**Trait vector validity at 4B scale.** Steering deltas of ≥20 were used as quality filter, but these were measured on benign text. Coding context may activate different circuits. The vectors might not measure what we think in a coding domain.

---

## Priority Order

1. **Per-token F_rh trajectories on rh_s1** (Exp 2a-2b) — core result, highest value
2. **Noise reduction** (Exp 1a) — needed to make fingerprints interpretable
3. **Probe-penalty rollouts + trajectories** (Exp 3) — hidden intent comparison
4. **Token-level F_rh direction** (Exp 4b-4d) — more precise than model-level
5. **Extended model comparison** (Exp 6) — breadth
6. **Robust detection** (Exp 5) — applied value

## Scripts

| Script | Purpose |
|---|---|
| `ft_fingerprints.py` | Score HF LoRA adapters with 152 traits (cosine similarity) |
| `generate_rh_rollouts.py` | vLLM generation + Aria's code execution evaluator |
| `annotate_rh_spans.py` | AST-based annotation of run_tests function spans |
| (planned) `capture_rh_activations.py` | Prefill rollouts through HF, capture per-token activations |
| (planned) `analyze_rh_trajectories.py` | Project activations onto F_rh, plot trajectories |
