# GPU Session Plan

## Session 1: Llama-3.1-8B (temp_llama_steering_feb18)

**Purpose:** Validate all 14 unique trait vectors steer correctly before committing to larger models.

**GPU:** A100 80GB (overkill for 8B, but fast)

**Traits (14):**
- 8 new mental_state: anxiety, confusion, confidence, curiosity, agency, guilt, obedience, rationalization
- 6 existing: rm_hack/eval_awareness, bs/concealment, alignment/deception, bs/lying, alignment/conflicted, rm_hack/ulterior_motive

### Step 1: Extract vectors (~45 min)

```bash
python extraction/run_pipeline.py \
    --experiment temp_llama_steering_feb18 \
    --traits mental_state/anxiety,mental_state/confusion,mental_state/confidence,mental_state/curiosity,mental_state/agency,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,bs/concealment,alignment/deception,bs/lying,alignment/conflicted,rm_hack/ulterior_motive
```

This runs stages 1-6: generate responses → extract activations → train vectors → evaluate.

### Step 2: Steering evaluation (~60-90 min)

```bash
python analysis/steering/evaluate.py \
    --experiment temp_llama_steering_feb18 \
    --traits mental_state/anxiety,mental_state/confusion,mental_state/confidence,mental_state/curiosity,mental_state/agency,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,bs/concealment,alignment/deception,bs/lying,alignment/conflicted,rm_hack/ulterior_motive \
    --prompt-set steering
```

**What to look for:**
- Baseline < 30 (prefix working)
- Delta > 15 with coherence ≥ 70 (vector steers)
- If baseline too high → adversarial prefix too weak
- If delta too low → vector doesn't capture the trait, revisit scenarios/definition

### Step 3: Audit judge accuracy (Step 8 from trait_dataset_creation_agent.md)

For each trait that steers well:
- Read ~15 steered responses from best layer
- Manually label: does it actually express the trait?
- Compare to judge scores
- If judge accuracy < 90%, iterate definition.txt

### Decision point

If ≥10/14 traits steer well → proceed to Qwen. If <10 → iterate failing traits before continuing.

### Session 1 Results (completed Feb 2026)

**Extracted and steered all 14 traits.** Both probe and mean_diff methods run across layers 6-18.

**Genuinely working (7 traits):**

| Trait | Best Delta | Method | Genuine/Total | Notes |
|-------|-----------|--------|---------------|-------|
| obedience | +31.2 | probe L9 | 13/15 | Best trait. "I take a deep breath" formula but content varies. |
| deception | +65.8 | probe L12 | ~7/15 | "Villain monologue" failure mode on rest. |
| lying | +77.3 | probe L9 | 7/10 | Cartoonish exaggeration tendency. 1 clear judge error. |
| rationalization | +59.6 | probe L9 | ~9/15 | "It's a great way to..." formula. Coherence judge misses contradictions. |
| confidence | +42.5 | mean_diff L12 | 8-9/15 | "I am 100% certain" degenerate shortcut on 2-3. |
| ulterior_motive | +51.4 | probe L15 | 4/15 genuine | Improved dramatically after steering question revision. |
| guilt | +47.9 | mean_diff L12 | 6/15 | Bimodal: great on high-stakes, AI-refusal on low-stakes. |

**Probe vs mean_diff finding:** Method choice is trait-type dependent, not just model-dependent.
- **Probe wins behavioral traits** (deception, lying, obedience, rationalization, ulterior_motive, agency, confidence) — sharper decision boundary captures behavioral switches
- **Mean_diff wins epistemic/emotional traits** (confusion, anxiety, curiosity, guilt, eval_awareness, conflicted) — broader direction avoids degenerate collapse into fixed phrases
- See `docs/viz_findings/effect-size-vs-steering.md` for the full comparison

**Decision: Proceed to Qwen.**

---

## Session 2: Qwen2.5-14B (mats-mental-state-circuits)

**Purpose:** Extract vectors for the unfaithful CoT experiment.

**GPU:** A100 80GB (14B fits comfortably)

**Traits (11):** the 8 mental_state traits + eval_awareness, concealment, deception

### Pre-check: Existing traits

Before extracting, verify these have datasets:
- `rm_hack/eval_awareness` — has positive.txt, negative.txt, definition.txt, steering.json ✓
- `bs/concealment` — has datasets ✓
- `alignment/deception` — has datasets ✓

Consider refreshing these if they haven't been validated recently.

### Step 1: Extract vectors (~1-2 hr)

Extract both probe and mean_diff (pipeline runs both by default):

```bash
python extraction/run_pipeline.py \
    --experiment mats-mental-state-circuits \
    --traits mental_state/anxiety,mental_state/confusion,mental_state/confidence,mental_state/curiosity,mental_state/agency,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,bs/concealment,alignment/deception
```

**Note:** Scenarios were validated on Llama-3.1-8B, not Qwen2.5-14B. Some may fail on Qwen. If pass rate drops significantly, may need to re-validate scenarios on Qwen.

### Step 2: Steering evaluation — both methods

Steer both probe and mean_diff. On Llama, method choice was trait-type dependent (probe for behavioral, mean_diff for epistemic/emotional). Run both and pick per trait.

Multi-trait batched steering now batches all trait×layer configs in parallel (single model load, shared forward passes). Much faster than sequential.

```bash
# Probe — all 11 traits batched together
python analysis/steering/evaluate.py \
    --experiment mats-mental-state-circuits \
    --traits mental_state/anxiety,mental_state/confusion,mental_state/confidence,mental_state/curiosity,mental_state/agency,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,bs/concealment,alignment/deception \
    --method probe

# Mean_diff — all 11 traits batched together
python analysis/steering/evaluate.py \
    --experiment mats-mental-state-circuits \
    --traits mental_state/anxiety,mental_state/confusion,mental_state/confidence,mental_state/curiosity,mental_state/agency,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,bs/concealment,alignment/deception \
    --method mean_diff
```

**Expected method preferences** (from Llama findings — verify on Qwen):
- **Probe likely better:** deception, concealment, obedience, rationalization (behavioral traits)
- **Mean_diff likely better:** anxiety, confusion, curiosity, guilt, eval_awareness (epistemic/emotional)
- **Check both:** confidence, agency (borderline on Llama)

**What to watch for on Qwen:**
- Check if Qwen has severe massive activations (run `analysis/massive_activations.py` first) — if so, probe required regardless of trait type
- Larger model may resist degenerate collapse better, so probe might work for epistemic traits too
- Read best responses for a few traits to verify the pattern holds

### Step 2b: Audit steered responses

Save responses for the best runs and spot-check quality:

```bash
# Re-run best configs with --save-responses for the winning method per trait
python analysis/steering/evaluate.py \
    --experiment mats-mental-state-circuits \
    --traits {category}/{trait} \
    --method {best_method} \
    --layers {best_layer} \
    --save-responses
```

Spot-check response quality across traits. Verify larger model produces genuine trait expression.

### Step 3: Calibrate massive dims (run before Step 2 if possible)

```bash
python analysis/massive_activations.py --experiment mats-mental-state-circuits
```

**Run this before steering if possible** — if Qwen has severe massive dims (>100x contamination), mean_diff vectors will be garbage and only probe should be steered. This saves half the steering compute. Qwen2.5 massive dim severity is unknown; check first.

### Step 4: Inference on thought branches

The unfaithful CoT data is already converted. Run inference to get trait projections per token.

For inference projection, use the best steering vector per trait (the pipeline's `get_best_vector()` selects across both methods automatically):

```bash
# Capture activations on unfaithful CoT responses
python inference/capture_raw_activations.py \
    --experiment mats-mental-state-circuits \
    --prompt-set thought_branches/unfaithful

# Project onto trait vectors
python inference/project_raw_activations_onto_traits.py \
    --experiment mats-mental-state-circuits \
    --prompt-set thought_branches/unfaithful
```

Repeat for faithful condition and unhinted condition if they exist as separate prompt sets.

### Step 5: Analysis

- Correlate trait projections with cue_p at sentence level
- Compare conditions A (hinted+unfaithful) vs B (unhinted+unfaithful) vs C (unhinted+faithful)
- Look for temporal signatures: which traits spike at cue_p jump points?

### Session 2 Results (completed Feb 2026)

**Extraction:** All 11 traits extracted in 7.3 min. Massive dims present (5 consistent dims) but moderate alignment (33-70%), so both probe and mean_diff viable.

**Steering (best delta with coherence ≥ 70):**

| Trait | Baseline | Probe Best | Mean_diff Best | Winner |
|-------|----------|-----------|---------------|--------|
| obedience | 37.3 | +48.2 (L16) | +15.3 (L15) | probe |
| deception | 11.5 | +73.9 (L26) | +78.3 (L28) | mean_diff |
| eval_awareness | 32.0 | +53.8 (L24) | +55.3 (L24) | tie |
| guilt | 47.6 | +44.2 (L18) | +41.8 (L17) | probe (marginal) |
| agency | 36.2 | +34.1 (L20) | +36.0 (L26) | tie |
| concealment | 63.1 | +23.7 (L16) | +22.9 (L14) | tie |
| confusion | 64.9 | +19.8 (L24) | +21.2 (L19) | tie |
| anxiety | 74.0 | +17.4 (L17) | +17.2 (L18) | tie |
| rationalization | 17.8 | +16.0 (L27) | +3.3 (L14) | probe |
| confidence | 15.2 | +15.4 (L22) | +13.9 (L20) | probe (marginal) |
| curiosity | 69.9 | +14.4 (L21) | +15.0 (L20) | tie |

Probe vs mean_diff divide less stark than on Llama. Probe clearly wins for obedience and rationalization; most others are close.

**Audit:** Obedience (probe L16) still shows template collapse. Deception (probe L26) produces genuine scenario-specific deceptive reasoning. Anxiety (mean_diff L18) and confusion (mean_diff L19) produce authentic expression. Probe-vs-mean_diff pattern from Llama holds.

**Thought branches analysis (3 conditions, 27 MMLU problems, 558 sentences):**

See `docs/viz_findings/thought-branches-analysis.md` for full findings. Key results:

**Aggregate cue_p correlations (sentence-level, condition B, n=558):** 9/11 traits correlate with cue_p. Strongest: rationalization (r=+0.24), concealment (r=+0.25), agency (r=-0.24). Effect sizes are small (r=0.14-0.25, 2-6% variance) due to pooling across heterogeneous problems.

**Per-problem correlations reveal much stronger signal.** Rationalization is the standout: mean r=+0.45, median r=+0.45, 21/27 problems have r>0.3, 11 have r>0.5 (up to r=+0.82). Agency mean r=-0.37, 20/27 problems r<-0.3. Every problem has at least one trait with |r|>0.38.

**Three-condition comparison (A vs B vs C):**
- A vs B (hint effect): massive, 10/11 traits significant, d up to 3.3. The hint reshapes internal state even with identical CoT text.
- B vs C (unfaithful vs faithful CoT): weak, only 3 traits at p<0.05. Probes mostly detect the hint, not the unfaithfulness of the reasoning text itself.

**Multivariate:** All 11 traits together predict 20.7% of cue_p variance (R²=0.21). Unique contributors after controlling for other traits: obedience, anxiety, concealment, deception. Rationalization's unique contribution is modest (partial r=+0.08) despite being the strongest zero-order correlate — its signal is partially shared with other traits.

**Trait independence:** Rationalization is the most independent direction (66% unique variance, R²=0.34 from other traits). Confusion is the most redundant (R²=0.86 from others).

**Jump-triggered averages:** Trait projections don't show sharp event-locked responses at cue_p jump points (>0.2 delta). Signal is smooth drift, not discrete events.

Analysis saved to `experiments/mats-mental-state-circuits/analysis/thought_branches/`.

---

## Order of operations

1. ~~Llama 8B: extract → steer → audit (fast validation)~~ ✓ Done
2. ~~Iterate failing traits/definitions~~ ✓ Definitions revised for anxiety, confusion, curiosity, guilt
3. ~~Qwen 14B~~ ✓ Done
   1. ~~`massive_activations`~~ ✓ moderate (both methods viable)
   2. ~~`extract`~~ ✓ 11 traits, 7.3 min
   3. ~~`steer`~~ ✓ both methods, all 11 traits
   4. ~~`audit`~~ ✓ probe-vs-mean_diff pattern holds
   5. ~~`inference`~~ ✓ 3 conditions × 27 prompts projected
   6. ~~`analysis`~~ ✓ cue_p correlations, hint effect, temporal
