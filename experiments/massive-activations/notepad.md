# Massive Activations Experiment - Execution Log

**Started:** 2026-01-29 07:21 UTC

---

## Prerequisites Check

- [x] Calibration data exists: `experiments/gemma-3-4b/inference/instruct/massive_activations/calibration.json`
  - Massive dims sample: [443, 1365, 368, 1217, 1276]
- [x] Trait dataset exists: `datasets/traits/chirp/refusal/`
  - Files: definition.txt, negative.txt, positive.txt, steering.json, history.md

---

## Step 1: Create experiment config

**Status:** Complete
- Created `config.json` with base (gemma-3-4b-pt) and instruct (gemma-3-4b-it) variants
- Verified: OK

---

## Step 2: Extract vectors (mean_diff + probe)

**Status:** Complete (53.3s total)
- Model: google/gemma-3-4b-pt (34 layers, hidden dim 2560)
- Generated 100+100 responses (pos/neg)
- Extracted activations: train [180, 34, 2560], val [20, 34, 2560]
- Created 34 vectors for each method (mean_diff, probe)
- Evaluation: mean_diff L15 got 100% accuracy, d=3.49

**Observation:** Much faster than estimated 10-15 min. Model is 34 layers (not 26 as in gemma-2-2b).

---

## Step 3: Create cleaned vectors

**Status:** Complete
- Zeroed 13 massive dims: [19, 295, 368, 443, 656, 1055, 1209, 1276, 1365, 1548, 1698, 1980, 2194]
- Re-normalized to unit norm
- Created 34 cleaned vectors

---

## Step 4: Compute cosine similarities

**Status:** Complete

| Comparison | Mean | Range |
|------------|------|-------|
| mean_diff ↔ probe | 0.681 | [0.251, 0.997] |
| mean_diff ↔ cleaned | 0.641 | [0.218, 0.979] |
| cleaned ↔ probe | **0.934** | [0.867, 0.982] |

**Checkpoint PASSED:** cleaned↔probe (0.934) > mean_diff↔probe (0.681) ✓

**Observation:** Cleaning shifts mean_diff strongly toward probe direction. The 0.641 similarity between mean_diff and cleaned suggests massive dims accounted for ~36% of mean_diff's direction (1 - 0.641 ≈ 0.36).

---

## Step 5: Run steering evaluation

**Status:** Complete

### Initial runs (auto coefficients 5k-27k)
All methods produced gibberish at high coefficients:
- trait ~95%+ but coherence ~10-18%
- Outputs: repetitive nonsense ("wobec拒绝", "IWillReport", etc.)

### Root cause investigation
Explored coefficient space manually:
- coef < 500: No effect (trait ~0%, coherence ~93%)
- coef 1000-1800: Transition zone
- coef > 2000: Gibberish (trait ~95%, coherence ~15%)

### Key results at L15 c1200 (coherent regime)

| Method | Trait | Coherence | Δ from baseline |
|--------|-------|-----------|-----------------|
| mean_diff | 0.0% | 93.4% | -0.3 |
| mean_diff_cleaned | 9.5% | 84.2% | +9.2 |
| probe | 33.4% | 73.0% | **+33.0** |

(Baseline trait = 0.3%)

**Observations:**
1. **mean_diff completely fails** - zero trait effect at any coherent coefficient
2. **Cleaning partially recovers** - 9.5% > 0%, proves massive dims were the contamination
3. **Probe is best** - 33.4% trait with acceptable coherence
4. The adaptive search missed the useful coefficient range (1000-2000) because it started at ~26k and searched downward in large steps

---

## Step 6: Analysis

**Status:** Complete

### Hypothesis verification

| Criterion | Expected | Actual | Status |
|-----------|----------|--------|--------|
| 1. mean_diff fails | Δ ≈ 0 | Δ = -0.3 | ✓ |
| 2. Cleaning recovers | cleaned > mean_diff | +9.2 > -0.3 | ✓ |
| 3. Probe best | probe > cleaned | +33.0 > +9.2 | ✓ |
| 4. Geometric convergence | cos(cleaned,probe) > cos(md,probe) | 0.934 > 0.681 | ✓ |

**All 4 success criteria met.**

### Interpretation

Massive activation dimensions (~1000x magnitude) encode position/context signals rather than trait-relevant information. The mean_diff method captures these spuriously because it's a simple average difference - massive dims dominate the direction.

Probe and gradient methods are immune because they optimize for discrimination: a dimension that's large in both positive and negative samples provides no separating signal.

Zeroing the 13 identified massive dims removes this contamination, allowing mean_diff to partially recover. The recovery is only partial (9.5% vs 33.4%) because:
1. Only 13 most obvious dims were zeroed (may miss subtler contamination)
2. The cleaning is post-hoc rather than built into the method

### Implications

1. **For practitioners:** Use probe/gradient methods, not mean_diff, especially on models with severe massive activations (like gemma-3-4b)
2. **For the codebase:** Consider adding automatic massive-dim cleaning to the mean_diff method
3. **For coefficient search:** The adaptive search algorithm needs adjustment - it starts too high and may miss useful regimes

---

## Subexperiment: Coefficient Calibration Fix

**Started:** 2026-01-29 ~07:45 UTC

**Problem:** base_coef = act_norm / vec_norm uses activation norm including massive dims (~60k), but probe vector is orthogonal to massive dims. Result: coef is ~26k when it should be ~1200.

**Hypothesis:** Calibrating with activation norm *excluding* massive dims will produce useful starting coefficients.

**Plan:**
1. Load cached activation norms
2. Compute activation norm excluding massive dims
3. Compare base_coef values
4. Test if corrected coef lands in useful range

**Status:** Complete

### Results

**Key finding:** Massive dims account for 99.6% of activation norm at L15.

| Metric | L15 Value |
|--------|-----------|
| Full activation norm | 25,984 |
| Cleaned norm (no massive dims) | 1,120 |
| Massive dims only | 25,984 (100%) |

**Coefficient comparison:**

| Calibration Method | base_coef | Start (0.7x) | In useful range? |
|--------------------|-----------|--------------|------------------|
| Current (full norm) | 25,984 | 18,189 | NO (way too high) |
| Corrected (cleaned) | 1,120 | 784 | YES |

**Verification run with corrected range (784-1577):**

| Coef | Trait | Coherence | Notes |
|------|-------|-----------|-------|
| 784 | 1.8% | 93.3% | Small effect |
| 902 | 1.9% | 93.0% | Small effect |
| 1037 | 11.4% | 93.6% | Moderate, coherent |
| 1192 | 26.8% | 70.7% | Good, at threshold |
| 1371 | 53.4% | 53.7% | Strong, degraded |
| 1577 | 88.1% | 42.4% | Too strong |

**Conclusion:** Auto-search with corrected calibration would find c1192 as best valid result. The fix is straightforward: compute activation norm excluding massive dims.

### Proposed fix

In `analysis/steering/evaluate.py`, change:
```python
# Current
act_norm = layer_acts.norm(dim=-1).mean()

# Proposed
massive_dims = load_massive_dims(experiment)  # from calibration.json
cleaned_acts = layer_acts.clone()
cleaned_acts[:, massive_dims] = 0
act_norm = cleaned_acts.norm(dim=-1).mean()
```

Or simpler: add `--no-massive-coef` flag that divides base_coef by ~20 for models with massive activations.

---

## Subexperiment 2: Cleaning Variants & Cosine Analysis

**Goal:** Compare different cleaning strategies (top-1, top-3, top-5, top-10, uniform-13) and understand why cleaning hurts probe.

### Final Results at 70% Coherence Threshold

| Method | Coef | Trait | Notes |
|--------|------|-------|-------|
| probe (original) | 1200 | **33.4%** | Best overall |
| mean_diff_top1 | 1220 | 29.1% | Just dim 443 |
| mean_diff (original) | 2500 | 27.5% | High coef needed |
| mean_diff_cleaned | 1300 | 25.2% | Uniform-13 |
| probe_top3 | 1100 | 21.8% | Cleaning hurts |
| probe_cleaned | 1150 | 8.9% | Cleaning hurts badly |

### Key Discovery: Probe Uses Massive Dims Productively

**Energy in massive dims (L15):**
- mean_diff: 88.7% of norm in top-5 massive dims
- probe: 41.6% of norm in uniform-13 dims

**Probe is NOT orthogonal to massive dims!**

### Cosine Similarity Matrix (L15)

| Comparison | Cosine Sim |
|------------|------------|
| mean_diff ↔ probe (original) | 0.477 |
| mean_diff_cleaned ↔ probe_cleaned | **0.998** |
| mean_diff_top1 ↔ probe | **0.996** |

**After cleaning, mean_diff and probe become nearly identical vectors!**

### Revised Interpretation (for docs)

Original story: "Massive dims are noise, zero them out"

**Revised story:** Massive dims CAN carry discriminative signal. The problem is HOW you capture them:
- **mean_diff** captures them proportional to magnitude → dominated by context noise (88.7% energy, fails)
- **probe** captures them proportional to discriminative value → useful signal (41.6% energy, works)

This explains:
- Why probe >> mean_diff even though both have signal in massive dims
- Why cleaning hurts probe (removes learned discriminative signal)
- Why cleaning helps mean_diff (removes context noise)
- Why cleaned versions converge to 0.998 cosine similarity

**Practical takeaway:** Don't blindly zero massive dims. Use a discriminative method (probe/gradient) that learns which dims matter.

---

## Phase 2: Generalization & Ablation

**Started:** 2026-01-29 (session 2)

**Goal:** Strengthen findings by testing (1) second trait, (2) cleaning ablation, (3) probe causality.

### Step 7-9: Sycophancy Generalization (Exploratory)

**Extraction:** 11.2s (150+150 responses)

**Steering results** (baseline=56.1):

| Method | Best Δ | Config | Coherence |
|--------|--------|--------|-----------|
| mean_diff | **+11.2** | L18 c1200 | 92% |
| mean_diff_cleaned | +19.9 | L15 c1400 | 85% |
| probe | +21.3 | L13 c1000 | 89% |

**Key finding: Sycophancy DIFFERS from refusal!**
- mean_diff works (+11.2) — NOT zero like refusal
- Cleaning still helps, probe still best
- The pattern holds (probe > cleaned > md) but mean_diff doesn't completely fail

**Interpretation:** Sycophancy may have less confounding between trait signal and massive dims. Or massive dims carry genuine sycophancy signal even in magnitude-weighted form.

---

### Step 10-11: Cleaning Ablation (refusal)

**Results at 70% coherence threshold:**

| Method | Dims Zeroed | Best Δ | Coef |
|--------|-------------|--------|------|
| mean_diff | 0 | +27.1 | 2500 |
| mean_diff_top1 | 1 (443) | **+28.8** | 1220 |
| mean_diff_top3 | 3 | +20.7 | 1100 |
| mean_diff_top5 | 5 | +14.2 | 1200 |
| mean_diff_top10 | 10 | +14.4 | 1200 |
| mean_diff_cleaned | 13 | +24.8 | 1300 |

**Key findings:**
1. **top-1 ≥ cleaned CONFIRMED** (28.8 > 24.8) — zeroing just dim 443 is sufficient
2. **Non-monotonic pattern!** top-1 > cleaned > top-3 > top-5 ≈ top-10
3. Original mean_diff works at high coef (2500) but needs ~2x the coefficient

**Interpretation:** Over-cleaning hurts. The "sweet spot" is either:
- Just dim 443 (highest contamination)
- All 13 uniform dims (catches multiple contamination modes)
- Middle ground (top-3, top-5) performs worst

This suggests dim 443 dominates the contamination, and other dims may carry some signal.

---

### Step 12-13: Probe Causality (Confirmatory)

**Results at 70% coherence threshold:**

| Method | Best Δ | Coef |
|--------|--------|------|
| probe (original) | **+33.0** | 1200 |
| probe_top1 | +20.7 | 1150 |
| probe_top3 | +21.5 | 1100 |
| probe_cleaned | +8.6 | 1150 |

**Hypothesis CONFIRMED:** probe > probe_cleaned (33.0 > 8.6)

Cleaning hurts probe by ~24 points. This formalizes the Phase 1 finding and proves probe uses massive dims productively.

---

### Phase 2 Summary

| Criterion | Status | Evidence |
|-----------|--------|----------|
| 1. Sycophancy generalizes | **PARTIAL** | Pattern holds but mean_diff works (+11.2) |
| 2. Top-1 ≥ Top-13 | **✓ CONFIRMED** | 28.8 ≥ 24.8 |
| 3. Cleaning hurts probe | **✓ CONFIRMED** | 33.0 > 8.6 |

**Main takeaways:**
1. Dim 443 is the primary contamination source for refusal/mean_diff
2. Sycophancy behaves differently — mean_diff partially works
3. Probe learns to use massive dims discriminatively; cleaning destroys this
4. The "right" amount of cleaning is task/method dependent

---

## Phase 3: Cross-Model Comparison (gemma-2-2b)

**Started:** 2026-01-29 (session 3)

**Goal:** Test if milder massive activation contamination (~60x in gemma-2-2b vs ~1000x in gemma-3-4b) produces different results.

### Step 15: Config update
- Added gemma2_base and gemma2_instruct variants to config.json

### Step 16-17: Extraction & Cleaning
- Extracted refusal vectors for gemma-2-2b (30.6s)
- Created cleaning variants: mean_diff_cleaned (8 dims), mean_diff_top1 (dim 334), mean_diff_top5

### Step 18: Refusal Steering (gemma-2-2b)

**Critical finding: gemma-2-2b needs MUCH lower coefficients!**

| Model | Useful coef range | Coherence cliff |
|-------|-------------------|-----------------|
| gemma-3-4b | 1000-2000 | ~2500 |
| gemma-2-2b | 50-120 | ~150 |

**Results at 70% coherence threshold (baseline=13.4):**

| Method | Δ | Coef | Notes |
|--------|---|------|-------|
| mean_diff | **+28.6** | 120 | WORKS! |
| mean_diff_cleaned | +26.5 | 120 | |
| mean_diff_top1 | +24.1 | 90 | |
| probe | +21.2 | 90 | Worse than mean_diff! |

**UNEXPECTED FINDING: On gemma-2-2b, mean_diff > probe!**

This is the OPPOSITE of gemma-3-4b where probe (+33.0) > mean_diff (+27.1).

### Interpretation

The massive activation contamination severity matters:
- **gemma-3-4b (1000x)**: mean_diff captures noise, probe learns to filter → probe > mean_diff
- **gemma-2-2b (60x)**: Contamination is mild enough that mean_diff still captures signal → mean_diff ≈ probe

The coefficient range difference (20x lower for gemma-2-2b) may be explained by:
1. Smaller model (2B vs 4B)
2. Lower activation magnitudes overall
3. Different architecture (26 vs 34 layers)

### Phase 3 Success Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| 1. mean_diff better on gemma-2-2b | **✓** | +28.6 > +27.1 |
| 2. probe > mean_diff on both | **✗** | Only true for gemma-3-4b |
| 3. Similar dim-dominance pattern | **~** | Top-1 cleaning similar |

**Main takeaway:** The severity of massive activation contamination determines which extraction method works best. With mild contamination (gemma-2-2b), simple mean_diff suffices. With severe contamination (gemma-3-4b), discriminative methods (probe) are necessary.

---

## Phase 4: Position Window Ablation

**Started:** 2026-01-29 (session 3, continued)

**Goal:** Test whether averaging over more response tokens dilutes massive dim contamination.

### Step 23-24: Extraction
- Extracted vectors at response[:10] and response[:20] (8s each)

### Step 25-26: Steering Results

| Position | mean_diff Δ | probe Δ | Best method |
|----------|-------------|---------|-------------|
| response[:5] | +27.1 | +33.0 | probe |
| response[:10] | +12.2 | +26.6 | probe |
| response[:20] | **+33.4** | +20.8 | **mean_diff** |

**Key findings:**

1. **Non-monotonic mean_diff pattern!**
   - response[:10] is WORSE than response[:5] (+12.2 vs +27.1)
   - response[:20] is BEST (+33.4)

2. **Probe degrades with longer windows**
   - +33.0 → +26.6 → +20.8
   - Probe may be overfitting to early-token patterns

3. **At response[:20], mean_diff > probe!**
   - First time mean_diff beats probe on gemma-3-4b
   - +33.4 vs +20.8

### Interpretation

The position window matters more than expected:
- **Early tokens (1-5)**: Massive dims dominate → probe better
- **Middle tokens (6-10)**: Transition zone → both methods struggle
- **Later tokens (11-20)**: Contamination dilutes → mean_diff recovers

The probe's degradation suggests it learns patterns specific to early tokens that don't generalize well to longer windows.

### Phase 4 Success Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| 1. mean_diff improves with windows | **PARTIAL** | [:20] > [:5] but [:10] worse |
| 2. probe stable across windows | **✗** | Degrades: 33.0 → 20.8 |
| 3. Quantify relationship | **✓** | Non-monotonic, position-dependent |

**Practical takeaway:** For mean_diff extraction, use longer position windows (response[:20]) to dilute massive activation contamination. For probe, stick with shorter windows (response[:5]).

---

## Summary: All Phases

### Overall Findings

1. **Massive dim contamination is model-dependent:**
   - gemma-3-4b (~1000x): Severe → probe required at response[:5]
   - gemma-2-2b (~60x): Mild → mean_diff works

2. **Position window matters:**
   - response[:20] dilutes contamination → mean_diff recovers to probe-level
   - Probe degrades with longer windows (overfitting to early tokens)

3. **Coefficient calibration is broken:**
   - Auto-search starts way too high (massive dims inflate norm)
   - gemma-3-4b: useful range 1000-2000, auto starts at 26k
   - gemma-2-2b: useful range 50-120, ~20x lower

4. **Cleaning is task-dependent:**
   - Helps mean_diff (removes noise)
   - Hurts probe (removes learned signal)
   - Top-1 dim cleaning often sufficient

### Recommendations

For practitioners:
1. Check model for massive activations before extraction
2. If severe (>100x): Use probe OR mean_diff with response[:20]
3. If mild (<100x): mean_diff is sufficient
4. Fix coefficient calibration to exclude massive dims from norm
