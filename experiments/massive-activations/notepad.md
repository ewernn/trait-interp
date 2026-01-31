# Experiment Notepad - Phase 2 Refined Ablations

## Machine
- GPU: NVIDIA GeForce RTX 5090, 32GB VRAM
- Date: 2026-01-30

## Important Context
**Plan's background assumptions were INCORRECT:**
- Plan says "mean_diff wins at [:20]" → FALSE (probe +49.4 > mean_diff +39.0)
- Plan says "top-1 cleaning helps mean_diff" → FALSE (top-1 hurts: +26.2 → +20.5)
- Actual: probe > mean_diff at ALL positions on BOTH models

## Progress
- [x] Prerequisites: Verified calibration and extraction data
- [x] Phase 1: Gap analysis (core methods all have coherence ≥68%)
- [x] Phase 2: Clean-before-extraction (preclean vectors)
- [x] Phase 3: Top-2 cleaning ablation
- [x] Phase 4: Coefficient pattern analysis

---

## Key Findings

### Phase 2: Preclean vs Postclean (Clean-before vs Clean-after)

**gemma-3-4b [:5] Results (coherence ≥68%):**

| Method | Δ | Coh | Coef |
|--------|---|-----|------|
| probe | +33.2 | 84.6% | 1200 |
| probe_preclean | **+57.9** | 71.8% | 1500 |
| probe_cleaned | +21.3 | 80.9% | 1200 |
| mean_diff | +26.2 | 82.1% | 2500 |
| mean_diff_preclean | +19.2 | 82.8% | 1200 |
| mean_diff_cleaned | +28.3 | 69.8% | 800 |

**Key insight: Order of cleaning matters!**
- **probe_preclean (+57.9) >> probe (+33.2) >> probe_cleaned (+21.3)**
  - Pre-cleaning activations HELPS probe significantly (+75% improvement!)
  - Post-cleaning vectors HURTS probe (-36%)
- **mean_diff_preclean (+19.2) < mean_diff (+26.2) < mean_diff_cleaned (+28.3)**
  - Pre-cleaning activations HURTS mean_diff (-27%)
  - Post-cleaning vectors helps slightly (+8%)

**Interpretation:** Probe learns to USE massive dims discriminatively. Pre-cleaning removes noise from training data, helping it learn better. Post-cleaning amputates learned weights.

### Phase 3: Cleaning Curve (mean_diff)

| Dims Cleaned | Δ | Coherence |
|--------------|---|-----------|
| 0 (original) | +26.2 | 82.1% |
| 1 (top-1) | +20.5 | 88.7% |
| 2 (top-2) | +72.1 | 72.2% |
| 5 (top-5) | +13.9 | 83.7% |
| 13 (all) | +28.3 | 69.8% |

**Unexpected pattern!** top-2 jumps to +72.1 (at edge of coherence threshold). Non-monotonic curve suggests complex interaction between dims.

### Phase 4: Coefficient Patterns

**gemma-3-4b:**
| Position | probe coef | mean_diff coef | Ratio |
|----------|-----------|----------------|-------|
| [:5] | 1200 | 2500 | 2.1x |
| [:10] | 2000 | 2000 | 1.0x |
| [:20] | 1000 | 1000 | 1.0x |

**gemma-2-2b:**
| Position | probe coef | mean_diff coef | Ratio |
|----------|-----------|----------------|-------|
| [:5] | 110 | 110 | 1.0x |
| [:10] | 110 | 130 | 1.2x |
| [:20] | 110 | 90 | 0.8x |

**Observation:** Coefficient ratio varies with position. The 2x ratio on gemma-3-4b at [:5] disappears at longer windows.

---

## Cross-Model Summary

| Model | Position | probe Δ | mean_diff Δ | Winner |
|-------|----------|---------|-------------|--------|
| gemma-3-4b | [:5] | +33.2 | +26.2 | probe |
| gemma-3-4b | [:10] | +93.2 | +38.8 | probe |
| gemma-3-4b | [:20] | +49.4 | +39.0 | probe |
| gemma-2-2b | [:5] | +54.6 | +42.1 | probe |
| gemma-2-2b | [:10] | +66.8 | +36.5 | probe |
| gemma-2-2b | [:20] | +81.2 | +35.7 | probe |

**probe wins in ALL conditions.**

---

## Success Criteria Assessment

- [x] All key method comparisons have results with coherence ≥68%
- [x] Clean-before vs clean-after comparison complete for probe
  - Finding: preclean > original > cleaned (opposite of what plan expected for mechanism)
- [x] top-2 cleaning fills gap between top-1 and top-3
  - Finding: Non-monotonic curve, top-2 unexpectedly best
- [x] Coefficient patterns documented with concrete recommendations

---

## Final Conclusions

1. **probe > mean_diff everywhere** - not just when massive activations are severe
2. **Pre-cleaning helps probe dramatically** (+57.9 vs +33.2) - this is the main new finding
3. **Post-cleaning hurts probe** (-36%) - confirms probe uses massive dims productively
4. **Cleaning curve is non-monotonic** - top-2 cleaning produces best results
5. **The "massive activations contaminate mean_diff" story is incomplete** - the real story is about how different methods interact with these dims during training vs inference

**Practical recommendation:** For best steering results:
- Use probe method (not mean_diff)
- Consider pre-cleaning activations before extraction (not post-cleaning vectors)
- Start with coefficients: gemma-3-4b ~1200, gemma-2-2b ~110

---

## Files Created
- `core/methods.py` - Added PreCleanedMethod class
- `extract_preclean_vectors.py` - Script to create preclean vectors
- `clean_vectors_top2.py` - Script to create top-2 cleaned vectors
