# Session Handoff - Projection Normalization & Cleanup

**Date:** 2025-12-17
**Focus:** Baseline investigation complete, implementing cosine normalization, rerunning inference

---

## Key Findings This Session

### 1. Baseline Problem Root Cause
- **NOT** caused by base vs IT model distribution shift
- Caused by measuring projection from ORIGIN instead of TRAINING CENTROID
- `baseline = (||mean_pos||² - ||mean_neg||²) / (2 * ||vector||)`
- Different extraction methods have different baselines:
  - mean_diff: ~50 baseline (huge)
  - probe: ~1 baseline (small)

### 2. Normalization Comparison
| Metric | Formula | Cohen's d | Centered? |
|--------|---------|-----------|-----------|
| vnorm | `a·v / ||v||` | 1.93 | No |
| cosine | `a·v / (||v||·||a||)` | 2.20 | ~Yes |

- Cosine has 14% better signal (higher Cohen's d)
- Cosine reduces baseline ~100-300x (bandaid, not proper fix)
- probe + cosine → baseline ≈ 0

### 3. BOS Token Issue
- BOS projects to EXTREME values (351 for refusal, -130 for intent)
- BOS is out-of-distribution for trait vectors (training used content only)
- Should be excluded from projections

---

## Implementation Plan

### Phase 1: Visualization Toggle (cosine primary)

**Files to modify:**

1. `visualization/views/trait-dynamics.js`
   - Add toggle state for normalization mode
   - Rename/restructure:
     - Current "Token Trajectory" → compute based on toggle
     - Remove separate "Normalized Trajectory" section
   - Add toggle UI: `[Cosine] [Magnitude]` buttons
   - Default to cosine

2. `visualization/core/state.js`
   - Add `projectionMode: 'cosine'` to state
   - Load/save preference from localStorage

3. `visualization/styles.css`
   - Add styles for toggle buttons (use existing `.pp-btn` pattern)

### Phase 2: Fix Projection Pipeline

**Files to modify:**

4. `extraction/extract_vectors.py`
   - Compute and store baseline: `baseline = center · v / ||v||`
   - Already stores bias for probe, extend to mean_diff/gradient
   - Save in metadata JSON alongside vector

5. `inference/project_raw_activations_onto_traits.py`
   - Load baseline from vector metadata
   - Subtract baseline from projections (optional flag `--centered`)
   - Skip BOS tokens (first 1-2 tokens) in output

6. `utils/vectors.py`
   - Add `load_vector_with_baseline()` helper
   - Returns (vector, baseline) tuple

### Phase 3: Re-extract and Re-run Inference

**Commands to run:**

```bash
# 1. Re-extract vectors with baseline (all traits)
python extraction/run_pipeline.py \
    --experiment gemma-2-2b \
    --traits all \
    --skip-responses  # Keep existing responses, just re-extract vectors

# 2. Re-run ALL inference (fixes double BOS + new projection format)
python inference/capture_raw_activations.py \
    --experiment gemma-2-2b \
    --prompt-set single_trait,multi_trait,dynamic,adversarial,baseline,real_world,harmful,benign

python inference/capture_raw_activations.py \
    --experiment gemma-2-2b \
    --prompt-set jailbreak

# 3. Verify jailbreak projections look correct
python -c "
import json
from pathlib import Path
p = Path('experiments/gemma-2-2b/inference/chirp/refusal/residual_stream/jailbreak/1.json')
data = json.load(open(p))
print('First 5 tokens:', data.get('prompt', {}).get('tokens', [])[:5])
print('First 5 projections:', data['projections']['prompt'][:5])
"
```

### Phase 4: Verify & Test

1. **Check pagination works** - Load jailbreak set, verify 50-per-page
2. **Check cosine toggle** - Switch between modes, verify values change
3. **Check baseline centering** - Traits should be near zero mean
4. **Check BOS exclusion** - No extreme values at start

---

## Files Summary

| File | Change |
|------|--------|
| `visualization/views/trait-dynamics.js` | Add toggle, restructure charts |
| `visualization/core/state.js` | Add projectionMode state |
| `visualization/styles.css` | Toggle button styles |
| `extraction/extract_vectors.py` | Store baseline in metadata |
| `inference/project_raw_activations_onto_traits.py` | Subtract baseline, skip BOS |
| `utils/vectors.py` | Add baseline loading helper |

---

## What Was Already Done This Session

1. ✅ Pagination for prompt picker (>50 prompts)
2. ✅ Investigated baseline issue thoroughly
3. ✅ Compared vnorm vs cosine (cosine wins)
4. ✅ Identified BOS as out-of-distribution

## What NOT to Do

- Don't use mean_diff as default (probe is better)
- Don't show BOS tokens in projections (meaningless)
- Don't remove vnorm entirely (useful for steering analysis)

---

## Quick Reference

### Projection Formulas
```
vnorm  = a · v / ||v||           # "How much trait signal"
cosine = a · v / (||a|| · ||v||) # "How aligned with trait"
```

### Baseline Formula
```
baseline = (mean_pos + mean_neg) / 2 · v / ||v||
centered_proj = proj - baseline
```

### Key Stats
- ||h|| varies 8x across layers (72 at L0 → 550 at L25)
- probe + cosine baseline ≈ -0.01 (essentially zero)
- mean_diff + vnorm baseline ≈ 50 (huge)
