# Fork A: Method Comparison

**Goal:** Compare steering effectiveness of probe vs mean_diff vs gradient vectors

**Date:** 2026-01-01

---

## Summary

**Results are mixed - method effectiveness is trait-dependent.**

| Trait | probe | mean_diff | gradient | Winner |
|-------|-------|-----------|----------|--------|
| refusal_v2 | +38.3 (L11) | +46.5 (L9) | **+73.9 (L11)** | gradient (1.9x) |
| refusal | **+70.8 (L10)** | - | +51.2 (L10) | probe (1.4x) |

Initial finding (gradient >> probe on refusal_v2) did NOT generalize to refusal.

---

## Detailed Results

### chirp/refusal_v2 (baseline=15.5)

| Method    | Best Layer | Delta  | Trait Score | Coherence |
|-----------|------------|--------|-------------|-----------|
| probe     | L11        | +38.3  | 53.8        | 73.8      |
| mean_diff | L9         | +46.5  | 62.0        | 80.2      |
| **gradient** | **L11** | **+73.9** | **89.5** | **85.7** |

Gradient achieves 1.9x the delta of probe with better coherence.

### chirp/refusal (baseline=17.1)

| Method    | Best Layer | Delta  | Trait Score | Coherence |
|-----------|------------|--------|-------------|-----------|
| **probe** | **L10**    | **+70.8** | **87.9** | **70.3** |
| gradient  | L10        | +51.2  | 68.3        | 76.9      |

Probe achieves 1.4x the delta of gradient.

---

## Key Findings

1. **Method effectiveness is trait-dependent** - No universal winner

2. **refusal_v2: gradient dominates**
   - +73.9 delta vs +38.3 for probe (1.9x improvement)
   - Better coherence too (85.7 vs 73.8)

3. **refusal: probe wins**
   - +70.8 delta vs +51.2 for gradient (1.4x improvement)
   - Gradient has slightly better coherence (76.9 vs 70.3)

4. **mean_diff is middle ground**
   - Only tested on refusal_v2 (+46.5)
   - Better than probe, worse than gradient

5. **Same optimal layer for both methods on each trait**
   - refusal_v2: both best at L11
   - refusal: both best at L10

---

## Why the Difference?

Possible explanations for trait-dependent method performance:

1. **Training data quality** - refusal_v2 may have cleaner separation that gradient optimization exploits better

2. **Gradient convergence** - The gradient optimization may have converged differently for each trait

3. **Trait structure** - Some traits may be more linearly separable (favoring probe) while others benefit from gradient's flexibility

4. **Vector norm differences** - Gradient vectors have much smaller norms (base_coef ~2-4) vs probe (base_coef ~55-120)

---

## Recommendations

1. **Don't assume gradient is always better** - Test both methods when extracting new traits

2. **Consider trait-specific method selection** - Use steering results to pick the best method per trait

3. **Current get_best_vector() logic is reasonable** - Using steering results as ground truth accounts for this variability

4. **Future work:** Extract gradient vectors for more traits and validate this pattern

---

## Commands Run

```bash
# refusal_v2 - mean_diff
python3 analysis/steering/evaluate.py \
    --experiment gemma-2-2b \
    --vector-from-trait gemma-2-2b/chirp/refusal_v2 \
    --method mean_diff --position "response[:]" \
    --layers 7-15 --subset 5 --max-new-tokens 64

# refusal_v2 - gradient
python3 analysis/steering/evaluate.py \
    --experiment gemma-2-2b \
    --vector-from-trait gemma-2-2b/chirp/refusal_v2 \
    --method gradient --position "response[:]" \
    --layers 7-15 --subset 5 --max-new-tokens 64

# refusal - probe
python3 analysis/steering/evaluate.py \
    --experiment gemma-2-2b \
    --vector-from-trait gemma-2-2b/chirp/refusal \
    --method probe --position "response[:]" \
    --layers 8-12 --subset 5 --max-new-tokens 64 --search-steps 10

# refusal - gradient
python3 analysis/steering/evaluate.py \
    --experiment gemma-2-2b \
    --vector-from-trait gemma-2-2b/chirp/refusal \
    --method gradient --position "response[:]" \
    --layers 7-15 --subset 5 --max-new-tokens 64
```

---

## Files Modified

Results appended to:
- `experiments/gemma-2-2b/steering/chirp/refusal_v2/response_all/results.json`
- `experiments/gemma-2-2b/steering/chirp/refusal/response_all/results.json`
