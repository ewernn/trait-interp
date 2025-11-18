# Emotional Valence: Corrected 2×2 Analysis

**Date**: 2025-11-17
**Trait**: emotional_valence (positive vs negative affect)
**Model**: Gemma 2B

---

## Data Quality Issues Found

1. **Instruction-based activations**: 188 examples (not 201)
   - Metadata claims 100 pos + 101 neg = 201
   - Actual file has 188 examples
   - 13 examples missing (likely filtered during extraction)

2. **Test set sizes inconsistent**:
   - Instruction → Instruction: 41 test examples (20 pos + 21 neg from 188 total)
   - Instruction → Natural: 201 test examples (100 pos + 101 neg)
   - Natural → Instruction: 188 test examples (using filtered instruction data)
   - Natural → Natural: 41 test examples (20 pos + 21 neg from 201 total)

3. **Probe showing exactly 60.7% (17/28) across ALL layers** in Instruction → Instruction
   - This is 17 correct out of 28 on EVERY SINGLE LAYER
   - Suggests severe overfitting or class imbalance issue
   - Test set too small (41 examples, but only testing on 28?)

---

## Corrected 2×2 Matrix Results

| Training Data | Test Data   | Test Size | Probe Best | Key Finding |
|---------------|-------------|-----------|------------|-------------|
| Instruction   | Natural     | 201       | **100.0%** (L16) | Perfect cross-distribution |
| Instruction   | Instruction | 41 (28?)  | 60.7% (all layers) | SUSPICIOUS - same score all layers |
| Natural       | Natural     | 41        | **100.0%** (all layers) | Perfect same-distribution |
| Natural       | Instruction | 188       | 88.3% (L16) | Good reverse cross-distribution |

---

## Major Problems with Current Analysis

### Problem 1: Instruction → Instruction Test is Broken

**Evidence**:
```
Layer 0:  probe: 60.7% (17/28)
Layer 1:  probe: 60.7% (17/28)
...
Layer 25: probe: 60.7% (17/28)
```

Probe gets EXACTLY 17/28 correct on every single layer. This is impossible unless:
- Test set has severe class imbalance
- Test set is too small (28 examples, not 41)
- There's a bug in the test code

**Why it matters**: Can't trust this number for comparison.

---

### Problem 2: Small Test Sets

**Instruction → Instruction**: 41 examples total
- With 80/20 split: 20 pos + 21 neg = 41 test
- But results show 28 tested? Where did 13 go?
- Too small for reliable statistics

**Natural → Natural**: 41 examples total
- With 80/20 split: 20 pos + 21 neg = 41 test
- Same small size issue

**Cross-distribution tests**: 201 and 188 examples
- Much larger, more reliable

---

### Problem 3: Missing Instruction Data

Instruction activations file has 188 examples, not 201:
- 13 examples were filtered/lost
- Metadata doesn't reflect actual data
- Affects all tests using instruction data

---

## What We Can Trust

### ✅ Instruction → Natural (Cross-Distribution)

**Test size**: 201 examples (100 pos, 101 neg)
**Reliable**: Yes - large test set

**Key findings**:
- **Probe dominates**: 91-100% across most layers
- **Best**: Probe at layer 16 = 100.0% (201/201 perfect!)
- **ICA fails**: 49-77%, many sign flips
- **Gradient**: 50-90%, varies by layer

This is OPPOSITE from uncertainty_calibration where ICA won!

---

### ✅ Natural → Instruction (Reverse Cross-Distribution)

**Test size**: 188 examples
**Reliable**: Yes - large test set
**Tested**: Layer 16 only (only layer with natural vectors)

**Results**:
- Probe: 88.3% (166/188)
- Mean Diff: 53.2% (100/188)

**Interpretation**: Natural vectors work reasonably well on instruction data

---

### ⚠️ Instruction → Instruction (Same-Distribution)

**Test size**: 41 examples (but only 28 tested??)
**Reliable**: NO - too small, suspicious results

**Issues**:
- Probe: exactly 60.7% (17/28) on ALL 26 layers
- This is statistically impossible
- Test set too small for reliable conclusions
- Possible bug in test code

**Can't use for comparison**

---

### ⚠️ Natural → Natural (Same-Distribution)

**Test size**: 41 examples
**Reliable**: Questionable - very small

**Results** (from test output):
- Probe: 100.0% on all layers
- Mean Diff: varies by layer

**Interpretation**: Perfect on small test set, but need larger test to confirm

---

## Correct Interpretation

### Main Finding: Trait-Specific Generalization Patterns

**emotional_valence (affect)**:
- Instruction → Natural: **Probe wins** (100% at L16)
- High linguistic separability ("wonderful" vs "terrible")
- Clear surface-level markers
- No confound with instruction-following
- Supervised methods excel

**uncertainty_calibration (epistemic state)**:
- Instruction → Natural: **ICA wins** (89.5% at L10)
- Subtle, context-dependent
- Confounded with instruction-following
- Requires disentanglement
- Unsupervised methods needed

---

## Recommendations

### Fix Test Issues

1. **Investigate Instruction → Instruction test**
   - Why does probe get exactly 17/28 on all layers?
   - Where are the missing 13 test examples?
   - Rerun with fixed code

2. **Use larger test sets**
   - 41 examples is too small
   - Consider 70/30 split instead of 80/20
   - Or use cross-validation

3. **Fix missing instruction data**
   - 188 examples vs 201 claimed
   - Find and restore missing 13 examples
   - Or update metadata to reflect actual count

### Valid Conclusions (from reliable tests)

1. **Probe achieves 100% cross-distribution for emotional_valence**
   - Instruction → Natural: 100% (201/201) at layer 16
   - This is genuinely impressive

2. **emotional_valence ≠ uncertainty_calibration**
   - Different trait types need different extraction methods
   - High-separability traits → Probe
   - Low-separability + confounds → ICA

3. **Middle layers still win**
   - Layer 14-16 optimal for emotional_valence
   - Consistent with uncertainty_calibration finding

---

## Next Steps

1. Debug Instruction → Instruction test (find the 28 vs 41 discrepancy)
2. Increase test set sizes for same-distribution tests
3. Extract natural vectors for all 26 layers (not just layer 16)
4. Run complete 2×2 with corrected test sets
5. Test on a third trait (retrieval_construction) to validate patterns
