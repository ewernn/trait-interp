# Emotional Valence: 2×2 Generalization Matrix

**Trait**: emotional_valence (positive vs negative affect)
**Model**: Gemma 2B (google/gemma-2-2b-it)
**Date**: 2025-11-17

---

## 2×2 Matrix Results

| Training Data | Test Data   | Best Probe Performance | Best Overall Performance | Key Finding |
|---------------|-------------|------------------------|---------------------------|-------------|
| **Instruction** | **Natural** | **100.0% (L16)** | **Probe 100.0% (L16)** ⭐ | **PERFECT cross-distribution!** |
| Instruction | Instruction | 60.7% (all layers) | Mean Diff 71.4% (L16) | Flat probe performance |
| Natural | Natural | **100.0% (all layers)** | **Probe 100.0% (all layers)** ⭐ | Perfect same-distribution |
| Natural | Instruction | 88.3% (L16) | Probe 88.3% (L16) | Good reverse generalization |

---

## Key Findings

### 1. Probe Dominates for emotional_valence

**Cross-Distribution (Instruction → Natural)**:
- **Probe achieves 100.0% accuracy at layer 16** (201/201 correct)
- **91-100% accuracy across layers 0-25** (near-perfect everywhere)
- Probe generalizes PERFECTLY to natural data

vs uncertainty_calibration:
- Probe only achieved 51.4% cross-distribution (failed to generalize)
- ICA won with 89.5%

**Opposite pattern!**

---

### 2. Perfect Same-Distribution Performance

**Natural → Natural**:
- **Probe: 100.0% accuracy at ALL 26 layers**
- Mean Diff: 48-88% (varies by layer)

**Instruction → Instruction**:
- Probe: 60.7% across ALL layers (completely flat)
- Mean Diff: 32-71% (peaks at layer 16)
- ICA: 28-71% (irregular)

---

### 3. Excellent Reverse Generalization

**Natural → Instruction** (layer 16 only):
- Probe: 88.3% (166/188)
- Mean Diff: 53.2% (100/188)

Natural vectors generalize well back to instruction data.

---

## Comparison to uncertainty_calibration

| Metric | emotional_valence | uncertainty_calibration |
|--------|-------------------|-------------------------|
| **Cross-dist winner** | **Probe (100.0%)** | **ICA (89.5%)** |
| Cross-dist: Probe | 100.0% (L16) | 51.4% (L16) - FAIL |
| Cross-dist: ICA | 49-77% | 89.5% (L10) - WIN |
| Cross-dist: Gradient | 80-91% | 57.5% (L16) |
| Same-dist: Probe | 60.7% (instruction) | 96.9% (instruction) |
| Same-dist: Probe | 100.0% (natural) | N/A |
| **Optimal layer** | **Layer 16** | **Layer 10** |

---

## Why Such Different Results?

### Hypothesis: Trait Separability

**emotional_valence** (affect):
- Positive/negative emotion is HIGHLY separable
- Clear, consistent linguistic markers ("wonderful", "terrible")
- Linear separation works perfectly
- Probe can learn robust decision boundary

**uncertainty_calibration** (epistemic state):
- Uncertainty is SUBTLE and context-dependent
- Confounded with instruction-following in training data
- Requires disentanglement (ICA) to separate genuine uncertainty from artifacts
- Probe overfits to instruction patterns

---

## Full Results

### Test 1: Instruction → Natural (Cross-Distribution)

**Top 5 layers (Probe)**:
1. Layer 16: 100.0% (201/201) ⭐
2. Layer 15: 99.5% (200/201)
3. Layer 14: 99.0% (199/201)
4. Layer 13: 98.5% (198/201)
5. Layer 12: 98.0% (197/201)

**All methods at layer 16**:
- Probe: 100.0% (201/201) ⭐
- Gradient: 84.6% (170/201)
- Mean Diff: 78.6% (158/201)
- ICA: 60.7% (122/201)

---

### Test 2: Instruction → Instruction (Same-Distribution)

**Top 5 overall**:
1. Mean Diff layer 16: 71.4% (20/28)
2. ICA layer 10: 71.4% (20/28)
3. ICA layer 25: 67.9% (19/28)
4. ICA layer 20: 64.3% (18/28)
5. Probe (all layers): 60.7% (17/28)

**Probe**: 60.7% across ALL 26 layers (completely flat!)

---

### Test 3: Natural → Natural (Same-Distribution)

**Probe**: 100.0% at ALL 26 layers (41/41 correct)

**Mean Diff** (varies by layer):
- Best: Layer 8: 87.8% (36/41)
- Worst: Layer 24-25: 48.8% (20/41)

---

### Test 4: Natural → Instruction (Cross-Distribution)

**Layer 16 only**:
- Probe: 88.3% (166/188)
- Mean Diff: 53.2% (100/188)

---

## Conclusions

1. **Trait-specific generalization patterns exist**
   - emotional_valence: Probe wins (100% cross-dist)
   - uncertainty_calibration: ICA wins (89.5% cross-dist)

2. **Separability determines best method**
   - High separability → Supervised methods (Probe) work perfectly
   - Low separability + confounds → Unsupervised methods (ICA) needed

3. **emotional_valence is an "easy" trait**
   - Perfect cross-distribution with simple probe
   - Consistent across all layers
   - No instruction confound

4. **Implications for deployment**
   - For affect-based traits: Use Probe at layer 16
   - For epistemic traits: Use ICA at layer 10
   - Test cross-distribution before deployment!

---

## Files Generated

- `/tmp/emotional_valence_cross_dist_output.txt` - Instruction → Natural results
- `/tmp/emotional_valence_instruction_to_instruction_output.txt` - Instruction → Instruction results
- `/tmp/emotional_valence_natural_to_natural_output.txt` - Natural → Natural results
- `/tmp/emotional_valence_natural_to_instruction_output.txt` - Natural → Instruction results
- `/tmp/emotional_valence_2x2_matrix_summary.md` - This file
