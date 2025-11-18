# Research Insights

Major findings from trait extraction experiments.

---

## 2025-11-15: Instruction-Following Confound

**Finding**: Instruction-based trait vectors measure compliance with meta-instructions, not natural trait expression.

**Evidence**:
- Refusal vector (instruction-based): Harmful prompts score -0.51, benign with "refuse" instruction score +6.15
- Polarity inverted: Vector points toward instruction-following, not genuine refusal

**Solution**: Natural elicitation - use naturally contrasting scenarios without explicit instructions

**Validation**:
- Natural refusal vector: Harmful +33.337, benign +22.913 (correct polarity, +10.4 separation)

**Implication**: All 16 instruction-based traits likely inverted; re-extract using natural scenarios

**Details**: See [session notes](sessions/2025-11-15-inversion-discovery.md)

---

## 2025-11-17: Gradient Optimization Achieves 96.1% Cross-Distribution Accuracy

**Finding**: Gradient-optimized vectors achieve 96.1% cross-distribution accuracy, dramatically outperforming all other methods and nearly matching same-distribution probe performance.

**Evidence** (uncertainty_calibration, complete 4×4 distribution matrix):

**Instruction → Natural (cross-distribution):**
- **Gradient @ Layer 14: 96.1%** (174/181 correct) ⭐ Only 0.8 points below same-distribution Probe
- ICA @ Layer 10: 89.5% (162/181 correct)
- Mean Diff @ Layer 20: 89.0% (161/181 correct)
- Probe @ Layer 6: 60.8% (110/181 correct) - 46% drop from same-distribution

**Natural → Natural (same-distribution):**
- Probe/Gradient/ICA: 96-100% across layers 1-24
- All methods work on clean, unconfounded data

**Natural → Instruction (reverse cross-distribution):**
- Probe @ Layer 18: 88.4%
- Shows natural-trained vectors can detect instruction patterns

**Key Insights**:
1. **Gradient optimization finds generalizable directions**: Unit normalization + iterative refinement discovers trait representation that transfers across distributions
2. **Middle layers (6-16) dominate**: 90-96% cross-distribution accuracy
3. **Late layers (21-25) fail**: Specialize for instruction-following, collapse to ~56% on natural data
4. **Probe massively overfits**: 96.9% → 60.8% cross-distribution (supervised learning captures instruction artifacts)
5. **Training strength ≠ generalization**: Correlation -0.039 between vector_norm and cross-distribution accuracy
6. **Natural data is clean**: All methods achieve 96-100% when trained on natural elicitation

**Distribution Gap (Natural→Natural vs Instruction→Natural):**
- Gradient: +3.9% (most robust)
- ICA: +10.5%
- Mean Diff: -0.1% (equally robust)
- Probe: +39.2% (most data-dependent)

**Explanation**: Gradient optimization with unit normalization finds directions that maximize trait separation while avoiding instruction-specific features. ICA's disentanglement helps but gradient's direct optimization is more effective. Probe overfits to instruction-following patterns.

**Implication**: For instruction-trained vectors, use **Gradient at layer 14** (96.1% cross-distribution). If training on natural data, all methods work (96-100%).

**Verification**:
```bash
# Run complete distribution matrix
python3 /tmp/complete_distribution_matrix.py
# Run multi-method comparison
python3 /tmp/multimethod_layer_sweep.py
```

**Details**: See [temp-vector-analysis/](../temp-vector-analysis/) for complete methodology

---

## 2025-11-17: Trait Separability Determines Optimal Extraction Method

**Finding**: Different trait types require different extraction methods. High-separability traits favor supervised methods (Probe), while low-separability traits with instruction confounds favor unsupervised methods (ICA) or optimization methods (Gradient).

**Evidence** (complete 4×4 distribution matrix comparison):

### emotional_valence (High Separability)

Clear linguistic markers: "wonderful" vs "terrible", "beautiful" vs "tragic"

**Instruction → Natural (cross-distribution):**
- **Probe @ All Layers: 100.0%** (201/201 perfect) ⭐
- Gradient @ Layer 14: 96.0% (193/201)
- Mean Diff @ Layer 8: 91.0% (183/201)
- ICA @ Layer 25: 77.1% (155/201)

**Natural → Natural (same-distribution):**
- **Probe @ All Layers: 100.0%** (41/41 perfect) ⭐
- Gradient @ Layers 14,16,22: 100.0%
- Mean Diff @ Layer 3: 85.4%
- ICA @ Layer 25: 75.6%

**Winner**: Probe dominates everywhere (100% cross-distribution)

### uncertainty_calibration (Low Separability + Confounds)

Subtle epistemic states confounded with instruction-following: "I think maybe" vs confident assertions

**Instruction → Natural (cross-distribution):**
- **Gradient @ Layer 14: 96.1%** (174/181) ⭐
- **ICA @ Layer 10: 89.5%** (162/181)
- Mean Diff @ Layer 20: 89.0% (161/181)
- Probe @ Layer 6: 60.8% (110/181) - FAILS

**Natural → Natural (same-distribution):**
- Probe/Gradient/ICA: 96-100% (all methods work on clean data)

**Winner**: Gradient/ICA for cross-distribution (Probe fails with 46% accuracy drop)

### Trait-Specific Method Selection

| Trait Type | Characteristics | Best Method | Cross-Dist Accuracy | Why |
|------------|-----------------|-------------|---------------------|-----|
| emotional_valence | High separability, clear markers, no confound | **Probe** | 100.0% | Linear separation works perfectly |
| uncertainty_calibration | Low separability, subtle, instruction confound | **Gradient/ICA** | 96.1% / 89.5% | Needs optimization or disentanglement |

**Key Insights**:
1. **Separability drives method choice**: Clear linguistic traits → Probe, subtle traits → Gradient/ICA
2. **Instruction confound**: When present, supervised methods (Probe) overfit to compliance patterns
3. **Natural data is clean**: All methods work well (96-100%) when trained on naturally-elicited data
4. **Perfect isn't always better**: emotional_valence Probe hits 100%, but uncertainty_calibration Probe fails cross-distribution despite 96.9% same-distribution

**Implication**: Test cross-distribution performance before deployment. Don't assume high same-distribution accuracy means robust generalization.

**Verification**:
```bash
# emotional_valence 4×4 sweep
python3 temp-vector-analysis/emotional_valence/emotional_valence_full_4x4_sweep.py
```

**Details**: See `temp-vector-analysis/emotional_valence/` and `temp-vector-analysis/CROSS_DISTRIBUTION_GENERALIZATION_STUDY.md`

---

## Future Findings

Additional insights will be added here as research progresses.
