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

## 2025-11-18: BFloat16 NumPy Incompatibility

**Discovery**: ICA method failed on traits with BFloat16 activations due to NumPy not supporting BFloat16 dtype.

**Root Cause**:
- `traitlens/methods.py:132` attempted `combined.cpu().numpy()` on BFloat16 tensors
- NumPy throws: `TypeError: Got unsupported ScalarType BFloat16`
- Affected traits: curiosity, confidence_doubt, defensiveness, enthusiasm

**Fix**:
```python
# OLD: combined_np = combined.cpu().numpy()
# NEW: combined_np = combined.float().cpu().numpy()
```

**Impact**: ICA extraction now works on all dtype formats (float16, bfloat16, float32)

**Related**: Similar fix was applied to Gradient method on Nov 17 (documented in numerical_stability_analysis.md)

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

## 2025-11-20: Behavioral Safety Metrics for Non-CoT Models

**Finding**: Token-by-token trait monitoring measures safety alignment quality in instruction-tuned models without requiring chain-of-thought traces.

**Context**: Google found "substantial propensity for strategic deception" in Gemini 3 Pro but deemed it safe because it "isn't competent enough yet." This raises: how do you measure genuine vs superficial safety alignment?

**Refusal Quality Metrics** (works on Gemma 2B IT, no CoT needed):
- **Early commitment** = genuine safety (crystallizes in first tokens)
- **Late commitment** = instruction-following (appears after processing full prompt)
- **High persistence** = internalized (holds throughout response)
- **Low persistence** = superficial (fades during generation)

**Evaluation Context Sensitivity**:
- Extract `evaluation_context_sensitivity` trait
- Compare: eval-formatted vs natural prompts
- Detects behavioral (not conscious) awareness of being tested

**Cross-Context Generalization**:
- Test which layers maintain trait separation across contexts
- Validates Google's finding: middle layers generalize, late layers don't
- Late layer dependence = instruction-following artifact

**Advantage**: Hidden states can't lie (no reasoning trace to manipulate). Works on any model architecture. Measures behavioral tendencies, not stated reasoning.

**Limitations**: Cannot detect explicit scheming or validate CoT legibility. Measures behavioral signatures, not conscious reasoning.

**Next Steps**: Build refusal dynamics dashboard, extract evaluation context sensitivity trait, run cross-context generalization study.

**Details**: Natural elicitation is critical - instruction-based extraction confounds safety behavior with instruction-following.

---

## 2025-11-21: Linear Representations Emerge from Near-Linear Activations

**Finding**: Linear representation hypothesis holds in middle layers (8-16) due to GELU/SiLU activation functions operating in near-linear regime, explaining extraction success and layer-dependent generalization.

**Evidence**:
- GELU is approximately linear for x ∈ [-1, 1] (near-identity function)
- Repeated composition of near-linear functions preserves linearity for ~16 layers (Jiang et al. 2024)
- Layers 0-16: Cross-distribution generalization works (96%+ accuracy)
- Layers 21-25: Nonlinearity accumulates, cross-distribution fails (~56% accuracy)
- Steering works via addition: f(x + δ) ≈ f(x) + f(δ) holds in middle layers

**Trait Complexity Hierarchy**:
- **Simple traits (truly 1-D)**: refusal, emotional_valence, temporal_focus - high separation (90-96%), all methods work
- **Complex traits (linear projections)**: paranoia_trust, power_dynamics, sycophancy - medium separation (70-85%), probes find dominant component
- **Circular/multi-dimensional**: days of week, cyclical patterns - not captured by 1-D extraction

**Known Limitations**:
1. **Euclidean inner product**: Using `v·w` instead of causal inner product `v^T Cov(γ)^(-1) w` (Park et al. 2023)
2. **1-D assumption**: Misses circular features and multi-dimensional concepts (Engels et al. 2024)
3. **Untested composition**: Late-layer traits may/may not compose linearly from early-layer features

**Mechanistic Picture**:
- **Layers 0-8**: Building blocks (syntax, semantics), activations in linear regime
- **Layers 8-16**: Goldilocks zone - compositional concepts, still linear, steering works
- **Layers 16-25**: Nonlinear specialization, task-specific circuits (instruction vs natural modes diverge)

**Implication**: Linear assumption is architecturally justified for layers 8-16. Extract and steer in this range for robust, generalizable trait vectors. Cross-distribution failure in late layers explained by nonlinearity accumulation, not invalid methodology.

**Reference**: Jiang et al. (2024) "On the origins of linear representations in LLMs", Park et al. (2023) "Linear representation hypothesis and geometry of LLMs"

---

## Future Findings

Additional insights will be added here as research progresses.
