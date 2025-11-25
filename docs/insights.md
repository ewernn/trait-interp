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

## 2025-11-21: Held-Out Validation Confirms Probe Generalization

**Finding**: Probe method generalizes well to held-out validation data with minimal overfitting (2.7% accuracy drop), while ICA fails completely.

**Evidence** (my_experiment, 10 traits, 20 val prompts each):

**Method Comparison (held-out validation accuracy):**
| Method | Train Acc | Val Acc | Drop | Polarity Correct |
|--------|-----------|---------|------|------------------|
| **Probe** | 100% | 85.8% | 2.7% | 97.7% |
| Gradient | 81.2% | 78.8% | -1.0% | 88.1% |
| Mean_diff | 81.4% | 70.9% | -0.3% | 80.8% |
| ICA | 48.2% | 47.6% | 0.1% | 45.0% |

**Best Layer (probe method, averaged across traits):**
| Layer | Val Accuracy |
|-------|--------------|
| 12 | **90.0%** |
| 17 | 89.7% |
| 14 | 89.3% |
| 16 | 89.0% |

**Per-Trait Best Vectors:**
| Trait | Best Vector | Val Acc | Cohen's d |
|-------|-------------|---------|-----------|
| confidence | probe_layer11 | 100% | 7.52 |
| defensiveness | probe_layer2 | 100% | 5.21 |
| uncertainty_expression | mean_diff_layer2 | 100% | 5.77 |
| context | probe_layer13 | 100% | 4.06 |
| retrieval | probe_layer1 | 97.5% | 3.79 |
| correction_impulse | probe_layer25 | 72.5% | 0.96 |
| search_activation | probe_layer8 | 70.0% | 1.09 |

**Cross-Trait Independence:**
- Diagonal mean: 89.0% (vectors work on their own trait)
- Off-diagonal mean: 48.1% (near chance on other traits)
- Independence score: 0.41 (good separation)

**Key Insights**:
1. **Probe generalizes**: Only 2.7% accuracy drop from train to validation
2. **Middle layers confirmed**: Layers 12-17 optimal on held-out data
3. **ICA useless**: 47.6% validation accuracy = worse than random, 45% polarity correct
4. **Two weak traits**: correction_impulse (72.5%) and search_activation (70.0%) need redesign
5. **Traits are independent**: Off-diagonal ~50% confirms vectors don't interfere

**Implication**: Previous 100% probe accuracy was training set performance. Real generalization is 85.8% - still best method, but not perfect. Use layers 12-17 for best held-out performance.

**Verification**:
```bash
python analysis/evaluate_on_validation.py --experiment my_experiment
```

---

## 2025-11-21: Impact of Normalization on Vector Evaluation

**Discovery**: Initial vector evaluation using raw dot products was confounded by significant variations in vector and activation norms across different extraction methods and model layers. This made direct comparisons of separation strength (and in some cases, accuracy) unreliable.

**Root Cause**:
- **Vector Norms:** Mean Difference vectors showed exploding norms in later layers (up to ~480), while Probe vectors decreased (from ~3.5 to ~0.4) due to regularization. Gradient vectors were unit-normalized (1.0).
- **Activation Norms:** Activations themselves showed a ~12x increase in magnitude from early (layer 0) to late (layer 24) layers.

**Fix**: Updated `analysis/evaluate_on_validation.py` to use **cosine similarity** by default for all projections. This involves normalizing both the trait vector and each activation vector to unit length *before* computing the dot product.

**Impact**:
- **Fairer Comparisons**: All projection scores are now on a standardized scale (-1 to 1), making accuracy, separation, and effect size directly comparable across different methods and layers.
- **Robustness**: The evaluation is now insensitive to arbitrary scale differences introduced by extraction methods or the growing magnitude of activations across layers.
- **Result Stability**: While overall method rankings (Probe best, ICA useless) remained consistent, the optimal layer for specific traits sometimes shifted, confirming the need for normalization.

**Implication**: All future validation results, including those in the visualization dashboard, are computed using cosine similarity, ensuring a more rigorous and comparable evaluation of trait vectors.

**Verification**:
```bash
python analysis/evaluate_on_validation.py --experiment my_experiment --no-normalize
# Compare output to default (normalized) run
```

---

## 2025-11-21: High-Dimensional Geometry and Random Projections

**Finding**: In high-dimensional activation spaces (e.g., 2304 dimensions), random vectors are almost always nearly orthogonal. This property governs the behavior of trait vectors when applied to unrelated data.

**Evidence**:
- **Concentration of Measure**: In high dimensions, the probability mass of random vectors concentrates near the "equator" (orthogonal space) of a hypersphere.
- **Cosine Similarity Distribution**: The cosine similarity between two truly random unit vectors in `d` dimensions forms a very narrow distribution centered at 0, with variance `1/d`. For `d=2304`, this variance is extremely small (`~0.0004`).

**Impact on Trait Evaluation**:
- **Unrelated Projections Near Zero**: When a trait vector is applied to activations entirely unrelated to that trait, the resulting projection scores will cluster tightly around zero.
- **Near-Zero Separation**: This leads to a separation score (difference between mean positive and negative projections) very close to zero.
- **~50% Accuracy for Unrelated Traits**: A decision boundary (threshold) set at zero (or near zero) will result in approximately 50% classification accuracy for unrelated data. This is because roughly half of the projections will be positive and half negative.

**Implication**: The observed ~50% accuracy on off-diagonal elements in the cross-trait independence matrix is not an artifact or a sign of poor evaluation; it is the **expected geometric behavior** of trait vectors operating in a high-dimensional space when the data is truly unrelated to the trait. This confirms the validity of the cross-trait matrix as a measure of independence.

---

## 2025-11-24: Layer Dynamics and Trait Monitoring Methodology

**Finding**: Raw velocity metrics are meaningless due to ~12x magnitude scaling across layers. After normalization, the true dynamics pattern emerges with distinct computational phases.

### Magnitude Normalization is Mandatory

**Evidence**:
- Representation magnitude scales ~12x from layer 0 to layer 24
- Raw velocity explosion at layers 23-24 was an artifact of this scaling
- After normalization, true pattern emerges:
  - **Layers 0-7**: High dynamics (0.5-0.67) - "deciding trajectory"
  - **Layers 8-19**: Stable (0.31-0.37) - "actual computation"
  - **Layers 20-24**: Rising (0.47-0.64) - "output preparation"

**Implication**: Use cosine similarity for projections (already done), but also track **radial vs angular velocity separately**. Magnitude changes and direction changes represent different computations.

### Trait Emergence is Layer-Variable

Different traits emerge at different layers - there is no universal emergence pattern. Held-out validation showed optimal layers ranging from 2 (defensiveness) to 25 (correction_impulse).

**What's established**:
- Early layers (0-7) show high dynamics - trajectory is being decided
- Middle layers (8-19) are stable - core computation
- Late layers (20-24) show rising dynamics - output preparation

**What's NOT established**:
- Any fixed hierarchy of trait types (behavioral vs cognitive vs meta-cognitive)
- Assumption that "layer 16 is good enough for all traits"

**Implication**: Optimal layer must be determined empirically per-trait via held-out validation. Don't assume layer 16 universally.

### Trait-Dynamics Correlation Split

Data revealed two distinct groups with different monitoring requirements:

| Group | Traits | Correlation | Interpretation |
|-------|--------|-------------|----------------|
| High correlation (0.56-0.63) | defensiveness, correction_impulse, formality, context | Change WITH layer dynamics | "Output modulation" - HOW to express |
| Low correlation (0.01-0.17) | uncertainty_expression, retrieval | Evolve on own schedule | "Content determination" - WHAT to express |

**Implication**: May need different monitoring strategies for each group. High-correlation traits track layer dynamics; low-correlation traits need independent tracking.

### Causal Sequentiality in Trait Detection

Later tokens have dramatically more computation behind them:
- Token 50: 26 layers × 49 previous positions worth of information
- Token 1: Almost nothing

**Implications**:
1. Trait projections on early tokens are inherently noisier
2. Errors accumulate - if model "decides" wrong at token 5, it can't revise
3. Weight early-token trait scores with appropriate uncertainty

### Universality Testing

The 8 dynamic prompts are the primary defense against overfitting interpretations:
- **8/8 prompts**: Pattern is probably architectural
- **2/8 prompts**: Pattern is probably noise or prompt-specific

Small multiples grid is not just visualization - it's a **validity check**. Always test patterns across the full prompt set before drawing conclusions.

### Framing Corrections

**Transformers are NOT dynamical systems with attractors.** Better framing: "26 layers of constrained flow through learned manifold."

- No feedback loops
- No settling to fixed points
- Traits aren't attractors - they're **channels** carved by training that tokens flow through
- Commitment point = where a token picks its channel, not where it settles into an attractor

The river/flow and differential equation framings are useful intuition pumps but don't take literally. The "commitment point" framing is the most mechanistically grounded - there really does seem to be a layer where acceleration drops and the model has "decided."

### Steering vs Measurement Geometry

These are different operations in the same space:
- **Measurement**: `score = h · trait_vector` (dot product, scalar output)
- **Steering**: `h' = h + α * trait_vector` (addition, move in space)

"Project trait onto activations" doesn't make geometric sense - both vectors live in the same activation space.

### Attention Interpretation Caution

Low attention weight to early tokens ≠ "ignoring" that information. By layer 16, information from early tokens is already encoded in hidden states. Attention is routing, not the only information flow mechanism. Don't over-interpret attention diffusion patterns.

### Commitment Point: Promising but Unvalidated

The commitment point detection heuristic (`where(|trait_velocity| < threshold)`) is a reasonable starting point but **not yet validated**:

**Open questions**:
- Is it robust across different traits? (Different traits emerge at different layers)
- Is it robust across different prompts? (Needs 8/8 universality testing)
- Is the threshold trait-dependent or universal?
- Does it correlate with actual behavioral commitment in generation?

**What's established**:
- Acceleration peaks at layer 23 are real (post-normalization)
- The general approach (tracking velocity/acceleration) is methodologically sound

**What's NOT established**:
- That a single "commitment point" exists for all traits
- That the current threshold is optimal
- That commitment point predicts anything about generation behavior

**Implication**: Treat commitment point as a hypothesis to test, not an established metric. Validate per-trait across the full prompt set before relying on it.

---

## Future Findings

Additional insights will be added here as research progresses.