# Future Ideas

Research extensions and technical improvements identified through experimentation.

---

## Validation & Diagnostics

### 1. Activation Range Diagnostics

**Method**: Check if activations fall within near-linear regime of GELU (x ∈ [-1, 1]).

**Test**: Flag layers where mean.abs() > 2 or std > 2 during extraction.

**Impact**: Validates when linear methods are appropriate, explains layer-dependent performance.

### 2. Non-Linearity Detection

**Method**: Run PCA on concatenated pos/neg activations, check first component variance ratio.

**Test**: If first component < 90% variance → trait is multi-dimensional.

**Impact**: Flags traits that need different extraction methods or are irreducibly complex.

### 3. CKA for Method Agreement

**Method**: Use Centered Kernel Alignment (CKA) to compare the similarity of vector spaces from different extraction methods (mean_diff, probe, ICA, gradient).

**Test**: Compute CKA score between vector sets (e.g., `cka(gradient_vectors, probe_vectors)`). A high score (>0.7) indicates methods find similar representations.

**Impact**: Validates that high-performing methods are converging on the same underlying trait structure, rather than just overfitting to the training data in different ways.

### 4. Cross-Layer Representation Similarity

**Method**: Compute the cosine similarity of a trait vector extracted from every pair of layers.

**Test**: Plot a heatmap of pairwise cosine similarities. If `cosine(vector_L10, vector_L16)` is high, the representation is stable.

**Impact**: Identifies the layer range where a trait representation is most stable and consistently represented (a "block" of high similarity). This provides a data-driven way to select the optimal extraction layer.

### 5. Holistic Vector Ranking

**Method**: Define a sophisticated ranking system to select the single "best" vector for a trait across all methods and layers, moving beyond single-metric sorting.

**Axes of Quality to Consider**:
- **Accuracy (`val_accuracy`):** Primary measure of correctness.
- **Robustness (`val_effect_size`):** Cleaner separation, less overlap between distributions.
- **Generalization (`accuracy_drop`):** Lower train-to-validation accuracy drop indicates less overfitting.
- **Specificity (Cross-Trait Independence):** Low accuracy on unrelated traits indicates a "purer" vector.

**Proposed Ranking Systems**:
- **A) Simple Tie-breaker:** Rank all 104 vectors by `val_accuracy`, then use `val_effect_size` as a secondary sort key.
- **B) Composite Quality Score:** Create a weighted score `(w1 * accuracy) + (w2 * effect_size) - (w3 * accuracy_drop)` to produce a single, holistic sortable value.
- **C) Pareto Frontier:** Identify the set of non-dominated vectors that represent optimal trade-offs between the different quality axes (e.g., highest accuracy vs. highest effect size).

**Impact**: Provides a more robust and nuanced method for identifying the most useful trait vectors, preventing suboptimal choices based on a single metric and surfacing vectors with different desirable properties.

---

## Mechanistic Understanding

### 5. Hierarchical Composition Testing

**Method**: Use Ridge regression to predict late-layer trait from early-layer trait projections.

**Test**: Can layer 16 refusal be predicted from layer 8 subfeatures (harm, uncertainty, instruction)?

**Impact**: Validates linear composition hypothesis, enables interpretable trait hierarchies.

### 6. SAE Feature Decomposition

**Method**: Project trait vectors into SAE feature space (16k features from GemmaScope).

**Test**: Which interpretable features contribute most to each trait vector?

**Impact**: Mechanistic decomposition (e.g., "Refusal = 0.5×harm_detection + 0.3×uncertainty + 0.2×instruction_boundary").

### 7. Cross-Model Validation

**Method**: Extract same traits on Llama 3.1 8B, Mistral, other architectures.

**Test**: Do traits transfer? Are layer numbers absolute or relative? Does middle-layer optimum generalize?

**Impact**: Tests whether findings are Gemma-specific or universal to transformer architectures.

---

## Causal Validation

### 8. Layer-Specific Trait Localization

**Method**: Run interchange interventions at all layers (0-26) for each trait.

**Test**: Find which layers actually mediate behavior when patched.

**Impact**: Identifies optimal extraction layer per trait, validates layer-dependent trait localization.

### 9. Component Ablation (Attention vs MLP)

**Method**: Patch QK, VO, and MLP components separately during interchange.

**Test**: Which architectural component actually mediates each trait?

**Impact**: Validates whether traits are "attention structure" vs residual stream features.

### 10. Cross-Trait Interference

**Method**: Patch two traits simultaneously with opposite signs (+refusal/-confidence).

**Test**: Do traits compose additively or interfere?

**Impact**: Validates trait orthogonality and correlation matrix predictions.

### 11. Temporal Causality Decay

**Method**: Patch at token T, measure effect at T+1, T+5, T+10, T+20.

**Test**: Quantify persistence window and decay rate per trait.

**Impact**: Tests KV cache propagation hypothesis, validates attention-based persistence claims.

### 12. Linearized Causal Validation

**Method**: During interchange interventions (patching), compare the effect of a regular forward pass vs. a linearized forward pass where the downstream network is treated as a frozen linear approximation.

**Test**: Intervene with and without linearization and measure the difference in behavioral change. `effect_regular = intervene(linearize=False)`, `effect_linearized = intervene(linearize=True)`.

**Impact**: Determines if a vector's causal effect is a first-order phenomenon or depends on non-linear network dynamics. Provides cleaner causal attribution, answering: "Does this vector *linearly* cause the behavior?"

### 13. Trait Correlation Matrix

**Method**: Compute pairwise correlations between all trait vector projections on shared prompts.

**Test**: Which traits are independent vs measuring same computation?

**Impact**: Quantifies confounding structure, validates whether framework is coherent or over-partitioned.

---

## References

- Park et al. (2023): "Linear representation hypothesis and geometry of LLMs"
- Jiang et al. (2024): "On the origins of linear representations in LLMs"
- Engels et al. (2024): "Not all language model features are one-dimensionally linear"
- Turner et al. (2023): "Steering language models with activation engineering"


