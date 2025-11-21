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

---

## Mechanistic Understanding

### 3. Hierarchical Composition Testing

**Method**: Use Ridge regression to predict late-layer trait from early-layer trait projections.

**Test**: Can layer 16 refusal be predicted from layer 8 subfeatures (harm, uncertainty, instruction)?

**Impact**: Validates linear composition hypothesis, enables interpretable trait hierarchies.

### 4. SAE Feature Decomposition

**Method**: Project trait vectors into SAE feature space (16k features from GemmaScope).

**Test**: Which interpretable features contribute most to each trait vector?

**Impact**: Mechanistic decomposition (e.g., "Refusal = 0.5×harm_detection + 0.3×uncertainty + 0.2×instruction_boundary").

### 5. Cross-Model Validation

**Method**: Extract same traits on Llama 3.1 8B, Mistral, other architectures.

**Test**: Do traits transfer? Are layer numbers absolute or relative? Does middle-layer optimum generalize?

**Impact**: Tests whether findings are Gemma-specific or universal to transformer architectures.

---

## Causal Validation

### 6. Layer-Specific Trait Localization

**Method**: Run interchange interventions at all layers (0-26) for each trait.

**Test**: Find which layers actually mediate behavior when patched.

**Impact**: Identifies optimal extraction layer per trait, validates layer-dependent trait localization.

### 7. Component Ablation (Attention vs MLP)

**Method**: Patch QK, VO, and MLP components separately during interchange.

**Test**: Which architectural component actually mediates each trait?

**Impact**: Validates whether traits are "attention structure" vs residual stream features.

### 8. Cross-Trait Interference

**Method**: Patch two traits simultaneously with opposite signs (+refusal/-confidence).

**Test**: Do traits compose additively or interfere?

**Impact**: Validates trait orthogonality and correlation matrix predictions.

### 9. Temporal Causality Decay

**Method**: Patch at token T, measure effect at T+1, T+5, T+10, T+20.

**Test**: Quantify persistence window and decay rate per trait.

**Impact**: Tests KV cache propagation hypothesis, validates attention-based persistence claims.

### 10. Trait Correlation Matrix

**Method**: Compute pairwise correlations between all trait vector projections on shared prompts.

**Test**: Which traits are independent vs measuring same computation?

**Impact**: Quantifies confounding structure, validates whether framework is coherent or over-partitioned.

---

## References

- Park et al. (2023): "Linear representation hypothesis and geometry of LLMs"
- Jiang et al. (2024): "On the origins of linear representations in LLMs"
- Engels et al. (2024): "Not all language model features are one-dimensionally linear"
- Turner et al. (2023): "Steering language models with activation engineering"
