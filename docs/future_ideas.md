# Future Ideas

Research extensions and technical improvements identified through experimentation.

---

## Priority List

### 🔴 High Priority
1. Implement causal inner product (Park et al. 2023)
2. Add activation range diagnostics (verify near-linear regime)
3. Document layer selection rationale based on activation functions

### 🟡 Medium Priority
4. Test hierarchical composition hypothesis (layer 8 features → layer 16 traits)
5. Add non-linearity detection to validation pipeline (PCA variance check)
6. Document trait complexity tiers (simple/complex/circular)

### 🟢 Low Priority
7. Explore multi-dimensional extraction for circular/complex traits
8. Expand SAE integration for feature decomposition
9. Test framework on additional models (Llama 3.1, Claude, etc.)

---

## Details

### 1. Causal Inner Product

**Problem**: Currently using Euclidean inner product `v·w`, but Park et al. (2023) proved causal inner product is more appropriate.

**Implementation**:
```python
# traitlens/compute.py
def causal_inner_product(v1, v2, model, tokenizer):
    """Compute v1^T Cov(γ)^(-1) v2 where γ = uniform vocab sample."""
    vocab_embeddings = model.get_output_embeddings().weight
    cov = torch.cov(vocab_embeddings.T)
    M = torch.linalg.inv(cov)
    return v1 @ M @ v2
```

**Impact**: More principled geometry, better semantic structure preservation.

### 2. Activation Range Diagnostics

**Problem**: Linear approximation only valid when activations in near-linear regime of GELU (x ∈ [-1, 1]).

**Implementation**:
```python
def check_linearity_regime(activations, layer_name):
    """Flag if activations outside linear regime."""
    mean, std = activations.mean(), activations.std()
    if mean.abs() > 2 or std > 2:
        print(f"⚠️  {layer_name}: Outside linear regime")
```

**Impact**: Validates when linear methods are appropriate, explains layer-dependent performance.

### 3. Hierarchical Composition Testing

**Question**: Do late-layer traits compose linearly from early-layer features?

**Test**:
```python
# Can layer 16 refusal be predicted from layer 8 subfeatures?
from sklearn.linear_model import Ridge

X = stack([
    projection(acts_l8, harm_vector_l8),
    projection(acts_l8, uncertainty_vector_l8),
    projection(acts_l8, instruction_vector_l8),
])
y = projection(acts_l16, refusal_vector_l16)

r2 = Ridge().fit(X, y).score(X, y)
# High R² → linear composition holds
# Low R² → nonlinear emergence
```

**Impact**: Validates mechanistic understanding, enables interpretable trait hierarchies.

### 4. Non-Linearity Detection

**Problem**: 1-D extraction misses circular features (days of week) and multi-dimensional concepts.

**Implementation**:
```python
def test_linearity(pos_acts, neg_acts):
    """Test if feature is truly 1-D linear."""
    pca = PCA(n_components=5)
    pca.fit(concat(pos, neg))
    # If first component < 90% variance → likely multi-dimensional
    return pca.explained_variance_ratio_[0]
```

**Impact**: Flag traits needing multi-dimensional extraction methods.

### 5. Multi-Dimensional Extraction

**Problem**: Some traits may be irreducibly multi-dimensional (Engels et al. 2024).

**Approach**: Extend extraction methods to output k-dimensional subspaces instead of 1-D vectors.

**Impact**: Capture circular features, multi-axis concepts (political ideology = economic + social).

### 6. SAE Feature Decomposition

**Current**: SAE integration in `sae/` directory, but not deeply integrated with extraction.

**Extension**: Use SAE features to decompose trait vectors into interpretable components.

**Impact**: "Refusal = 0.5×harm_detection + 0.3×uncertainty + 0.2×instruction_boundary"

### 7. Cross-Model Validation

**Current**: 38 traits on Gemma 2B only.

**Extension**: Test on Llama 3.1 8B, Claude models, Mistral.

**Questions**:
- Do traits transfer across architectures?
- Are layer numbers architecture-dependent or relative?
- Does the Goldilocks zone (layers 8-16) generalize?

---

## References

- Park et al. (2023): "Linear representation hypothesis and geometry of LLMs" - Causal inner product theory
- Jiang et al. (2024): "On the origins of linear representations" - Near-linear activation explanation
- Engels et al. (2024): "Not all language model features are one-dimensionally linear" - Multi-dimensional critique
- Turner et al. (2023): "Steering language models with activation engineering" - Practical steering validation
