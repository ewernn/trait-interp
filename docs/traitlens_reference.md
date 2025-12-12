# traitlens Quick Reference

**traitlens** is the extraction toolkit powering trait-interp. It's installed as a pip package (`pip install git+https://github.com/ewernn/traitlens.git`) and provides primitives for trait vector extraction and analysis.

**Full documentation:** `/Users/ewern/code/trait-stuff/traitlens/docs/API.md`

---

## Quick Discovery

```bash
# List all available functions
python -c "import traitlens; print('\n'.join(traitlens.__all__))"

# Get help on any function
python -c "import traitlens; help(traitlens.projection)"

# Interactive exploration
python -c "import traitlens; help(traitlens)"
```

---

## Core Classes (2)

```python
from traitlens import HookManager, ActivationCapture

# Capture activations from any layer
capture = ActivationCapture()
with HookManager(model) as hooks:
    hooks.add_forward_hook("model.layers.16", capture.make_hook("layer_16"))
    output = model.generate(**inputs)

activations = capture.get("layer_16")  # [batch, seq_len, hidden_dim]
```

---

## Extraction Methods (5)

```python
from traitlens import MeanDifferenceMethod, ProbeMethod, GradientMethod, ICAMethod, get_method

# Factory pattern
method = get_method('probe')  # or 'mean_diff', 'gradient', 'ica', 'random_baseline'
result = method.extract(pos_acts, neg_acts)
vector = result['vector']

# Direct instantiation
method = ProbeMethod()
result = method.extract(pos_acts, neg_acts, penalty='l1')  # Sparse probe
vector = result['vector']
train_acc = result['train_acc']
```

**Methods:**
- `MeanDifferenceMethod` - Baseline: `vector = mean(pos) - mean(neg)`
- `ProbeMethod` - Logistic regression (best accuracy)
- `GradientMethod` - Gradient optimization (best generalization)
- `ICAMethod` - Independent Component Analysis (disentangle confounds)
- `get_method(name)` - Factory function

---

## Compute Functions (13)

### Vector Operations
```python
from traitlens import projection, cosine_similarity, normalize_vectors, magnitude

# Project activations onto vector
scores = projection(activations, trait_vector)

# Compare vectors
similarity = cosine_similarity(refusal_vec, evil_vec)

# Normalize to unit length
normalized = normalize_vectors(vectors)

# Compute magnitude
mags = magnitude(hidden_states)
```

### Temporal Dynamics
```python
from traitlens import compute_derivative, compute_second_derivative, radial_velocity, angular_velocity

# Velocity (rate of change)
velocity = compute_derivative(trajectory)

# Acceleration (commitment points)
acceleration = compute_second_derivative(trajectory)
commitment_point = (acceleration.norm(dim=-1) < threshold).nonzero()[0]

# Magnitude change
radial_vel = radial_velocity(trajectory)  # Positive = growing, negative = shrinking

# Direction change
angular_vel = angular_velocity(trajectory)  # 0 = same direction, 2 = opposite
```

### Utilities
```python
from traitlens import mean_difference, pca_reduce, attention_entropy

# Extract trait vector (baseline)
vector = mean_difference(pos_acts, neg_acts)

# Dimensionality reduction
reduced = pca_reduce(hidden_states, n_components=2)

# Attention analysis
entropy = attention_entropy(attention_weights)  # Higher = more diffuse
```

---

## Evaluation Metrics (16)

### Separation Metrics
```python
from traitlens import separation, accuracy, effect_size

sep = separation(pos_proj, neg_proj)       # Higher = better
acc = accuracy(pos_proj, neg_proj)         # 0.0 to 1.0
d = effect_size(pos_proj, neg_proj)        # 0.2=small, 0.5=medium, 0.8=large
```

### Statistical Metrics
```python
from traitlens import p_value, polarity_correct

p = p_value(pos_proj, neg_proj)            # Lower = significant
if not polarity_correct(pos_proj, neg_proj):
    vector = -vector                        # Flip polarity
```

### Stability Metrics
```python
from traitlens import bootstrap_stability, noise_robustness, subsample_stability

stability = bootstrap_stability(vector, pos_acts, neg_acts)
robustness = noise_robustness(vector, pos_acts, neg_acts)
subsample = subsample_stability(vector, pos_acts, neg_acts)
```

### Vector Properties
```python
from traitlens import sparsity, effective_rank, top_k_concentration

sparse_frac = sparsity(vector)             # Fraction of near-zero components
rank = effective_rank(vector)              # Effective dimensionality
concentration = top_k_concentration(vector, k=10)  # Fraction in top k dims
```

### Cross-Vector Metrics
```python
from traitlens import orthogonality, cross_trait_accuracy

ortho = orthogonality(vectors)             # Lower = more independent
acc = cross_trait_accuracy(vec, other_pos, other_neg)  # Should be ~0.5
```

### Convenience Functions
```python
from traitlens import evaluate_vector, evaluate_vector_properties

# All separation + statistical metrics
metrics = evaluate_vector(vector, pos_acts, neg_acts)
# Returns: {separation, accuracy, effect_size, p_value, polarity_correct}

# All vector properties + stability
props = evaluate_vector_properties(vector, pos_acts, neg_acts)
# Returns: {sparsity, effective_rank, top_k_concentration, bootstrap_stability, ...}
```

---

## Common Patterns

### Extract Vector
```python
from traitlens import ProbeMethod

method = ProbeMethod()
result = method.extract(pos_acts, neg_acts)
vector = result['vector']
```

### Monitor During Generation
```python
from traitlens import HookManager, projection

capture = ActivationCapture()
with HookManager(model) as hooks:
    hooks.add_forward_hook("model.layers.16", capture.make_hook("layer_16"))
    output = model.generate(**inputs)

acts = capture.get("layer_16")
scores = projection(acts, trait_vector)  # Trait expression per token
```

### Analyze Dynamics
```python
from traitlens import compute_derivative, compute_second_derivative

# Velocity and acceleration
velocity = compute_derivative(trajectory)
acceleration = compute_second_derivative(trajectory)

# Find commitment point
accel_magnitude = acceleration.norm(dim=-1)
commitment_idx = (accel_magnitude < threshold).nonzero()[0]
```

### Evaluate Quality
```python
from traitlens import evaluate_vector

metrics = evaluate_vector(vector, pos_acts, neg_acts)
print(f"Accuracy: {metrics['accuracy']:.1%}")
print(f"Effect size: {metrics['effect_size']:.2f}")
print(f"p-value: {metrics['p_value']:.2e}")
```

---

## Installation

```bash
# Editable install (for development)
cd /Users/ewern/code/trait-stuff/traitlens
pip install -e .

# From GitHub
pip install git+https://github.com/ewernn/traitlens.git
```

**Current version:** 0.4.0
**Location:** `/Users/ewern/code/trait-stuff/traitlens`

---

## See Also

- **Full API docs:** `/Users/ewern/code/trait-stuff/traitlens/docs/API.md`
- **README:** `/Users/ewern/code/trait-stuff/traitlens/README.md`
- **Philosophy:** `/Users/ewern/code/trait-stuff/traitlens/docs/philosophy.md`
- **Math:** `/Users/ewern/code/trait-stuff/traitlens/docs/math.md`
