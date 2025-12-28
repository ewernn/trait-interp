# core/ Quick Reference

Primitives for trait vector extraction and analysis.

---

## Hooks

```python
from core import CaptureHook, MultiLayerCapture, SteeringHook, get_hook_path

# Capture from one layer
with CaptureHook(model, "model.layers.16") as hook:
    model(**inputs)
activations = hook.get()  # [batch, seq, hidden]

# Capture from multiple layers
with MultiLayerCapture(model, layers=[14, 15, 16]) as capture:
    model(**inputs)
acts = capture.get(16)
all_acts = capture.get_all()  # {14: tensor, 15: tensor, 16: tensor}

# Capture all layers
with MultiLayerCapture(model) as capture:  # layers=None = all
    model(**inputs)

# Steer generation
vector = torch.load('vectors/probe_layer16.pt')
with SteeringHook(model, vector, "model.layers.16", coefficient=1.5):
    output = model.generate(**inputs)

# Path helper (layer + component -> string)
get_hook_path(16)                    # "model.layers.16"
get_hook_path(16, "attn_out")        # "model.layers.16.self_attn.o_proj"
get_hook_path(16, "mlp_out")         # "model.layers.16.mlp.down_proj"
# Components: residual, attn_out, mlp_out, k_cache, v_cache
```

---

## Extraction Methods

```python
from core import get_method

method = get_method('probe')  # or 'mean_diff', 'gradient', 'random_baseline'
result = method.extract(pos_acts, neg_acts)
vector = result['vector']
```

**Available methods** (all return unnormalized vectors):
- `mean_diff` - Baseline: `vector = mean(pos) - mean(neg)`
- `probe` - Logistic regression weights
- `gradient` - Gradient optimization to maximize separation
- `random_baseline` - Random vector (sanity check, ~50% accuracy)

---

## Math Functions

```python
from core import projection, batch_cosine_similarity, cosine_similarity, orthogonalize

# Project activations onto vector (normalizes vector only)
scores = projection(activations, trait_vector)  # [n_samples]

# Cosine similarity (normalizes both activations and vector)
scores = batch_cosine_similarity(activations, trait_vector)  # [n_samples] in [-1, 1]

# Compare two vectors
similarity = cosine_similarity(refusal_vec, evil_vec)  # scalar in [-1, 1]

# Remove one vector's component from another
clean_vec = orthogonalize(trait_vector, confound_vector)
```

**Metrics (operate on projection scores):**
```python
from core import separation, accuracy, effect_size, p_value, polarity_correct

# First compute projections
pos_proj = batch_cosine_similarity(pos_acts, vector)
neg_proj = batch_cosine_similarity(neg_acts, vector)

# Then compute metrics
sep = separation(pos_proj, neg_proj)                  # Higher = better
acc = accuracy(pos_proj, neg_proj)                    # 0.0 to 1.0
d = effect_size(pos_proj, neg_proj)                   # 0.2=small, 0.5=medium, 0.8=large
d = effect_size(pos_proj, neg_proj, signed=True)      # Preserve sign (pos > neg = positive)
p = p_value(pos_proj, neg_proj)                       # Lower = significant
```

**Vector/distribution analysis:**
```python
from core import vector_properties, distribution_properties

# Vector properties
props = vector_properties(vector)  # {norm, sparsity}

# Distribution properties (for projection scores)
dist = distribution_properties(pos_proj, neg_proj)
# {pos_std, neg_std, overlap_coefficient, separation_margin}
```

---

## Files

```
core/
├── __init__.py      # Public API exports
├── hooks.py         # get_hook_path, CaptureHook, SteeringHook, MultiLayerCapture, HookManager
├── methods.py       # Extraction methods (probe, mean_diff, gradient)
└── math.py          # projection, batch_cosine_similarity, metrics, vector/distribution properties
```
