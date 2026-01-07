# core/ Quick Reference

Primitives for trait vector extraction and analysis.

---

## Hooks

```python
from core import CaptureHook, MultiLayerCapture, SteeringHook, get_hook_path, detect_contribution_paths

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

# Contribution components (auto-detect architecture)
get_hook_path(16, "attn_contribution", model=model)
# Gemma-2: "model.layers.16.post_attention_layernorm"
# Others:  "model.layers.16.self_attn.o_proj"

# Components: residual, attn_out, mlp_out, attn_contribution*, mlp_contribution*, k_proj, v_proj
# *contribution components require model parameter
```

**Architecture detection:**
```python
from core import detect_contribution_paths

paths = detect_contribution_paths(model)
# Gemma-2: {'attn_contribution': 'post_attention_layernorm', 'mlp_contribution': 'post_feedforward_layernorm'}
# Others:  {'attn_contribution': 'self_attn.o_proj', 'mlp_contribution': 'mlp.down_proj'}
```

---

## Extraction Methods

```python
from core import get_method

method = get_method('probe')  # or 'mean_diff', 'gradient', 'random_baseline'
result = method.extract(pos_acts, neg_acts)
vector = result['vector']
```

**Available methods** (all return unit-normalized vectors):
- `mean_diff` - Baseline: `vector = mean(pos) - mean(neg)`, then normalized
- `probe` - Logistic regression on row-normalized activations, then normalized
- `gradient` - Gradient optimization to maximize separation, normalized
- `random_baseline` - Random unit vector (sanity check, ~50% accuracy)

**Note:** All vectors are unit-normalized for consistent steering coefficients across models.
Probe uses row normalization (each sample scaled to unit norm) so LogReg coefficients are ~1 magnitude regardless of model activation scale.

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

## Massive Activations

Certain dimensions have values 100-1000x larger than median (Sun et al. 2024). These create fixed biases in projections.

**Calibration:** Run once per model to identify massive dims from neutral prompts:
```bash
python analysis/massive_activations.py --experiment gemma-2-2b
```

This uses a calibration dataset (50 Alpaca prompts) and saves results to `experiments/{exp}/inference/massive_activations/calibration.json`. The projection script embeds this data for interactive cleaning in the visualization.

**Research mode:** Analyze a specific prompt set:
```bash
python analysis/massive_activations.py --experiment gemma-2-2b --prompt-set jailbreak_subset --per-token
```

**Visualization:** The Trait Dynamics view has a "Clean" dropdown with options:
- "No cleaning" — Raw projections
- "Top 5, 3+ layers" — Dims in top-5 at 3+ layers (recommended)
- "All candidates" — All massive dims

---

## Files

```
core/
├── __init__.py      # Public API exports
├── hooks.py         # get_hook_path, detect_contribution_paths, CaptureHook, SteeringHook, MultiLayerCapture, HookManager
├── methods.py       # Extraction methods (probe, mean_diff, gradient)
└── math.py          # projection, batch_cosine_similarity, metrics, vector/distribution properties
```
