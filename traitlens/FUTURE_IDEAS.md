# Future Ideas for traitlens

This document captures ideas discussed but not implemented in v0.1. These may become features, examples, or recipes.

## Potential Core Additions

### Advanced Hook Types
- **Backward hooks** - For gradient capture
  ```python
  hooks.add_backward_hook("model.layers.16", capture.make_hook("gradients"))
  ```
- **Modification hooks** - For steering/intervention
  ```python
  hooks.add_modification_hook("model.layers.16", lambda x: x + vector)
  ```

### Batch Processing
```python
class BatchExtractor:
    """Efficiently extract from multiple prompts"""
    def extract_batch(self, prompts, hook_locations, batch_size=8):
        # Parallel processing for efficiency
```

### Attention-Based Operations
```python
def attention_weighted_mean(activations, attention_weights):
    """Weight tokens by how much future tokens attend to them"""
    # Useful for finding "important" tokens
```

### Statistical Extractors
```python
def extract_ica_components(activations_dict, n_components):
    """ICA for trait disentanglement"""
    # ✅ IMPLEMENTED in traitlens/methods.py as ICAMethod

def extract_probe_direction(pos_acts, neg_acts):
    """Logistic regression for optimal boundary"""
    # ✅ IMPLEMENTED in traitlens/methods.py as ProbeMethod

def extract_pca_direction(activations):
    """Principal component extraction"""
    # ❌ NOT YET IMPLEMENTED (consider adding as PCAMethod)
```

## Analysis Tools (Build on Top)

### Commitment Detection
```python
def find_commitment_point(trajectory, threshold=0.1):
    """Where does variance drop / model 'lock in'?"""
    velocity = compute_derivative(trajectory)
    variance = velocity.var(dim=-1)
    return (variance < threshold).nonzero()[0]
```

### Trait Correlation
```python
def trait_correlation_matrix(vectors_dict):
    """Measure confounding between traits"""
    correlations = {}
    for t1, v1 in vectors_dict.items():
        for t2, v2 in vectors_dict.items():
            correlations[f"{t1}_{t2}"] = cosine_similarity(v1, v2)
    return correlations
```

### Phase Transitions
```python
def detect_phase_transitions(trajectory, window=5):
    """Find sudden changes in activation patterns"""
    # Sliding window variance
    # Spike detection
```

### Decay Analysis
```python
def measure_decay_rate(trajectory, trait_vector):
    """How long does trait expression persist?"""
    projections = projection(trajectory, trait_vector)
    # Fit exponential decay
    # Return half-life
```

## Integration Ideas

### SAE Feature Decomposition
```python
def project_to_sae_features(vector, sae_model):
    """Decompose trait vector into SAE features"""
    # For use with GemmaScope or similar
    features = sae_model.encode(vector)
    return features
```

### Cross-Layer Analysis
```python
def layer_rotation(acts_layer1, acts_layer2):
    """Measure how much representations rotate between layers"""
    # Procrustes analysis or simpler metrics
```

### KV Cache Extraction
```python
# Special handling for key/value caches
def extract_from_kv_cache(model, layer):
    # Keys and values have different shapes/meanings
    # Might need special treatment
```

## Experimental Concepts

### Resonance Detection
Some prompts might "resonate" with internal structures:
```python
def find_resonant_prompts(model, prompt_variations):
    """Which prompts cause maximum activation variance?"""
    # Sweep through variations
    # Measure response diversity
```

### Attention Sink Detection
```python
def find_attention_sinks(attention_patterns):
    """Tokens that receive disproportionate attention"""
    incoming_attention = attention_patterns.sum(dim=-2)
    return incoming_attention.topk(k=5)
```

### Temporal Interpolation
```python
def interpolate_trajectory(trajectory, factor=2):
    """Smooth trajectory by interpolating between tokens"""
    # For better derivative estimation
```

## Model-Specific Helpers (Maybe Never Add)

These violate the "model-agnostic" principle but users keep asking:

```python
def get_gemma_residual_stream(layer):
    """Helper for Gemma models"""
    return f"model.layers.{layer}"

def get_llama_attention_out(layer):
    """Helper for Llama models"""
    return f"model.layers.{layer}.self_attn.o_proj"

# Maybe provide as documentation/examples instead?
```

## Performance Optimizations

### Hook Caching
```python
class CachedHookManager(HookManager):
    """Reuse hook handles across multiple forward passes"""
    # Avoid re-registering hooks
```

### Memory Management
```python
def streaming_capture(max_buffer_size=1000):
    """For very long sequences, stream to disk"""
    # Write activations to HDF5/zarr
```

### Multi-GPU Support
```python
def distributed_extraction(model, prompts, world_size=4):
    """Distribute extraction across GPUs"""
    # Using torch.distributed
```

## Integration with Existing Tools

### TransformerLens Backend
```python
class TLBackend:
    """Optional TransformerLens compatibility"""
    def __init__(self, tl_model):
        self.model = tl_model

    def add_hook(self, tl_hook_name, capture_fn):
        # Map to TL's hook system
```

### Captum Integration
```python
def to_captum_format(activations):
    """Convert for use with Captum interpretability"""
```

## Validation Suite

### Behavioral Validation
```python
def validate_steering(model, vector, test_prompts, coefficients):
    """Does applying vector actually change behavior?"""
    # Apply at different strengths
    # Measure behavioral change
```

### Cross-Method Comparison
```python
def compare_extraction_methods(methods_dict, test_data):
    """Which extraction method works best?"""
    # Compare separation, correlation, steering effectiveness
```

## Things We Decided NOT to Do

### ❌ Model Wrapping
- traitlens will never wrap models
- Direct model access only

### ❌ Built-in Analyses
- No logit lens, attention patterns, etc.
- Users build their own

### ❌ Standardization
- No unified hook naming across models
- Users specify exact paths

### ❌ Visualization
- No built-in plots or dashboards
- Export data, visualize elsewhere

---

**Note:** This document is a collection of ideas, not a roadmap. Most of these should be examples or user code, not core features. The goal is to keep traitlens minimal while documenting useful patterns.