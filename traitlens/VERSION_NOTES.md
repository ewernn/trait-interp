# traitlens v0.2

traitlens is complete and functional. It provides minimal, composable primitives for trait vector extraction and analysis.

## Core Components (750 lines total)

1. **hooks.py** (~116 lines)
   - `HookManager` class for managing forward hooks
   - Context manager support for automatic cleanup
   - Module path navigation (e.g., "model.layers.16.self_attn")

2. **activations.py** (~161 lines)
   - `ActivationCapture` class for storing activations
   - `make_hook()` factory method for creating hook functions
   - Memory usage tracking
   - Batch concatenation support

3. **compute.py** (~222 lines)
   - `mean_difference()` - Compute mean difference
   - `compute_derivative()` - Velocity of trait expression
   - `compute_second_derivative()` - Acceleration
   - `projection()` - Project onto trait vectors
   - `cosine_similarity()` - Compare vectors
   - `normalize_vectors()` - Unit normalization

4. **methods.py** (~347 lines)
   - `ExtractionMethod` - Abstract base class
   - `MeanDifferenceMethod` - Simple baseline extraction
   - `ICAMethod` - Independent component analysis
   - `ProbeMethod` - Linear probe via logistic regression
   - `GradientMethod` - Gradient-based optimization
   - `get_method()` - Factory function

5. **__init__.py** (~50 lines)
   - Clean exports of all public functions

## Usage Example

```python
from traitlens import HookManager, ActivationCapture, ProbeMethod

# Capture activations
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
capture = ActivationCapture()

with HookManager(model) as hooks:
    hooks.add_forward_hook("model.layers.16", capture.make_hook("layer_16"))
    outputs = model.generate(...)

# Extract trait vector with any method
method = ProbeMethod()  # or MeanDifferenceMethod(), ICAMethod(), GradientMethod()
result = method.extract(pos_activations, neg_activations)
trait_vector = result['vector']
```

## Philosophy Achieved

✅ **Minimal** - Just 5 files, ~750 lines
✅ **Composable** - Primitives combine naturally
✅ **Unopinionated** - Multiple extraction methods, you choose
✅ **Transparent** - You see exactly what happens
✅ **Model-agnostic** - Works with any PyTorch model

## Testing

✅ All unit tests pass (`test_basic.py`, `test_methods.py`)
- Hook manager functionality
- Activation capture and storage
- Compute functions
- All extraction methods (mean_diff, gradient)
- ICA and Probe methods (with scikit-learn)

## Requirements

- PyTorch
- transformers (for examples only)
- scikit-learn (optional, for ICA and Probe methods)

---

Built: November 2024
Lines of code: ~750
