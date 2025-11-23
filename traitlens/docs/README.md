# traitlens Documentation

## Overview

traitlens is a minimal toolkit for extracting and analyzing trait vectors from transformer language models. Unlike full frameworks, traitlens provides only the essential primitives—you build your extraction strategy.

## Quick Links

- [API Reference](api.md) - Complete API documentation
- [Philosophy](philosophy.md) - Design principles and comparisons
- [Recipes](recipes/) - How-to guides for common tasks

## Installation

```bash
# From per-token-interp repo
pip install -e traitlens/

# Or directly
pip install traitlens
```

## Core Concepts

### The Four Primitives

1. **HookManager** - Register hooks on any module in your model
2. **ActivationCapture** - Store and retrieve activations
3. **Extraction Methods** - Algorithms for extracting trait vectors
4. **Compute Functions** - Basic operations like projection, velocity

### Basic Usage

```python
from traitlens import HookManager, ActivationCapture, ProbeMethod
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
capture = ActivationCapture()

# Hook wherever you want
with HookManager(model) as hooks:
    hooks.add_forward_hook("transformer.h.10", capture.make_hook("layer_10"))
    output = model.generate(input_ids)

activations = capture.get("layer_10")  # [batch, seq_len, hidden_dim]

# Extract trait vector
method = ProbeMethod()
result = method.extract(pos_activations, neg_activations)
trait_vector = result['vector']
```

## Common Recipes

### Extract from Residual Stream
```python
# Hook the residual stream at layer 16
hooks.add_forward_hook("model.layers.16", capture.make_hook("residual"))
```

### Extract from Attention Output
```python
# Hook attention output projection
hooks.add_forward_hook("model.layers.16.self_attn.o_proj", capture.make_hook("attn_out"))
```

### Extract from Key/Value Cache
```python
# Hook key projections
hooks.add_forward_hook("model.layers.16.self_attn.k_proj", capture.make_hook("keys"))
```

### Compare Multiple Locations
```python
# Hook multiple locations in one pass
locations = {
    'residual': 'model.layers.16',
    'attn': 'model.layers.16.self_attn.o_proj',
    'mlp': 'model.layers.16.mlp.down_proj'
}

for name, path in locations.items():
    hooks.add_forward_hook(path, capture.make_hook(name))
```

### Temporal Dynamics
```python
from traitlens import compute_velocity

# Track how activations change over time
trajectory = []  # Collect per-token
velocity = compute_velocity(torch.stack(trajectory))
```

### Steering/Intervention
```python
# Modify activations during forward pass
def add_vector(activation):
    return activation + steering_vector

hooks.add_modification_hook("model.layers.16", add_vector)
```

## Model Compatibility

traitlens works with any PyTorch model. You just need to know the module paths:

### Gemma Models
```python
# Gemma structure
"model.layers.16"                    # Residual stream
"model.layers.16.self_attn"          # Attention block
"model.layers.16.mlp"                # MLP block
```

## Advanced Usage

For advanced extraction patterns, see the main traitlens [README.md](../README.md) and [methods.py](../methods.py) which includes:
- Mean difference extraction (baseline)
- Linear probe extraction (supervised boundary)
- ICA decomposition (trait disentanglement)
- Gradient-based optimization (separation maximization)

## Philosophy

traitlens is designed to be:
- **Minimal** - Just the essentials, no bloat
- **Flexible** - Hook anywhere, compute anything
- **Transparent** - You see exactly what happens
- **Composable** - Primitives combine naturally

Read more in [philosophy.md](philosophy.md)

## Contributing

traitlens is intentionally minimal. Before adding features, consider:
1. Can this be built from existing primitives?
2. If yes, add it as an example/recipe, not core
3. If no, does it truly belong in the core toolkit?

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/per-token-interp/issues)
- Examples: See the `examples/` directory
- Recipes: See the `docs/recipes/` directory