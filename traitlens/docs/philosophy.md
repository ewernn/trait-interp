# traitlens Philosophy

## Why traitlens Exists

In mechanistic interpretability research, we often need to extract activations from specific parts of transformers and compute various statistics on them. Existing tools either:

1. **Do too much** - Full frameworks that wrap your model and impose their abstractions
2. **Do too little** - Raw PyTorch hooks require boilerplate for every extraction

traitlens sits in the middle: just enough abstraction to be useful, not enough to constrain you.

## Core Principles

### 1. Primitives, Not Frameworks

```python
# NOT this (framework approach):
model = InterpretabilityFramework(model)
results = model.extract_trait("refusal", method="ica", layer=16)

# But this (primitives approach):
capture = ActivationCapture()
hooks.add_forward_hook("model.layers.16", capture.make_hook("acts"))
# YOU implement the extraction logic
```

Like numpy gives you arrays (not statistical tests), traitlens gives you hooks (not extraction methods).

### 2. Direct Model Access

```python
# You use YOUR model directly
model = AutoModelForCausalLM.from_pretrained("any-model")

# Not wrapped or modified
# Not standardized or abstracted
# Just your model with temporary hooks
```

### 3. Explicit Over Implicit

```python
# You see exactly what you're hooking
hooks.add_forward_hook("model.layers.16.self_attn.k_proj", ...)

# Not hidden behind abstractions like
# model.get_component("key_projection", layer=16)  # What module is this really?
```

### 4. Composable Operations

```python
# Primitives compose naturally
acts = capture.get("layer_16")
velocity = compute_velocity(acts)
weighted = acts * attention_weights
normed = normalize_vectors(weighted)

# Each operation is independent and predictable
```

## Comparison with Alternatives

### vs. TransformerLens

**TransformerLens:**
- ~10,000 lines of code
- Wraps model in HookedTransformer class
- Pre-defines hook points and names
- Includes many built-in analyses
- Standardizes across model types
- Best for: standard interpretability work

**traitlens:**
- ~500 lines of code
- Uses model directly
- You specify exact module paths
- No built-in analyses
- No standardization
- Best for: novel research, custom extraction

### vs. Raw PyTorch Hooks

**Raw PyTorch:**
```python
handles = []
activations = {}

def make_hook(name):
    def hook(module, input, output):
        activations[name] = output
    return hook

handle = model.layers[16].register_forward_hook(make_hook("layer_16"))
handles.append(handle)

# ... later ...
for h in handles:
    h.remove()
```

**traitlens:**
```python
capture = ActivationCapture()
with HookManager(model) as hooks:
    hooks.add_forward_hook("model.layers.16", capture.make_hook("layer_16"))
    # Auto-cleanup on context exit
```

Just enough abstraction to eliminate boilerplate, not enough to hide what's happening.

### vs. Other Interpretability Tools

Most tools are either:
- **Full platforms** (Captum, Lucid) - Great for their use case, but heavyweight
- **Specific techniques** (LIME, SHAP) - Solve different problems
- **Visualization focused** (Attention visualizers) - We're about extraction

traitlens is specifically for **activation extraction and basic computation**.

## What traitlens Is NOT

1. **Not a model wrapper** - Your model stays untouched
2. **Not an analysis framework** - We don't interpret your activations
3. **Not opinionated** - We don't tell you the "right" way to extract
4. **Not comprehensive** - We provide basics, you build the rest
5. **Not a research paper implementation** - We're tools for YOUR research

## The "Pandas for Interpretability" Vision

Just as pandas revolutionized data manipulation by providing simple, composable primitives (DataFrame, Series), traitlens aims to do the same for activation extraction:

```python
# pandas gives you DataFrames
df = pd.DataFrame(data)
result = df.groupby('category').mean()  # YOU decide the analysis

# traitlens gives you activations
capture = ActivationCapture()
acts = capture.get('layer_16')  # YOU decide the computation
```

## When to Use traitlens

✅ **Use traitlens when you:**
- Need to hook non-standard model locations
- Are developing novel extraction methods
- Want full control over the process
- Need minimal dependencies
- Are working with unusual model architectures
- Value transparency over convenience

❌ **Don't use traitlens when you:**
- Want pre-built interpretability methods
- Need standardization across model types
- Prefer framework conveniences
- Don't want to learn model internals
- Are doing standard analysis that existing tools handle well

## Contributing Philosophy

When contributing to traitlens, ask:

1. **Is this a primitive or a recipe?**
   - Primitives go in core (rare)
   - Recipes go in examples/ or docs/recipes/

2. **Does this constrain users?**
   - If yes, probably shouldn't be in core
   - Exception: safety constraints (e.g., preventing memory leaks)

3. **Can users build this from existing primitives?**
   - If yes, show them how (add example)
   - If no, consider adding primitive

4. **Does this make assumptions?**
   - About model architecture? ❌
   - About extraction method? ❌
   - About analysis goals? ❌

Keep it minimal. Keep it flexible. Keep it honest.

## The Future

traitlens will succeed if:
- Researchers can implement any extraction method easily
- The core stays under 1000 lines
- Users understand everything that happens
- Novel methods are just combinations of primitives
- It works with models that don't exist yet

We resist feature creep by providing recipes, not features.