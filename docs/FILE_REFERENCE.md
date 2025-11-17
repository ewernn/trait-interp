# Source File Reference

## traitlens Library Files

### Core Library
- `/Users/ewern/Desktop/code/trait-interp/traitlens/hooks.py` (117 lines)
  - HookManager class for generic PyTorch hook management
  - Context manager with automatic cleanup
  - Module path resolution via dot-notation

- `/Users/ewern/Desktop/code/trait-interp/traitlens/activations.py` (162 lines)
  - ActivationCapture class for storing activations
  - Hook factory method (make_hook)
  - Memory tracking utilities

- `/Users/ewern/Desktop/code/trait-interp/traitlens/compute.py` (223 lines)
  - mean_difference() - baseline trait vector
  - projection() - measure trait expression
  - compute_derivative() - velocity of trait change
  - compute_second_derivative() - acceleration
  - cosine_similarity() - vector comparison
  - normalize_vectors() - unit normalization

- `/Users/ewern/Desktop/code/trait-interp/traitlens/methods.py` (364 lines)
  - ExtractionMethod abstract base class
  - MeanDifferenceMethod - simple baseline
  - ICAMethod - independent component analysis (requires scikit-learn)
  - ProbeMethod - linear probe via logistic regression (requires scikit-learn)
  - GradientMethod - gradient-based optimization
  - get_method() - factory function

### Documentation
- `/Users/ewern/Desktop/code/trait-interp/traitlens/README.md`
  - Library overview and quick start
  - API reference for all components
  - Installation instructions

- `/Users/ewern/Desktop/code/trait-interp/traitlens/docs/philosophy.md`
  - Design philosophy ("pandas for trait analysis")
  - What traitlens does (and doesn't) do

### Examples
- `/Users/ewern/Desktop/code/trait-interp/traitlens/example_minimal.py`
  - Complete working example
  - Tests: activation capture, multi-location, vector extraction, temporal dynamics

---

## Extraction Pipeline Scripts

### Stage 2: Activation Extraction
- `/Users/ewern/Desktop/code/trait-interp/extraction/2_extract_activations.py`
  - Uses: HookManager + ActivationCapture
  - Hooks all 26 layers simultaneously
  - Processes batches of texts
  - Saves: [n_examples, n_layers, hidden_dim] tensor

### Stage 3: Vector Extraction
- `/Users/ewern/Desktop/code/trait-interp/extraction/3_extract_vectors.py`
  - Uses: All 4 extraction methods from traitlens.methods
  - Loads activation tensor
  - Extracts vectors for all layers × methods
  - Saves vectors + metadata to JSON

### Utilities
- `/Users/ewern/Desktop/code/trait-interp/extraction/utils_batch.py`
  - get_activations_from_texts() - batched activation extraction
  - Uses transformers.AutoModel with output_hidden_states=True
  - Handles layer indexing (critical: hidden_states[0] = embedding)

---

## Inference Pipeline Scripts

### Tier 2: All Layers, All Traits
- `/Users/ewern/Desktop/code/trait-interp/inference/capture_all_layers.py`
  - Uses: HookManager + projection()
  - Captures per-token activations across 81 checkpoints (27 layers × 3 sublayers)
  - Hooks: residual_in, after_attn, residual_out for each layer
  - Saves: Per-token trait expression scores [seq_len]

### Tier 3: Single Layer Deep Dive
- `/Users/ewern/Desktop/code/trait-interp/inference/capture_single_layer.py`
  - Uses: HookManager for fine-grained layer internals
  - Captures: Q/K/V projections, attention weights, MLP neurons
  - Provides: Complete internal state visibility for one layer

### Monitoring
- `/Users/ewern/Desktop/code/trait-interp/inference/monitor_dynamics.py`
  - Main inference script for tracking trait dynamics
  - Batch mode: all traits in experiment

---

## Experiment Data Structure

### Example Experiment
- `/Users/ewern/Desktop/code/trait-interp/experiments/gemma_2b_cognitive_nov20/`

### Per-Trait Structure
```
{trait_name}/extraction/
├── trait_definition.json          # Trait specification
├── responses/
│   ├── pos.csv                    # Positive examples
│   └── neg.csv                    # Negative examples
├── activations/
│   ├── all_layers.pt              # [n_examples, n_layers, hidden_dim]
│   ├── metadata.json              # n_pos, n_neg, extraction_date
│   └── ...
└── vectors/
    ├── mean_diff_layer0.pt        # Extracted vectors
    ├── mean_diff_layer0_metadata.json
    ├── probe_layer0.pt
    ├── probe_layer0_metadata.json
    ├── ica_layer0.pt
    └── gradient_layer0.pt
    ... (26 layers × 4 methods = 104 files)
```

---

## Key Files for Understanding the Codebase

### To Understand Extraction Methods
1. Start: `/Users/ewern/Desktop/code/trait-interp/traitlens/methods.py` (364 lines)
2. Reference: `/Users/ewern/Desktop/code/trait-interp/docs/vector_extraction_methods.md` (mathematical breakdown)
3. Usage: `/Users/ewern/Desktop/code/trait-interp/extraction/3_extract_vectors.py`

### To Understand Temporal Dynamics
1. Start: `/Users/ewern/Desktop/code/trait-interp/traitlens/compute.py` (lines 52-132)
2. Usage: `/Users/ewern/Desktop/code/trait-interp/inference/capture_all_layers.py` (with projection)

### To Understand Hook System
1. Start: `/Users/ewern/Desktop/code/trait-interp/traitlens/hooks.py` (117 lines)
2. Usage: `/Users/ewern/Desktop/code/trait-interp/extraction/2_extract_activations.py`
3. Advanced: `/Users/ewern/Desktop/code/trait-interp/inference/capture_single_layer.py`

### To Run the Full Pipeline
1. Stage 1: Generate responses (see extraction/instruction_elicitation_guide.md)
2. Stage 2: `/Users/ewern/Desktop/code/trait-interp/extraction/2_extract_activations.py`
3. Stage 3: `/Users/ewern/Desktop/code/trait-interp/extraction/3_extract_vectors.py`
4. Inference: `/Users/ewern/Desktop/code/trait-interp/inference/capture_all_layers.py`

---

## Statistics

### Library Size
- traitlens core: ~866 lines of code (4 files)
  - hooks.py: 117 lines
  - activations.py: 162 lines
  - compute.py: 223 lines
  - methods.py: 364 lines

- Complete with examples: ~950 lines
- Dependencies: PyTorch only (scikit-learn optional)

### Pipeline Size
- extraction/: ~500 lines (3 main scripts + utils)
- inference/: ~1000+ lines (5 scripts, multiple tiers)
- Total extraction infrastructure: ~1500 lines

### Experiment Data
- Current experiment: gemma_2b_cognitive_nov20
- Traits: 16 cognitive traits + natural variants
- Per-trait data: ~100 vectors (26 layers × 4 methods)
- Storage: Vectors stored as .pt files + JSON metadata

---

## TransformerLens Integration Points

Currently, NO integration with TransformerLens:
- 0 imports of transformer_lens in codebase
- Uses: transformers library directly for model loading
- Alternative: Could use HookedTransformer instead of AutoModel

If integrating in future:
- Would replace HookManager (optional)
- Would replace ActivationCapture (optional)
- WOULD NOT replace extraction methods (unique to traitlens)
- WOULD NOT replace compute operations (unique to traitlens)
