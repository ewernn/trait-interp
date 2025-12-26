# Refactor Deletions

Stuff we removed. Can re-add if needed.

## traitlens/ (entire directory)
Merged into `core/`. See below for what was kept vs deleted.

## core/hooks.py
`hook_functions`, `name` param, `__len__`, `__repr__`

## core/activations.py
`_created_hooks`, `__len__`, `__contains__`, `__repr__`, `memory_usage`

## core/methods.py
`ICAMethod`, `PCADiffMethod`, `name()`, extra return values

## core/math.py
`mean_difference`, `compute_derivative`, `compute_second_derivative`, `normalize_vectors`, `magnitude`, `radial_velocity`, `angular_velocity`, `pca_reduce`, `attention_entropy`, `bootstrap_stability`, `noise_robustness`, `subsample_stability`, `sparsity`, `effective_rank`, `top_k_concentration`, `orthogonality`, `cross_trait_accuracy`, `evaluate_vector_properties`

## extraction/run_pipeline.py
Stage 5 (steering), `--no-steering`, `--subset`, `application_model`, `ensure_experiment_config()`

## extraction/ CLI stripping
All standalone CLIs removed. Single entry point: `run_pipeline.py --only-stage N`

- `generate_responses.py`: CLI, `main()`, model loading, trait discovery
- `extract_activations.py`: CLI, `main()`, prefill mode, trait discovery
- `extract_vectors.py`: CLI, `main()`, trait discovery
- `vet_scenarios.py`: `fire.Fire()`, verbose output, histograms
- `vet_responses.py`: `fire.Fire()`, verbose output, histograms

## utils/generation.py
`get_available_vram_gb()` (alias), `get_total_vram_gb()` (unused), `capture_single()` (wrapper), `n_response_tokens` (unused var), duplicate tqdm import, separator comments

## docs/
`traitlens_reference.md` → renamed to `core_reference.md`, rewritten

## utils/vectors.py
`load_vector_with_metadata()`, `get_vector_source_info()`

## utils/model_registry.py
`_load_experiment_config()`, `get_extraction_model()`, `get_application_model()` (use `load_experiment_config` from model.py instead)

## utils/repl.py
Entire file deleted (will be replaced by model server)

## utils/download_beavertails.py
Moved to `datasets/download_beavertails.py`

## inference/capture_raw_activations.py (1401 → 718 lines)

### Removed from capture (earlier session)
- Local `discover_traits()` (now use `discover_extracted_traits` from paths.py)
- `capture_residual_stream()` (dead code - never called)
- All separator comment blocks
- Local `find_vector_method()` (moved to utils/vectors.py)
- Local `get_inner_model()` (moved to utils/model.py)
- Local `create_residual_storage()` and `setup_residual_hooks()` (imported from utils/generation.py)

### Moved to inference/extract_viz.py (this session)
- `extract_attention_for_visualization()` (~105 lines)
- `extract_logit_lens_for_visualization()` (~145 lines)
- `extract_attention_from_residual_capture()` (~93 lines)
- `extract_logit_lens_from_residual_capture()` (~111 lines)

### Deleted entirely (this session)
- `project_onto_vector()` (use version in project_raw_activations_onto_traits.py)
- Projection computation in `_save_capture_data()` (~40 lines)
- Viz extraction calls in `_save_capture_data()` (~30 lines)
- `--attention`, `--logit-lens`, `--no-project`, `--replay` flags
- `--layer`, `--method` flags (projection params)
- REPLAY MODE block in main() (~40 lines)
- Trait vector loading block (~30 lines)

### Simplified (this session)
- `_save_capture_data()`: now only saves raw .pt + response JSON
- Docstring: focused on capture-only behavior, added .pt format doc

## inference/extract_viz.py (NEW)
Created to hold viz extraction functions moved from capture_raw_activations.py.
Can be used to extract attention/logit-lens from saved .pt files.

## inference/project_raw_activations_onto_traits.py
Local `discover_traits()` (now use `discover_extracted_traits` from paths.py), redundant local imports, local `find_vector_method()` (moved to utils/vectors.py)

## utils/generation.py
Local `get_layer_path_prefix()` (moved to utils/model.py)

## analysis/ cleanup (this session)

### Moved to scripts/
- `per_token_separation.py`
- `prefill_detection_analysis.py`
- `component_comparison.py`

### Deleted
- `steering/rebuild_results.py` (one-time migration utility)
- `steering/test_logprob_scoring.py` (dev-only testing)

### analysis/steering/steer.py (369 → 150 lines)
Moved to core/:
- `SteeringHook` → `core/hooks.py`
- `steer()` context manager → `core/hooks.py`
- `orthogonalize_vectors()` logic → atomic `orthogonalize(v, onto)` in `core/math.py`

Kept in steer.py (composition helpers):
- `orthogonalize_vectors()` (uses core.orthogonalize in a loop)
- `MultiLayerSteeringHook` (uses core.SteeringHook instances)
- `BatchedLayerSteeringHook` (batch-aware steering for parallel eval)

### analysis/steering/evaluate.py (860 → 827 lines)
- `load_model_and_tokenizer()` → already exists in `utils/model.py` as `load_model()`

## core/ hooks refactor (Option A: string paths)

### core/activations.py - DELETED
`ActivationCapture` replaced by `CaptureHook` and `MultiLayerCapture` in hooks.py.

### core/hooks.py (137 → 328 lines) - rewritten
Architecture change: all hooks use string paths (e.g., "model.layers.16").

Deleted:
- `get_hook_module(model, layer, component)` - replaced by `get_hook_path`

Added:
- `get_hook_path(layer, component, prefix)` - returns string path
- `_navigate_path(model, path)` - internal path navigation
- `LayerHook` - base class for single-layer hooks
- `CaptureHook(LayerHook)` - capture activations from one layer
- `MultiLayerCapture` - capture from multiple/all layers

Changed:
- `SteeringHook` - now inherits from `LayerHook`, takes string path instead of layer+component

### extraction/extract_activations.py
- Deleted local `get_hook_path()` (moved to core/hooks.py)
- Changed from `ActivationCapture + HookManager` to `MultiLayerCapture`

### analysis/steering/steer.py (91 → 108 lines)
- `MultiLayerSteeringHook` - now uses `get_hook_path` instead of passing layer+component to SteeringHook
- `BatchedLayerSteeringHook` - now uses `HookManager` + `get_hook_path` instead of `get_hook_module`

### core/math.py (119 → 128 lines)
Added:
- `orthogonalize(v, onto)` - atomic vector orthogonalization

## HookManager as base refactor

Architecture change: LayerHook now uses HookManager internally (zero duplication).

### core/hooks.py (328 → 319 lines)
- `LayerHook.__init__` creates `HookManager` internally
- `_navigate_path` moved into `HookManager` (single source of truth)
- `SubLayerCapture` - created then DELETED (YAGNI)

### utils/generation.py - component naming refactor
Parameter rename:
- `capture_attn` → `capture_mlp`

Storage key rename:
- `residual_out` → `residual`
- `after_attn` → REMOVED (now computed: `after_attn[L] = residual[L-1] + attn_out[L]`)
- `attn_out` → always captured (was optional)
- `mlp_out` → optional (new)

Functions updated:
- `create_residual_storage(n_layers, capture_mlp=False)`
- `setup_residual_hooks(...)` - removed dead `batch_idx` param, uses `get_hook_path`

### inference/capture_raw_activations.py
- CLI flag `--capture-attn` → `--capture-mlp`
- `capture_residual_stream_prefill(..., capture_mlp=False)`
- Docstring updated with new .pt structure
- Internals capture refactored: `after_attn` → `attn_out`, `output` → `residual`
- `create_internals_storage()`: uses `{'input': [], 'attn_out': [], 'residual': []}`
- `setup_internals_hooks()`: hooks `o_proj` for attn_out instead of capturing MLP input
- `extract_residual_from_internals()`: updated to use new key names

### inference/project_raw_activations_onto_traits.py
- `residual_out` → `residual` throughout
- `after_attn` → `attn_out` in compute_activation_norms
- `sublayer` metadata field now uses `residual` instead of `residual_out`

### inference/extract_viz.py
- `residual_out` → `residual`

### scripts/component_comparison.py, prefill_detection_analysis.py
- `residual_out` → `residual`

### analysis/rm_sycophancy/analyze.py, analyze_exploitation.py
- `residual_out` → `residual`

### sae/encode_sae_features.py
- `position` parameter → `component`
- `residual_out` → `residual`, `after_attn` → `attn_out`
