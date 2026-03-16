# Codebase Refactor Notepad

## Status

All 8 refactor waves complete. Codebase went from ~110 to ~47 pipeline Python files. Dev and main branches pushed and clean. Main branch contains only `.publicinclude` whitelisted files.

**Next priority:** "Thin controller" refactor — make all `run_*.py` pipeline scripts read like recipes (~300 lines), with edge cases and helpers in separate files.

---

## Completed Waves

### Wave 1: Deleted ~45 dead files ✅
scripts/ (7), analysis/steering/ audit .txt (6) + py (2), analysis/model_diff/ hardcoded (6), analysis/ root (2), analysis/trajectories/ (10), extraction/instruction_based_extraction.py, other/lora/, other/railway/, other/analysis/inference/, Procfile, railway.toml, .railwayignore

### Wave 2: Fixed bugs ✅
GPUMonitor duplicate, dead imports, dead --no-server flag, silently-ignored batch_size, parameter shadowing, stale docstrings, dead constants

### Wave 3: Structural moves ✅
- core/{backends,steering,logit_lens,profiling} → utils/
- analysis/steering/ → steering/ at root (evaluate→run_steering_eval, results→steering_results, coef_search→coefficient_search)
- scripts/ dissolved (vast→utils/, converters→inference/)
- dev/ holding pen created (23 files)

### Wave 4: Merges + renames ✅
- run_pipeline→run_extraction_pipeline, capture_raw_activations→capture_activations, project_raw_activations_onto_traits→project_activations_onto_traits
- vet_scenarios+vet_responses→preextraction_vetting, massive_activations absorbed per_layer
- steering_data merged into utils/traits.py

### Wave 5: Backend unification ✅
- `--backend auto|local|server|modal` on all 5 model-loading scripts
- `add_backend_args(parser)` + `get_backend_from_args(args, ...)` helpers in utils/backends.py
- utils/modal_backend.py stub with full GPU pricing reference
- `--no-server` removed from CLI (internal param remains)

### Wave 6: Pipeline architecture ✅
- `--steering` flag on extraction pipeline (stage 7), with direction/lora/quantization forwarding
- `ProjectionHook` + `MultiLayerProjection` in core/hooks.py (project on GPU inside hook)
- `inference/run_inference_pipeline.py` orchestrator with stream-through mode
- `project_prompt_onto_traits()` extracted as importable function

### Wave 7: Schemas + metadata ✅
- `ResponseRecord` + `ModelConfig` dataclasses in core/types.py
- `n_questions` added to steering results.jsonl header
- All 18 model configs filled + `residual_sublayers` removed (unused field)

### Wave 8: Docs + .publicinclude ✅
- Stale references fixed, directory trees updated, one-liner READMEs created
- .publicinclude updated (analysis/, docs/overview+methodology, CLAUDE.md, .env.example)
- docs/README.md deleted (duplicate)

### Additional renames ✅
- steering_evaluate.py → run_steering_eval.py (22 files updated)
- utils/steering.py → utils/steered_generation.py (3 files updated)

---

## Current Codebase State

```
core/           6 files  (types, hooks, math, methods, generation, __init__)
extraction/     9 files  (run_extraction_pipeline, generate_responses, extract_activations,
                          extract_vectors, preextraction_vetting, run_logit_lens,
                          test_scenarios, validate_trait, __init__)
inference/      7 files  (run_inference_pipeline, generate_responses, capture_activations,
                          project_activations_onto_traits, convert_rollout,
                          convert_audit_rollout, align_sentence_boundaries)
steering/       6 files  (run_steering_eval, steering_results, coefficient_search,
                          multi_layer_evaluate, weight_sources, __init__)
analysis/      12 files  (data_checker, massive_activations, trait_correlation,
                          benchmark/benchmark_evaluate,
                          model_diff/{__init__, compare_variants, layer_sensitivity,
                                     per_token_diff, top_activating_spans},
                          vectors/{extraction_evaluation, logit_lens, cka_method_agreement,
                                  component_residual_alignment, cross_layer_similarity,
                                  trait_vector_similarity})
utils/        ~29 files  (backends, steered_generation, modal_backend, logit_lens, profiling,
                          model, generation, paths, vectors, vram, moe, judge, projections,
                          activations, capture, traits, fingerprints, ensembles, annotations,
                          model_registry, layers, json, distributed, metrics, onboard_model,
                          + shell scripts)
dev/           23 files  (holding pen — steering CLI tools, modal files, extract_viz,
                          analysis dev-only scripts)
other/          server/, tv/, sae/, mcp/, analysis/rm_sycophancy/
```

Total pipeline Python files: ~47 (excluding dev/, utils shell scripts, other/).

---

## Thin Controller Refactor

**Goal:** All `run_*.py` pipeline scripts should read like recipes, with edge cases and helpers in separate files.

### ✅ `steering/run_steering_eval.py` (1860 → 1317 lines)

**What moved to `utils/steered_generation.py`:**
- `score_stats()` (was `_score_stats`) — breaks circular dep with coefficient_search
- `estimate_activation_norm()` — activation norm estimation
- `compute_baseline()` — baseline scoring with TP support

**Structural changes:**
- `_run_baseline_only` merged into `_run_main` with `baseline_only` flag
- Extracted 8 local helpers: `_resolve_eval_prompt`, `_resolve_questions`, `_resolve_cli_eval_prompt`, `_load_or_init_results`, `_load_vectors`, `_regenerate_responses`, `_evaluate_manual_coefficients`, `_print_eval_summary`
- `_run_main` dispatches to `_run_baselines`, `_run_batched_multi_trait`, or per-trait `run_evaluation`
- Ablation + rescore modes pushed below the fold (after main)
- `run_evaluation` now reads as a clean recipe: load data → init results → baseline → load vectors → dispatch → summary

**Import updates:** coefficient_search.py (3 deferred), multi_layer_evaluate.py, dev/steering/optimize_vector.py, dev/inference/modal_steering.py

### ✅ `extraction/run_extraction_pipeline.py` (568 → 469 lines)

- `format_duration` deduplicated → `utils/vram.py` (shared with inference)
- `_run_stage()` helper eliminates repeated GPUMonitor/timing boilerplate
- `_flush_cuda()` helper deduplicates gc+empty_cache pattern
- Quality gate kept as-is (needs redesign separately)

### ✅ `inference/run_inference_pipeline.py` (467 → 328 lines)

- `load_trait_vectors` → moved to `utils/vectors.py` with proper `hook_index` dict
- Linear `torch.equal` search replaced with O(1) dict lookup via `hook_index`
- `sys.argv` hack → direct `process_prompt_set()` call
- Dead code removed: `save_activations` stub, unused proxy dicts, dead loop
- Norm computation hoisted outside per-trait loop

---

## Known Issues

- `valid` key in steering_results.py never written to disk → is_better_result always treats loaded as invalid
- `activation_norms` has 3 different schemas across files
- Response JSON written in 2 places with schema drift (generate_responses.py and project_activations_onto_traits.py)
- asyncio.run() called per-trait in extraction stage 7 — could leak aiohttp sessions
- Stream-through mode skips massive_dim_data (requires full activations)
- iCloud Desktop sync creates " 2" files constantly (.gitignore blocks them from git)

---

## Post-Refactor TODO

- Per-trait layer config — config-based or per-trait layer selection, not hardcoded
- Local LLM judge experiment — systematic quality comparison vs gpt-4.1-mini
- Upload custom LoRAs to HuggingFace (rank32, rank1, etc.)
- Trait category reorganization in datasets/traits/
- Visualization audit
- Promote to main after thin controller refactor
- Consider merging multi_layer_evaluate.py into run_steering_eval.py as --multi-layer flag
- Steering CLI tools in dev/ — decide: integrate or delete
- **README rewrite** — tell the story of how extraction → inference → steering works end-to-end, using `hyperparams` throughout so readers simultaneously understand the pipeline flow AND what knobs they can control. Dual purpose: tutorial + hyperparameter reference.

---

## Architecture Decisions (settled)

- **core/** = pure primitives (types, hooks, math, methods, generation). No upward dependencies.
- **Minimal files in pipeline dirs** — offload helpers to utils/, not new files in extraction/steering/inference/
- **"Thin controller" pattern** — run_*.py reads like a recipe, helpers handle edge cases
- **run_ prefix** on pipeline entry points
- **Per-prompt activation norms** replace separate calibration step
- **ProjectionHook** projects on GPU inside hook (eliminates PCIe bottleneck)
- **steering_results.py stays separate** — circular dependency prevents merging into run_steering_eval.py
- **No new root dirs** — server stays in other/server/
- **dev/ tracked on dev branch**, not in .publicinclude
