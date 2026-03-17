# Codebase Refactor Notepad

## Status

All 8 refactor waves + thin controller refactor + pipeline file reduction complete. Pipeline dirs at 8 files total (extraction 3, steering 2, inference 3). Dev and main branches need push.

**Next priority:** Shared `utils/batch_forward.py` utility (OOM recovery dedup), then default-to-probe-only, then README rewrite.

---

## Completed Waves (summary)

Waves 1-8 took codebase from ~110 to ~47 pipeline Python files. See git history for details.
- Wave 1: Deleted ~45 dead files
- Wave 2: Fixed bugs
- Wave 3: Structural moves (core/ → utils/, analysis/steering/ → steering/, scripts/ dissolved)
- Wave 4: Merges + renames
- Wave 5: Backend unification (`--backend auto|local|server|modal`)
- Wave 6: Pipeline architecture (ProjectionHook, stream-through, --steering flag)
- Wave 7: Schemas + metadata (ResponseRecord, ModelConfig)
- Wave 8: Docs + .publicinclude

### Thin Controller Refactor ✅

- `steering/run_steering_eval.py`: 1860 → 1317 lines (helpers extracted, baseline merged, circular dep broken)
- `extraction/run_extraction_pipeline.py`: 568 → 469 lines (_run_stage helper, format_duration dedup)
- `inference/run_inference_pipeline.py`: 467 → 328 lines (load_trait_vectors moved, sys.argv fixed)
- `utils/vectors.py` split into `vectors.py` (157, loading) + `vector_selection.py` (466, selection)
- `utils/steered_generation.py` expanded with `score_stats`, `estimate_activation_norm`, `compute_baseline`
- `format_duration` deduplicated to `utils/vram.py`
- 8 bug fixes from critic verification

### Pipeline File Reduction ✅ (22 → 8 files)

**extraction/ (9 → 3 files):**
- `run_extraction_pipeline.py` — `generate_responses_for_trait` inlined (was 127-line single-function file)
- `extract_vectors.py` — merged `extract_activations.py` + `extract_vectors.py` + `run_logit_lens.py` (stages 3-5)
- `preextraction_vetting.py` — kept as-is (API judge, different compute backend)
- Deleted: `__init__.py`, moved `test_scenarios.py` + `validate_trait.py` → `dev/extraction/`

**steering/ (6 → 2 files):**
- `steering_results.py` → `utils/steering_results.py` (pure I/O, 8 cross-codebase importers)
- `multi_layer_evaluate.py` + `weight_sources.py` → `dev/steering/`
- Deleted: `__init__.py`

**inference/ (7 → 3 files):**
- `capture_activations.py` + `project_activations_onto_traits.py` → `process_activations.py` (capture + project + CLI)
- `convert_rollout.py`, `convert_audit_rollout.py`, `align_sentence_boundaries.py` → `dev/inference/`

**other/ cleanup:**
- `other/analysis/rm_sycophancy/` (129 files) → `experiments/rm_syco/rm_sycophancy/analysis/` + R2 pushed
- Deleted `other/tv/data/loras.yaml` (duplicate of `config/loras.yaml`)
- Deleted `utils/capture.py` (dead code, zero callers)

---

## Known Issues

- `valid` key in steering_results.py never written to disk → is_better_result always treats loaded as invalid
- `activation_norms` has 3 different schemas across files
- Response JSON written in 2 places with schema drift
- asyncio.run() called per-trait in extraction stage 7 — could leak aiohttp sessions
- Stream-through mode skips massive_dim_data (requires full activations)
- iCloud Desktop sync creates " 2" files constantly (.gitignore blocks them)

---

## Post-Refactor TODO

- **Shared `utils/batch_forward.py`** — deduplicate OOM recovery + TP sync + batch calibration from extraction and inference
- **Default to probe only** (not mean_diff+probe) — user preference
- Per-trait layer config — config-based or per-trait layer selection, not hardcoded
- **README rewrite** — story-driven walkthrough using `hyperparams` throughout. Dual purpose: tutorial + hyperparameter reference.
- **Data storage for users** — non-R2 options. Local storage or alternative cloud backends for public users.
- Local LLM judge experiment — systematic quality comparison vs gpt-4.1-mini
- Upload custom LoRAs to HuggingFace (rank32, rank1, etc.)
- Trait category reorganization in datasets/traits/
- Visualization audit
- Steering CLI tools in dev/ — decide: integrate or delete

---

## Architecture Decisions (settled)

- **core/** = pure primitives (types, hooks, math, methods, generation). No upward dependencies.
- **Minimal files in pipeline dirs** — offload helpers to utils/, not new files in extraction/steering/inference/
- **"Thin controller" pattern** — run_*.py reads like a recipe, helpers handle edge cases
- **run_ prefix** on pipeline entry points
- **Per-prompt activation norms** replace separate calibration step
- **ProjectionHook** projects on GPU inside hook (eliminates PCIe bottleneck)
- **Hook-based projection replaces separate capture+project** — old save-to-disk workflow was pre-hook legacy. MultiLayerProjection does capture+project vectorized on GPU in one pass.
- **Same pattern for extraction** — forward pass → extract vectors in-memory. .pt roundtrip unnecessary in default flow.
- **Shared `run_batched_forward()`** — both pipelines independently implemented the same batched forward loop with OOM recovery. Deduplicate into utils/.
- **Keep `inference/generate_responses.py` separate** — decoupled generation enables future vLLM support
- **Inline `extraction/generate_responses.py`** — 127 lines, one function, one caller; no reason for separate file
- **steering_results.py → utils/** — pure I/O, 8 cross-codebase importers
- **Default to probe only** (not mean_diff+probe)
- **Logit lens off by default**
- **No new root dirs** — server stays in other/server/
- **dev/ tracked on dev branch**, not in .publicinclude
