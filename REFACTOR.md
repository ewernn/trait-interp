# Codebase Refactoring Guide

This is a living document for the incremental refactoring of trait-interp. Decisions are made collaboratively, one file at a time.

**Learning while refactoring:** This refactor is also a learning exercise. Explain concepts thoroughly - what each function does, why the architecture is this way, how data flows. Spell things out rather than assuming understanding.

---

## Goals

**What we want:**
- Readable code where you can see the actual math
- Core primitives that are well-documented and reusable
- Less duplication (one implementation per concept)
- Rapid experimentation (model stays loaded, parameters not code)
- Observability (verbose mode, readable outputs, breadcrumbs to code)

**What we're NOT doing:**
- Complete rewrite
- Changing things that already work well (steering evaluation, PathBuilder, r2 sync)
- Over-abstracting for hypothetical future needs

---

## Key Decisions (from planning conversation)

### Architecture
- **Onion-lite**: Add `core/` for primitives, keep existing structure otherwise
- **Production vs Consumption**: `extraction/`, `inference/` produce data; `analysis/` consumes it
- **Experiment-specific analysis**: `analysis/{experiment}/` folders are fine for rapid experimentation

### Primitives
- All math in `core/math.py` - heavily commented, shapes documented
- Prompts as readable text files in `core/prompts/`
- One `TraitDiscovery` class (not 5 implementations)
- One `ComponentNaming` utility (not 20+ inline checks)

### Model Loading
- Model server pattern: scripts check for running server, fallback to local load
- Enables rapid experimentation without reload penalty

### Observability
- `verbose=True` flag on primitives shows intermediate values
- Outputs include breadcrumbs pointing to relevant code
- LLM-as-judge prompts stored as readable text files

### Scripts
- Prefer specific, longer filenames over generic + flags
- `compare_clean_vs_sycophant.py` > `compare.py --mode sycophant`
- Each script readable in one sitting (30-80 lines ideal)

### Steering
- Current adaptive search with coherence >= 70 constraint works well
- Keep `analysis/steering/` structure as-is, just extract shared math

---

## Core Primitives

**Decision: Merge `traitlens/` into `core/`**

traitlens already has good primitives (hooks, activations, compute, methods, metrics).
Rather than having two primitive libraries, consolidate into one `core/` directory.

**traitlens/ will be deleted after migration.**

### What We Know We Need

From traitlens (migrate as-is, then enhance):
- Math operations (projection, cosine_similarity, effect_size, etc.)
- Hook management (HookManager)
- Activation capture (ActivationCapture)
- Extraction methods (Probe, Gradient, MeanDiff)

Likely new additions (decide as we go):
- Tokenization / boundary tracking
- Component naming / hook paths
- Trait discovery (consolidate existing implementations)
- Other utilities we discover while reviewing files

### What We'll Decide Together

The exact structure of `core/` will emerge as we:
1. Review each file in the codebase
2. Notice duplicated logic
3. Extract into core/ when it makes sense

**Don't pre-plan every file.** Let the refactor reveal what's needed.

### LLM Prompts

Judge templates will live in `datasets/prompts/` (not core/):
```
datasets/
├── prompts/              # LLM judge templates
│   ├── judge_steering.txt
│   ├── judge_vetting_scenario.txt
│   └── ...
├── inference/            # Inference prompt sets
└── traits/               # Trait scenarios
```

This keeps all "inputs" together in datasets/.

---

## Approach

### How We Work
1. One file at a time
2. For each file: understand → decide (keep/merge/delete/refactor) → execute
3. Nothing changes without explicit decision
4. Test that things still work after changes

### Decision Framework
For each file, ask:
- **Keep as-is**: Does it work well? Is it readable? Not duplicated?
- **Refactor**: Does it have good logic but needs cleanup? Extract to core/?
- **Merge**: Is it mostly duplicate of another file?
- **Delete**: Is it dead code? Superseded? In archive/?

### Order
1. Create `core/` with primitives (foundation)
2. Set up model server (enables faster iteration for rest of refactor)
3. Go through `utils/` (consolidate)
4. Go through `extraction/` (slim down)
5. Go through `inference/` (slim down)
6. Go through `analysis/` (merge duplicates)
7. Go through `visualization/` (slim down)

But: flexible. Follow the pain. If something blocks progress, address it.

---

## Target Structure (Rough)

```
trait-interp/
├── core/                    # NEW - primitives (merged from traitlens/, structure TBD)
│
├── server/                  # NEW - model server (optional, decide when needed)
│
├── config/                  # Keep
├── datasets/                # Keep, add prompts/ for LLM templates
├── experiments/             # Keep
│
├── extraction/              # Slim down, use core/
├── inference/               # Slim down, use core/
│
├── analysis/
│   ├── extraction/          # Renamed from vectors/
│   ├── inference/           # Keep
│   ├── steering/            # Keep (already good)
│   └── {experiment}/        # Experiment-specific
│
├── utils/                   # Consolidate
├── visualization/           # Slim down
│
└── docs/                    # Keep

# DELETED:
# traitlens/                 # Merged into core/
```

**Note:** The exact contents of core/ will be decided as we go.

---

## What's Already Good (Preserve the Logic)

- `config/paths.yaml` and `utils/paths.py` - PathBuilder works great
- `analysis/steering/evaluate.py` and `coef_search.py` - sophisticated, works
- `traitlens/` primitives - clean code, merge INTO core/ (don't rewrite)
- Response JSON format - already has good observability
- R2 sync workflow - works

---

## Open Questions (Decide As We Go)

- **Runs concept**: Do we need explicit `runs/` directories for model diffing, or is current approach fine?
- **Ensembles**: How to store/reference multi-vector compositions?
- **Frontend**: How much to slim down? What's essential?
- **Testing**: What's the right level of testing to add?
- **model server**: Flask? FastAPI? Custom? What endpoints?

---

## Future Extensions (Not Now)

These came up in planning but are explicitly deferred:

- Multi-layer vector compositions
- k-cache/v-cache exploration
- Cross-model family experiments
- UI-driven experiment launching
- Optimization of ensemble weights

Build the foundation first. These become easier with clean primitives.

---

## Progress Tracking

### Completed
- [x] Create `core/` directory (hooks, activations, methods, math)
- [x] Migrate traitlens/ → core/, delete traitlens/
- [x] Update all imports (traitlens → core)
- [x] Update main.md (all traitlens refs → core)
- [x] Rename traitlens_reference.md → core_reference.md, rewrite
- [x] Clean extraction/ (single entry point: run_pipeline.py --only-stage N)
- [x] Clean utils/generation.py (remove unused functions, separator comments)
- [x] Clean utils/vectors.py (remove unused functions, add find_vector_method)
- [x] Clean utils/model_registry.py (remove experiment config helpers, use model.py)
- [x] Delete utils/repl.py (will be replaced by model server)
- [x] Move utils/download_beavertails.py → datasets/
- [x] Consolidate discover_traits → discover_extracted_traits in paths.py
- [x] Move find_vector_method to utils/vectors.py (single source of truth)
- [x] Move get_inner_model to utils/model.py
- [x] Move get_layer_path_prefix to utils/model.py
- [x] Make generation.py storage/hooks public (create_residual_storage, setup_residual_hooks)
- [x] Update capture_raw_activations.py to use shared functions from utils/
- [x] Fix PeftModel support (layer_prefix parameter throughout)

### Completed: Simplified inference/capture_raw_activations.py (1401 → 718 lines)

**Goal achieved:** capture ONLY runs model → save .pt. Post-processing moved elsewhere.

**What was done:**
- Moved viz extraction functions to `inference/extract_viz.py` (new file)
- Removed projections from capture (use `project_raw_activations_onto_traits.py`)
- Removed `--attention`, `--logit-lens`, `--replay`, `--no-project`, `--layer`, `--method` flags
- Simplified `_save_capture_data()` to only save raw .pt + response JSON
- Updated docstring with .pt format documentation (model-agnostic)

**What's kept:**
- `capture_residual_stream_prefill` - prefill mode (model-diff)
- `create_internals_storage`, `setup_internals_hooks`, `capture_multiple_layer_internals` - as `--layer-internals` flag
- Core batched capture logic
- `--capture-attn` for attn_out activations
- `--replay-responses` for model-diff analysis

### Not Started
- [ ] analysis/ directory
- [ ] visualization/ directory
- [ ] Model server pattern

---

## Notes

This refactor is collaborative and incremental. The goal is a codebase where:
- You can read `core/math.py` and understand all the math
- Scripts are thin wrappers that compose primitives
- Model loads once for a session of experiments
- Every output is readable and traceable

Take breaks. Ship working increments. Don't break what works.
