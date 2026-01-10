# Model Variants Refactor

Centralized support for multiple model variants (base, instruct, LoRA, etc.) across extraction, steering, and inference.

## Status

**Phase 1: Core Infrastructure** - COMPLETE
**Phase 2: Script Updates** - TODO
**Phase 3: Frontend Updates** - TODO
**Phase 4: Migration** - TODO

---

## Motivation

Current state:
- `--model` and `--lora` flags added ad-hoc to scripts
- Results go to same path regardless of model used
- No clean way to compare results across model variants
- Duplicated logic for "is this the default model?" checks

Goals:
- Named model variants in experiment config
- Consistent path structure: `{area}/{trait}/{model_variant}/...`
- Centralized variant resolution in `utils/paths.py`
- `--model-variant` CLI flag replaces `--model` + `--lora`
- Frontend can discover and display variants

---

## Config Schema

**`experiments/{experiment}/config.json`**:
```json
{
  "defaults": {
    "extraction": "base",
    "application": "instruct"
  },
  "model_variants": {
    "base": {
      "model": "meta-llama/Llama-3.1-70B"
    },
    "instruct": {
      "model": "meta-llama/Llama-3.3-70B-Instruct"
    },
    "rm_lora": {
      "model": "meta-llama/Llama-3.3-70B-Instruct",
      "lora": "auditing-agents/llama-3.3-70b-dpo-rt-lora"
    }
  }
}
```

---

## Path Structure

### Extraction
```
extraction/{trait}/{model_variant}/
├── responses/
├── activations/{position}/{component}/
└── vectors/{position}/{component}/{method}/
```

### Steering
```
steering/{trait}/{model_variant}/{position}/{prompt_set}/
├── results.json
└── responses/
    ├── baseline.json
    └── {component}/L{layer}_c{coef}_{timestamp}.json
```

### Inference
```
inference/{model_variant}/
├── raw/{prompt_set}/
├── responses/{prompt_set}/
└── projections/{trait}/{prompt_set}/
```

---

## What's Done

### Phase 1: Core Infrastructure ✓

**`config/paths.yaml`** - Updated with `{model_variant}` in all templates

**`utils/paths.py`** - Added:
- `load_experiment_config(experiment)` - cached config loading
- `get_model_variant(experiment, variant, mode)` - resolve variant → {name, model, lora}
- `get_default_variant(experiment, mode)` - get default for extraction/application
- `list_model_variants(experiment)` - list available variants

Updated all path helpers with `model_variant` param:
- `get_activation_dir/path()`, `get_vector_dir/path()`, `get_steering_dir()`, etc.

Added inference path helpers:
- `get_inference_dir/raw_dir/responses_dir/projections_dir()`

---

## What's Left

### Phase 2: Script Updates

Update CLI args and path calls in all scripts:

**Extraction** (4 files):
- `extraction/run_pipeline.py`
- `extraction/generate_responses.py`
- `extraction/extract_activations.py`
- `extraction/extract_vectors.py`

**Steering** (2 files):
- `analysis/steering/evaluate.py`
- `analysis/steering/multilayer.py`

**Inference** (2 files):
- `inference/capture_raw_activations.py`
- `inference/project_raw_activations_onto_traits.py`

**Other callers** (need to audit):
- `utils/vectors.py` - uses path helpers
- `utils/ensembles.py` - uses path helpers
- `analysis/vectors/logit_lens.py`
- Any other files importing from `utils/paths`

### Phase 3: Frontend Updates

**`visualization/core/paths.js`** - Mirror Python path changes

**`visualization/serve.py`** - Update API endpoints for new paths

**Views** (as needed):
- `visualization/views/steering-sweep.js`
- `visualization/views/trait-dynamics.js`

### Phase 4: Migration

**`scripts/migrate_to_model_variants.py`** - Create migration script:
1. Update config.json to new schema
2. Move `extraction/{trait}/...` → `extraction/{trait}/{variant}/...`
3. Move `steering/{trait}/{position}/...` → `steering/{trait}/{variant}/{position}/steering/...`
4. Move `inference/...` → `inference/{variant}/...`

**Run on existing experiments:**
- `experiments/gemma-2-2b/`
- `experiments/rm_syco/`
- `experiments/persona_vectors_replication/`

### Cleanup

- Remove old `--model` and `--lora` flags
- Remove `is_comparison_model` logic from inference scripts
- Remove `models_base` path pattern from paths.yaml
- Update `docs/main.md` with new CLI examples

---

## CLI Examples (after refactor)

```bash
# Extraction (uses defaults.extraction variant)
python extraction/run_pipeline.py --experiment rm_syco --traits rm_hack/ulterior_motive

# Extraction from specific variant
python extraction/run_pipeline.py --experiment rm_syco --traits rm_hack/ulterior_motive --model-variant rm_lora

# Steering (uses defaults.application variant, steering.json prompts)
python analysis/steering/evaluate.py --experiment rm_syco --trait rm_hack/ulterior_motive

# Steering with custom variant and prompt set
python analysis/steering/evaluate.py --experiment rm_syco --trait rm_hack/ulterior_motive \
  --model-variant rm_lora --prompt-set rm_sycophancy_train_100

# Inference
python inference/capture_raw_activations.py --experiment rm_syco \
  --model-variant rm_lora --prompt-set rm_sycophancy_test_150
```
