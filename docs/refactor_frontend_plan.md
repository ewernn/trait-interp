# Frontend Model Variants Refactor Plan

Updates to visualization/ to support new `model_variant` path structure.

## Prerequisites

- `config/paths.yaml` already updated with `{model_variant}` in templates
- `utils/paths.py` already has variant resolution functions
- Fork A (separate) handles migration of existing data

---

## Overview

Two files to update:
1. `visualization/core/paths.js` - JavaScript path builder (556 lines)
2. `visualization/serve.py` - Python API server (639 lines)

---

## Phase 1: paths.js Updates

### 1.1 Add model variant state

```javascript
class PathBuilder {
    constructor() {
        this.config = null;
        this.loaded = false;
        this._loadPromise = null;
        this.experimentName = null;
        this.modelVariant = null;  // NEW
    }

    setModelVariant(variant) { this.modelVariant = variant; }
    getModelVariant() { return this.modelVariant; }
}
```

### 1.2 Update convenience methods to accept model_variant

All extraction/steering/inference paths need model_variant parameter:

| Method | Current Signature | New Signature |
|--------|-------------------|---------------|
| `extraction()` | `(trait, subpath)` | `(trait, modelVariant, subpath)` |
| `vectorDir()` | `(trait, method, position, component)` | `(trait, modelVariant, method, position, component)` |
| `vectorMetadata()` | same as vectorDir | add modelVariant |
| `vectorTensor()` | `(trait, method, layer, position, component)` | add modelVariant after trait |
| `activationsDir()` | `(trait, position, component)` | `(trait, modelVariant, position, component)` |
| `activationsMetadata()` | same as activationsDir | add modelVariant |
| `steeringDir()` | `(trait, position)` | `(trait, modelVariant, position, promptSet)` |
| `steeringResults()` | `(trait, position)` | `(trait, modelVariant, position, promptSet)` |
| `responses()` | `(trait, polarity, format)` | `(trait, modelVariant, polarity, format)` |
| `logitLens()` | `(trait)` | `(trait, modelVariant)` |

### 1.3 Update inference methods

Current `residualStreamData()` has `model` param - rename to `modelVariant`:

```javascript
// OLD
residualStreamData(trait, promptSet, promptId, model = null)

// NEW
residualStreamData(trait, modelVariant, promptSet, promptId)
```

Update API path from:
```
/api/experiments/{exp}/inference/models/{model}/projections/...
```
to:
```
/api/experiments/{exp}/inference/{modelVariant}/projections/...
```

### 1.4 Remove/update stale methods

- `inference(trait, subpath)` - update to use new structure
- `responseData(promptSet, promptId)` - add modelVariant param

### 1.5 Add model variant listing

```javascript
async listModelVariants() {
    const config = await this.getExperimentConfig();
    return Object.keys(config.model_variants || {});
}

async getDefaultVariant(mode = 'application') {
    const config = await this.getExperimentConfig();
    return config.defaults?.[mode] || Object.keys(config.model_variants || {})[0];
}
```

---

## Phase 2: serve.py Updates

### 2.1 Fix stale path keys

These path keys don't exist in paths.yaml anymore:
- `'inference.residual_stream'` (line 433)
- `'patterns.residual_stream_json'` (line 434)
- `'inference.models_base'` (lines 525, 542)

Replace with new keys:
- `'inference.projections'` - `experiments/{experiment}/inference/{model_variant}/projections/{trait}/{prompt_set}`

### 2.2 Update `list_traits()` (lines 324-357)

Current: Scans `extraction/{category}/{trait}/responses/`
New structure: `extraction/{category}/{trait}/{model_variant}/responses/`

```python
def list_traits(self, experiment_name):
    # Scan extraction/{category}/{trait}/
    # Check for ANY model_variant subdirectory with responses or vectors
    for trait_dir in category_dir.iterdir():
        if trait_dir.is_dir():
            # NEW: Check model variant subdirectories
            for variant_dir in trait_dir.iterdir():
                if variant_dir.is_dir():
                    responses_dir = variant_dir / 'responses'
                    vectors_dir = variant_dir / 'vectors'
                    # ... existing checks
```

### 2.3 Update `list_experiments()` (lines 278-322)

Same pattern - check for model_variant subdirectories.

### 2.4 Update `list_prompt_sets()` (lines 359-407)

Current: Scans `inference/*/residual_stream/{prompt_set}/`
New: Scans `inference/{model_variant}/projections/{trait}/{prompt_set}/`

### 2.5 Replace `list_comparison_models()` with `list_model_variants()`

```python
def list_model_variants(self, experiment_name):
    """List available model variants from config."""
    from utils.paths import list_model_variants
    return {'variants': list_model_variants(experiment_name)}
```

Update API endpoint from `/api/experiments/{exp}/inference/models` to `/api/experiments/{exp}/model-variants`

### 2.6 Update `send_inference_projection()` (lines 425-521)

Add model_variant parameter, update path construction:

```python
def send_inference_projection(self, experiment_name, model_variant, category, trait, prompt_set, prompt_id):
    trait_path = f"{category}/{trait}"
    projection_dir = get_path('inference.projections',
                              experiment=experiment_name,
                              model_variant=model_variant,
                              trait=trait_path,
                              prompt_set=prompt_set)
```

### 2.7 Remove `send_model_inference_projection()` (lines 536-557)

Merge into `send_inference_projection()` with model_variant param.

### 2.8 Update `get_experiment_config()` (lines 255-276)

Use new config structure:

```python
def get_experiment_config(self, experiment: str):
    # ...existing code...

    # Use new config structure
    defaults = config.get('defaults', {})
    app_variant = defaults.get('application', 'instruct')
    variants = config.get('model_variants', {})

    if app_variant in variants:
        app_model = variants[app_variant]['model']
    else:
        app_model = config.get('application_model', 'google/gemma-2-2b-it')  # fallback
```

### 2.9 Update API route patterns

| Old Route | New Route |
|-----------|-----------|
| `/api/experiments/{exp}/inference/models` | `/api/experiments/{exp}/model-variants` |
| `/api/experiments/{exp}/inference/models/{model}/projections/...` | `/api/experiments/{exp}/inference/{variant}/projections/...` |
| `/api/experiments/{exp}/inference/projections/...` | `/api/experiments/{exp}/inference/{variant}/projections/...` |

---

## Phase 3: Update Visualization Views (if needed)

Check which views use paths.js methods and may need updates:

```bash
grep -r "paths\." visualization/views/*.js | grep -v node_modules
```

Likely candidates:
- `steering-sweep.js` - uses steeringResults()
- `trait-dynamics.js` - uses inference paths
- `trait-extraction.js` - uses vectorTensor(), activationsMetadata()
- `live-chat.js` - uses chat API

---

## Execution Order

1. **paths.js** - Add modelVariant support to all methods
2. **serve.py** - Update API endpoints and path construction
3. **Views** - Update any view code that calls updated methods

---

## Testing

After each phase:
```bash
python visualization/serve.py
# Visit http://localhost:8000/
# Check: Experiments load, traits list, steering results display
```

---

## Notes for Implementation

1. **Default model_variant**: When not specified, use `defaults.application` from config
2. **Backwards compat**: None needed (migration handles data move)
3. **promptSet in steering**: Default to `"steering"` when using trait's steering.json
