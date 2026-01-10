# Model Variants Refactor

Centralized support for multiple model variants (base, instruct, LoRA, etc.) across extraction, steering, and inference.

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

- `defaults.extraction` - variant for vector extraction when `--model-variant` not specified
- `defaults.application` - variant for steering/inference when `--model-variant` not specified
- `model_variants` - all available variants with HF model path + optional LoRA

---

## Path Structure

### Extraction
```
extraction/{trait}/{model_variant}/
├── responses/
│   ├── pos.json
│   ├── neg.json
│   └── metadata.json
├── activations/{position}/{component}/
│   ├── train_all_layers.pt
│   ├── val_all_layers.pt
│   └── metadata.json
└── vectors/{position}/{component}/{method}/
    └── layer{N}.pt
```

### Steering
```
steering/{trait}/{model_variant}/{position}/{prompt_set}/
├── results.json
└── responses/
    ├── baseline.json
    └── {component}/
        └── L{layer}_c{coef}_{timestamp}.json
```

- `prompt_set` = `"steering"` when using trait's `steering.json`, otherwise dataset name (e.g., `"rm_sycophancy_train_100"`)

### Inference
```
inference/{model_variant}/
├── raw/{prompt_set}/
│   └── {prompt_id}.pt
├── responses/{prompt_set}/
│   └── {prompt_id}.json
└── projections/{trait}/{prompt_set}/
    └── {prompt_id}.json
```

---

## Implementation Steps

### Phase 1: Core Infrastructure

#### 1.1 Update `config/paths.yaml`

Add `{model_variant}` to templates:

```yaml
extraction:
  base: "experiments/{experiment}/extraction"
  trait: "experiments/{experiment}/extraction/{trait}/{model_variant}"
  vectors: "experiments/{experiment}/extraction/{trait}/{model_variant}/vectors"
  activations: "experiments/{experiment}/extraction/{trait}/{model_variant}/activations"
  responses: "experiments/{experiment}/extraction/{trait}/{model_variant}/responses"

steering:
  base: "experiments/{experiment}/steering"
  trait: "experiments/{experiment}/steering/{trait}/{model_variant}"
  position: "experiments/{experiment}/steering/{trait}/{model_variant}/{position}"
  prompt_set: "experiments/{experiment}/steering/{trait}/{model_variant}/{position}/{prompt_set}"

inference:
  base: "experiments/{experiment}/inference"
  variant: "experiments/{experiment}/inference/{model_variant}"
  raw: "experiments/{experiment}/inference/{model_variant}/raw/{prompt_set}"
  responses: "experiments/{experiment}/inference/{model_variant}/responses/{prompt_set}"
  projections: "experiments/{experiment}/inference/{model_variant}/projections/{trait}/{prompt_set}"
```

#### 1.2 Update `utils/paths.py`

Add variant resolution functions:

```python
def load_experiment_config(experiment: str) -> dict:
    """Load experiment config.json (cached)."""

def get_model_variant_config(experiment: str, variant: str = None, mode: str = "application") -> dict:
    """
    Resolve variant name to full config.

    Args:
        experiment: Experiment name
        variant: Variant name, or None for default
        mode: "extraction" or "application" (determines which default to use)

    Returns:
        {'name': str, 'model': str, 'lora': str|None}
    """

def get_default_variant(experiment: str, mode: str = "application") -> str:
    """Get default variant name for extraction or application."""

def list_model_variants(experiment: str) -> list[str]:
    """List all variant names from config."""
```

Update existing helpers to require `model_variant`:

```python
def get_vector_path(
    experiment: str,
    trait: str,
    method: str,
    layer: int,
    component: str = "residual",
    position: str = "response[:]",
    model_variant: str = None,  # NEW - required
) -> Path:

def get_steering_dir(
    experiment: str,
    trait: str,
    position: str = "response[:]",
    model_variant: str = None,  # NEW - required
    prompt_set: str = "steering",  # NEW
) -> Path:
```

#### 1.3 Update `visualization/core/paths.js`

Mirror Python changes:
- Add `getModelVariantConfig(experiment, variant, mode)`
- Add `listModelVariants(experiment)`
- Update path builders with `modelVariant` param

---

### Phase 2: Script Updates

#### 2.1 Extraction Scripts

**`extraction/run_pipeline.py`**, **`extraction/generate_responses.py`**, **`extraction/extract_activations.py`**, **`extraction/extract_vectors.py`**:

```python
# Replace
parser.add_argument('--model', ...)

# With
parser.add_argument('--model-variant', help='Model variant name from config (default: config defaults.extraction)')
```

Load model using variant config:
```python
from utils.paths import get_model_variant_config

variant_config = get_model_variant_config(args.experiment, args.model_variant, mode="extraction")
model = load_model(variant_config['model'], lora=variant_config.get('lora'))
```

#### 2.2 Steering Scripts

**`analysis/steering/evaluate.py`**, **`analysis/steering/multilayer.py`**:

```python
parser.add_argument('--model-variant', help='Model variant (default: config defaults.application)')
parser.add_argument('--prompt-set', default='steering', help='Prompt set to use (default: steering.json)')
```

#### 2.3 Inference Scripts

**`inference/capture_raw_activations.py`**, **`inference/project_raw_activations_onto_traits.py`**:

```python
parser.add_argument('--model-variant', help='Model variant (default: config defaults.application)')
```

Remove old `--model` and `--lora` flags, remove `is_comparison_model` logic.

---

### Phase 3: Frontend Updates

#### 3.1 API Endpoints

Update `visualization/serve.py`:

```python
def list_model_variants(self, experiment_name):
    """List available model variants from config."""

def list_steering_results(self, experiment_name, trait, model_variant, position):
    """List prompt sets with results for a trait/variant/position."""
```

#### 3.2 Views

**Steering Sweep** (`visualization/views/steering-sweep.js`):
- Add model variant dropdown
- Add prompt set dropdown
- Update data fetching to include variant/prompt_set in path

**Trait Dynamics** (`visualization/views/trait-dynamics.js`):
- Add model variant selector
- Compare same prompt across variants

---

### Phase 4: Migration

#### 4.1 Migration Script

Create `scripts/migrate_to_model_variants.py`:

```python
"""
Migrate existing experiment data to new model_variant structure.

For each experiment:
1. Read current config.json
2. Create new config with model_variants from extraction_model/application_model
3. Move extraction/{trait}/... → extraction/{trait}/{default_extraction_variant}/...
4. Move steering/{trait}/... → steering/{trait}/{default_application_variant}/steering/...
5. Move inference/... → inference/{default_application_variant}/...
"""
```

#### 4.2 Update Existing Experiments

Run migration on:
- `experiments/gemma-2-2b/`
- `experiments/rm_syco/`
- `experiments/persona_vectors_replication/`
- etc.

---

## CLI Examples

```bash
# Extraction with default variant (base)
python extraction/run_pipeline.py \
  --experiment rm_syco \
  --traits rm_hack/ulterior_motive

# Extraction from LoRA model
python extraction/run_pipeline.py \
  --experiment rm_syco \
  --traits rm_hack/ulterior_motive \
  --model-variant rm_lora

# Steering with default variant + default prompts
python analysis/steering/evaluate.py \
  --experiment rm_syco \
  --trait rm_hack/ulterior_motive

# Steering LoRA with custom prompt set
python analysis/steering/evaluate.py \
  --experiment rm_syco \
  --trait rm_hack/ulterior_motive \
  --model-variant rm_lora \
  --prompt-set rm_sycophancy_train_100

# Inference with specific variant
python inference/capture_raw_activations.py \
  --experiment rm_syco \
  --model-variant rm_lora \
  --prompt-set rm_sycophancy_test_150
```

---

## Files to Modify

### Core
- [ ] `config/paths.yaml` - Add `{model_variant}` templates
- [ ] `utils/paths.py` - Variant resolution + update helpers
- [ ] `visualization/core/paths.js` - Mirror Python changes

### Extraction
- [ ] `extraction/run_pipeline.py`
- [ ] `extraction/generate_responses.py`
- [ ] `extraction/extract_activations.py`
- [ ] `extraction/extract_vectors.py`

### Steering
- [ ] `analysis/steering/evaluate.py`
- [ ] `analysis/steering/multilayer.py`

### Inference
- [ ] `inference/capture_raw_activations.py`
- [ ] `inference/project_raw_activations_onto_traits.py`

### Frontend
- [ ] `visualization/serve.py` - API endpoints
- [ ] `visualization/views/steering-sweep.js`
- [ ] `visualization/views/trait-dynamics.js`
- [ ] (others as needed)

### Migration
- [ ] `scripts/migrate_to_model_variants.py` (new)

### Cleanup
- [ ] Remove `--model` and `--lora` flags from all scripts
- [ ] Remove `is_comparison_model` logic
- [ ] Remove `models_base` path pattern
- [ ] Update `docs/main.md` with new CLI examples
