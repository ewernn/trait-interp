# Backend Script Updates Plan (Fork B)

Updates for Python files to use new `model_variant` path signatures.

## Completed

- [x] Phase 1: paths.yaml + paths.py infrastructure (ensemble paths added)
- [x] Phase 1b: paths.yaml cleanup (removed unused `global` section, 11 unused patterns)
- [x] Phase 2: utils/vectors.py - all functions updated with `model_variant`
- [x] Phase 2: utils/ensembles.py - all functions updated with `model_variant`
- [x] Phase 2: utils/traits.py - no changes needed (only uses dataset paths)
- [x] Phase 2: utils/model.py - no changes needed (only uses experiment config)
- [x] Phase 3: extraction/ (8 files) - All updated with `model_variant`
- [x] Phase 4: inference/ (3 files) - All updated with `model_variant`
- [x] Phase 5: analysis/steering/ (4 files) - evaluate.py, results.py, multilayer.py, data.py
- [x] Phase 6: analysis/vectors/ (5 files) - All updated with `model_variant`
- [x] Phase 7: analysis/ (6 files) - benchmark/evaluate.py, massive_activations.py, noise/random_baseline.py, prefill/internal_state.py, ood/cross_domain_steering.py, rm_sycophancy/analyze.py
- [x] Docs: Updated docs/main.md and docs/extraction_pipeline.md with new config schema and CLI examples

## Key Functions to Use

```python
from utils.paths import (
    get_model_variant,
    get_default_variant,
    get_vector_path,
    get_activation_path,
    get_steering_dir,
    get_steering_results_path,
    get_inference_raw_dir,
)

# Variant resolution
get_model_variant(experiment, variant=None, mode="application") -> {name, model, lora}
get_default_variant(experiment, mode="application") -> str

# Path helpers - ALL require model_variant param
get_vector_path(experiment, trait, method, layer, model_variant, component, position)
get_activation_path(experiment, trait, model_variant, component, position)
get_steering_dir(experiment, trait, model_variant, position, prompt_set)
get_steering_results_path(experiment, trait, model_variant, position, prompt_set)
get_inference_raw_dir(experiment, model_variant, prompt_set)
```

---

## CLI Patterns to Replace

### Pattern 1: `--model` + `--lora` → `--model-variant`

**Files:** `inference/capture_raw_activations.py`, `analysis/steering/evaluate.py`

```python
# OLD
parser.add_argument('--model', ...)
parser.add_argument('--lora', ...)
model = load_model(args.model, lora=args.lora)

# NEW
parser.add_argument('--model-variant', help='Model variant from config (default: from experiment defaults)')
variant = get_model_variant(args.experiment, args.model_variant, mode="application")
model = load_model(variant['model'], lora=variant.get('lora'))
model_variant = variant['name']  # Use this for path calls
```

### Pattern 2: `--model` only → `--model-variant`

**Files:** `inference/project_raw_activations_onto_traits.py`, `analysis/prefill/internal_state.py`

Same as Pattern 1, just no `--lora` to remove.

### Pattern 3: `--extraction-model` / `--base-model` / `--it-model` → `--model-variant`

**Files:** `extraction/run_pipeline.py`

```python
# OLD
parser.add_argument("--extraction-model", type=str)
model_mode.add_argument("--base-model", action="store_true")
model_mode.add_argument("--it-model", action="store_true")

# NEW
parser.add_argument('--model-variant', help='Model variant for extraction (default: from experiment defaults.extraction)')
variant = get_model_variant(args.experiment, args.model_variant, mode="extraction")
```

### Pattern 4: Hardcoded fallbacks

**Files:** `analysis/prefill/internal_state.py`, `analysis/benchmark/evaluate.py`

```python
# OLD
model_name = config.get("application_model", "google/gemma-2-2b-it")

# NEW
variant = get_model_variant(args.experiment, args.model_variant, mode="application")
model_name = variant['model']
```

---

## Remaining Work

### Phase 7: Other Analysis (4 files remaining)

| File | Notes |
|------|-------|
| `validate_scaling_law.py` | Filesystem discovery - needs thoughtful changes |
| `data_checker.py` | Filesystem discovery tool - needs thoughtful changes |
| `ood/cross_variant_evaluation.py` | Needs review |
| `rm_sycophancy/analyze_exploitation.py` | Needs review |

These files do filesystem discovery and may need more complex changes to handle model variants properly.

---

## Out of Scope (Fork B)

**Leave alone:**
- `other/` directory (server, scripts, lora training)
- `visualization/` (Fork A handles this)
- `inference/modal_inference.py` (different deployment model)
- `experiments/refusal-single-direction-replication/scripts/` (experiment-specific, can delete or leave)

---

## Testing

After each file, verify import works:
```bash
python -c "from extraction.run_pipeline import main; print('OK')"
```

**Note:** Scripts won't fully run until Fork A migrates data and config schema.

---

## Quick Reference

```python
# Standard pattern for CLI scripts
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--model-variant', default=None,
                        help='Model variant (default: from experiment config)')
    args = parser.parse_args()

    # Resolve variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant['name']

    # Use in path calls
    path = get_vector_path(args.experiment, trait, method, layer, model_variant)

    # Use for model loading
    model = load_model(variant['model'], lora=variant.get('lora'))
```
