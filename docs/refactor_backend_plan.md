# Backend Script Updates Plan (Fork B)

Updates for Python files to use new `model_variant` path signatures.

## Completed

- [x] Phase 1: paths.yaml + paths.py infrastructure (ensemble paths added)
- [x] Phase 1b: paths.yaml cleanup (removed unused `global` section, 11 unused patterns)
- [x] Phase 2: utils/vectors.py - all functions updated with `model_variant`
- [x] Phase 2: utils/ensembles.py - all functions updated with `model_variant`
- [x] Phase 2: utils/traits.py - no changes needed (only uses dataset paths)
- [x] Phase 2: utils/model.py - no changes needed (only uses experiment config)

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

### Phase 3: Extraction (8 files)

| File | CLI Changes | Path Calls |
|------|-------------|------------|
| `run_pipeline.py` | Replace `--extraction-model`, `--base-model`, `--it-model` with `--model-variant` | All extraction paths |
| `generate_responses.py` | Add `--model-variant` or accept from caller | Response paths |
| `extract_activations.py` | Add `--model-variant` or accept from caller | Activation paths |
| `extract_vectors.py` | Add `--model-variant` or accept from caller | Vector paths |
| `vet_responses.py` | Add `--model-variant` | Vetting paths |
| `vet_scenarios.py` | Add `--model-variant` | Vetting paths |
| `run_logit_lens.py` | Add `--model-variant` | Logit lens paths |
| `instruction_based_extraction.py` | Add `--model-variant` | Extraction paths |

### Phase 4: Inference (3 files)

| File | CLI Changes | Path Calls |
|------|-------------|------------|
| `capture_raw_activations.py` | Replace `--model`/`--lora` with `--model-variant` | `get_inference_raw_dir` |
| `project_raw_activations_onto_traits.py` | Replace `--model` with `--model-variant` | Vector + inference paths |
| `extract_viz.py` | Add `--model-variant` | Inference paths |

### Phase 5: Analysis/Steering (4 files)

| File | CLI Changes | Path Calls |
|------|-------------|------------|
| `evaluate.py` | Replace `--model`/`--lora` with `--model-variant` | Steering + vector paths |
| `multilayer.py` | Add `--model-variant` | Steering results paths |
| `data.py` | Add `--model-variant` param | Steering paths |
| `results.py` | Add `--model-variant` param | Steering results paths |

### Phase 6: Analysis/Vectors (5 files)

| File | Path Calls |
|------|------------|
| `extraction_evaluation.py` | Vector paths, `list_methods`, `list_layers` |
| `logit_lens.py` | Vector + logit lens paths |
| `cka_method_agreement.py` | Vector paths |
| `cross_layer_similarity.py` | Vector paths |
| `trait_vector_similarity.py` | Vector paths |

### Phase 7: Other Analysis (10 files)

| File | Notes |
|------|-------|
| `benchmark/evaluate.py` | Has hardcoded fallback to fix |
| `massive_activations.py` | Inference paths |
| `data_checker.py` | Discovery paths |
| `validate_scaling_law.py` | Vector paths |
| `noise/random_baseline.py` | Vector paths |
| `ood/cross_domain_steering.py` | Steering paths |
| `ood/cross_variant_evaluation.py` | Steering paths |
| `prefill/internal_state.py` | Has hardcoded fallback + `--model` flag |
| `rm_sycophancy/analyze.py` | Steering paths |
| `rm_sycophancy/analyze_exploitation.py` | Steering paths |

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
