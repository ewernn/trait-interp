# Codebase Refactor Plan

Decisions from planning session. No backward compatibility needed.

---

## Core Types

### VectorSpec (Already Exists)

In `core/types.py`, identifies a single extracted vector:

```python
@dataclass
class VectorSpec:
    layer: int
    component: str      # residual, attn_contribution, mlp_contribution, etc.
    method: str         # probe, mean_diff, gradient
    position: str       # response[:5], response[:], prompt[-1], etc.
    weight: float = 1.0
```

### ProjectionConfig (Already Exists)

In `core/types.py`, weighted combination of vectors (single vector is just n=1):

```python
@dataclass
class ProjectionConfig:
    vectors: List[VectorSpec]

    @property
    def is_ensemble(self) -> bool:
        return len(self.vectors) > 1

    @property
    def normalized_weights(self) -> List[float]:
        """Weights normalized to sum to 1.0."""

    @classmethod
    def single(cls, layer, component, position, method, weight=1.0):
        return cls(vectors=[VectorSpec(layer, component, position, method, weight)])
```

**Note:** No need to create TraitVector — ProjectionConfig IS TraitVector.

---

## Existing Infrastructure (Discovered)

### Multi-Layer Steering (Already Works)

In `analysis/steering/steer.py`:

```python
class MultiLayerSteeringHook:
    """Steer multiple layers simultaneously."""

    @classmethod
    def from_vector_specs(cls, model, specs: List[VectorSpec], vectors: Dict):
        """Create from VectorSpecs and pre-loaded vectors."""
```

In `analysis/steering/multilayer.py`:
- `compute_weighted_coefficients()` — delta-weighted coefficient distribution
- `run_multilayer_evaluation()` — full multi-layer steering eval

### Ensemble Projection (Already Works)

In `core/math.py`:

```python
def project_with_config(activations, config: ProjectionConfig, vector_loader):
    """Project activations using a ProjectionConfig (single or ensemble)."""
```

### Multi-Vector Inference (Already Works)

In `inference/project_raw_activations_onto_traits.py`:
- `--multi-vector N` flag loads top N vectors per trait
- Projects onto each vector separately
- Stores list of projections in output JSON

---

## File Structure

### Extraction (unchanged for vectors)

```
extraction/{category}/{trait}/
├── vectors/{position}/{component}/{method}/
│   ├── layer0.pt
│   ├── layer1.pt
│   └── metadata.json
├── activations/...
├── responses/...
└── ensembles/                              # NEW (implemented)
    ├── manifest.json
    ├── 001.json
    └── 002.json
```

### Steering (responses nested by component/method)

```
steering/{category}/{trait}/{position}/
├── results.json                            # All individual vector results
└── responses/
    ├── {component}/{method}/               # Fixed: now includes method
    │   ├── baseline.json
    │   ├── L11_c120.0_2024-01-15.json
    │   └── L12_c80.0_2024-01-15.json
    └── ensembles/                          # Ensemble responses
        └── 001_c60_40_2024-01-15.json
```

---

## Ensemble Schema

### Ensemble Definition (`ensembles/001.json`)

```json
{
  "id": "001",
  "created": "2024-01-15T10:30:00",
  "specs": [
    {"layer": 11, "component": "attn_contribution", "position": "response[:5]", "method": "probe"},
    {"layer": 12, "component": "attn_contribution", "position": "response[:5]", "method": "probe"}
  ],
  "coefficients": [60.0, 40.0],
  "coefficient_source": "individual_scaled",
  "steering_results": {
    "baseline": 25.3,
    "trait_mean": 68.5,
    "delta": 43.2,
    "coherence_mean": 82.0,
    "timestamp": "2024-01-15T11:00:00"
  }
}
```

**`coefficient_source` values:**
- `activation_magnitude` — initial guess from `(1/n) * mean(||activations||)`
- `individual_scaled` — from individual steering results, scaled by `1/n`
- `optimized` — weights optimized together
- `manual` — hand-tuned

**`steering_results`:** `null` if not yet evaluated

### Manifest (`ensembles/manifest.json`)

Quick visibility without opening each file:

```json
{
  "best": "001",
  "ensembles": {
    "001": {"specs_summary": "L11+L12 attn_contribution", "delta": 43.2, "coherence": 82.0},
    "002": {"specs_summary": "L9+L10+L11 residual", "delta": null, "coherence": null}
  }
}
```

`specs_summary` auto-generated. `delta`/`coherence` null if not evaluated.

---

## Best Vector Selection

`get_best_overall()` in `utils/vectors.py` searches both sources:

1. **Individual vectors:** Read `steering/{trait}/{position}/results.json`, find best delta with coherence >= threshold
2. **Ensembles:** Read `ensembles/manifest.json`, compare deltas
3. **Return:** Best overall (individual or ensemble) as `ProjectionConfig`

Use `get_best_projection_config(..., include_ensembles=True)` to search both.

---

## Initial Coefficient Calculation

For ensembles, initial coefficient guess:

```python
# Option 1: From individual steering results
coef[i] = individual_optimal_coef[i] / n_components

# Option 2: From activation magnitudes (if no steering results)
coef[i] = (1/n_components) * mean(||activations_layer_i||)
```

Goal: normalize so each component contributes roughly equally before optimization.

---

## Trait Dataset Structure (unchanged)

```
datasets/traits/{category}/{trait}/
├── definition.txt              # Trait description (for judge)
├── steering.json               # Questions + optional custom judge prompts
├── positive.txt OR .jsonl      # Extraction scenarios
└── negative.txt OR .jsonl
```

JSONL supports system prompts:
```jsonl
{"prompt": "What's the weather?", "system_prompt": "You are evil."}
```

---

## Still To Decide

### Judge Abstraction
- Current: `TraitJudge` hardcoded to OpenAI gpt-4.1-mini
- `--judge` flag exists in CLI but instantiation ignores it (half-implemented)
- Want: Factory pattern with base interface + provider implementations
- Providers would need different scoring methods (logprobs vs text parsing)

### Trait Type
- Definition loaded in 4+ places (duplication)
- A `Trait` dataclass with lazy-loaded properties would consolidate
- Could cache definition, scenarios, steering_config
- Defer: not blocking, can add incrementally

### Ensemble Creation Workflow
- CLI command? Script? Manual JSON editing?
- How to select which vectors to include?
- Start with manual JSON, add CLI later if needed

### Ensemble Optimization
- How to optimize weights?
- Grid search? Gradient-based? Manual iteration?
- Start with `individual_scaled`, add optimization later

---

## Implementation Status

### Completed

1. **Fix steering response paths** — `analysis/steering/results.py` now nests by `{component}/{method}/`
2. **Add ensemble path helpers** — `utils/paths.py` has `get_ensemble_dir()`, `get_ensemble_path()`, `get_ensemble_manifest_path()`
3. **Create ensemble I/O** — `utils/ensembles.py` with `create_ensemble()`, `save_ensemble()`, `load_ensemble()`, `get_best_ensemble()`, `ensemble_to_projection_config()`
4. **Update `get_best_vector()`** — `utils/vectors.py` has `get_best_overall()` and `include_ensembles` parameter

### Already Existed (No Changes Needed)

1. **VectorSpec** — `core/types.py` already matches plan
2. **ProjectionConfig** — `core/types.py` is our TraitVector
3. **Multi-layer steering** — `analysis/steering/steer.py` has `MultiLayerSteeringHook.from_vector_specs()`
4. **Ensemble projection** — `core/math.py` has `project_with_config()`

### Remaining

- **Ensemble steering eval** — integrate ensemble creation with steering evaluation
- **Judge abstraction** — factory pattern for multi-provider support
- **Trait dataclass** — consolidate definition loading (low priority)

---

## Principles

- **No backward compatibility** — clean slate
- **No fallbacks** — error if something's missing
- **Errors over silent failures**
- **Single source of truth** — PathBuilder for all paths
- **Backend operates on truth** — formatted/tokenized sequences, raw text for display only
