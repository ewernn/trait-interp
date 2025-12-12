# Gaussian-Weighted Trait Vector Ensemble

Combines trait vectors across layers using Gaussian weighting instead of picking a single best layer.

## Motivation

Traits are likely distributed across multiple layers, not concentrated in one. A single "best layer" discards information from neighboring layers that also encode the trait. This module learns an optimal Gaussian distribution over layers for each trait.

## Method

Instead of using a single layer's vector, compute a weighted combination:

```
combined_vector = Σ w_i * vector_layer_i
where w_i = exp(-(i - μ)² / 2σ²) / Z  (normalized)
```

**Parameters:**
- **μ (mu)**: Center layer where the Gaussian peaks
- **σ (sigma)**: Spread - how many neighboring layers contribute
- **coef**: Global steering coefficient (Phase 2 only)

## Usage

### Phase 1: Classification Grid Search

Find optimal (μ, σ) per trait using validation data:

```bash
# Full grid search (26 μ × 5 σ = 130 combinations per trait)
python analysis/ensemble/classification_search.py \
    --experiment gemma-2-2b \
    --traits all

# Quick mode (fix σ=2, sweep μ only - 26 combinations per trait)
python analysis/ensemble/classification_search.py \
    --experiment gemma-2-2b \
    --traits all \
    --quick

# Specific traits
python analysis/ensemble/classification_search.py \
    --experiment gemma-2-2b \
    --traits epistemic/confidence,epistemic/optimism
```

**Output:** `experiments/{experiment}/extraction/ensemble_evaluation.json`

### Phase 2: Steering Evaluation

Validate the ensemble via causal intervention:

```bash
# Use best (μ, σ) from Phase 1
python analysis/ensemble/steering_evaluation.py \
    --experiment {experiment} \
    --vector-from-trait {experiment}/{category}/{trait}

# Manual (μ, σ) specification
python analysis/ensemble/steering_evaluation.py \
    --experiment {experiment} \
    --vector-from-trait {experiment}/{category}/{trait} \
    --mu 12 --sigma 2 \
    --coefficients 50,100,150

# Compare to single-layer baseline
python analysis/ensemble/steering_evaluation.py \
    --experiment {experiment} \
    --vector-from-trait {experiment}/{category}/{trait} \
    --compare-baseline
```

**Output:** Appends to `experiments/{experiment}/steering/{trait}/results.json`

## API

### Core Functions

```python
from analysis.ensemble.gaussian import (
    compute_gaussian_weights,  # (mu, sigma, layers) → Dict[layer, weight]
    create_ensemble_vector,    # (vectors, weights) → combined vector
    load_vectors_for_trait,    # Load all layer vectors
    get_active_layers,         # Layers with weight > threshold
)

# Example
weights = compute_gaussian_weights(mu=12, sigma=2, layers=range(26))
# {10: 0.06, 11: 0.15, 12: 0.24, 13: 0.24, 14: 0.15, ...}

vectors = load_vectors_for_trait('gemma-2-2b', 'epistemic/confidence', method='probe')
ensemble = create_ensemble_vector(vectors, weights)
```

### Best Ensemble Lookup

```python
from utils.vectors import get_best_ensemble

params = get_best_ensemble('gemma-2-2b', 'epistemic/confidence')
# {'mu': 12, 'sigma': 2, 'val_accuracy': 0.92, ...}
```

## Output Format

### ensemble_evaluation.json

```json
{
  "grid_config": {
    "mu_values": [0, 1, ..., 25],
    "sigma_values": [1, 2, 3, 4, 5],
    "method": "probe"
  },
  "all_results": [
    {"trait": "...", "mu": 12, "sigma": 2, "val_accuracy": 0.92, ...}
  ],
  "best_per_trait": {
    "epistemic/confidence": {"mu": 12, "sigma": 2, "val_accuracy": 0.92}
  },
  "comparison": {
    "epistemic/confidence": {
      "single_layer_best": {"layer": 14, "val_accuracy": 0.88},
      "ensemble_best": {"mu": 12, "sigma": 2, "val_accuracy": 0.92},
      "accuracy_improvement": 0.04
    }
  }
}
```

### Steering results.json (ensemble run)

```json
{
  "config": {
    "ensemble": {
      "mu": 12,
      "sigma": 2,
      "global_coefficient": 100,
      "active_layers": [10, 11, 12, 13, 14],
      "layer_coefficients": {"10": 6, "11": 15, "12": 24, ...},
      "layer_weights": {"10": 0.06, "11": 0.15, "12": 0.24, ...}
    },
    "method": "probe",
    "component": "residual"
  },
  "result": {
    "trait_mean": 85.2,
    "coherence_mean": 78.5
  }
}
```

## Files

```
analysis/ensemble/
├── __init__.py                # Module exports
├── gaussian.py                # Core Gaussian weight computation
├── classification_search.py   # Phase 1: grid search
├── steering_evaluation.py     # Phase 2: steering eval
└── README.md                  # This file
```
