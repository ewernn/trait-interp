"""
Gaussian-weighted trait vector ensemble.

Combines vectors across layers using Gaussian weights instead of picking a single best layer.
"""

from analysis.ensemble.gaussian import (
    compute_gaussian_weights,
    create_ensemble_vector,
    load_vectors_for_trait,
)

__all__ = [
    'compute_gaussian_weights',
    'create_ensemble_vector',
    'load_vectors_for_trait',
]
