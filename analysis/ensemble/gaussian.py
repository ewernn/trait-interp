"""
Gaussian weight computation for layer ensembles.

Input:
    - mu: Center layer (Gaussian peak)
    - sigma: Spread (standard deviation)
    - layers: List of layer indices

Output:
    - Normalized weights for each layer

Usage:
    from analysis.ensemble.gaussian import compute_gaussian_weights, create_ensemble_vector

    weights = compute_gaussian_weights(mu=12, sigma=2, layers=range(26))
    ensemble_vec = create_ensemble_vector(vectors, weights)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from utils.paths import get as get_path


def compute_gaussian_weights(
    mu: float,
    sigma: float,
    layers: List[int],
    min_weight: float = 0.01,
) -> Dict[int, float]:
    """
    Compute normalized Gaussian weights for each layer.

    Args:
        mu: Center layer (Gaussian peak)
        sigma: Spread (standard deviation)
        layers: List of layer indices
        min_weight: Minimum weight threshold (layers below this are excluded)

    Returns:
        Dict mapping layer -> normalized weight (sum to 1)

    Formula:
        w_i = exp(-(i - mu)^2 / (2 * sigma^2)) / Z
        where Z = sum of all unnormalized weights
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    # Compute unnormalized weights
    raw_weights = {}
    for layer in layers:
        weight = np.exp(-((layer - mu) ** 2) / (2 * sigma ** 2))
        if weight >= min_weight:
            raw_weights[layer] = weight

    # Normalize
    total = sum(raw_weights.values())
    if total == 0:
        raise ValueError(f"All weights below threshold {min_weight} for mu={mu}, sigma={sigma}")

    return {layer: w / total for layer, w in raw_weights.items()}


def create_ensemble_vector(
    vectors: Dict[int, torch.Tensor],
    weights: Dict[int, float],
) -> torch.Tensor:
    """
    Create weighted ensemble of layer vectors.

    Args:
        vectors: Dict mapping layer -> vector tensor [hidden_dim]
        weights: Dict mapping layer -> weight (should sum to ~1)

    Returns:
        Weighted sum of vectors: sum(w_i * v_i)
    """
    # Find common layers
    common_layers = set(vectors.keys()) & set(weights.keys())
    if not common_layers:
        raise ValueError("No common layers between vectors and weights")

    # Get reference for shape/dtype/device
    ref_layer = next(iter(common_layers))
    ref_vec = vectors[ref_layer]

    # Renormalize weights over available layers
    total = sum(weights[l] for l in common_layers)
    norm_weights = {l: weights[l] / total for l in common_layers}

    # Compute weighted sum
    result = torch.zeros_like(ref_vec, dtype=torch.float32)
    for layer in common_layers:
        result += norm_weights[layer] * vectors[layer].float()

    return result


def load_vectors_for_trait(
    experiment: str,
    trait: str,
    method: str = "probe",
    component: str = "residual",
    layers: Optional[List[int]] = None,
) -> Dict[int, torch.Tensor]:
    """
    Load all layer vectors for a trait.

    Args:
        experiment: Experiment name
        trait: Trait path (category/trait)
        method: Extraction method (probe, mean_diff, gradient)
        component: residual, attn_out, mlp_out, k_cache, v_cache
        layers: Optional layer filter (default: all available)

    Returns:
        Dict mapping layer -> vector tensor
    """
    vectors_dir = get_path('extraction.vectors', experiment=experiment, trait=trait)

    if not vectors_dir.exists():
        raise FileNotFoundError(f"Vectors directory not found: {vectors_dir}")

    prefix = "" if component == "residual" else f"{component}_"
    pattern = f"{prefix}{method}_layer*.pt"

    vectors = {}
    for path in vectors_dir.glob(pattern):
        # Extract layer number from filename
        # e.g., "probe_layer16.pt" -> 16
        name = path.stem  # "probe_layer16"
        layer_str = name.split("_layer")[-1]  # "16"
        try:
            layer = int(layer_str)
        except ValueError:
            continue

        if layers is not None and layer not in layers:
            continue

        vectors[layer] = torch.load(path, weights_only=True)

    if not vectors:
        raise FileNotFoundError(
            f"No vectors found for {experiment}/{trait} with method={method}, component={component}"
        )

    return vectors


def load_all_validation_activations(
    experiment: str,
    trait: str,
    component: str = "residual",
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Load validation activations for all layers.

    Args:
        experiment: Experiment name
        trait: Trait path (category/trait)
        component: residual, attn_out, etc.

    Returns:
        Tuple of (pos_activations, neg_activations) where each is Dict[layer -> tensor]
    """
    val_dir = get_path('extraction.val_activations', experiment=experiment, trait=trait)

    if not val_dir.exists():
        raise FileNotFoundError(f"Validation activations not found: {val_dir}")

    prefix = "" if component == "residual" else f"{component}_"

    pos_acts = {}
    neg_acts = {}

    for path in val_dir.glob(f"{prefix}val_pos_layer*.pt"):
        layer_str = path.stem.split("_layer")[-1]
        try:
            layer = int(layer_str)
        except ValueError:
            continue

        neg_path = val_dir / f"{prefix}val_neg_layer{layer}.pt"
        if neg_path.exists():
            pos_acts[layer] = torch.load(path, weights_only=True)
            neg_acts[layer] = torch.load(neg_path, weights_only=True)

    if not pos_acts:
        raise FileNotFoundError(f"No validation activations found in {val_dir}")

    return pos_acts, neg_acts


def get_active_layers(
    mu: float,
    sigma: float,
    num_layers: int = 26,
    min_weight: float = 0.01,
) -> List[int]:
    """
    Get list of layers with significant weight (above threshold).

    Useful for knowing which layers actually contribute to the ensemble.

    Args:
        mu: Center layer
        sigma: Spread
        num_layers: Total number of layers
        min_weight: Minimum normalized weight to include

    Returns:
        List of layer indices with weight >= min_weight
    """
    weights = compute_gaussian_weights(mu, sigma, list(range(num_layers)), min_weight=min_weight)
    return sorted(weights.keys())


def compute_effective_width(sigma: float, threshold: float = 0.95) -> float:
    """
    Compute effective width of Gaussian (how many layers contain threshold% of weight).

    For a Gaussian, ~95% of weight is within 2*sigma of the mean.

    Args:
        sigma: Gaussian spread
        threshold: Fraction of weight to contain (default: 0.95)

    Returns:
        Effective width in layers
    """
    # For normal distribution, z for 95% is ~1.96
    # So effective width = 2 * 1.96 * sigma ~ 4 * sigma
    if threshold == 0.95:
        return 4 * sigma
    else:
        from scipy.stats import norm
        z = norm.ppf((1 + threshold) / 2)
        return 2 * z * sigma
