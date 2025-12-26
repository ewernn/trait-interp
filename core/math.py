"""
Math primitives for trait vector analysis.

projection: measure trait expression in activations
evaluate_vector: compute all quality metrics for a vector
activation_norms: mean ||h|| per layer (for steering coefficient estimation)
vector_properties: norm, sparsity of a vector
distribution_properties: std, overlap, margin of projection distributions
"""

import torch
import numpy as np
from typing import Dict, List, Union
from scipy import stats


def projection(
    activations: torch.Tensor,
    vector: torch.Tensor,
    normalize_vector: bool = True
) -> torch.Tensor:
    """
    Project activations onto a trait vector.

    Args:
        activations: [*, hidden_dim]
        vector: [hidden_dim]
        normalize_vector: if True, normalize vector to unit length

    Returns:
        [*] projection scores (higher = more trait expression)
    """
    if normalize_vector:
        vector = vector / (vector.norm() + 1e-8)
    return torch.matmul(activations, vector)


def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two vectors. Returns scalar in [-1, 1]."""
    v1 = vec1 / (vec1.norm() + 1e-8)
    v2 = vec2 / (vec2.norm() + 1e-8)
    return (v1 * v2).sum()


def orthogonalize(v: torch.Tensor, onto: torch.Tensor) -> torch.Tensor:
    """Remove onto's component from v. Returns v with onto projected out."""
    v_flat, onto_flat = v.flatten(), onto.flatten()
    norm_sq = (onto_flat @ onto_flat)
    if norm_sq < 1e-10:
        return v
    proj = (v_flat @ onto_flat) / norm_sq * onto_flat
    return (v_flat - proj).view_as(v)


def separation(pos_proj: torch.Tensor, neg_proj: torch.Tensor) -> float:
    """Absolute difference between mean projections."""
    return (pos_proj.mean() - neg_proj.mean()).abs().item()


def accuracy(pos_proj: torch.Tensor, neg_proj: torch.Tensor, threshold: float = None) -> float:
    """Classification accuracy. Positive should score above threshold, negative below."""
    if threshold is None:
        threshold = (pos_proj.mean() + neg_proj.mean()) / 2
    pos_correct = (pos_proj > threshold).float().mean().item()
    neg_correct = (neg_proj <= threshold).float().mean().item()
    return (pos_correct + neg_correct) / 2


def effect_size(pos_proj: torch.Tensor, neg_proj: torch.Tensor, signed: bool = False) -> float:
    """Cohen's d: separation in units of std. 0.2=small, 0.5=medium, 0.8=large.

    Args:
        signed: If True, preserve sign (positive = pos > neg). Default False (absolute value).
    """
    pooled_std = torch.sqrt((pos_proj.std()**2 + neg_proj.std()**2) / 2)
    if pooled_std <= 0:
        return 0.0
    d = ((pos_proj.mean() - neg_proj.mean()) / pooled_std).item()
    return d if signed else abs(d)


def p_value(pos_proj: torch.Tensor, neg_proj: torch.Tensor) -> float:
    """Two-tailed t-test p-value. Lower = more significant separation."""
    _, p = stats.ttest_ind(pos_proj.cpu().numpy(), neg_proj.cpu().numpy())
    return float(p)


def polarity_correct(pos_proj: torch.Tensor, neg_proj: torch.Tensor) -> bool:
    """Check if positive examples score higher than negative."""
    return bool(pos_proj.mean() > neg_proj.mean())


def evaluate_vector(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    vector: torch.Tensor,
    normalize: bool = True,
) -> Dict[str, Union[float, bool]]:
    """
    Compute all evaluation metrics for a trait vector.

    Args:
        pos_acts: [n_pos, hidden_dim]
        neg_acts: [n_neg, hidden_dim]
        vector: [hidden_dim]
        normalize: if True, normalize vector and activations (cosine similarity)

    Returns:
        accuracy, separation, effect_size, p_value, polarity_correct, pos_mean, neg_mean
    """
    pos_acts = pos_acts.float()
    neg_acts = neg_acts.float()
    vector = vector.float()

    if normalize:
        vector = vector / (vector.norm() + 1e-8)
        pos_acts = pos_acts / (pos_acts.norm(dim=1, keepdim=True) + 1e-8)
        neg_acts = neg_acts / (neg_acts.norm(dim=1, keepdim=True) + 1e-8)

    pos_proj = pos_acts @ vector
    neg_proj = neg_acts @ vector

    return {
        'accuracy': accuracy(pos_proj, neg_proj),
        'separation': separation(pos_proj, neg_proj),
        'effect_size': effect_size(pos_proj, neg_proj),
        'p_value': p_value(pos_proj, neg_proj),
        'polarity_correct': polarity_correct(pos_proj, neg_proj),
        'pos_mean': pos_proj.mean().item(),
        'neg_mean': neg_proj.mean().item(),
    }


def vector_properties(vector: torch.Tensor) -> Dict[str, float]:
    """
    Compute properties of a vector.

    Args:
        vector: [hidden_dim] tensor

    Returns:
        {'norm': float, 'sparsity': float (fraction of components < 0.01)}
    """
    vector_np = vector.float().cpu().numpy()
    return {
        'norm': float(np.linalg.norm(vector_np)),
        'sparsity': float(np.mean(np.abs(vector_np) < 0.01)),
    }


def distribution_properties(
    pos_proj: torch.Tensor,
    neg_proj: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute properties of projection score distributions.

    Args:
        pos_proj: [n_pos] projection scores for positive examples
        neg_proj: [n_neg] projection scores for negative examples

    Returns:
        pos_std, neg_std, overlap_coefficient, separation_margin
    """
    pos_mean = pos_proj.mean().item()
    neg_mean = neg_proj.mean().item()
    pos_std = float(pos_proj.std())
    neg_std = float(neg_proj.std())

    # Overlap coefficient: estimate using normal approximation
    if pos_std > 0 and neg_std > 0:
        pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
        z_score = abs(pos_mean - neg_mean) / (pooled_std + 1e-8)
        overlap = max(0, 1 - z_score / 4.0)  # z=4 → no overlap
    else:
        overlap = 0.5

    # Separation margin: gap between distributions (positive = good separation)
    margin = (pos_mean - pos_std) - (neg_mean + neg_std)

    return {
        'pos_std': pos_std,
        'neg_std': neg_std,
        'overlap_coefficient': float(overlap),
        'separation_margin': float(margin),
    }
