"""Fingerprint comparison metrics.

cosine_sim: cosine similarity between numpy vectors
spearman_corr: Spearman rank correlation
short_name: extract short trait name from path
fingerprint_delta: per-trait delta between two score dicts
"""

import numpy as np
from scipy.stats import spearmanr


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation between two vectors. Returns (rho, p-value)."""
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


def short_name(trait: str) -> str:
    """Extract short trait name: 'alignment/deception' -> 'deception'."""
    return trait.split("/")[-1]


def fingerprint_delta(scores_a: dict[str, float], scores_b: dict[str, float]) -> dict[str, float]:
    """Per-trait delta between two score dicts (a - b).

    Args:
        scores_a: {trait: mean_score} for condition A
        scores_b: {trait: mean_score} for condition B

    Returns:
        {trait: delta} for traits present in both
    """
    return {trait: scores_a[trait] - scores_b[trait]
            for trait in scores_a if trait in scores_b}
