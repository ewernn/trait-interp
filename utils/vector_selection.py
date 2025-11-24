#!/usr/bin/env python3
"""
Utility for selecting the best vector per trait using evaluation metrics.

Uses the same ranking formula as the visualization dashboard:
    score = 0.5 * val_accuracy + 0.5 * (val_effect_size / max_effect_size_for_trait)
"""

import json
from pathlib import Path
from typing import Tuple, Optional

from utils.paths import get as get_path


def get_best_vector(experiment: str, trait: str, fallback_method: str = 'probe', fallback_layer: int = 16) -> Tuple[str, int]:
    """Get best (method, layer) for a trait using evaluation metrics.

    Args:
        experiment: Experiment name
        trait: Trait name (e.g., 'behavioral_tendency/refusal')
        fallback_method: Method to use if evaluation not found (default: 'probe')
        fallback_layer: Layer to use if evaluation not found (default: 16)

    Returns:
        (method, layer) tuple for the best vector

    Example:
        >>> method, layer = get_best_vector('gemma_2b_cognitive_nov21', 'behavioral_tendency/refusal')
        >>> vec_path = f"{method}_layer{layer}.pt"
    """
    eval_path = get_path('extraction_eval.evaluation', experiment=experiment)

    if not eval_path.exists():
        return (fallback_method, fallback_layer)

    try:
        with open(eval_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return (fallback_method, fallback_layer)

    # Filter to this trait
    trait_results = [r for r in data.get('all_results', []) if r.get('trait') == trait]

    if not trait_results:
        return (fallback_method, fallback_layer)

    # Find max effect size for this trait (for normalization)
    max_effect = max((r.get('val_effect_size', 0) or 0) for r in trait_results)
    if max_effect == 0:
        max_effect = 1.0

    # Calculate composite score (same as JS: trait-dashboard.js line 172-175)
    def calculate_score(r):
        acc = r.get('val_accuracy')
        effect = r.get('val_effect_size', 0) or 0
        if acc is None:
            return 0.0
        return 0.5 * acc + 0.5 * (effect / max_effect)

    # Find best by score
    best = max(trait_results, key=calculate_score)

    return (best['method'], best['layer'])


def load_best_vectors(experiment: str, traits: list = None, fallback_method: str = 'probe', fallback_layer: int = 16) -> dict:
    """Load best vectors for multiple traits.

    Args:
        experiment: Experiment name
        traits: List of trait names, or None to load all from evaluation
        fallback_method: Method to use if evaluation not found
        fallback_layer: Layer to use if evaluation not found

    Returns:
        Dict mapping trait_name -> {'method': str, 'layer': int, 'score': float}
    """
    eval_path = get_path('extraction_eval.evaluation', experiment=experiment)

    if not eval_path.exists():
        return {}

    try:
        with open(eval_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

    all_results = data.get('all_results', [])

    # If traits not specified, get all unique traits
    if traits is None:
        traits = list(set(r['trait'] for r in all_results if 'trait' in r))

    best_vectors = {}

    for trait in traits:
        trait_results = [r for r in all_results if r.get('trait') == trait]

        if not trait_results:
            best_vectors[trait] = {
                'method': fallback_method,
                'layer': fallback_layer,
                'score': 0.0,
                'fallback': True
            }
            continue

        # Find max effect size for this trait
        max_effect = max((r.get('val_effect_size', 0) or 0) for r in trait_results)
        if max_effect == 0:
            max_effect = 1.0

        # Calculate scores
        def calculate_score(r):
            acc = r.get('val_accuracy')
            effect = r.get('val_effect_size', 0) or 0
            if acc is None:
                return 0.0
            return 0.5 * acc + 0.5 * (effect / max_effect)

        best = max(trait_results, key=calculate_score)

        best_vectors[trait] = {
            'method': best['method'],
            'layer': best['layer'],
            'score': calculate_score(best),
            'val_accuracy': best.get('val_accuracy'),
            'val_effect_size': best.get('val_effect_size'),
            'fallback': False
        }

    return best_vectors
