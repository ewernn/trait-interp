"""Shared projection file reader. Handles both single-vector and multi-vector formats.

Input: Projection JSON files from inference/project_raw_activations_onto_traits.py
Output: Normalized projection data dicts

Usage:
    from utils.projections import read_projection, read_response_projections
    proj = read_projection(path)  # {prompt, response, token_norms, layer, method, baseline, selection_source}
    scores = read_response_projections(path)  # Just response scores
"""

import json
from pathlib import Path
from typing import Optional


def read_projection(path, layer=None) -> dict:
    """Read a projection file and return normalized data.

    Handles both single-vector and multi-vector formats transparently.

    Args:
        path: Path to projection JSON file
        layer: Layer to extract (for multi-vector files). If None, uses first/only entry.

    Returns:
        Dict with keys: prompt, response, token_norms, layer, method, baseline, selection_source
    """
    with open(path) as f:
        data = json.load(f)

    projections = data['projections']

    if isinstance(projections, list):
        # Multi-vector format: projections is a list of dicts
        if layer is not None:
            entry = next((e for e in projections if e['layer'] == layer), None)
            if entry is None:
                available = [e['layer'] for e in projections]
                raise ValueError(f"Layer {layer} not found in {path}. Available: {available}")
        else:
            entry = projections[0]

        # token_norms: per-entry (new format) or top-level (old multi-vector)
        if 'token_norms' in entry:
            token_norms = entry['token_norms']
        else:
            token_norms = data.get('token_norms', {})

        return {
            'prompt': entry.get('prompt', []),
            'response': entry.get('response', []),
            'token_norms': token_norms,
            'layer': entry['layer'],
            'method': entry.get('method'),
            'baseline': entry.get('baseline', 0.0),
            'selection_source': entry.get('selection_source'),
        }
    else:
        # Single-vector format: projections is {prompt: [...], response: [...]}
        vector_source = data.get('metadata', {}).get('vector_source', {})

        return {
            'prompt': projections.get('prompt', []),
            'response': projections.get('response', []),
            'token_norms': data.get('token_norms', {}),
            'layer': vector_source.get('layer'),
            'method': vector_source.get('method'),
            'baseline': vector_source.get('baseline', 0.0),
            'selection_source': vector_source.get('selection_source'),
        }


def read_response_projections(path, layer=None) -> list:
    """Convenience: returns just the response projection array."""
    return read_projection(path, layer=layer)['response']
