"""
Layer specification parsing.

Input: Layer spec string (e.g., "25,30,35,40", "0-75:5", "30%-60%", "all")
Output: Sorted list of unique layer indices

Usage:
    from utils.layers import parse_layers
    layers = parse_layers("25,30,35,40", n_layers=80)
"""

from typing import List


def parse_layers(layers_str: str, n_layers: int) -> List[int]:
    """Parse layer specification string into list of layer indices.

    Supports:
        - "all" — all layers
        - "16" — single layer
        - "5,10,15" — comma-separated list
        - "5-20" — inclusive range
        - "0-75:5" — range with step
        - "30%-60%" — percentage range (30% to 60% of depth)
        - Mixed: "5,10-15,20-30:5" — combines all above

    Args:
        layers_str: Layer specification string
        n_layers: Total number of model layers (for validation and percentage calc)

    Returns:
        Sorted list of unique layer indices within [0, n_layers)
    """
    if layers_str.strip().lower() == "all":
        return list(range(n_layers))

    layers = []
    for part in layers_str.split(','):
        part = part.strip()
        if not part:
            continue

        if '%' in part:
            # Percentage range: "30%-60%"
            pct_parts = part.replace('%', '').split('-')
            start_pct = int(pct_parts[0]) / 100
            end_pct = int(pct_parts[1]) / 100 if len(pct_parts) > 1 else start_pct
            start = int(n_layers * start_pct)
            end = int(n_layers * end_pct)
            layers.extend(range(start, end + 1))
        elif ':' in part:
            # Range with step: "0-75:5"
            range_part, step = part.split(':')
            start, end = range_part.split('-')
            layers.extend(range(int(start), int(end) + 1, int(step)))
        elif '-' in part and not part.startswith('-'):
            # Simple range: "5-20"
            start, end = part.split('-')
            layers.extend(range(int(start), int(end) + 1))
        else:
            # Single layer: "16"
            layers.append(int(part))

    return sorted(set(l for l in layers if 0 <= l < n_layers))
