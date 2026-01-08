"""
Trait data loading utilities.

Input: Trait name (e.g., "category/trait_name")
Output: Trait definition, scenarios, steering config

Usage:
    from utils.traits import load_trait_definition, load_scenarios

    definition = load_trait_definition("chirp/refusal_v2")
    scenarios = load_scenarios("chirp/refusal_v2")  # {"positive": [...], "negative": [...]}
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from utils.paths import get as get_path


def load_trait_definition(trait: str) -> str:
    """
    Load trait definition from datasets/traits/{trait}/definition.txt.

    Returns the definition text, or a generated fallback if file doesn't exist.
    """
    def_file = get_path('datasets.trait_definition', trait=trait)
    if def_file.exists():
        return def_file.read_text().strip()
    # Fallback: generate from trait name
    trait_name = trait.split('/')[-1].replace('_', ' ')
    return f"The trait '{trait_name}'"


def load_scenarios(trait: str, polarity: str = None) -> Dict[str, List[dict]]:
    """
    Load scenarios from datasets/traits/{trait}/.

    Supports both formats:
    - JSONL: {"prompt": "...", "system_prompt": "..."} per line
    - TXT: One prompt per line (no system_prompt)

    Args:
        trait: Trait path like "category/trait_name"
        polarity: If specified, only load "positive" or "negative".
                  If None, load both.

    Returns:
        Dict with "positive" and/or "negative" keys, each containing
        list of {"prompt": str, "system_prompt": Optional[str]}
    """
    trait_dir = get_path('datasets.trait', trait=trait)
    polarities = [polarity] if polarity else ['positive', 'negative']

    result = {}
    for pol in polarities:
        result[pol] = _load_polarity(trait_dir, pol)

    return result


def _load_polarity(trait_dir: Path, polarity: str) -> List[dict]:
    """Load scenarios for a single polarity (positive or negative)."""
    jsonl_file = trait_dir / f'{polarity}.jsonl'
    txt_file = trait_dir / f'{polarity}.txt'

    if jsonl_file.exists():
        scenarios = []
        with open(jsonl_file, 'r') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{jsonl_file} line {i+1}: invalid JSON: {e}")
                if 'prompt' not in item:
                    raise ValueError(f"{jsonl_file} line {i+1}: missing 'prompt' field")
                scenarios.append(item)
        return scenarios

    elif txt_file.exists():
        with open(txt_file, 'r') as f:
            return [{'prompt': line.strip()} for line in f if line.strip()]

    else:
        raise FileNotFoundError(f"No scenario file found: {jsonl_file} or {txt_file}")


def get_scenario_count(trait: str) -> Dict[str, int]:
    """Get count of scenarios per polarity without loading full data."""
    trait_dir = get_path('datasets.trait', trait=trait)
    counts = {}

    for polarity in ['positive', 'negative']:
        jsonl_file = trait_dir / f'{polarity}.jsonl'
        txt_file = trait_dir / f'{polarity}.txt'

        if jsonl_file.exists():
            with open(jsonl_file, 'r') as f:
                counts[polarity] = sum(1 for line in f if line.strip())
        elif txt_file.exists():
            with open(txt_file, 'r') as f:
                counts[polarity] = sum(1 for line in f if line.strip())
        else:
            counts[polarity] = 0

    return counts
