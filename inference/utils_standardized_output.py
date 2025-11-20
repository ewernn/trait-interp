#!/usr/bin/env python3
"""
Utilities for standardized inference output format.

Saves inference results in deduplicated format:
- Shared prompts: experiments/{exp}/inference/prompts/prompt_N.json
- Per-trait projections: experiments/{exp}/inference/projections/{trait}/prompt_N.json
"""

import json
from pathlib import Path
from typing import Dict, List


def save_standardized_inference(
    result: Dict,
    experiment: str,
    prompt_idx: int,
    layer: int,
    method: str
) -> None:
    """
    Save inference result in standardized format.

    Args:
        result: Result dict from run_inference_with_dynamics
        experiment: Experiment name
        prompt_idx: Prompt index (0, 1, 2, ...)
        layer: Layer number used for inference
        method: Method name used (probe, mean_diff, etc.)
    """
    exp_inference_dir = Path('experiments') / experiment / 'inference'

    # Create directories
    prompts_dir = exp_inference_dir / 'prompts'
    projections_dir = exp_inference_dir / 'projections'
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Save shared prompt data (only if doesn't exist)
    prompt_file = prompts_dir / f'prompt_{prompt_idx}.json'
    if not prompt_file.exists():
        shared_data = {
            'prompt': result['prompt'],
            'response': result['response'],
            'tokens': result['tokens'],
            'prompt_idx': prompt_idx
        }
        with open(prompt_file, 'w') as f:
            json.dump(shared_data, f, indent=2)

    # Save per-trait projections
    for trait_name, scores in result['trait_scores'].items():
        trait_dir = projections_dir / trait_name
        trait_dir.mkdir(parents=True, exist_ok=True)

        projection_file = trait_dir / f'prompt_{prompt_idx}.json'
        projection_data = {
            'prompt_idx': prompt_idx,
            'trait': trait_name,
            'scores': scores,
            'dynamics': result['dynamics'].get(trait_name, {}),
            'metadata': {
                'layer': layer,
                'method': method
            }
        }

        with open(projection_file, 'w') as f:
            json.dump(projection_data, f, indent=2)


def load_standardized_inference(
    experiment: str,
    prompt_idx: int,
    traits: List[str] = None
) -> Dict:
    """
    Load inference result from standardized format.

    Args:
        experiment: Experiment name
        prompt_idx: Prompt index
        traits: List of trait names to load (None = all available)

    Returns:
        Combined result dict (same format as run_inference_with_dynamics)
    """
    exp_inference_dir = Path('experiments') / experiment / 'inference'

    # Load shared prompt
    prompt_file = exp_inference_dir / 'prompts' / f'prompt_{prompt_idx}.json'
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file) as f:
        result = json.load(f)

    # Load trait projections
    projections_dir = exp_inference_dir / 'projections'

    if traits is None:
        # Auto-discover all traits
        traits = [d.name for d in projections_dir.iterdir() if d.is_dir()]

    trait_scores = {}
    dynamics = {}

    for trait in traits:
        projection_file = projections_dir / trait / f'prompt_{prompt_idx}.json'
        if projection_file.exists():
            with open(projection_file) as f:
                data = json.load(f)
                trait_scores[trait] = data['scores']
                dynamics[trait] = data.get('dynamics', {})

    result['trait_scores'] = trait_scores
    result['dynamics'] = dynamics

    return result
