#!/usr/bin/env python3
"""
Test script for prompt standardization.

Tests the standardized output format by running a quick inference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from inference.utils_inference import save_inference, load_inference


def test_save_and_load():
    """Test saving and loading standardized inference data."""

    # Create a mock result (same format as run_inference_with_dynamics)
    mock_result = {
        'prompt': 'What is the capital of France?',
        'response': 'The capital of France is Paris.',
        'tokens': ['What', ' is', ' the', ' capital', ' of', ' France', '?', ' The', ' capital', ' of', ' France', ' is', ' Paris', '.'],
        'trait_scores': {
            'refusal': [0.1, 0.2, 0.15, 0.3, 0.25, 0.2, 0.1, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -0.8],
            'uncertainty': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.2]
        },
        'dynamics': {
            'refusal': {
                'commitment_point': 7,
                'peak_velocity': 0.4,
                'avg_velocity': 0.2,
                'persistence': 6
            },
            'uncertainty': {
                'commitment_point': 6,
                'peak_velocity': 0.3,
                'avg_velocity': 0.15,
                'persistence': 4
            }
        }
    }

    # Test experiment name
    experiment = 'test_standardization'
    prompt_idx = 0
    layer = 16
    method = 'probe'

    print("Testing prompt standardization...")
    print()

    # Save inference data
    print("1. Saving inference data...")
    save_inference(mock_result, experiment, prompt_idx, layer, method)
    print(f"   ✓ Saved to experiments/{experiment}/inference/")
    print()

    # Check files exist
    prompt_file = Path(f'experiments/{experiment}/inference/prompts/prompt_0.json')
    refusal_file = Path(f'experiments/{experiment}/inference/projections/refusal/prompt_0.json')
    uncertainty_file = Path(f'experiments/{experiment}/inference/projections/uncertainty/prompt_0.json')

    assert prompt_file.exists(), f"Prompt file not found: {prompt_file}"
    assert refusal_file.exists(), f"Refusal projection file not found: {refusal_file}"
    assert uncertainty_file.exists(), f"Uncertainty projection file not found: {uncertainty_file}"
    print("2. Files created:")
    print(f"   ✓ {prompt_file}")
    print(f"   ✓ {refusal_file}")
    print(f"   ✓ {uncertainty_file}")
    print()

    # Load back and verify
    print("3. Loading inference data...")
    loaded_result = load_inference(experiment, prompt_idx, traits=['refusal', 'uncertainty'])
    print("   ✓ Data loaded successfully")
    print()

    # Verify content
    print("4. Verifying data integrity...")
    assert loaded_result['prompt'] == mock_result['prompt'], "Prompt mismatch"
    assert loaded_result['response'] == mock_result['response'], "Response mismatch"
    assert len(loaded_result['tokens']) == len(mock_result['tokens']), "Token count mismatch"
    assert 'refusal' in loaded_result['trait_scores'], "Refusal scores missing"
    assert 'uncertainty' in loaded_result['trait_scores'], "Uncertainty scores missing"
    assert len(loaded_result['trait_scores']['refusal']) == len(mock_result['trait_scores']['refusal']), "Refusal scores length mismatch"
    print("   ✓ All data verified")
    print()

    # Show file sizes
    prompt_size = prompt_file.stat().st_size
    refusal_size = refusal_file.stat().st_size
    uncertainty_size = uncertainty_file.stat().st_size
    total_standardized = prompt_size + refusal_size + uncertainty_size

    print("5. Storage comparison:")
    print(f"   Shared prompt: {prompt_size} bytes")
    print(f"   Refusal projections: {refusal_size} bytes")
    print(f"   Uncertainty projections: {uncertainty_size} bytes")
    print(f"   Total (standardized): {total_standardized} bytes")
    print()
    print(f"   If we had 20 traits with old format:")
    print(f"   Old format (duplicated): ~{total_standardized * 20 // 3} bytes")
    print(f"   New format (deduplicated): ~{prompt_size + (refusal_size + uncertainty_size) * 10} bytes")
    print(f"   Savings: ~{100 - (prompt_size + (refusal_size + uncertainty_size) * 10) * 100 // (total_standardized * 20 // 3)}%")
    print()

    print("✅ All tests passed!")
    print()
    print("Inference format is ready to use.")
    print("Run inference: python inference/monitor_dynamics.py --experiment gemma_2b_cognitive_nov20 --prompts 'Your prompt here'")

    return True


if __name__ == '__main__':
    test_save_and_load()
