#!/usr/bin/env python3
"""
Stage 1: Generate responses for natural elicitation.

Input:
    - experiments/{experiment}/extraction/{category}/{trait}/positive.txt
    - experiments/{experiment}/extraction/{category}/{trait}/negative.txt

Output:
    - experiments/{experiment}/extraction/{category}/{trait}/responses/pos.json
    - experiments/{experiment}/extraction/{category}/{trait}/responses/neg.json
    - experiments/{experiment}/extraction/{category}/{trait}/generation_metadata.json

Usage:
    # Single trait
    python extraction/generate_responses.py --experiment my_exp --trait category/my_trait

    # All traits
    python extraction/generate_responses.py --experiment my_exp --trait all
"""

import sys
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get as get_path


def discover_traits(experiment: str) -> list[str]:
    """Find all traits with positive.txt and negative.txt files."""
    extraction_dir = get_path('extraction.base', experiment=experiment)
    traits = []
    if not extraction_dir.is_dir():
        return []

    for category_dir in extraction_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        for trait_dir in category_dir.iterdir():
            if not trait_dir.is_dir():
                continue
            # Check for scenario files
            if (trait_dir / 'positive.txt').exists() and (trait_dir / 'negative.txt').exists():
                traits.append(f"{category_dir.name}/{trait_dir.name}")
    return sorted(traits)


def load_scenarios(scenario_file: Path) -> list[str]:
    """Load scenarios from text file (one per line)."""
    with open(scenario_file, 'r') as f:
        scenarios = [line.strip() for line in f if line.strip()]
    return scenarios


def generate_responses_for_trait(
    experiment: str,
    trait: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 200,
    batch_size: int = 8,
) -> tuple[int, int]:
    """
    Generate responses for natural scenarios.

    Args:
        experiment: Experiment name.
        trait: Trait path like "category/trait_name".
        model: Pre-loaded HuggingFace model.
        tokenizer: Pre-loaded HuggingFace tokenizer.
        max_new_tokens: Max tokens to generate per response.
        batch_size: Batch size for generation.

    Returns:
        Tuple of (n_positive, n_negative) responses generated.
    """
    print(f"  [Stage 1] Generating responses for '{trait}'...")

    trait_dir = get_path('extraction.trait', experiment=experiment, trait=trait)
    responses_dir = get_path('extraction.responses', experiment=experiment, trait=trait)
    responses_dir.mkdir(parents=True, exist_ok=True)

    # Load scenario files
    pos_file = trait_dir / 'positive.txt'
    neg_file = trait_dir / 'negative.txt'

    if not pos_file.exists() or not neg_file.exists():
        print(f"    ERROR: Scenario files not found in {trait_dir}")
        print(f"    Expected: positive.txt, negative.txt")
        return 0, 0

    pos_scenarios = load_scenarios(pos_file)
    neg_scenarios = load_scenarios(neg_file)

    def generate_batch(scenarios: list[str], label: str) -> list[dict]:
        results = []
        for i in tqdm(range(0, len(scenarios), batch_size), desc=f"    Generating {label}", leave=False):
            batch_scenarios = scenarios[i:i + batch_size]
            inputs = tokenizer(
                batch_scenarios,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            for j, output in enumerate(outputs):
                prompt_length = inputs['input_ids'][j].shape[0]
                response = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
                results.append({
                    'question': batch_scenarios[j],
                    'answer': response.strip(),
                    'full_text': tokenizer.decode(output, skip_special_tokens=True)
                })
        return results

    pos_results = generate_batch(pos_scenarios, 'positive')
    neg_results = generate_batch(neg_scenarios, 'negative')

    # Save results
    with open(responses_dir / 'pos.json', 'w') as f:
        json.dump(pos_results, f, indent=2)
    with open(responses_dir / 'neg.json', 'w') as f:
        json.dump(neg_results, f, indent=2)

    # Save metadata
    metadata = {
        'experiment': experiment,
        'trait': trait,
        'model_name': model.config.name_or_path,
        'n_positive': len(pos_results),
        'n_negative': len(neg_results),
        'total': len(pos_results) + len(neg_results),
    }
    with open(trait_dir / 'generation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved {len(pos_results)} positive and {len(neg_results)} negative responses.")
    return len(pos_results), len(neg_results)


def main():
    parser = argparse.ArgumentParser(description='Generate natural elicitation responses.')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--trait', type=str, required=True,
                        help='Trait name (e.g., "category/my_trait") or "all" for all traits')
    parser.add_argument('--model', type=str, default='google/gemma-2-2b-it', help='Model name')
    parser.add_argument('--max-new-tokens', type=int, default=200, help='Max tokens to generate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for generation')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cuda, cpu, mps)')

    args = parser.parse_args()

    # Determine traits to process
    if args.trait.lower() == 'all':
        traits = discover_traits(args.experiment)
        if not traits:
            print(f"No traits found in experiment '{args.experiment}'")
            return
        print(f"Found {len(traits)} traits to process")
    else:
        traits = [args.trait]

    print("=" * 80)
    print("GENERATE RESPONSES")
    print(f"Experiment: {args.experiment}")
    print(f"Traits: {len(traits)}")
    print(f"Model: {args.model}")
    print("=" * 80)

    # Load model and tokenizer once
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    model.eval()
    print("Model loaded.\n")

    # Process each trait
    total_pos, total_neg = 0, 0
    for trait in traits:
        n_pos, n_neg = generate_responses_for_trait(
            experiment=args.experiment,
            trait=trait,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size
        )
        total_pos += n_pos
        total_neg += n_neg

    print(f"\nDONE: Generated {total_pos} positive + {total_neg} negative = {total_pos + total_neg} total responses.")


if __name__ == '__main__':
    main()
