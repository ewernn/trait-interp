#!/usr/bin/env python3
"""
Expand Persona Vectors factored format to our JSONL format.

Input: PV's {instructions, questions, eval_prompt} JSON
Output: Our {positive.jsonl, negative.jsonl, definition.txt, steering.json}

Usage:
    python scripts/expand_pv_format.py \
        --input experiments/persona_vectors_replication/their_data/evil_extract.json \
        --output datasets/traits/persona_vectors_instruction/evil

    # Process all PV traits
    python scripts/expand_pv_format.py --all
"""

import json
import argparse
from pathlib import Path


def load_pv_format(input_path: Path) -> dict:
    """Load PV's factored format."""
    with open(input_path) as f:
        return json.load(f)


def expand_to_jsonl(data: dict) -> tuple[list[dict], list[dict]]:
    """
    Expand instructions × questions to scenario lists.

    Returns (positive_scenarios, negative_scenarios)
    """
    instructions = data['instruction']
    questions = data['questions']

    positive = []
    negative = []

    for instr in instructions:
        for question in questions:
            positive.append({
                'prompt': question,
                'system_prompt': instr['pos']
            })
            negative.append({
                'prompt': question,
                'system_prompt': instr['neg']
            })

    return positive, negative


def extract_definition(data: dict, trait_name: str) -> str:
    """Extract a short definition from eval_prompt or generate from trait name."""
    # Try to extract from eval_prompt if it exists
    eval_prompt = data.get('eval_prompt', '')

    # Look for "This involves..." pattern in PV prompts
    if 'This involves' in eval_prompt:
        start = eval_prompt.find('This involves')
        end = eval_prompt.find('.', start)
        if end > start:
            description = eval_prompt[start:end + 1]
            return f"{trait_name.title()}: {description[14:]}"  # Skip "This involves "

    # Fallback: generate from trait name
    return f"The trait '{trait_name.replace('_', ' ')}'"


def create_steering_json(data: dict) -> dict:
    """Create steering.json with questions and optional eval_prompt."""
    result = {
        'questions': data['questions']
    }

    if 'eval_prompt' in data:
        result['eval_prompt'] = data['eval_prompt']

    return result


def expand_pv_trait(input_path: Path, output_dir: Path, trait_name: str = None):
    """Expand a single PV trait file to our format."""
    print(f"Expanding: {input_path}")

    # Load PV format
    data = load_pv_format(input_path)

    # Derive trait name from filename if not provided
    if trait_name is None:
        trait_name = input_path.stem.replace('_extract', '')

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Expand to JSONL
    positive, negative = expand_to_jsonl(data)

    # Write positive.jsonl
    pos_path = output_dir / 'positive.jsonl'
    with open(pos_path, 'w') as f:
        for scenario in positive:
            f.write(json.dumps(scenario) + '\n')
    print(f"  {pos_path.name}: {len(positive)} scenarios")

    # Write negative.jsonl
    neg_path = output_dir / 'negative.jsonl'
    with open(neg_path, 'w') as f:
        for scenario in negative:
            f.write(json.dumps(scenario) + '\n')
    print(f"  {neg_path.name}: {len(negative)} scenarios")

    # Write definition.txt
    definition = extract_definition(data, trait_name)
    def_path = output_dir / 'definition.txt'
    def_path.write_text(definition)
    print(f"  {def_path.name}: {len(definition)} chars")

    # Write steering.json
    steering = create_steering_json(data)
    steering_path = output_dir / 'steering.json'
    with open(steering_path, 'w') as f:
        json.dump(steering, f, indent=2)
    print(f"  {steering_path.name}: {len(steering['questions'])} questions")

    print(f"  Done: {len(data['instruction'])} instructions × {len(data['questions'])} questions = {len(positive)} scenarios/polarity")


def main():
    parser = argparse.ArgumentParser(description="Expand PV format to JSONL")
    parser.add_argument('--input', type=Path, help="Input JSON file (PV format)")
    parser.add_argument('--output', type=Path, help="Output directory")
    parser.add_argument('--all', action='store_true', help="Process all PV traits")
    args = parser.parse_args()

    if args.all:
        # Process all PV traits
        pv_data_dir = Path('experiments/persona_vectors_replication/their_data')
        output_base = Path('datasets/traits/persona_vectors_instruction')

        trait_files = [
            ('evil_extract.json', 'evil'),
            ('sycophantic_extract.json', 'sycophancy'),
            ('hallucinating_extract.json', 'hallucination'),
        ]

        for filename, trait_name in trait_files:
            input_path = pv_data_dir / filename
            if input_path.exists():
                output_dir = output_base / trait_name
                expand_pv_trait(input_path, output_dir, trait_name)
            else:
                print(f"Skipping {filename}: not found")

    elif args.input and args.output:
        expand_pv_trait(args.input, args.output)
    else:
        parser.error("Either --all or both --input and --output required")


if __name__ == '__main__':
    main()
