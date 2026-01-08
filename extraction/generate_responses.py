"""
Generate responses for trait extraction.

Called by run_pipeline.py (stage 1).
"""

import json
from datetime import datetime

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.paths import get as get_path
from utils.model import format_prompt
from utils.generation import generate_batch as _generate_batch
from utils.traits import load_scenarios


def generate_responses_for_trait(
    experiment: str,
    trait: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    rollouts: int = 1,
    temperature: float = 0.0,
    chat_template: bool = False,
) -> tuple[int, int]:
    """
    Generate responses for scenarios. Returns (n_positive, n_negative).

    Uses utils.generation.generate_batch for auto batch sizing and OOM recovery.
    """

    try:
        scenarios = load_scenarios(trait)
        pos_scenarios = scenarios['positive']
        neg_scenarios = scenarios['negative']
    except FileNotFoundError as e:
        print(f"    ERROR: {e}")
        return 0, 0

    responses_dir = get_path('extraction.responses', experiment=experiment, trait=trait)
    responses_dir.mkdir(parents=True, exist_ok=True)

    def generate_for_scenarios(scenarios: list[dict], label: str) -> list[dict]:
        results = []

        # Extract prompts and system prompts
        prompts = [s['prompt'] for s in scenarios]
        system_prompts = [s.get('system_prompt') for s in scenarios]

        # Log if system prompts are being used
        n_with_system = sum(1 for sp in system_prompts if sp)
        if n_with_system > 0:
            print(f"      {label}: {n_with_system}/{len(scenarios)} scenarios have system prompts")

        # Format prompts (applies chat template if needed, with system_prompt)
        formatted_prompts = [
            format_prompt(p, tokenizer, use_chat_template=chat_template, system_prompt=sp)
            for p, sp in zip(prompts, system_prompts)
        ]

        # Get token counts for formatted prompts (match add_special_tokens with generation)
        prompt_token_counts = [len(tokenizer.encode(p, add_special_tokens=not chat_template)) for p in formatted_prompts]

        for rollout_idx in tqdm(range(rollouts), desc=f"    {label}", leave=False, disable=rollouts == 1):
            # Generate all responses (handles batching + OOM internally)
            # Set add_special_tokens=False if chat template was used (it already added BOS)
            responses = _generate_batch(model, tokenizer, formatted_prompts, max_new_tokens, temperature,
                                        add_special_tokens=not chat_template)

            for i, (scenario, response, prompt_tokens) in enumerate(zip(scenarios, responses, prompt_token_counts)):
                results.append({
                    'scenario_idx': i,
                    'rollout_idx': rollout_idx,
                    'prompt': scenario['prompt'],
                    'system_prompt': scenario.get('system_prompt'),
                    'response': response,
                    'full_text': formatted_prompts[i] + response,  # Use formatted prompt for accurate tokenization
                    'prompt_token_count': prompt_tokens,
                    'response_token_count': len(tokenizer.encode(response, add_special_tokens=False)),
                })

        print(f"      {label}: {len(results)} responses")
        return results

    pos_results = generate_for_scenarios(pos_scenarios, 'positive')
    neg_results = generate_for_scenarios(neg_scenarios, 'negative')

    with open(responses_dir / 'pos.json', 'w') as f:
        json.dump(pos_results, f, indent=2)
    with open(responses_dir / 'neg.json', 'w') as f:
        json.dump(neg_results, f, indent=2)

    metadata = {
        'model': model.config.name_or_path,
        'experiment': experiment,
        'trait': trait,
        'max_new_tokens': max_new_tokens,
        'chat_template': chat_template,
        'rollouts': rollouts,
        'temperature': temperature,
        'n_pos': len(pos_results),
        'n_neg': len(neg_results),
        'n_scenarios_pos': len(pos_scenarios),
        'n_scenarios_neg': len(neg_scenarios),
        'has_system_prompts': any(s.get('system_prompt') for s in pos_scenarios + neg_scenarios),
        'timestamp': datetime.now().isoformat(),
    }
    with open(responses_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"      Saved {len(pos_results)} + {len(neg_results)} responses")
    return len(pos_results), len(neg_results)
