"""
Compare multi-layer steering configurations on Qwen-2.5-7B-Instruct.

Tests two layer spread strategies with equal total coefficient (~12):
1. 4 layers @ 3.0 each: L10, L13, L16, L19
2. 2 layers @ 6.0 each: L12, L17
"""

import torch
import json
import asyncio
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from analysis.steering.steer import SteeringHook
from analysis.steering.judge import TraitJudge

def load_vectors(experiment, trait, layers):
    """Load steering vectors for specified layers."""
    vectors = {}
    for layer in layers:
        vector_path = Path(f'experiments/{experiment}/extraction/{trait}/vectors/probe_layer{layer}.pt')
        if not vector_path.exists():
            raise FileNotFoundError(f"Vector not found: {vector_path}")
        vectors[layer] = torch.load(vector_path, map_location='cpu', weights_only=True)
        print(f"Loaded vector for layer {layer}, norm: {vectors[layer].norm():.2f}")
    return vectors

def generate_with_steering(model, tokenizer, prompt, steering_config, device):
    """
    Generate with multi-layer steering.

    Args:
        steering_config: dict mapping layer -> (vector, coefficient)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Sort layers to apply hooks in consistent order
    sorted_layers = sorted(steering_config.keys())

    # Build nested context managers
    contexts = []
    for layer in sorted_layers:
        vector, coef = steering_config[layer]
        vector = vector.to(device)
        contexts.append(SteeringHook(model, vector, layer=layer, coefficient=coef))

    # Enter all contexts
    for ctx in contexts:
        ctx.__enter__()

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response
    finally:
        # Exit all contexts in reverse order
        for ctx in reversed(contexts):
            ctx.__exit__(None, None, None)

async def run_config(model, tokenizer, questions, eval_prompt, vectors, config_name, layer_coefs, judge, device):
    """Run a single configuration and return results."""
    print(f"\n{'='*60}")
    print(f"CONFIGURATION: {config_name}")
    print(f"{'='*60}")

    # Build steering config
    steering_config = {}
    for layer, coef in layer_coefs.items():
        steering_config[layer] = (vectors[layer], coef)

    results = []
    scores = []

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question[:60]}...")

        # Format prompt with chat template
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate with steering
        response = generate_with_steering(model, tokenizer, prompt, steering_config, device)

        # Score response (async)
        score = await judge.score(eval_prompt, question, response)
        if score is None:
            score = 0.0  # Treat refusal/error as 0
        scores.append(score)

        print(f"  Score: {score:.1f}/100")
        print(f"  Response: {response[:100]}...")

        results.append({
            "question": question,
            "response": response,
            "score": score
        })

    mean_score = sum(scores) / len(scores)
    print(f"\n{config_name} Mean Score: {mean_score:.1f}")

    return {
        "config_name": config_name,
        "layer_coefficients": {str(k): v for k, v in layer_coefs.items()},
        "mean_score": mean_score,
        "all_scores": scores,
        "results": results
    }

async def main_async():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    experiment = "qwen_optimism_instruct"
    trait = "mental_state/optimism"

    # Load questions
    prompts_file = Path("analysis/steering/prompts/optimism.json")
    with open(prompts_file) as f:
        data = json.load(f)
    questions = data["questions"]
    eval_prompt = data["eval_prompt"]

    print(f"Loaded {len(questions)} questions")

    # Initialize judge
    print("\nInitializing TraitJudge...")
    judge = TraitJudge(provider="openai")  # Use OpenAI for judging

    # Load model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Load all required vectors
    required_layers = [10, 12, 13, 16, 17, 19]
    print(f"\nLoading vectors for layers: {required_layers}")
    vectors = load_vectors(experiment, trait, required_layers)

    # Configuration 1: 4 layers @ 3.0 each
    config1_layers = {10: 3.0, 13: 3.0, 16: 3.0, 19: 3.0}
    results1 = await run_config(
        model, tokenizer, questions, eval_prompt, vectors,
        "4-layer spread (L10,13,16,19 @ 3.0)",
        config1_layers, judge, device
    )

    # Configuration 2: 2 layers @ 6.0 each
    config2_layers = {12: 6.0, 17: 6.0}
    results2 = await run_config(
        model, tokenizer, questions, eval_prompt, vectors,
        "2-layer concentrated (L12,17 @ 6.0)",
        config2_layers, judge, device
    )

    # Summary comparison
    baseline_score = 64.4
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline (single layer): {baseline_score:.1f}")
    print(f"\nConfig 1 (4-layer spread):   {results1['mean_score']:.1f} (Δ{results1['mean_score'] - baseline_score:+.1f})")
    print(f"Config 2 (2-layer conc):     {results2['mean_score']:.1f} (Δ{results2['mean_score'] - baseline_score:+.1f})")
    print(f"\nDifference (Config1 - Config2): {results1['mean_score'] - results2['mean_score']:+.1f}")

    # Save results
    output = {
        "baseline_score": baseline_score,
        "config1": results1,
        "config2": results2,
        "summary": {
            "winner": "Config 1" if results1['mean_score'] > results2['mean_score'] else "Config 2",
            "config1_delta": results1['mean_score'] - baseline_score,
            "config2_delta": results2['mean_score'] - baseline_score,
            "difference": results1['mean_score'] - results2['mean_score']
        }
    }

    output_file = Path("analysis/steering/multi_layer_comparison_results.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
