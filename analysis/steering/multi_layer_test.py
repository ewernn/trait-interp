"""
Multi-layer steering test for Qwen-2.5-7B-Instruct with optimism vectors.

Compares:
- Baseline (no steering)
- Single-layer L13 coef=8
- Multi-layer L10+L13+L16 coef=3 each

Usage:
    python analysis/steering/multi_layer_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from analysis.steering.steer import SteeringHook
from analysis.steering.judge import TraitJudge
from utils.paths import get


def load_vectors(experiment: str, trait: str, layers: list[int]) -> dict[int, torch.Tensor]:
    """Load vectors for specified layers."""
    vectors = {}
    vectors_dir = Path(get('extraction.vectors', experiment=experiment, trait=trait))

    for layer in layers:
        vector_path = vectors_dir / f"probe_layer{layer}.pt"
        if not vector_path.exists():
            raise FileNotFoundError(f"Vector not found: {vector_path}")
        vectors[layer] = torch.load(vector_path, weights_only=True)
        print(f"Loaded vector for layer {layer}: norm={vectors[layer].norm():.2f}")

    return vectors


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.0
) -> str:
    """Generate a single response."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    # Configuration
    experiment = "qwen_optimism_instruct"
    trait = "mental_state/optimism"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    layers = [10, 13, 16]
    multi_coef = 3.0
    single_layer = 13
    single_coef = 8.0
    temperature = 0.0

    print("=" * 80)
    print("Multi-Layer Steering Test: Qwen-2.5-7B-Instruct + Optimism")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Experiment: {experiment}")
    print(f"Trait: {trait}")
    print(f"Multi-layer config: L{layers} @ coef={multi_coef} each")
    print(f"Single-layer config: L{single_layer} @ coef={single_coef}")
    print(f"Temperature: {temperature}")
    print()

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded: {model.device}")
    print()

    # Load vectors
    print("Loading vectors...")
    vectors = load_vectors(experiment, trait, layers)
    print()

    # Load questions
    prompts_path = Path("analysis/steering/prompts/optimism.json")
    with open(prompts_path) as f:
        data = json.load(f)
        questions = data["questions"]
        eval_prompt = data["eval_prompt"]
    print(f"Loaded {len(questions)} questions from {prompts_path}")
    print()

    # Initialize judge
    judge = TraitJudge(provider="openai")  # Use OpenAI for reliable scoring
    print("Initialized TraitJudge")
    print()

    # Run experiments
    results = {
        "baseline": [],
        "single_layer": [],
        "multi_layer": []
    }

    async def run_experiments():
        for i, question in enumerate(tqdm(questions, desc="Testing questions")):
            print(f"\n--- Question {i+1}/{len(questions)} ---")
            print(f"Q: {question[:100]}...")

            # Baseline
            response_baseline = generate_response(model, tokenizer, question, temperature=temperature)
            score_baseline = await judge.score(eval_prompt, question, response_baseline)
            if score_baseline is not None:
                results["baseline"].append(score_baseline)
                print(f"Baseline: {score_baseline:.1f} | {response_baseline[:80]}...")
            else:
                print(f"Baseline: NONE (judge failed) | {response_baseline[:80]}...")

            # Single-layer L13
            with SteeringHook(model, vectors[single_layer], layer=single_layer, coefficient=single_coef):
                response_single = generate_response(model, tokenizer, question, temperature=temperature)
            score_single = await judge.score(eval_prompt, question, response_single)
            if score_single is not None:
                results["single_layer"].append(score_single)
                print(f"Single L{single_layer}@{single_coef}: {score_single:.1f} | {response_single[:80]}...")
            else:
                print(f"Single L{single_layer}@{single_coef}: NONE (judge failed) | {response_single[:80]}...")

            # Multi-layer L10+L13+L16
            with SteeringHook(model, vectors[10], layer=10, coefficient=multi_coef):
                with SteeringHook(model, vectors[13], layer=13, coefficient=multi_coef):
                    with SteeringHook(model, vectors[16], layer=16, coefficient=multi_coef):
                        response_multi = generate_response(model, tokenizer, question, temperature=temperature)
            score_multi = await judge.score(eval_prompt, question, response_multi)
            if score_multi is not None:
                results["multi_layer"].append(score_multi)
                print(f"Multi L{layers}@{multi_coef}: {score_multi:.1f} | {response_multi[:80]}...")
            else:
                print(f"Multi L{layers}@{multi_coef}: NONE (judge failed) | {response_multi[:80]}...")

    # Run async experiments
    import asyncio
    asyncio.run(run_experiments())

    # Compute statistics
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    mean_baseline = sum(results["baseline"]) / len(results["baseline"])
    mean_single = sum(results["single_layer"]) / len(results["single_layer"])
    mean_multi = sum(results["multi_layer"]) / len(results["multi_layer"])

    delta_single = mean_single - mean_baseline
    delta_multi = mean_multi - mean_baseline

    print(f"\nBaseline (no steering):              {mean_baseline:.1f}")
    print(f"Single-layer L{single_layer}@{single_coef}:         {mean_single:.1f}  (Δ = {delta_single:+.1f})")
    print(f"Multi-layer L{layers}@{multi_coef}:  {mean_multi:.1f}  (Δ = {delta_multi:+.1f})")
    print()

    if delta_multi > delta_single:
        print(f"✓ Multi-layer WINS: {delta_multi:.1f} > {delta_single:.1f} (+{delta_multi - delta_single:.1f})")
    elif delta_single > delta_multi:
        print(f"✗ Single-layer WINS: {delta_single:.1f} > {delta_multi:.1f} (+{delta_single - delta_multi:.1f})")
    else:
        print("= TIE")

    print("\n" + "=" * 80)

    # Save results
    output_path = Path("analysis/steering/multi_layer_results.json")
    output_data = {
        "config": {
            "model": model_name,
            "experiment": experiment,
            "trait": trait,
            "temperature": temperature,
            "single_layer": single_layer,
            "single_coef": single_coef,
            "multi_layers": layers,
            "multi_coef": multi_coef
        },
        "scores": results,
        "summary": {
            "baseline_mean": mean_baseline,
            "single_mean": mean_single,
            "multi_mean": mean_multi,
            "single_delta": delta_single,
            "multi_delta": delta_multi
        }
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
