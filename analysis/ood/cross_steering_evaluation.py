#!/usr/bin/env python3
"""
Cross-domain steering evaluation for OOD formality experiments.

Tests if vectors extracted from one variant (e.g., English) work on another
variant (e.g., Spanish) using steering.

Input:
    - experiments/{experiment}/extraction/{trait}/vectors/...
    - datasets/traits/{trait}/steering.json

Output:
    - Prints cross-domain steering deltas

Usage:
    python analysis/ood/cross_steering_evaluation.py --experiment gemma-2-2b
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import json
import fire
import asyncio
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from utils.paths import get, get_vector_path, get_steering_results_path
from utils.model import load_experiment_config, load_model
from utils.judge import TraitJudge
from utils.generation import generate_batch
from core import SteeringHook


async def evaluate_steering(
    model,
    tokenizer,
    questions: List[str],
    trait_definition: str,
    vector: torch.Tensor,
    layer: int,
    coefficient: float,
    judge: TraitJudge,
    max_new_tokens: int = 128,
) -> Tuple[float, float, float, float]:
    """
    Evaluate steering effect.

    Returns: (baseline_trait, steered_trait, baseline_coherence, steered_coherence)
    """
    # Generate baseline
    baseline_responses = generate_batch(
        model, tokenizer, questions,
        max_new_tokens=max_new_tokens,
        use_chat_template=True,
    )

    # Generate with steering
    hook = SteeringHook(
        model=model,
        layer=layer,
        vector=vector,
        coefficient=coefficient,
        component="residual",
    )
    steered_responses = generate_batch(
        model, tokenizer, questions,
        max_new_tokens=max_new_tokens,
        use_chat_template=True,
    )
    hook.remove()

    # Judge responses
    trait_name = "formality"
    baseline_scores = await judge.score_batch(
        questions, baseline_responses, trait_name, trait_definition
    )
    steered_scores = await judge.score_batch(
        questions, steered_responses, trait_name, trait_definition
    )

    baseline_trait = sum(s["trait_score"] for s in baseline_scores) / len(baseline_scores)
    steered_trait = sum(s["trait_score"] for s in steered_scores) / len(steered_scores)
    baseline_coh = sum(s.get("coherence_score", 100) for s in baseline_scores) / len(baseline_scores)
    steered_coh = sum(s.get("coherence_score", 100) for s in steered_scores) / len(steered_scores)

    return baseline_trait, steered_trait, baseline_coh, steered_coh


def load_vector(experiment: str, trait: str, layer: int, method: str = "probe") -> Optional[torch.Tensor]:
    """Load trait vector."""
    path = get_vector_path(experiment, trait, method, layer, "residual", "response[:]")
    if not path.exists():
        return None
    return torch.load(path, weights_only=True)


def load_steering_questions(trait: str) -> Tuple[List[str], str]:
    """Load steering questions and definition for a trait."""
    questions_path = get('datasets.trait_steering', trait=trait)
    with open(questions_path) as f:
        data = json.load(f)

    definition_path = get('datasets.trait_definition', trait=trait)
    with open(definition_path) as f:
        definition = f.read().strip()

    return data["questions"], definition


async def cross_steering_matrix(
    experiment: str,
    variants: List[str],
    layer: int = 12,
    coefficient: float = 100.0,
    method: str = "probe",
    subset: int = 3,
) -> Dict:
    """
    Evaluate cross-domain steering.

    For each source variant, apply its vector to all target variants' questions
    and measure trait delta.
    """
    print(f"Loading model...")
    config = load_experiment_config(experiment)
    model_name = config.get('application_model') or config.get('model')
    model, tokenizer = load_model(model_name)

    judge = TraitJudge()

    # Load all vectors and questions
    vectors = {}
    questions = {}
    definitions = {}
    for v in variants:
        trait = f"formality_variations/{v}"
        vec = load_vector(experiment, trait, layer, method)
        if vec is not None:
            vectors[v] = vec
        try:
            q, d = load_steering_questions(trait)
            questions[v] = q[:subset] if subset > 0 else q
            definitions[v] = d
        except FileNotFoundError:
            print(f"Warning: No steering questions for {v}")

    # Cross-evaluate
    results = {}
    for source in tqdm(variants, desc="Source variants"):
        if source not in vectors:
            continue
        results[source] = {}

        for target in variants:
            if target not in questions:
                continue

            baseline, steered, base_coh, steer_coh = await evaluate_steering(
                model=model,
                tokenizer=tokenizer,
                questions=questions[target],
                trait_definition=definitions[target],
                vector=vectors[source],
                layer=layer,
                coefficient=coefficient,
                judge=judge,
            )

            delta = steered - baseline
            results[source][target] = {
                "baseline": baseline,
                "steered": steered,
                "delta": delta,
                "baseline_coh": base_coh,
                "steered_coh": steer_coh,
            }

            marker = "*" if source == target else ""
            print(f"  {source} -> {target}: delta={delta:+.1f}{marker} (coh: {steer_coh:.0f}%)")

    await judge.close()
    return results


def print_delta_matrix(results: Dict, variants: List[str], title: str):
    """Print delta matrix."""
    print(f"\n{title} - Steering Deltas")
    print("=" * (15 + 10 * len(variants)))

    header = f"{'Vec \\ Qs':<15}"
    for v in variants:
        header += f"{v[:8]:<10}"
    print(header)
    print("-" * (15 + 10 * len(variants)))

    deltas = []
    in_domain = []
    cross_domain = []

    for source in variants:
        if source not in results:
            continue
        row = f"{source[:14]:<15}"
        for target in variants:
            if target in results[source]:
                d = results[source][target]["delta"]
                deltas.append(d)
                if source == target:
                    in_domain.append(d)
                    row += f"{d:+.1f}*    "
                else:
                    cross_domain.append(d)
                    row += f"{d:+.1f}     "
            else:
                row += f"{'N/A':<10}"
        print(row)

    print("-" * (15 + 10 * len(variants)))
    if in_domain:
        print(f"In-domain mean delta: {sum(in_domain)/len(in_domain):+.1f}")
    if cross_domain:
        print(f"Cross-domain mean delta: {sum(cross_domain)/len(cross_domain):+.1f}")


async def main(
    experiment: str = "gemma-2-2b",
    layer: int = 12,
    coefficient: float = 100.0,
    method: str = "probe",
    subset: int = 3,
):
    """
    Run cross-domain steering evaluation.

    Args:
        experiment: Experiment name
        layer: Layer to test
        coefficient: Steering coefficient
        method: Vector method (probe, mean_diff, gradient)
        subset: Number of questions per variant (0 for all)
    """
    print("Cross-Domain Steering Evaluation")
    print(f"Experiment: {experiment}, Layer: {layer}, Coef: {coefficient}")

    # Cross-language
    print("\n" + "="*60)
    print("CROSS-LANGUAGE")
    print("="*60)
    lang_variants = ["english", "spanish", "french", "chinese"]
    lang_results = await cross_steering_matrix(
        experiment, lang_variants, layer, coefficient, method, subset
    )
    print_delta_matrix(lang_results, lang_variants, "Cross-Language")

    # Cross-topic
    print("\n" + "="*60)
    print("CROSS-TOPIC")
    print("="*60)
    topic_variants = ["business", "academic", "social", "technical"]
    topic_results = await cross_steering_matrix(
        experiment, topic_variants, layer, coefficient, method, subset
    )
    print_delta_matrix(topic_results, topic_variants, "Cross-Topic")

    return {
        "cross_language": lang_results,
        "cross_topic": topic_results,
    }


if __name__ == "__main__":
    fire.Fire(lambda **kwargs: asyncio.run(main(**kwargs)))
