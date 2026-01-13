#!/usr/bin/env python3
"""
CMA-ES optimization of vector direction.

Optimizes the vector direction by searching in a random subspace around
an existing vector (from get_best_vector) or from random initialization.

Input: experiment, trait, layers, component
Output: Optimized vector direction per layer

Usage:
    python analysis/steering/optimize_vector.py \
        --experiment gemma-2-2b \
        --trait chirp/refusal \
        --layers 8,9,10,11,12 \
        --component residual

    # Single layer with explicit position
    python analysis/steering/optimize_vector.py \
        --experiment gemma-2-2b \
        --trait chirp/refusal \
        --layers 10 \
        --component residual \
        --position "response[:5]"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cma
import json
import torch
import asyncio
import argparse
import time
from typing import List, Tuple, Optional, Dict
from datetime import datetime

from analysis.steering.data import load_steering_data
from core import BatchedLayerSteeringHook
from analysis.steering.results import init_results_file, append_run, save_responses
from utils.paths import get_vector_dir, get_model_variant, get_steering_results_path
from utils.model import load_model, format_prompt
from utils.generation import generate_batch
from utils.judge import TraitJudge
from utils.vectors import MIN_COHERENCE, load_vector, get_best_vector
from core import VectorSpec


def random_orthonormal_basis(dim: int, n_components: int, seed: int = None) -> torch.Tensor:
    """
    Generate random orthonormal basis vectors.

    Args:
        dim: Hidden dimension (e.g., 2304 for Gemma-2-2B)
        n_components: Number of basis vectors
        seed: Optional random seed for reproducibility

    Returns:
        basis: [n_components, dim] - orthonormal basis vectors
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Generate random matrix and orthonormalize via QR decomposition
    random_matrix = torch.randn(dim, n_components)
    Q, _ = torch.linalg.qr(random_matrix)
    return Q.T[:n_components]  # [n_components, dim]


def get_init_vector_and_position(
    experiment: str,
    trait: str,
    layer: int,
    component: str,
    position: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], str, Optional[str], Optional[Dict]]:
    """
    Get initial vector and position for optimization.

    Tries to find existing best vector for this layer/component.
    Falls back to random if nothing exists.

    Returns:
        (init_vector, position, method, best_info) where:
        - init_vector: Starting vector or None for random
        - position: Position to use (discovered or default)
        - method: Method of init vector (e.g., 'probe') or None
        - best_info: Full info from get_best_vector or None
    """
    # Try to find existing vector for this component
    try:
        best = get_best_vector(experiment, trait, component=component)

        # Check if best is for this layer or different layer
        if best['layer'] == layer:
            # Perfect - use this vector
            init_vector = load_vector(
                experiment, trait, layer,
                best.get('extraction_variant', get_model_variant(experiment, mode="extraction")['name']),
                best['method'], component, best['position']
            )
            return init_vector, best['position'], best['method'], best
        else:
            # Best is different layer - try to load same method/position for our layer
            try:
                init_vector = load_vector(
                    experiment, trait, layer,
                    best.get('extraction_variant', get_model_variant(experiment, mode="extraction")['name']),
                    best['method'], component, best['position']
                )
                return init_vector, best['position'], best['method'], best
            except FileNotFoundError:
                # No vector at this layer with best's method/position
                # Use position from best but random init
                return None, best['position'] if position is None else position, None, best

    except (FileNotFoundError, ValueError) as e:
        # No steering results found - try to find any vector for this component
        extraction_variant = get_model_variant(experiment, mode="extraction")['name']

        # Try common positions
        for try_position in [position, "response[:5]", "response[:3]"]:
            if try_position is None:
                continue
            for method in ["probe", "mean_diff", "gradient"]:
                try:
                    init_vector = load_vector(
                        experiment, trait, layer,
                        extraction_variant, method, component, try_position
                    )
                    return init_vector, try_position, method, None
                except FileNotFoundError:
                    continue

        # Nothing found - use defaults
        default_position = position if position else "response[:5]"
        return None, default_position, None, None


def reconstruct_vector(
    init_vector: Optional[torch.Tensor],
    weights: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct vector from initial vector + weighted perturbations.

    Args:
        init_vector: Starting vector [hidden_dim] or None for pure basis combination
        weights: [n_components] - weights for each basis vector
        basis: [n_components, hidden_dim] - orthonormal basis

    Returns:
        vector: [hidden_dim] - reconstructed vector
    """
    # Weighted sum of basis vectors (perturbation)
    perturbation = (weights.unsqueeze(1) * basis).sum(dim=0)  # [hidden_dim]

    if init_vector is not None:
        return init_vector + perturbation
    else:
        return perturbation


async def run_cma_es_single_layer(
    model,
    tokenizer,
    experiment: str,
    trait: str,
    layer: int,
    component: str,
    position: str,
    init_vector: Optional[torch.Tensor],
    init_method: Optional[str],
    steering_data,
    judge: TraitJudge,
    n_components: int = 20,
    n_generations: int = 15,
    popsize: int = 8,
    max_new_tokens: int = 12,
    n_questions: int = 5,
    coherence_threshold: float = MIN_COHERENCE,
    base_coef: float = 90.0,
    seed: int = None,
) -> Optional[Dict]:
    """
    Run CMA-ES optimization for a single layer.
    """
    questions = steering_data.questions[:n_questions]
    use_chat_template = tokenizer.chat_template is not None

    # Format questions once
    formatted_questions = [
        format_prompt(q, tokenizer, use_chat_template=use_chat_template)
        for q in questions
    ]

    # Get hidden dim from model
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    hidden_dim = config.hidden_size

    # Generate random orthonormal basis for perturbation search
    basis = random_orthonormal_basis(hidden_dim, n_components, seed=seed)
    basis = basis.to(model.device).float()

    # Prepare init vector
    if init_vector is not None:
        init_vector = init_vector.to(model.device).float()
        # Normalize init vector - magnitude will come from perturbations
        init_vector = init_vector / (init_vector.norm() + 1e-8) * base_coef
        print(f"  Starting from {init_method} vector (norm={init_vector.norm():.1f})")
    else:
        print(f"  Starting from random (no existing vector found)")

    # Initial weights: zeros (start exactly at init_vector) or small random for pure random start
    if init_vector is not None:
        x0 = [0.0] * n_components  # Start at init_vector
        sigma0 = base_coef * 0.3  # Explore around it
    else:
        x0 = [base_coef / n_components] * n_components  # Random start
        sigma0 = base_coef * 0.5  # Larger exploration

    # CMA-ES setup
    opts = {
        'popsize': popsize,
        'maxiter': n_generations,
        'verbose': -9,
    }

    print(f"\n{'='*60}")
    print(f"CMA-ES Vector Direction Optimization - Layer {layer}")
    print(f"{'='*60}")
    print(f"Component: {component}")
    print(f"Position: {position}")
    print(f"Subspace dims: {n_components}")
    print(f"Population: {popsize}/gen, {n_generations} generations")
    print(f"Questions: {n_questions}, Tokens: {max_new_tokens}")
    print(f"Fitness: trait if coh >= {coherence_threshold} else -1000")
    print(f"{'='*60}\n")

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_result = None
    best_responses = None
    best_weights = None
    generation = 0
    fitness_history = []

    while not es.stop():
        generation += 1
        candidates = es.ask()
        n_candidates = len(candidates)

        print(f"--- Generation {generation}/{n_generations} ---")

        # Build vectors for each candidate
        candidate_vectors = []
        for weights in candidates:
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=model.device)
            vector = reconstruct_vector(init_vector, weights_tensor, basis)
            candidate_vectors.append(vector)

        # Build batched prompts
        batched_prompts = []
        for _ in candidates:
            batched_prompts.extend(formatted_questions)

        # Build steering configs (coef=1.0 since magnitude is in the vector)
        steering_configs = []
        for cand_idx, vector in enumerate(candidate_vectors):
            batch_start = cand_idx * n_questions
            batch_end = (cand_idx + 1) * n_questions
            steering_configs.append((
                layer,
                vector,
                1.0,  # magnitude baked into vector
                (batch_start, batch_end)
            ))

        # Generate all responses
        t0 = time.time()
        with BatchedLayerSteeringHook(model, steering_configs, component=component):
            all_responses = generate_batch(
                model, tokenizer, batched_prompts,
                max_new_tokens=max_new_tokens
            )
        gen_time = time.time() - t0

        # Build QA pairs
        all_qa_pairs = []
        for cand_idx in range(n_candidates):
            start = cand_idx * n_questions
            end = (cand_idx + 1) * n_questions
            for q, r in zip(questions, all_responses[start:end]):
                all_qa_pairs.append((q, r))

        # Score
        t0 = time.time()
        all_scores = await judge.score_steering_batch(
            all_qa_pairs,
            steering_data.trait_name,
            steering_data.trait_definition
        )
        score_time = time.time() - t0

        print(f"  Generated {len(batched_prompts)} responses ({gen_time:.1f}s), scored ({score_time:.1f}s)")

        # Compute fitness
        fitnesses = []
        gen_best_fitness = -1000
        for cand_idx, weights in enumerate(candidates):
            start = cand_idx * n_questions
            end = (cand_idx + 1) * n_questions
            cand_scores = all_scores[start:end]

            trait_scores = [s["trait_score"] for s in cand_scores if s["trait_score"] is not None]
            coherence_scores = [s["coherence_score"] for s in cand_scores if s.get("coherence_score") is not None]

            trait_mean = sum(trait_scores) / len(trait_scores) if trait_scores else 0
            coherence_mean = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0

            # Fitness: hard floor on coherence
            if coherence_mean >= coherence_threshold:
                fitness = trait_mean
            else:
                fitness = -1000

            fitnesses.append(-fitness)  # CMA-ES minimizes
            gen_best_fitness = max(gen_best_fitness, fitness)

            # Compute magnitude
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=model.device)
            opt_vector = reconstruct_vector(init_vector, weights_tensor, basis)
            magnitude = opt_vector.norm().item()

            # Track best
            if best_result is None or fitness > best_result.get("fitness", -999):
                best_result = {
                    "trait_mean": trait_mean,
                    "coherence_mean": coherence_mean,
                    "fitness": fitness,
                    "magnitude": magnitude,
                }
                best_weights = list(weights)
                # Track best responses for saving
                cand_responses = all_responses[start:end]
                best_responses = [
                    {"question": q, "response": r, "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
                    for q, r, s in zip(questions, cand_responses, cand_scores)
                ]

            print(f"  [{cand_idx+1}/{n_candidates}] trait={trait_mean:.1f} coh={coherence_mean:.1f} fit={fitness:.1f} mag={magnitude:.1f}")

        es.tell(candidates, fitnesses)
        fitness_history.append(gen_best_fitness)

        if best_result:
            print(f"  Best so far: trait={best_result['trait_mean']:.1f}, coh={best_result['coherence_mean']:.1f}")

    # Final result
    print(f"\n{'='*60}")
    print(f"LAYER {layer} OPTIMIZATION COMPLETE")
    print(f"{'='*60}")

    if best_result and best_result['fitness'] > -1000:
        print(f"Best trait: {best_result['trait_mean']:.1f}")
        print(f"Best coherence: {best_result['coherence_mean']:.1f}")
        print(f"Best magnitude: {best_result['magnitude']:.1f}")

        # Compute final vector
        weights_tensor = torch.tensor(best_weights, dtype=torch.float32, device=model.device)
        opt_vector = reconstruct_vector(init_vector, weights_tensor, basis)

        # Compute similarity to init
        if init_vector is not None:
            init_dir = init_vector / (init_vector.norm() + 1e-8)
            opt_dir = opt_vector / (opt_vector.norm() + 1e-8)
            similarity = (opt_dir @ init_dir).item()
            print(f"Similarity to init vector: {similarity:.4f}")
        else:
            similarity = None

        # Save optimized vector
        extraction_variant = get_model_variant(experiment, mode="extraction")['name']
        output_dir = get_vector_dir(experiment, trait, "cma_es", extraction_variant, component, position)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"layer{layer}.pt"
        torch.save(opt_vector.cpu(), output_path)
        print(f"Saved optimized vector: {output_path}")

        # Save metadata with fitness history
        metadata = {
            "layer": layer,
            "component": component,
            "position": position,
            "n_components": n_components,
            "n_generations": n_generations,
            "popsize": popsize,
            "base_coef": base_coef,
            "coherence_threshold": coherence_threshold,
            "init_method": init_method,
            "best_trait": best_result['trait_mean'],
            "best_coherence": best_result['coherence_mean'],
            "best_fitness": best_result['fitness'],
            "best_magnitude": best_result['magnitude'],
            "similarity_to_init": similarity,
            "fitness_history": fitness_history,
            "timestamp": datetime.now().isoformat(),
        }
        metadata_path = output_dir / f"layer{layer}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")

        # Save to steering results.jsonl
        model_name = model.config.name_or_path
        prompt_set = "steering"
        results_path = get_steering_results_path(experiment, trait, extraction_variant, position, prompt_set)

        if not results_path.exists():
            init_results_file(
                experiment, trait, extraction_variant, steering_data.prompts_file,
                model_name, experiment, "openai", position, prompt_set
            )

        config = {
            "vectors": [VectorSpec(
                layer=layer,
                component=component,
                position=position,
                method="cma_es",
                weight=1.0,
            ).to_dict()]
        }
        result = {
            "trait_mean": best_result['trait_mean'],
            "coherence_mean": best_result['coherence_mean'],
            "n": n_questions,
        }

        append_run(experiment, trait, extraction_variant, config, result, position, prompt_set)
        print(f"Saved to results.jsonl")

        # Save responses
        if best_responses:
            timestamp = datetime.now().isoformat()
            save_responses(best_responses, experiment, trait, extraction_variant, position, prompt_set, config, timestamp)
            print(f"Saved responses")

        return best_result
    else:
        print("No valid result found (all candidates below coherence threshold)")
        return None


async def run_cma_es(
    experiment: str,
    trait: str,
    layers: List[int],
    component: str,
    position: Optional[str] = None,
    n_components: int = 20,
    n_generations: int = 15,
    popsize: int = 8,
    max_new_tokens: int = 12,
    n_questions: int = 5,
    coherence_threshold: float = MIN_COHERENCE,
    base_coef: float = 90.0,
    seed: int = None,
):
    """
    Run CMA-ES optimization for multiple layers sequentially.
    """
    # Load steering data
    steering_data = load_steering_data(trait)

    # Load model once
    variant = get_model_variant(experiment, mode="application")
    model_name = variant['model']
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name)

    # Initialize judge
    judge = TraitJudge()

    results = {}

    try:
        for layer in layers:
            print(f"\n{'#'*60}")
            print(f"# LAYER {layer}")
            print(f"{'#'*60}")

            # Get init vector and position for this layer
            init_vector, layer_position, init_method, _ = get_init_vector_and_position(
                experiment, trait, layer, component, position
            )

            result = await run_cma_es_single_layer(
                model=model,
                tokenizer=tokenizer,
                experiment=experiment,
                trait=trait,
                layer=layer,
                component=component,
                position=layer_position,
                init_vector=init_vector,
                init_method=init_method,
                steering_data=steering_data,
                judge=judge,
                n_components=n_components,
                n_generations=n_generations,
                popsize=popsize,
                max_new_tokens=max_new_tokens,
                n_questions=n_questions,
                coherence_threshold=coherence_threshold,
                base_coef=base_coef,
                seed=seed,
            )

            results[layer] = result

    finally:
        await judge.close()

    # Summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    for layer, result in results.items():
        if result and result['fitness'] > -1000:
            print(f"  L{layer}: trait={result['trait_mean']:.1f}, coh={result['coherence_mean']:.1f}")
        else:
            print(f"  L{layer}: no valid result")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="CMA-ES vector direction optimization")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--layers", required=True, help="Comma-separated layers, e.g., '8,9,10,11,12'")
    parser.add_argument("--component", default="residual")
    parser.add_argument("--position", default=None, help="Position (default: auto-discover from existing vectors)")
    parser.add_argument("--n-components", type=int, default=20, help="Subspace dimensionality")
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--coherence-threshold", type=float, default=MIN_COHERENCE)
    parser.add_argument("--base-coef", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Parse layers
    layers = [int(x.strip()) for x in args.layers.split(",")]

    asyncio.run(run_cma_es(
        experiment=args.experiment,
        trait=args.trait,
        layers=layers,
        component=args.component,
        position=args.position,
        n_components=args.n_components,
        n_generations=args.generations,
        popsize=args.popsize,
        max_new_tokens=args.max_new_tokens,
        n_questions=args.n_questions,
        coherence_threshold=args.coherence_threshold,
        base_coef=args.base_coef,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
