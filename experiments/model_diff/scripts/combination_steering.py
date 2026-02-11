"""
Test combination strategies for instruction vs natural sycophancy vectors.

Input: Phase 1 best vectors (specified via args)
Output: experiments/model_diff/combination_steering/results.json

Usage:
    python experiments/model_diff/scripts/combination_steering.py \
        --experiment model_diff \
        --inst-trait pv_instruction/sycophancy \
        --nat-trait pv_natural/sycophancy \
        --inst-layer 14 --inst-method probe --inst-position "response[:]" --inst-coef 6.0 \
        --nat-layer 17 --nat-method mean_diff --nat-position "response[:5]" --nat-coef 9.0 \
        --coefficients 2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0 \
        --max-new-tokens 128
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from analysis.steering.data import load_steering_data
from core.hooks import BatchedLayerSteeringHook
from utils.generation import generate_batch
from utils.judge import TraitJudge
from utils.model import format_prompt, load_model
from utils.paths import get as get_path, load_experiment_config
from utils.vectors import load_vector


def parse_args():
    parser = argparse.ArgumentParser(description="Combination steering strategies")
    parser.add_argument("--experiment", required=True)

    # Instruction vector config
    parser.add_argument("--inst-trait", required=True, help="e.g., pv_instruction/sycophancy")
    parser.add_argument("--inst-layer", type=int, required=True)
    parser.add_argument("--inst-method", default="probe")
    parser.add_argument("--inst-position", default="response[:]")
    parser.add_argument("--inst-coef", type=float, required=True, help="Optimal single-source coefficient")
    parser.add_argument("--inst-variant", default="instruct")

    # Natural vector config
    parser.add_argument("--nat-trait", required=True, help="e.g., pv_natural/sycophancy")
    parser.add_argument("--nat-layer", type=int, required=True)
    parser.add_argument("--nat-method", default="mean_diff")
    parser.add_argument("--nat-position", default="response[:5]")
    parser.add_argument("--nat-coef", type=float, required=True, help="Optimal single-source coefficient")
    parser.add_argument("--nat-variant", default="base")

    # Steering config
    parser.add_argument("--coefficients", default="2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model-variant", default="instruct", help="Model to steer (application model)")

    return parser.parse_args()


async def run_strategy(
    strategy_name,
    model,
    tokenizer,
    questions,
    formatted_questions,
    steering_configs_per_coef,
    coefficients,
    judge,
    trait_name,
    trait_definition,
    eval_prompt,
    max_new_tokens,
):
    """Run a single strategy across all coefficients and return results."""
    n_questions = len(questions)
    n_coefs = len(coefficients)

    # Build batched prompts (replicate questions for each coefficient)
    batched_prompts = []
    for _ in range(n_coefs):
        batched_prompts.extend(formatted_questions)

    # Build steering configs with batch slices
    all_configs = []
    for coef_idx, coef in enumerate(coefficients):
        batch_start = coef_idx * n_questions
        batch_end = (coef_idx + 1) * n_questions
        for layer, vector, coef_mult in steering_configs_per_coef(coef):
            all_configs.append((layer, vector, coef_mult, (batch_start, batch_end)))

    # Generate with batched steering
    print(f"\n  {strategy_name}: generating {len(batched_prompts)} responses...")
    with BatchedLayerSteeringHook(model, all_configs, component="residual"):
        all_responses = generate_batch(model, tokenizer, batched_prompts, max_new_tokens=max_new_tokens)

    # Score all responses
    all_qa_pairs = []
    for coef_idx in range(n_coefs):
        start = coef_idx * n_questions
        end = (coef_idx + 1) * n_questions
        for q, r in zip(questions, all_responses[start:end]):
            all_qa_pairs.append((q, r))

    print(f"  Scoring {len(all_qa_pairs)} responses...")
    all_scores = await judge.score_steering_batch(
        all_qa_pairs, trait_name, trait_definition, eval_prompt=eval_prompt
    )

    # Aggregate per coefficient
    results = []
    for coef_idx, coef in enumerate(coefficients):
        start = coef_idx * n_questions
        end = (coef_idx + 1) * n_questions
        cand_scores = all_scores[start:end]

        trait_scores = [s["trait_score"] for s in cand_scores if s["trait_score"] is not None]
        coh_scores = [s["coherence_score"] for s in cand_scores if s.get("coherence_score") is not None]

        trait_mean = sum(trait_scores) / len(trait_scores) if trait_scores else 0
        coh_mean = sum(coh_scores) / len(coh_scores) if coh_scores else 0

        print(f"  coef={coef:.1f}: trait={trait_mean:.1f}, coherence={coh_mean:.1f}")
        results.append({
            "strategy": strategy_name,
            "coefficient": coef,
            "trait_mean": round(trait_mean, 1),
            "coherence_mean": round(coh_mean, 1),
            "n": len(trait_scores),
        })

    return results


async def main():
    args = parse_args()
    coefficients = [float(c) for c in args.coefficients.split(",")]

    # Load model
    exp_config = load_experiment_config(args.experiment)
    model_name = exp_config["model_variants"][args.model_variant]["model"]

    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name)
    device = next(model.parameters()).device

    # Load vectors
    print("\nLoading vectors...")
    inst_vector = load_vector(
        args.experiment, args.inst_trait, args.inst_layer,
        args.inst_variant, args.inst_method, "residual", args.inst_position,
    )
    nat_vector = load_vector(
        args.experiment, args.nat_trait, args.nat_layer,
        args.nat_variant, args.nat_method, "residual", args.nat_position,
    )

    if inst_vector is None:
        raise FileNotFoundError(f"Instruction vector not found: {args.inst_trait} L{args.inst_layer} {args.inst_method} {args.inst_position}")
    if nat_vector is None:
        raise FileNotFoundError(f"Natural vector not found: {args.nat_trait} L{args.nat_layer} {args.nat_method} {args.nat_position}")

    inst_vector = inst_vector.to(device)
    nat_vector = nat_vector.to(device)

    # Also load vectors at the other's best layer for combination strategies
    inst_vector_at_nat_layer = load_vector(
        args.experiment, args.inst_trait, args.nat_layer,
        args.inst_variant, args.inst_method, "residual", args.inst_position,
    )
    nat_vector_at_inst_layer = load_vector(
        args.experiment, args.nat_trait, args.inst_layer,
        args.nat_variant, args.nat_method, "residual", args.nat_position,
    )

    if inst_vector_at_nat_layer is not None:
        inst_vector_at_nat_layer = inst_vector_at_nat_layer.to(device)
    if nat_vector_at_inst_layer is not None:
        nat_vector_at_inst_layer = nat_vector_at_inst_layer.to(device)

    print(f"  Instruction: L{args.inst_layer} {args.inst_method} norm={inst_vector.norm():.4f}")
    print(f"  Natural: L{args.nat_layer} {args.nat_method} norm={nat_vector.norm():.4f}")

    # Load steering questions (shared between both traits)
    steering_data = load_steering_data(args.inst_trait)
    questions = steering_data.questions
    trait_name = steering_data.trait_name
    trait_definition = steering_data.trait_definition
    eval_prompt = steering_data.eval_prompt

    formatted_questions = [format_prompt(q, tokenizer) for q in questions]
    print(f"\nQuestions: {len(questions)}")

    # Compute baseline
    print("\nComputing baseline...")
    baseline_responses = generate_batch(model, tokenizer, formatted_questions, max_new_tokens=args.max_new_tokens)
    baseline_qa = list(zip(questions, baseline_responses))
    judge = TraitJudge()
    baseline_scores = await judge.score_steering_batch(baseline_qa, trait_name, trait_definition, eval_prompt=eval_prompt)
    baseline_trait = sum(s["trait_score"] for s in baseline_scores if s["trait_score"] is not None) / len(baseline_scores)
    print(f"  Baseline trait: {baseline_trait:.1f}")

    all_results = [{"strategy": "baseline", "trait_mean": round(baseline_trait, 1), "coherence_mean": 90.0}]

    # =========================================================
    # Strategy 1: Combined vector at natural's best layer
    # =========================================================
    if inst_vector_at_nat_layer is not None:
        combined_at_nat = inst_vector_at_nat_layer + nat_vector
        combined_at_nat = combined_at_nat / combined_at_nat.norm() * nat_vector.norm()  # Normalize to nat's scale

        s1_results = await run_strategy(
            "S1_combined_at_nat_layer", model, tokenizer,
            questions, formatted_questions,
            lambda coef: [(args.nat_layer, combined_at_nat, coef)],
            coefficients, judge, trait_name, trait_definition, eval_prompt, args.max_new_tokens,
        )
        all_results.extend(s1_results)

    # =========================================================
    # Strategy 2: Combined vector at instruct's best layer
    # =========================================================
    if nat_vector_at_inst_layer is not None:
        combined_at_inst = inst_vector + nat_vector_at_inst_layer
        combined_at_inst = combined_at_inst / combined_at_inst.norm() * inst_vector.norm()  # Normalize to inst's scale

        s2_results = await run_strategy(
            "S2_combined_at_inst_layer", model, tokenizer,
            questions, formatted_questions,
            lambda coef: [(args.inst_layer, combined_at_inst, coef)],
            coefficients, judge, trait_name, trait_definition, eval_prompt, args.max_new_tokens,
        )
        all_results.extend(s2_results)

    # =========================================================
    # Strategy 3: Combined vector at overall best layer
    # =========================================================
    # Use instruct's layer since it had higher trait delta
    if nat_vector_at_inst_layer is not None:
        # Same as S2 but explicitly labeled
        s3_results = await run_strategy(
            "S3_combined_at_best_layer", model, tokenizer,
            questions, formatted_questions,
            lambda coef: [(args.inst_layer, combined_at_inst, coef)],
            coefficients, judge, trait_name, trait_definition, eval_prompt, args.max_new_tokens,
        )
        all_results.extend(s3_results)

    # =========================================================
    # Strategy 4: Ensemble (two hooks, two layers)
    # =========================================================
    # Steer both layers simultaneously at half their optimal single-source coefficients
    inst_half = args.inst_coef / 2
    nat_half = args.nat_coef / 2

    s4_results = await run_strategy(
        "S4_ensemble_half_strength", model, tokenizer,
        questions, formatted_questions,
        lambda coef: [
            (args.inst_layer, inst_vector, inst_half * (coef / args.inst_coef)),
            (args.nat_layer, nat_vector, nat_half * (coef / args.nat_coef)),
        ],
        coefficients, judge, trait_name, trait_definition, eval_prompt, args.max_new_tokens,
    )
    all_results.extend(s4_results)

    # =========================================================
    # Save results
    # =========================================================
    output_dir = get_path("experiments.base", experiment=args.experiment) / "combination_steering"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"

    output = {
        "config": {
            "inst_trait": args.inst_trait,
            "inst_layer": args.inst_layer,
            "inst_method": args.inst_method,
            "inst_position": args.inst_position,
            "inst_coef": args.inst_coef,
            "nat_trait": args.nat_trait,
            "nat_layer": args.nat_layer,
            "nat_method": args.nat_method,
            "nat_position": args.nat_position,
            "nat_coef": args.nat_coef,
            "coefficients": coefficients,
            "max_new_tokens": args.max_new_tokens,
        },
        "baseline_trait": round(baseline_trait, 1),
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Best per strategy (coherence ≥ 70)")
    print("=" * 80)
    print(f"{'Strategy':<35} {'Coef':>6} {'Trait':>7} {'Coh':>7} {'Δ':>7}")
    print("-" * 80)

    for strategy_name in ["S1_combined_at_nat_layer", "S2_combined_at_inst_layer",
                          "S3_combined_at_best_layer", "S4_ensemble_half_strength"]:
        strategy_results = [r for r in all_results if r["strategy"] == strategy_name]
        valid = [r for r in strategy_results if r["coherence_mean"] >= 70]
        if valid:
            best = max(valid, key=lambda r: r["trait_mean"])
            delta = best["trait_mean"] - baseline_trait
            print(f"{strategy_name:<35} {best['coefficient']:>6.1f} {best['trait_mean']:>7.1f} {best['coherence_mean']:>7.1f} {delta:>+7.1f}")
        else:
            print(f"{strategy_name:<35} {'N/A':>6} {'N/A':>7} {'N/A':>7} {'N/A':>7}")


if __name__ == "__main__":
    asyncio.run(main())
