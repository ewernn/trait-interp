"""Validate probe fingerprints against LLM judge scores.

Scores LoRA-generated responses on all 23 traits using GPT-4.1-mini judge,
then correlates judge scores with probe activation fingerprints per trait.

Input: pxs_grid_14b/responses/, pxs_grid_14b/probe_scores/, datasets/inference/
Output: analysis/judge_validation/scores/, analysis/judge_validation/summary.json
Usage:
    # Pilot (3 traits × 2 variants, ~$2.60)
    python experiments/mats-emergent-misalignment/analysis_judge_validation.py \
        --traits new_traits/aggression chirp/refusal new_traits/warmth \
        --variants em_rank32 angry_refusal --n-responses 5

    # Full run (all 23 traits × 6 variants, ~$30)
    python experiments/mats-emergent-misalignment/analysis_judge_validation.py \
        --n-responses 5 --skip-existing

    # Analysis only
    python experiments/mats-emergent-misalignment/analysis_judge_validation.py --analyze-only
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.judge import TraitJudge
from utils.traits import load_trait_definition

# --- Constants ---

EXP_DIR = Path("experiments/mats-emergent-misalignment")
RESPONSES_DIR = EXP_DIR / "analysis/pxs_grid_14b/responses"
PROBE_SCORES_DIR = EXP_DIR / "analysis/pxs_grid_14b/probe_scores"
PROMPTS_DIR = Path("datasets/inference")

ALL_TRAITS = [
    "alignment/deception", "alignment/conflicted",
    "bs/lying", "bs/concealment",
    "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
    "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
    "mental_state/obedience", "mental_state/rationalization",
    "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
    "pv_natural/sycophancy", "chirp/refusal",
    "new_traits/aggression", "new_traits/amusement", "new_traits/contempt",
    "new_traits/frustration", "new_traits/hedging", "new_traits/sadness",
    "new_traits/warmth",
]

ALL_VARIANTS = [
    "em_rank32", "em_rank1", "mocking_refusal", "angry_refusal",
    "curt_refusal", "clean_instruct",
]

LORA_VARIANTS = [v for v in ALL_VARIANTS if v != "clean_instruct"]

ALL_EVAL_SETS = [
    "em_generic_eval", "em_medical_eval", "emotional_vulnerability",
    "ethical_dilemmas", "identity_introspection", "interpersonal_advice",
    "sriram_diverse", "sriram_factual", "sriram_harmful", "sriram_normal",
]

# Traits where the judge should reliably detect behavioral differences
TEXT_OBSERVABLE = {
    "new_traits/aggression", "new_traits/warmth", "new_traits/contempt",
    "new_traits/frustration", "new_traits/amusement", "new_traits/sadness",
    "new_traits/hedging", "chirp/refusal", "pv_natural/sycophancy",
    "mental_state/confidence", "mental_state/confusion",
}


# --- Data loading ---

def load_prompts(eval_set: str) -> dict[str, str]:
    """Load {prompt_id: prompt_text} from datasets/inference/{eval_set}.json."""
    path = PROMPTS_DIR / f"{eval_set}.json"
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return {item["id"]: item["prompt"] for item in data}


def load_responses(variant: str, eval_set: str) -> dict[str, list[str]]:
    """Load {prompt_id: [response1, ...]} from pxs_grid responses."""
    path = RESPONSES_DIR / f"{variant}_x_{eval_set}.json"
    if not path.exists():
        raise FileNotFoundError(f"Responses not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_probe_scores(variant: str, eval_set: str, score_type: str) -> dict[str, float]:
    """Load probe scores for a cell. score_type: combined, text_only, reverse_model."""
    path = PROBE_SCORES_DIR / f"{variant}_x_{eval_set}_{score_type}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# --- Scoring ---

async def score_cell(
    judge: TraitJudge,
    variant: str,
    eval_set: str,
    traits: list[str],
    trait_defs: dict[str, str],
    n_responses: int,
    output_dir: Path,
    skip_existing: bool,
) -> dict[str, list[dict]]:
    """Score all responses in a cell for given traits. Returns {trait: [scores]}."""
    prompts = load_prompts(eval_set)
    responses = load_responses(variant, eval_set)
    cell_dir = output_dir / "scores" / f"{variant}_x_{eval_set}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    cell_results = {}

    for trait in traits:
        trait_name = trait.split("/")[-1]
        trait_file = cell_dir / f"{trait_name}.json"

        if skip_existing and trait_file.exists():
            with open(trait_file) as f:
                cell_results[trait] = json.load(f)
            continue

        # Build (question, answer) pairs — sample n_responses per prompt
        qa_pairs = []
        qa_metadata = []  # track which prompt/response for saving
        for prompt_id, prompt_text in prompts.items():
            if prompt_id not in responses:
                continue
            resp_list = responses[prompt_id]
            sampled = resp_list[:n_responses] if len(resp_list) <= n_responses else random.sample(resp_list, n_responses)
            for resp in sampled:
                qa_pairs.append((prompt_text, resp))
                qa_metadata.append({"prompt_id": prompt_id, "response_preview": resp[:100]})

        if not qa_pairs:
            print(f"  WARNING: No responses for {variant} x {eval_set}")
            continue

        # Score using score_steering directly (skip coherence, save cost)
        semaphore = asyncio.Semaphore(20)
        async def score_one(q: str, a: str) -> float | None:
            async with semaphore:
                return await judge.score_steering(q, a, trait_name, trait_defs[trait])

        tasks = [score_one(q, a) for q, a in qa_pairs]
        scores = await asyncio.gather(*tasks)

        # Save per-response results
        results = []
        for i, score in enumerate(scores):
            results.append({
                "prompt_id": qa_metadata[i]["prompt_id"],
                "response_preview": qa_metadata[i]["response_preview"],
                "trait_score": score,
            })

        with open(trait_file, "w") as f:
            json.dump(results, f, indent=2)

        cell_results[trait] = results
        valid = [r["trait_score"] for r in results if r["trait_score"] is not None]
        mean_score = np.mean(valid) if valid else float("nan")
        print(f"  {trait_name}: mean={mean_score:.1f} (n={len(valid)}/{len(results)})")

    return cell_results


# --- Analysis ---

def compute_cell_means(output_dir: Path, traits: list[str], variants: list[str], eval_sets: list[str]) -> dict:
    """Compute mean judge score per (variant, eval_set, trait). Returns nested dict."""
    judge_means = {}  # {variant: {eval_set: {trait: mean_score}}}
    for variant in variants:
        judge_means[variant] = {}
        for eval_set in eval_sets:
            judge_means[variant][eval_set] = {}
            for trait in traits:
                trait_name = trait.split("/")[-1]
                score_file = output_dir / "scores" / f"{variant}_x_{eval_set}" / f"{trait_name}.json"
                if not score_file.exists():
                    judge_means[variant][eval_set][trait] = None
                    continue
                with open(score_file) as f:
                    results = json.load(f)
                valid = [r["trait_score"] for r in results if r["trait_score"] is not None]
                judge_means[variant][eval_set][trait] = float(np.mean(valid)) if valid else None
    return judge_means


def analyze(output_dir: Path, traits: list[str], variants: list[str], eval_sets: list[str]):
    """Run correlation analysis between judge scores and probe scores."""
    judge_means = compute_cell_means(output_dir, traits, variants, eval_sets)

    # Only use LoRA variants for correlation (clean_instruct excluded)
    corr_variants = [v for v in variants if v != "clean_instruct"]

    results = {}
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS: Judge scores vs Probe scores")
    print("=" * 80)

    for probe_type in ["combined", "text_only"]:
        print(f"\n--- Probe type: {probe_type} ---")
        print(f"{'Trait':<30} {'Pooled ρ':>10} {'p-value':>10} {'Judge SD':>10} {'Category':>15}")
        print("-" * 80)

        type_results = {}
        for trait in traits:
            trait_name = trait.split("/")[-1]

            # Within-variant Spearman: for each variant, correlate across eval sets
            variant_rhos = []
            all_judge = []
            all_probe = []

            for variant in corr_variants:
                judge_vals = []
                probe_vals = []
                for eval_set in eval_sets:
                    j_score = judge_means.get(variant, {}).get(eval_set, {}).get(trait)
                    p_scores = load_probe_scores(variant, eval_set, probe_type)
                    p_score = p_scores.get(trait)
                    if j_score is not None and p_score is not None:
                        judge_vals.append(j_score)
                        probe_vals.append(p_score)

                if len(judge_vals) >= 4:  # need enough points for Spearman
                    rho, _ = stats.spearmanr(judge_vals, probe_vals)
                    if not np.isnan(rho):
                        variant_rhos.append(rho)

                all_judge.extend(judge_vals)
                all_probe.extend(probe_vals)

            # Fisher z-transform to pool within-variant rhos
            if variant_rhos:
                z_vals = [np.arctanh(min(max(r, -0.999), 0.999)) for r in variant_rhos]
                z_mean = np.mean(z_vals)
                pooled_rho = float(np.tanh(z_mean))
                # Approximate p-value: z-test on mean Fisher z
                z_se = 1.0 / np.sqrt(len(z_vals)) * np.std(z_vals) if len(z_vals) > 1 else 1.0
                z_stat = z_mean / z_se if z_se > 0 else 0
                p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
            else:
                pooled_rho = float("nan")
                p_value = 1.0

            judge_sd = float(np.std(all_judge)) if all_judge else 0.0
            category = "text-obs" if trait in TEXT_OBSERVABLE else "internal"

            type_results[trait] = {
                "pooled_rho": pooled_rho,
                "p_value": p_value,
                "judge_mean": float(np.mean(all_judge)) if all_judge else None,
                "judge_sd": judge_sd,
                "n_variants_with_rho": len(variant_rhos),
                "per_variant_rhos": {v: r for v, r in zip(corr_variants, variant_rhos)} if variant_rhos else {},
                "category": category,
                "n_cells": len(all_judge),
            }

            flag = "***" if p_value < 0.05 and not np.isnan(pooled_rho) else "   "
            low_var = " LOW-VAR" if judge_sd < 5 else ""
            print(f"  {trait_name:<28} {pooled_rho:>10.3f} {p_value:>10.4f} {judge_sd:>10.1f} {category:>12} {flag}{low_var}")

        results[probe_type] = type_results

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for probe_type in ["combined", "text_only"]:
        tr = results[probe_type]
        sig_traits = [t for t, v in tr.items() if v["p_value"] < 0.05 and not np.isnan(v["pooled_rho"])]
        text_obs_rhos = [v["pooled_rho"] for t, v in tr.items() if v["category"] == "text-obs" and not np.isnan(v["pooled_rho"])]
        internal_rhos = [v["pooled_rho"] for t, v in tr.items() if v["category"] == "internal" and not np.isnan(v["pooled_rho"])]
        low_var = [t.split("/")[-1] for t, v in tr.items() if v["judge_sd"] < 5]

        print(f"\n{probe_type}:")
        print(f"  Significant traits (p<0.05): {len(sig_traits)}/23")
        if text_obs_rhos:
            print(f"  Text-observable mean ρ: {np.mean(text_obs_rhos):.3f} (n={len(text_obs_rhos)})")
        if internal_rhos:
            print(f"  Internal mean ρ: {np.mean(internal_rhos):.3f} (n={len(internal_rhos)})")
        if low_var:
            print(f"  Low-variance traits (SD<5): {', '.join(low_var)}")

    # Save
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {summary_path}")

    return results


# --- Main ---

async def main():
    parser = argparse.ArgumentParser(description="Validate probe fingerprints against LLM judge")
    parser.add_argument("--traits", nargs="+", default=ALL_TRAITS, help="Traits to score")
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS, help="Variants to score")
    parser.add_argument("--eval-sets", nargs="+", default=ALL_EVAL_SETS, help="Eval sets to score")
    parser.add_argument("--n-responses", type=int, default=5, help="Responses per prompt to sample")
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR / "analysis/judge_validation",
                        help="Output directory")
    parser.add_argument("--skip-existing", action="store_true", help="Skip cells already scored")
    parser.add_argument("--analyze-only", action="store_true", help="Skip scoring, run analysis only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for response sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        # Load trait definitions
        trait_defs = {}
        for trait in args.traits:
            trait_defs[trait] = load_trait_definition(trait)
            if trait_defs[trait].startswith("The trait '"):
                print(f"WARNING: No definition.txt for {trait}, using fallback")

        # Count total API calls for cost estimate
        total_calls = 0
        for variant in args.variants:
            for eval_set in args.eval_sets:
                prompts = load_prompts(eval_set)
                responses = load_responses(variant, eval_set)
                n_prompts = sum(1 for pid in prompts if pid in responses)
                total_calls += len(args.traits) * n_prompts * args.n_responses
        est_cost = total_calls * 0.0005
        print(f"Estimated: {total_calls} API calls (~${est_cost:.2f})")
        print(f"Scoring {len(args.traits)} traits × {len(args.variants)} variants × {len(args.eval_sets)} eval sets")
        print(f"Sampling {args.n_responses} responses per prompt\n")

        judge = TraitJudge()
        try:
            for variant in args.variants:
                for eval_set in args.eval_sets:
                    print(f"\n[{variant} × {eval_set}]")
                    await score_cell(
                        judge, variant, eval_set, args.traits, trait_defs,
                        args.n_responses, args.output_dir, args.skip_existing,
                    )
        finally:
            await judge.close()

    # Run analysis
    analyze(args.output_dir, args.traits, args.variants, args.eval_sets)


if __name__ == "__main__":
    asyncio.run(main())
