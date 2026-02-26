#!/usr/bin/env python3
"""
Probe-weighted multi-layer steering evaluation.

Uses two-stage probe weights from multi_layer_probe.py to steer with
multiple layer×component hooks simultaneously. Each hook's contribution
is scaled by its probe weight relative to the total.

Input: Probe results from analysis/multi_layer_probe.py
Output: Steering results in standard format (results.jsonl + responses/)

Usage:
    python analysis/steering/probe_weighted_evaluate.py \
        --experiment mats-emergent-misalignment \
        --traits rm_hack/eval_awareness,pv_natural/sycophancy

    # Custom weight threshold (fraction of total weight to include)
    python analysis/steering/probe_weighted_evaluate.py \
        --experiment mats-emergent-misalignment \
        --traits mental_state/obedience \
        --weight-threshold 0.02

Note: For additional weight strategies (causal, intersection, topk),
use weighted_multi_layer_evaluate.py --weight-source <strategy>.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import asyncio

from analysis.steering.multi_layer_evaluate import run_multi_layer_evaluation
from analysis.steering.weight_sources import build_weighted_configs
from core import LocalBackend
from utils.judge import TraitJudge
from utils.model import load_model_with_lora
from utils.paths import get_default_variant, get_model_variant, load_experiment_config
from utils.vectors import MIN_COHERENCE


def build_probe_configs(experiment, trait, extraction_variant, method, position,
                        weight_threshold=0.01, device=None):
    """Build steering configs from probe weights. Delegates to weight_sources."""
    return build_weighted_configs(
        "probe", experiment, trait, extraction_variant, method, position,
        weight_threshold=weight_threshold, device=device,
    )


async def main():
    parser = argparse.ArgumentParser(description="Probe-weighted multi-layer steering")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--traits", required=True, help="Comma-separated traits")
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--position", default="response[:5]")
    parser.add_argument("--weight-threshold", type=float, default=0.01,
                        help="Min probe weight as fraction of total (default: 0.01)")
    parser.add_argument("--prompt-set", default="steering")
    parser.add_argument("--subset", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--judge", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--search-steps", type=int, default=8)
    parser.add_argument("--min-coherence", type=float, default=MIN_COHERENCE)
    parser.add_argument("--save-responses", choices=["all", "best", "none"], default="best")
    parser.add_argument("--direction", choices=["positive", "negative"], default="positive")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--extraction-variant", default=None)
    parser.add_argument("--vector-experiment", default=None)
    args = parser.parse_args()

    traits = [t.strip() for t in args.traits.split(",")]

    exp_config = load_experiment_config(args.experiment)
    model_variant = args.model_variant or get_default_variant(args.experiment, mode='application')
    model_info = get_model_variant(args.experiment, model_variant)
    model_name = model_info['model']
    vector_experiment = args.vector_experiment or args.experiment
    extraction_variant = args.extraction_variant or get_default_variant(vector_experiment, mode='extraction')

    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_with_lora(model_name)
    backend = LocalBackend.from_model(model, tokenizer)
    judge = TraitJudge()

    for trait in traits:
        print(f"\n{'='*60}")
        print(f"PROBE-WEIGHTED STEERING: {trait}")
        print(f"{'='*60}")

        vectors_and_layers, per_hook_weights, base_coef = build_probe_configs(
            vector_experiment, trait, extraction_variant,
            args.method, args.position,
            weight_threshold=args.weight_threshold,
            device=model.device,
        )

        if not vectors_and_layers:
            print(f"  No valid configs for {trait}, skipping")
            continue

        try:
            await run_multi_layer_evaluation(
                experiment=args.experiment,
                trait=trait,
                vector_experiment=vector_experiment,
                model_variant=model_variant,
                extraction_variant=extraction_variant,
                method=args.method,
                component="residual",  # default, overridden by per-config components
                position=args.position,
                layers=[],  # unused when override is provided
                prompt_set=args.prompt_set,
                judge_provider=args.judge,
                model_name=model_name,
                subset=args.subset,
                backend=backend,
                judge=judge,
                n_search_steps=args.search_steps,
                max_new_tokens=args.max_new_tokens,
                min_coherence=args.min_coherence,
                save_mode=args.save_responses,
                relevance_check=True,
                direction=args.direction,
                force=args.force,
                vectors_and_layers_override=vectors_and_layers,
                per_hook_weights=per_hook_weights,
                base_coef_override=base_coef,
            )
        except Exception as e:
            print(f"\nERROR on {trait}: {e}")
            import traceback
            traceback.print_exc()
            continue

    await judge.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
