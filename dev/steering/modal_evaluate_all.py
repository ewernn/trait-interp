#!/usr/bin/env python3
"""
Run steering eval for all traits across multiple Modal GPUs.

Reads layer_selection.json for per-trait layers, splits by direction,
greedy bin-packs into N shards, launches parallel Modal calls.

Input:
    - experiments/{experiment}/steering/layer_selection.json
    - datasets/traits/{category}/{trait}/steering.json (for direction)
    - Experiment config, vectors, datasets

Output:
    - Steering results for all traits pulled to local filesystem

Usage:
    # Dry run — show shard assignments
    python dev/steering/modal_evaluate_all.py --experiment emotion_set --dry-run

    # Full run
    python dev/steering/modal_evaluate_all.py --experiment emotion_set --force
"""

import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.paths import load_experiment_config, get_model_variant


def load_trait_directions(experiment: str) -> dict[str, str]:
    """Read direction from each trait's steering.json."""
    datasets_dir = Path("datasets/traits")
    directions = {}

    layer_sel = load_layer_selection(experiment)
    for trait in layer_sel:
        steering_file = datasets_dir / experiment / trait / "steering.json"
        direction = "positive"
        if steering_file.exists():
            with open(steering_file) as f:
                data = json.load(f)
                direction = data.get("direction", "positive")
        directions[trait] = direction

    return directions


def load_layer_selection(experiment: str) -> dict:
    """Load layer_selection.json."""
    path = Path(f"experiments/{experiment}/steering/layer_selection.json")
    if not path.exists():
        raise FileNotFoundError(f"No layer_selection.json at {path}")
    with open(path) as f:
        return json.load(f)


def build_shards(experiment: str, n_gpus: int = 9, n_neg_gpus: int = 2) -> list[dict]:
    """Build direction-aware shard assignments via greedy bin-packing.

    Returns list of shard dicts: {traits, direction, trait_layers, total_layers}
    """
    layer_sel = load_layer_selection(experiment)
    directions = load_trait_directions(experiment)

    # Split by direction
    pos_traits = []
    neg_traits = []
    for trait, info in layer_sel.items():
        layers = info["layers"]
        entry = {"trait": trait, "layers": layers, "n_layers": len(layers)}
        if directions.get(trait, "positive") == "negative":
            neg_traits.append(entry)
        else:
            pos_traits.append(entry)

    n_pos_gpus = n_gpus - n_neg_gpus

    def greedy_pack(traits: list[dict], n_bins: int) -> list[list[dict]]:
        """Pack traits into bins minimizing max bin size (by total layers)."""
        # Sort largest first for better packing
        traits_sorted = sorted(traits, key=lambda t: t["n_layers"], reverse=True)
        bins = [[] for _ in range(n_bins)]
        bin_sizes = [0] * n_bins

        for t in traits_sorted:
            # Put in lightest bin
            min_idx = min(range(n_bins), key=lambda i: bin_sizes[i])
            bins[min_idx].append(t)
            bin_sizes[min_idx] += t["n_layers"]

        return bins

    neg_bins = greedy_pack(neg_traits, n_neg_gpus)
    pos_bins = greedy_pack(pos_traits, n_pos_gpus)

    shards = []
    for i, bin_traits in enumerate(neg_bins):
        trait_layers = {}
        for t in bin_traits:
            trait_layers[f"{experiment}/{t['trait']}"] = ",".join(str(l) for l in t["layers"])
        shards.append({
            "id": i,
            "direction": "negative",
            "traits": [f"{experiment}/{t['trait']}" for t in bin_traits],
            "trait_layers": trait_layers,
            "total_layers": sum(t["n_layers"] for t in bin_traits),
        })

    for i, bin_traits in enumerate(pos_bins):
        trait_layers = {}
        for t in bin_traits:
            trait_layers[f"{experiment}/{t['trait']}"] = ",".join(str(l) for l in t["layers"])
        shards.append({
            "id": n_neg_gpus + i,
            "direction": "positive",
            "traits": [f"{experiment}/{t['trait']}" for t in bin_traits],
            "trait_layers": trait_layers,
            "total_layers": sum(t["n_layers"] for t in bin_traits),
        })

    return shards


def print_shards(shards: list[dict]):
    """Print shard assignment summary."""
    for shard in shards:
        dir_label = "NEG" if shard["direction"] == "negative" else "POS"
        print(f"  GPU {shard['id']} [{dir_label}]: {len(shard['traits'])} traits, "
              f"{shard['total_layers']} layer-slots")
        for trait in shard["traits"]:
            layers = shard["trait_layers"][trait]
            short = trait.split("/")[-1]
            print(f"    {short}: L[{layers}]")

    total_traits = sum(len(s["traits"]) for s in shards)
    total_layers = sum(s["total_layers"] for s in shards)
    max_layers = max(s["total_layers"] for s in shards)
    print(f"\n  Total: {total_traits} traits, {total_layers} layer-slots across {len(shards)} GPUs")
    print(f"  Bottleneck: {max_layers} layer-slots (GPU {max(shards, key=lambda s: s['total_layers'])['id']})")


def main():
    parser = argparse.ArgumentParser(description="Run steering eval across multiple Modal GPUs")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--n-gpus", type=int, default=9)
    parser.add_argument("--n-neg-gpus", type=int, default=2)
    parser.add_argument("--method", default="probe")
    parser.add_argument("--component", default="residual")
    parser.add_argument("--position", default="response[:5]")
    parser.add_argument("--search-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--save-responses", default="best", choices=["all", "best", "none"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-pull", action="store_true", help="Skip downloading results")
    parser.add_argument("--dry-run", action="store_true", help="Print shard assignments only")
    parser.add_argument("--traits", nargs="+", default=None,
                        help="Subset of traits to run (default: all from layer_selection.json)")
    args = parser.parse_args()

    config = load_experiment_config(args.experiment)
    model_variant = config.get("defaults", {}).get("application")
    extraction_variant = config.get("defaults", {}).get("extraction")
    variant_config = get_model_variant(args.experiment, model_variant)
    model_name = variant_config.model

    print(f"Experiment: {args.experiment}")
    print(f"Model: {model_name} ({model_variant})")
    print(f"Extraction: {extraction_variant}")
    print()

    # Build shards
    shards = build_shards(args.experiment, args.n_gpus, args.n_neg_gpus)

    # Filter to subset if specified
    if args.traits:
        trait_set = set(args.traits)
        for shard in shards:
            shard["traits"] = [t for t in shard["traits"] if t in trait_set or t.split("/")[-1] in trait_set]
            shard["trait_layers"] = {t: l for t, l in shard["trait_layers"].items()
                                     if t in shard["traits"]}
            shard["total_layers"] = sum(len(l.split(",")) for l in shard["trait_layers"].values())
        shards = [s for s in shards if s["traits"]]

    print(f"Shard assignments ({len(shards)} GPUs):")
    print_shards(shards)

    if args.dry_run:
        return

    # Sync all traits to volumes
    all_traits = []
    for shard in shards:
        all_traits.extend(shard["traits"])

    print(f"\nSyncing {len(all_traits)} traits to Modal volumes...")
    from dev.steering.modal_evaluate import sync_to_volumes
    t0 = time.time()
    sync_to_volumes(args.experiment, all_traits, extraction_variant)
    print(f"  Sync: {time.time() - t0:.1f}s")

    # Launch all shards in parallel
    import modal
    steering_fn = modal.Function.from_name("trait-steering", "steering_eval_remote")

    # Use fallback layers (won't be used since trait_layers covers all traits)
    fallback_layers = [20]

    print(f"\nLaunching {len(shards)} parallel GPU jobs...")
    t0 = time.time()

    handles = []
    for shard in shards:
        handle = steering_fn.spawn(
            model_name=model_name,
            experiment=args.experiment,
            traits=shard["traits"],
            model_variant=model_variant,
            extraction_variant=extraction_variant,
            layers=fallback_layers,
            trait_layers=shard["trait_layers"],
            method=args.method,
            component=args.component,
            position=args.position,
            direction=shard["direction"],
            search_steps=args.search_steps,
            max_new_tokens=args.max_new_tokens,
            subset=args.subset,
            force=args.force,
            save_responses=args.save_responses,
        )
        handles.append((shard, handle))
        print(f"  GPU {shard['id']}: spawned ({len(shard['traits'])} traits, {shard['direction']})")

    # Wait for all to complete
    print(f"\nWaiting for {len(handles)} jobs...")
    results = {}
    for shard, handle in handles:
        try:
            result = handle.get()
            load_time = result.get("model_load_time", 0)
            n_results = len(result.get("results", {}))
            elapsed = time.time() - t0
            print(f"  GPU {shard['id']}: done ({n_results} traits, model load {load_time:.0f}s, "
                  f"wall {elapsed:.0f}s)")
            results[shard["id"]] = result
        except Exception as e:
            print(f"  GPU {shard['id']}: FAILED — {e}")
            results[shard["id"]] = {"error": str(e)}

    total_time = time.time() - t0
    successful = sum(1 for r in results.values() if "error" not in r)
    print(f"\n{successful}/{len(shards)} GPUs completed in {total_time:.0f}s")

    # Pull results
    if not args.no_pull:
        print(f"\nPulling results...")
        from dev.steering.modal_evaluate import pull_results
        t_pull = time.time()
        pull_results(args.experiment, all_traits, model_variant, args.position)
        print(f"  Pull: {time.time() - t_pull:.1f}s")

    # Summary
    print(f"\nTotal wall time: {time.time() - t0:.0f}s")

    # Report any failures
    failed_shards = [sid for sid, r in results.items() if "error" in r]
    if failed_shards:
        print(f"\nFailed shards: {failed_shards}")
        print("Re-run with --traits to retry specific traits")


if __name__ == "__main__":
    main()
