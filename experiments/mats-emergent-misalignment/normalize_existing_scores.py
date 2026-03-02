#!/usr/bin/env python3
"""Post-hoc normalize existing probe scores by activation magnitude.

For checkpoint_method_b JSONs and pxs_grid probe_score JSONs that can't be
re-run (missing adapters), this applies the same normalization that scoring
pipelines now do at score time: score / norms_per_layer[probe_layer].

Input: Existing unnormalized JSONs + activation norms file
Output: Overwrites JSONs with normalized scores, adds metadata flags

Usage:
    # Normalize checkpoint JSONs
    python experiments/mats-emergent-misalignment/normalize_existing_scores.py \
        --checkpoint-jsons analysis/checkpoint_method_b/turner_r64.json \
                           analysis/checkpoint_method_b/sports.json

    # Normalize pxs_grid probe_scores directory
    python experiments/mats-emergent-misalignment/normalize_existing_scores.py \
        --pxs-dir analysis/pxs_grid_4b/probe_scores \
        --norms-file analysis/activation_norms_4b.json

    # Normalize everything that isn't already normalized
    python experiments/mats-emergent-misalignment/normalize_existing_scores.py --all
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from utils.projections import load_activation_norms
from utils.vectors import get_best_vector

EXPERIMENT = "mats-emergent-misalignment"
BASE_DIR = Path(__file__).parent

TRAITS = [
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


def get_trait_layers(experiment, extraction_variant="base", steering_variant="instruct"):
    """Get best layer per trait from steering results."""
    layers = {}
    for trait in TRAITS:
        try:
            info = get_best_vector(
                experiment, trait,
                extraction_variant=extraction_variant,
                steering_variant=steering_variant,
            )
            layers[trait] = info["layer"]
        except (FileNotFoundError, ValueError):
            pass
    return layers


def normalize_checkpoint_json(path, norms, trait_layers_override=None):
    """Normalize a checkpoint_method_b JSON in place."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if data.get("metadata", {}).get("normalized", False):
        print(f"  SKIP {path.name}: already normalized")
        return False

    # Get trait layers from the JSON's own probes section
    if "probes" in data:
        trait_layers = {t: p["layer"] for t, p in data["probes"].items()}
    elif trait_layers_override:
        trait_layers = trait_layers_override
    else:
        raise ValueError(f"No probes section in {path} and no trait_layers_override provided")

    def normalize_scores(scores):
        return {
            t: s / float(norms[trait_layers[t]]) if t in trait_layers and float(norms[trait_layers[t]]) > 1e-8 else s
            for t, s in scores.items()
        }

    # Normalize baseline
    if "baseline" in data:
        data["baseline"]["scores"] = normalize_scores(data["baseline"]["scores"])
        if "per_eval" in data["baseline"]:
            for eval_name in data["baseline"]["per_eval"]:
                data["baseline"]["per_eval"][eval_name] = normalize_scores(
                    data["baseline"]["per_eval"][eval_name]
                )

    # Normalize each checkpoint
    for ckpt in data.get("checkpoints", []):
        # reverse_model scores
        if "reverse_model" in ckpt:
            ckpt["reverse_model"] = normalize_scores(ckpt["reverse_model"])
        if "reverse_model_per_eval" in ckpt:
            for eval_name in ckpt["reverse_model_per_eval"]:
                ckpt["reverse_model_per_eval"][eval_name] = normalize_scores(
                    ckpt["reverse_model_per_eval"][eval_name]
                )

        # Recompute model_delta from normalized scores
        if "reverse_model" in ckpt and "baseline" in data:
            ckpt["model_delta"] = {
                t: ckpt["reverse_model"][t] - data["baseline"]["scores"][t]
                for t in ckpt["reverse_model"]
            }
        if "reverse_model_per_eval" in ckpt and "per_eval" in data.get("baseline", {}):
            ckpt["model_delta_per_eval"] = {}
            for eval_name in ckpt["reverse_model_per_eval"]:
                if eval_name in data["baseline"]["per_eval"]:
                    ckpt["model_delta_per_eval"][eval_name] = {
                        t: ckpt["reverse_model_per_eval"][eval_name][t] -
                           data["baseline"]["per_eval"][eval_name][t]
                        for t in ckpt["reverse_model_per_eval"][eval_name]
                    }

    # Mark as normalized
    data.setdefault("metadata", {})["normalized"] = True
    data["metadata"]["norms_source"] = "post_hoc"

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    n_ckpts = len(data.get("checkpoints", []))
    print(f"  OK {path.name}: {n_ckpts} checkpoints normalized")
    return True


def normalize_pxs_scores(scores_dir, norms, trait_layers):
    """Normalize pxs_grid probe_score JSONs in place."""
    scores_dir = Path(scores_dir)
    count = 0
    for f in sorted(scores_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)

        # Simple {trait: score} format
        if not isinstance(data, dict):
            continue

        # Skip if already has normalized marker (we add a __metadata key)
        if "__normalized" in data:
            continue

        normalized = {}
        for trait, score in data.items():
            if trait.startswith("__"):
                normalized[trait] = score
                continue
            if trait in trait_layers:
                layer = trait_layers[trait]
                norm = float(norms[layer])
                normalized[trait] = score / norm if norm > 1e-8 else score
            else:
                normalized[trait] = score

        normalized["__normalized"] = True

        with open(f, "w") as fh:
            json.dump(normalized, fh, indent=2)
        count += 1

    print(f"  Normalized {count} probe score files in {scores_dir.name}/")
    return count


def main():
    parser = argparse.ArgumentParser(description="Post-hoc normalize existing probe scores")
    parser.add_argument("--checkpoint-jsons", nargs="+", default=None,
                        help="Checkpoint method B JSONs to normalize")
    parser.add_argument("--pxs-dir", type=str, default=None,
                        help="pxs_grid probe_scores directory to normalize")
    parser.add_argument("--norms-file", type=str, default=None,
                        help="Activation norms JSON (auto-detected if not specified)")
    parser.add_argument("--all", action="store_true",
                        help="Find and normalize all unnormalized data")
    parser.add_argument("--experiment", default=EXPERIMENT)
    args = parser.parse_args()

    if not args.checkpoint_jsons and not args.pxs_dir and not args.all:
        parser.print_help()
        sys.exit(1)

    # Determine norms file
    if args.norms_file:
        norms_path = Path(args.norms_file)
        if not norms_path.is_absolute():
            norms_path = BASE_DIR / norms_path
    else:
        norms_path = BASE_DIR / "analysis" / "activation_norms_14b.json"

    print(f"Loading norms: {norms_path.name}")
    norms = load_activation_norms(norms_path)

    # Get trait layers for pxs_grid normalization
    trait_layers = get_trait_layers(args.experiment)
    print(f"Loaded {len(trait_layers)} trait layers")

    if args.all:
        # Normalize all checkpoint JSONs
        ckpt_dir = BASE_DIR / "analysis" / "checkpoint_method_b"
        print(f"\n=== Checkpoint JSONs ({ckpt_dir.name}/) ===")
        for f in sorted(ckpt_dir.glob("*.json")):
            normalize_checkpoint_json(f, norms, trait_layers)

        # Normalize 14B pxs_grid
        pxs_14b = BASE_DIR / "analysis" / "pxs_grid_14b" / "probe_scores"
        if pxs_14b.exists():
            print(f"\n=== pxs_grid_14b probe_scores ===")
            normalize_pxs_scores(pxs_14b, norms, trait_layers)

        # Normalize 4B pxs_grid (different norms)
        norms_4b_path = BASE_DIR / "analysis" / "activation_norms_4b.json"
        pxs_4b = BASE_DIR / "analysis" / "pxs_grid_4b" / "probe_scores"
        if pxs_4b.exists() and norms_4b_path.exists():
            print(f"\n=== pxs_grid_4b probe_scores ===")
            norms_4b = load_activation_norms(norms_4b_path)
            # 4B uses different extraction/steering variants
            trait_layers_4b = get_trait_layers(
                args.experiment,
                extraction_variant="qwen3_4b_base",
                steering_variant="qwen3_4b_instruct",
            )
            normalize_pxs_scores(pxs_4b, norms_4b, trait_layers_4b)

        # Normalize aria_rl checkpoint
        aria_ckpt = BASE_DIR.parent / "aria_rl" / "analysis" / "method_b"
        if aria_ckpt.exists():
            print(f"\n=== aria_rl checkpoint JSONs ===")
            norms_4b = load_activation_norms(norms_4b_path) if norms_4b_path.exists() else None
            if norms_4b is not None:
                for f in sorted(aria_ckpt.glob("*.json")):
                    normalize_checkpoint_json(f, norms_4b)
            else:
                print("  SKIP: no 4B norms file")

        return

    if args.checkpoint_jsons:
        print(f"\n=== Checkpoint JSONs ===")
        for json_path in args.checkpoint_jsons:
            p = Path(json_path)
            if not p.is_absolute():
                p = BASE_DIR / p
            normalize_checkpoint_json(p, norms, trait_layers)

    if args.pxs_dir:
        print(f"\n=== pxs_grid probe_scores ===")
        p = Path(args.pxs_dir)
        if not p.is_absolute():
            p = BASE_DIR / p
        normalize_pxs_scores(p, norms, trait_layers)


if __name__ == "__main__":
    main()
