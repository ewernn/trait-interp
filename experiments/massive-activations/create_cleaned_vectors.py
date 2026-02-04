#!/usr/bin/env python3
"""
Create cleaned vector variants for massive-activations experiment.

Input: Existing vectors + activations from extraction pipeline
Output: Cleaned vector variants under new method names

Usage:
    python experiments/massive-activations/create_cleaned_vectors.py \
        --trait chirp/refusal \
        --extraction-variant base \
        --calibration-experiment gemma-3-4b \
        --calibration-variant base
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import torch
from collections import Counter

from core.methods import MeanDifferenceMethod, ProbeMethod
from core.math import remove_massive_dims
from utils.paths import get_vector_path, get_activation_path, get_activation_metadata_path


def load_calibration(experiment, variant):
    """Load calibration data: top_dims_by_layer and global massive dims."""
    calib_path = Path(f"experiments/{experiment}/inference/{variant}/massive_activations/calibration.json")
    with open(calib_path) as f:
        calib = json.load(f)

    top_dims_by_layer = calib["aggregate"]["top_dims_by_layer"]

    # Global dims: appearing in top-5 at 3+ layers
    all_dims = []
    for layer_dims in top_dims_by_layer.values():
        all_dims.extend(layer_dims[:5])
    dim_counts = Counter(all_dims)
    global_dims = sorted([d for d, c in dim_counts.items() if c >= 3], key=lambda d: -dim_counts[d])

    return top_dims_by_layer, global_dims


def create_postclean_vectors(experiment, trait, extraction_variant, top_dims_by_layer, global_dims, position, component="residual"):
    """Create postclean variants (uniform + layer-aware, top 1/2/3)."""
    methods = ["mean_diff", "probe"]

    for method in methods:
        for n_dims in [1, 2, 3]:
            # --- Uniform: same global dims at all layers ---
            uniform_name = f"{method}_postclean_u{n_dims}"
            dims_to_zero = global_dims[:n_dims]

            layer = 0
            while True:
                src_path = get_vector_path(experiment, trait, method, layer, extraction_variant, component, position)
                if not src_path.exists():
                    break

                vector = torch.load(src_path, weights_only=True)
                cleaned = remove_massive_dims(vector.unsqueeze(0), dims_to_zero, clone=True).squeeze(0)
                cleaned = cleaned / (cleaned.norm() + 1e-8)

                dst_path = get_vector_path(experiment, trait, uniform_name, layer, extraction_variant, component, position)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(cleaned, dst_path)
                layer += 1

            if layer > 0:
                print(f"  {uniform_name}: {layer} layers (dims: {dims_to_zero})")

            # --- Layer-aware: per-layer dims ---
            la_name = f"{method}_postclean_la{n_dims}"

            layer = 0
            while True:
                src_path = get_vector_path(experiment, trait, method, layer, extraction_variant, component, position)
                if not src_path.exists():
                    break

                layer_key = str(layer)
                if layer_key in top_dims_by_layer:
                    dims_to_zero = top_dims_by_layer[layer_key][:n_dims]
                else:
                    dims_to_zero = global_dims[:n_dims]  # fallback

                vector = torch.load(src_path, weights_only=True)
                cleaned = remove_massive_dims(vector.unsqueeze(0), dims_to_zero, clone=True).squeeze(0)
                cleaned = cleaned / (cleaned.norm() + 1e-8)

                dst_path = get_vector_path(experiment, trait, la_name, layer, extraction_variant, component, position)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(cleaned, dst_path)
                layer += 1

            if layer > 0:
                print(f"  {la_name}: {layer} layers")


def create_preclean_vectors(experiment, trait, extraction_variant, top_dims_by_layer, position, component="residual"):
    """Create preclean variants (layer-aware top 1/2/3)."""
    act_path = get_activation_path(experiment, trait, extraction_variant, component, position)
    meta_path = get_activation_metadata_path(experiment, trait, extraction_variant, component, position)

    if not act_path.exists():
        print(f"  ERROR: Activations not found at {act_path}")
        return

    all_acts = torch.load(act_path, weights_only=True)
    with open(meta_path) as f:
        meta = json.load(f)

    n_pos = meta["n_examples_pos"]
    n_layers = meta.get("n_layers", all_acts.shape[1])

    base_methods = {"mean_diff": MeanDifferenceMethod(), "probe": ProbeMethod()}

    for method_name, base_method in base_methods.items():
        for n_dims in [1, 2, 3]:
            variant_name = f"{method_name}_preclean_la{n_dims}"
            n_extracted = 0

            for layer_idx in range(n_layers):
                layer_key = str(layer_idx)
                if layer_key in top_dims_by_layer:
                    dims_to_zero = top_dims_by_layer[layer_key][:n_dims]
                else:
                    continue

                layer_acts = all_acts[:, layer_idx, :]
                pos_acts = layer_acts[:n_pos]
                neg_acts = layer_acts[n_pos:]

                # Clean activations BEFORE extraction
                pos_clean = remove_massive_dims(pos_acts, dims_to_zero, clone=True)
                neg_clean = remove_massive_dims(neg_acts, dims_to_zero, clone=True)

                result = base_method.extract(pos_clean, neg_clean)
                vector = result['vector']

                dst_path = get_vector_path(experiment, trait, variant_name, layer_idx, extraction_variant, component, position)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(vector, dst_path)
                n_extracted += 1

            if n_extracted > 0:
                print(f"  {variant_name}: {n_extracted} layers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create cleaned vector variants")
    parser.add_argument("--trait", required=True, help="e.g., chirp/refusal")
    parser.add_argument("--extraction-variant", default="base")
    parser.add_argument("--calibration-experiment", required=True, help="e.g., gemma-3-4b")
    parser.add_argument("--calibration-variant", default="instruct")
    parser.add_argument("--position", default="response[:3]")
    args = parser.parse_args()

    # Pass raw position string — path helpers call sanitize_position() internally
    position = args.position

    print(f"Loading calibration from {args.calibration_experiment}/{args.calibration_variant}...")
    top_dims_by_layer, global_dims = load_calibration(args.calibration_experiment, args.calibration_variant)
    print(f"  Global massive dims ({len(global_dims)}): {global_dims[:5]}...")

    print(f"\nCreating postclean variants for {args.trait}...")
    create_postclean_vectors("massive-activations", args.trait, args.extraction_variant,
                             top_dims_by_layer, global_dims, position)

    print(f"\nCreating preclean variants for {args.trait}...")
    create_preclean_vectors("massive-activations", args.trait, args.extraction_variant,
                            top_dims_by_layer, position)

    print("\nDone!")
