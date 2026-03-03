"""Export tonal detection vectors as .pt files for probe_monitor.py.

Creates two files:
  - best1.pt: single best detection layer per trait
  - avg3.pt: 3-layer window (best ± 1) per trait

Format:
  best1: {trait: {"layer": L, "vector": tensor[2560]}}
  avg3:  {trait: {"layers": [L-1, L, L+1], "vectors": [tensor, tensor, tensor]}}

Usage:
    python /tmp/export_detection_vectors.py --layers '{"angry_register": 20, "bureaucratic": 21, ...}'
    # Or after heuristic is found:
    python /tmp/export_detection_vectors.py --heuristic steering_peak
"""

import argparse
import json
import torch
from pathlib import Path

VEC_BASE = Path("/home/dev/trait-interp/experiments/aria_rl/extraction/tonal")
OUTPUT_DIR = Path("/home/dev/persona-generalization/methods/probe_vectors")

TRAITS = ["angry_register", "bureaucratic", "confused_processing",
          "disappointed_register", "mocking", "nervous_register"]

# Trait name mapping: how they'll appear in probe_monitor
TRAIT_KEYS = {
    "angry_register": "tonal/angry_register",
    "bureaucratic": "tonal/bureaucratic",
    "confused_processing": "tonal/confused_processing",
    "disappointed_register": "tonal/disappointed_register",
    "mocking": "tonal/mocking",
    "nervous_register": "tonal/nervous_register",
}

MAX_LAYER = 34
MIN_LAYER = 1


def load_vector(trait, layer):
    path = VEC_BASE / trait / "qwen3_4b_base" / "vectors" / "response__5" / "residual" / "probe" / f"layer{layer}.pt"
    vec = torch.load(path, weights_only=True, map_location="cpu").float()
    return vec / vec.norm()


def export(layer_choices: dict):
    """Export best1.pt and avg3.pt given {trait: layer} mapping."""

    best1 = {}
    avg3 = {}

    for trait in TRAITS:
        L = layer_choices[trait]
        key = TRAIT_KEYS[trait]

        # Best-1
        vec = load_vector(trait, L)
        best1[key] = {"layer": L, "vector": vec}

        # Avg-3: L-1, L, L+1 (clamped to valid range)
        layers_3 = [max(MIN_LAYER, L - 1), L, min(MAX_LAYER, L + 1)]
        vectors_3 = [load_vector(trait, l) for l in layers_3]
        avg3[key] = {"layers": layers_3, "vectors": vectors_3}

        print(f"  {trait:<24} best1=L{L}  avg3=L{layers_3}")

    # Save
    best1_path = OUTPUT_DIR / "qwen3_4b_tonal_best1.pt"
    avg3_path = OUTPUT_DIR / "qwen3_4b_tonal_avg3.pt"

    torch.save(best1, best1_path)
    torch.save(avg3, avg3_path)

    print(f"\nSaved {best1_path}")
    print(f"Saved {avg3_path}")

    # Verify
    for path in [best1_path, avg3_path]:
        d = torch.load(path, weights_only=True, map_location="cpu")
        print(f"  {path.name}: {len(d)} traits")


def get_steering_peaks():
    """Get best coherent steering layer per trait from results.jsonl."""
    steer_base = Path("/home/dev/trait-interp/experiments/aria_rl/steering/tonal")
    peaks = {}
    for trait in TRAITS:
        results_file = steer_base / trait / "qwen3_4b_instruct/response__5/steering/results.jsonl"
        baseline = None
        best_per_layer = {}
        with open(results_file) as f:
            for line in f:
                d = json.loads(line)
                if d.get("type") == "header": continue
                if d.get("type") == "baseline":
                    baseline = d["result"]["trait_mean"]; continue
                layer = d["config"]["vectors"][0]["layer"]
                delta = d["result"]["trait_mean"] - baseline
                coh = d["result"]["coherence_mean"]
                if coh >= 70:
                    if layer not in best_per_layer or delta > best_per_layer[layer]:
                        best_per_layer[layer] = delta
        if best_per_layer:
            peaks[trait] = max(best_per_layer, key=best_per_layer.get)
        else:
            print(f"  WARNING: {trait} has no coherent steering results, using L20 fallback")
            peaks[trait] = 20
    return peaks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, help="JSON dict of {trait: layer}")
    parser.add_argument("--heuristic", type=str, choices=["steering_peak", "steering_plus_2", "steering_plus_5"],
                        help="Auto-compute layers from steering results")
    parser.add_argument("--offset", type=int, default=0, help="Add offset to heuristic layers")
    args = parser.parse_args()

    if args.layers:
        layer_choices = json.loads(args.layers)
    elif args.heuristic:
        peaks = get_steering_peaks()
        offset = {"steering_peak": 0, "steering_plus_2": 2, "steering_plus_5": 5}[args.heuristic] + args.offset
        layer_choices = {t: min(MAX_LAYER, max(MIN_LAYER, L + offset)) for t, L in peaks.items()}
        print(f"Heuristic: {args.heuristic} + offset={args.offset}")
        print(f"Steering peaks: {peaks}")
    else:
        parser.error("Provide --layers or --heuristic")

    export(layer_choices)
