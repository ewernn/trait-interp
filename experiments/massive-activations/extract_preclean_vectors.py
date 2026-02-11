#!/usr/bin/env python3
"""Extract vectors with pre-cleaned activations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import json
from core.methods import MeanDifferenceMethod, ProbeMethod, PreCleanedMethod

# Load massive dims from calibration
def load_massive_dims(model: str) -> list:
    if 'gemma-3' in model or model == 'base':
        calib_path = Path("experiments/gemma-3-4b/inference/instruct/massive_activations/calibration.json")
    else:
        calib_path = Path("experiments/gemma-2-2b/inference/instruct/massive_activations/calibration.json")

    with open(calib_path) as f:
        calib = json.load(f)

    # Get dims appearing in top-5 at 3+ layers
    from collections import Counter
    all_dims = []
    for layer_dims in calib["aggregate"]["top_dims_by_layer"].values():
        all_dims.extend(layer_dims[:5])
    dim_counts = Counter(all_dims)
    return [d for d, c in dim_counts.items() if c >= 3]

# Configs
CONFIGS = [
    ("base", "experiments/massive-activations/extraction/chirp/refusal/base", 34),
    ("gemma2_base", "experiments/massive-activations/extraction/chirp/refusal/gemma2_base", 26),
]

for variant, base_dir, n_layers in CONFIGS:
    massive_dims = load_massive_dims(variant)
    print(f"\n{variant}: cleaning {len(massive_dims)} dims: {sorted(massive_dims)[:5]}...")

    for position in ["response__5"]:
        act_dir = Path(base_dir) / "activations" / position / "residual"
        vec_dir = Path(base_dir) / "vectors" / position / "residual"

        # Load activations
        train_path = act_dir / "train_all_layers.pt"
        if not train_path.exists():
            print(f"  {position}: activations not found")
            continue

        all_acts = torch.load(train_path, weights_only=True)

        # Load metadata for pos/neg split
        with open(act_dir / "metadata.json") as f:
            meta = json.load(f)
        n_pos = meta["n_examples_pos"]

        # Extract preclean vectors for each layer
        for method_name, MethodClass in [("mean_diff", MeanDifferenceMethod), ("probe", ProbeMethod)]:
            output_dir = vec_dir / f"{method_name}_preclean"
            output_dir.mkdir(parents=True, exist_ok=True)

            base_method = MethodClass()
            preclean_method = PreCleanedMethod(base_method, massive_dims)

            for layer in range(n_layers):
                layer_acts = all_acts[:, layer, :]
                pos_acts = layer_acts[:n_pos]
                neg_acts = layer_acts[n_pos:]

                result = preclean_method.extract(pos_acts, neg_acts)
                torch.save(result['vector'], output_dir / f"layer{layer}.pt")

            print(f"  {position}/{method_name}_preclean: {n_layers} layers")

print("\nDone!")
