#!/usr/bin/env python3
"""Create top-2 cleaned vectors."""

import torch
import json
from pathlib import Path
from collections import Counter

# Load calibration for gemma-3-4b
with open("experiments/gemma-3-4b/inference/instruct/massive_activations/calibration.json") as f:
    calib = json.load(f)

# Get global dims (appearing in top-5 at 3+ layers) for consistency with existing scripts
all_dims = []
for layer_dims in calib["aggregate"]["top_dims_by_layer"].values():
    all_dims.extend(layer_dims[:5])
dim_counts = Counter(all_dims)
global_dims = sorted([d for d, c in dim_counts.items() if c >= 3], key=lambda d: -dim_counts[d])

# Top-2 of the global massive dims
TOP_2_DIMS = global_dims[:2]
print(f"Top-2 global dims: {TOP_2_DIMS}")

VECTOR_DIR = Path("experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__5/residual")

for base_method in ["mean_diff", "probe"]:
    input_dir = VECTOR_DIR / base_method
    output_dir = VECTOR_DIR / f"{base_method}_top2"
    output_dir.mkdir(parents=True, exist_ok=True)

    for pt_file in sorted(input_dir.glob("layer*.pt")):
        vector = torch.load(pt_file, weights_only=True)
        cleaned = vector.clone()

        for dim in TOP_2_DIMS:
            if dim < cleaned.shape[0]:
                cleaned[dim] = 0.0

        cleaned = cleaned / cleaned.norm()
        torch.save(cleaned, output_dir / pt_file.name)

    print(f"Created {base_method}_top2")

print("Done!")
