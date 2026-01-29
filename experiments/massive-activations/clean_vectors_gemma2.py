#!/usr/bin/env python3
"""Zero out massive dims from gemma-2-2b mean_diff vectors."""

import torch
from pathlib import Path

# Massive dims from gemma-2-2b calibration (L12)
MASSIVE_DIMS_GEMMA2 = [334, 1068, 1570, 1807, 682, 535, 1393, 1645]

# Top dims by magnitude at mid-layers
MASSIVE_DIMS_TOP1 = [334]
MASSIVE_DIMS_TOP5 = [334, 1068, 1570, 1807, 682]

VECTOR_DIR = Path("experiments/massive-activations/extraction/chirp/refusal/gemma2_base/vectors/response__5/residual")
INPUT_DIR = VECTOR_DIR / "mean_diff"

VARIANTS = {
    "mean_diff_cleaned": MASSIVE_DIMS_GEMMA2,  # All 8
    "mean_diff_top1": MASSIVE_DIMS_TOP1,        # Just dim 334
    "mean_diff_top5": MASSIVE_DIMS_TOP5,        # Top 5
}

for variant_name, dims_to_zero in VARIANTS.items():
    output_dir = VECTOR_DIR / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for pt_file in sorted(INPUT_DIR.glob("layer*.pt")):
        vector = torch.load(pt_file, weights_only=True)
        cleaned = vector.clone()
        for dim in dims_to_zero:
            if dim < cleaned.shape[0]:
                cleaned[dim] = 0.0
        cleaned = cleaned / cleaned.norm()
        torch.save(cleaned, output_dir / pt_file.name)

    print(f"Created {variant_name}: zeroed {len(dims_to_zero)} dims")

print("\nDone!")
