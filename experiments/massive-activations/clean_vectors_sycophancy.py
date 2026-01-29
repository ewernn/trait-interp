#!/usr/bin/env python3
"""Zero out massive dims from sycophancy mean_diff vectors."""

import torch
from pathlib import Path

# Same massive dims from gemma-3-4b calibration
MASSIVE_DIMS = [19, 295, 368, 443, 656, 1055, 1209, 1276, 1365, 1548, 1698, 1980, 2194]

VECTOR_DIR = Path("experiments/massive-activations/extraction/pv_natural/sycophancy/base/vectors/response__5/residual")
INPUT_DIR = VECTOR_DIR / "mean_diff"
OUTPUT_DIR = VECTOR_DIR / "mean_diff_cleaned"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for pt_file in sorted(INPUT_DIR.glob("layer*.pt")):
    vector = torch.load(pt_file, weights_only=True)
    cleaned = vector.clone()
    for dim in MASSIVE_DIMS:
        if dim < cleaned.shape[0]:
            cleaned[dim] = 0.0
    cleaned = cleaned / cleaned.norm()
    torch.save(cleaned, OUTPUT_DIR / pt_file.name)
    print(f"Cleaned {pt_file.name}")

print(f"\nSaved {len(list(OUTPUT_DIR.glob('*.pt')))} files to {OUTPUT_DIR}")
