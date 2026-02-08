#!/usr/bin/env python3
"""Zero out massive dims from mean_diff vectors."""

import torch
from pathlib import Path

# Massive dims from gemma-3-4b calibration (appear in top-5 at 2+ layers)
MASSIVE_DIMS = [19, 295, 368, 443, 656, 1055, 1209, 1276, 1365, 1548, 1698, 1980, 2194]

# Paths
VECTOR_DIR = Path("experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__5/residual")
INPUT_DIR = VECTOR_DIR / "mean_diff"
OUTPUT_DIR = VECTOR_DIR / "mean_diff_cleaned"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for pt_file in sorted(INPUT_DIR.glob("layer*.pt")):
    vector = torch.load(pt_file, weights_only=True)

    # Zero out massive dims
    cleaned = vector.clone()
    for dim in MASSIVE_DIMS:
        if dim < cleaned.shape[0]:
            cleaned[dim] = 0.0

    # Re-normalize to unit norm
    cleaned = cleaned / cleaned.norm()

    # Save
    output_path = OUTPUT_DIR / pt_file.name
    torch.save(cleaned, output_path)
    print(f"Cleaned {pt_file.name}: zeroed {len(MASSIVE_DIMS)} dims")

print(f"\nSaved to {OUTPUT_DIR}")
print(f"Total files: {len(list(OUTPUT_DIR.glob('*.pt')))}")
