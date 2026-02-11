#!/usr/bin/env python3
"""Create mean_diff_top5 cleaning variant for refusal."""

import torch
from pathlib import Path

# Massive dims ordered by L15 magnitude (from calibration.json)
# 443 (~1697) >> 1365 (~25) > 1980 (~19) > 295 (~18) > 1698 (~15)
MASSIVE_DIMS_TOP5 = [443, 1365, 1980, 295, 1698]

VECTOR_DIR = Path("experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__5/residual")
INPUT_DIR = VECTOR_DIR / "mean_diff"
OUTPUT_DIR = VECTOR_DIR / "mean_diff_top5"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for pt_file in sorted(INPUT_DIR.glob("layer*.pt")):
    vector = torch.load(pt_file, weights_only=True)
    cleaned = vector.clone()
    for dim in MASSIVE_DIMS_TOP5:
        if dim < cleaned.shape[0]:
            cleaned[dim] = 0.0
    cleaned = cleaned / cleaned.norm()
    torch.save(cleaned, OUTPUT_DIR / pt_file.name)

print(f"Created mean_diff_top5: zeroed 5 dims {MASSIVE_DIMS_TOP5}")
print(f"Saved {len(list(OUTPUT_DIR.glob('*.pt')))} files to {OUTPUT_DIR}")
