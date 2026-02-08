#!/usr/bin/env python3
"""Compute pairwise cosine similarities between vector methods."""

import torch
import json
from pathlib import Path

VECTOR_DIR = Path("experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__5/residual")
METHODS = ["mean_diff", "mean_diff_cleaned", "probe"]

def load_vectors(method: str) -> dict:
    """Load all layer vectors for a method."""
    method_dir = VECTOR_DIR / method
    vectors = {}
    for pt_file in sorted(method_dir.glob("layer*.pt")):
        layer = int(pt_file.stem.replace("layer", ""))
        vectors[layer] = torch.load(pt_file, weights_only=True).float()
    return vectors

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a @ b / (a.norm() * b.norm())).item()

# Load all vectors
all_vectors = {m: load_vectors(m) for m in METHODS}
layers = sorted(all_vectors["mean_diff"].keys())

# Compute pairwise similarities
results = {
    "layers": layers,
    "comparisons": {}
}

pairs = [
    ("mean_diff", "probe"),
    ("mean_diff", "mean_diff_cleaned"),
    ("mean_diff_cleaned", "probe"),
]

for m1, m2 in pairs:
    key = f"{m1}_vs_{m2}"
    sims = []
    for layer in layers:
        sim = cosine_sim(all_vectors[m1][layer], all_vectors[m2][layer])
        sims.append(round(sim, 4))
    results["comparisons"][key] = sims

    # Summary stats
    avg = sum(sims) / len(sims)
    print(f"{key}:")
    print(f"  mean: {avg:.3f}")
    print(f"  range: [{min(sims):.3f}, {max(sims):.3f}]")
    print()

# Save
output_path = Path("experiments/massive-activations/cosine_similarities.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {output_path}")
