"""Compute separation ratio on model_delta vectors vs raw fingerprints.

Model delta = reverse_model score - baseline score (element-wise, 23D).
This isolates what the LoRA adds beyond the base model's behavior.

Input: probe_scores/{variant}_x_{eval}_reverse_model.json, clean_instruct_x_{eval}_combined.json
Output: Printed separation metrics comparing model_delta vs raw fingerprints.
Usage: python experiments/mats-emergent-misalignment/analysis_model_delta_separation.py
"""

import json
import numpy as np
from pathlib import Path
from itertools import combinations

SCORES_DIR = Path("experiments/mats-emergent-misalignment/analysis/pxs_grid_14b/probe_scores")

VARIANTS = ["em_rank32", "em_rank1", "mocking_refusal", "angry_refusal", "curt_refusal"]
EVAL_SETS = [
    "em_generic_eval", "em_medical_eval", "emotional_vulnerability",
    "ethical_dilemmas", "identity_introspection", "interpersonal_advice",
    "sriram_diverse", "sriram_factual", "sriram_harmful", "sriram_normal",
]


def load_json(path: Path) -> dict[str, float]:
    with open(path) as f:
        return json.load(f)


def dict_to_vector(d: dict[str, float], trait_order: list[str]) -> np.ndarray:
    return np.array([d[t] for t in trait_order])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_cosine_mean(vectors: list[np.ndarray]) -> float:
    """Mean cosine similarity over all pairs."""
    sims = [cosine_sim(a, b) for a, b in combinations(vectors, 2)]
    return np.mean(sims) if sims else 0.0


def pairwise_euclidean_distances(vectors: list[np.ndarray]) -> list[float]:
    """All pairwise Euclidean distances."""
    return [float(np.linalg.norm(a - b)) for a, b in combinations(vectors, 2)]


def compute_separation_metrics(variant_vectors: dict[str, list[np.ndarray]]):
    """
    Compute within-variant spread, between-variant distance, and separation ratio.

    Within-variant spread: mean pairwise Euclidean distance among eval sets for the same variant.
    Between-variant distance: mean pairwise Euclidean distance between variant centroids.
    Separation ratio: between / within.
    """
    # Within-variant: mean pairwise distance per variant, then average
    within_distances_per_variant = {}
    for variant, vecs in variant_vectors.items():
        dists = pairwise_euclidean_distances(vecs)
        within_distances_per_variant[variant] = np.mean(dists) if dists else 0.0

    mean_within = np.mean(list(within_distances_per_variant.values()))

    # Centroids
    centroids = {v: np.mean(vecs, axis=0) for v, vecs in variant_vectors.items()}

    # Between-variant: pairwise distances between centroids
    centroid_list = list(centroids.values())
    between_dists = pairwise_euclidean_distances(centroid_list)
    mean_between = np.mean(between_dists) if between_dists else 0.0

    # Also compute all-pairs between-variant (not just centroids)
    all_between_dists = []
    variant_names = list(variant_vectors.keys())
    for i, v1 in enumerate(variant_names):
        for v2 in variant_names[i + 1:]:
            for vec1 in variant_vectors[v1]:
                for vec2 in variant_vectors[v2]:
                    all_between_dists.append(float(np.linalg.norm(vec1 - vec2)))
    mean_between_all_pairs = np.mean(all_between_dists) if all_between_dists else 0.0

    # Within-variant cosine similarity
    within_cosine_per_variant = {}
    for variant, vecs in variant_vectors.items():
        within_cosine_per_variant[variant] = pairwise_cosine_mean(vecs)

    mean_within_cosine = np.mean(list(within_cosine_per_variant.values()))

    # Between-variant cosine (centroids)
    centroid_cosines = [cosine_sim(a, b) for a, b in combinations(centroid_list, 2)]
    mean_between_cosine = np.mean(centroid_cosines) if centroid_cosines else 0.0

    separation_ratio_centroid = mean_between / mean_within if mean_within > 1e-12 else float("inf")
    separation_ratio_all = mean_between_all_pairs / mean_within if mean_within > 1e-12 else float("inf")

    return {
        "within_distances": within_distances_per_variant,
        "mean_within_distance": mean_within,
        "mean_between_distance_centroids": mean_between,
        "mean_between_distance_all_pairs": mean_between_all_pairs,
        "separation_ratio_centroids": separation_ratio_centroid,
        "separation_ratio_all_pairs": separation_ratio_all,
        "within_cosine": within_cosine_per_variant,
        "mean_within_cosine": mean_within_cosine,
        "mean_between_cosine_centroids": mean_between_cosine,
    }


def main():
    # Determine trait order from first file
    sample = load_json(SCORES_DIR / f"{VARIANTS[0]}_x_{EVAL_SETS[0]}_reverse_model.json")
    trait_order = sorted(sample.keys())
    print(f"Traits ({len(trait_order)}): {trait_order}\n")

    # Load baselines (clean_instruct)
    baselines = {}
    for eval_set in EVAL_SETS:
        path = SCORES_DIR / f"clean_instruct_x_{eval_set}_combined.json"
        baselines[eval_set] = dict_to_vector(load_json(path), trait_order)

    # Build raw fingerprints and model deltas
    raw_vectors = {v: [] for v in VARIANTS}
    delta_vectors = {v: [] for v in VARIANTS}

    for variant in VARIANTS:
        for eval_set in EVAL_SETS:
            # Raw fingerprint: combined scores
            raw_path = SCORES_DIR / f"{variant}_x_{eval_set}_combined.json"
            raw = dict_to_vector(load_json(raw_path), trait_order)
            raw_vectors[variant].append(raw)

            # Model delta: reverse_model - baseline
            rm_path = SCORES_DIR / f"{variant}_x_{eval_set}_reverse_model.json"
            rm = dict_to_vector(load_json(rm_path), trait_order)
            delta = rm - baselines[eval_set]
            delta_vectors[variant].append(delta)

    # Compute metrics for both
    print("=" * 80)
    print("RAW FINGERPRINTS (combined scores)")
    print("=" * 80)
    raw_metrics = compute_separation_metrics(raw_vectors)
    print_metrics(raw_metrics)

    print()
    print("=" * 80)
    print("MODEL DELTAS (reverse_model - baseline)")
    print("=" * 80)
    delta_metrics = compute_separation_metrics(delta_vectors)
    print_metrics(delta_metrics)

    # Comparison
    print()
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"  Raw fingerprint separation ratio (centroids):    {raw_metrics['separation_ratio_centroids']:.3f}")
    print(f"  Model delta separation ratio (centroids):        {delta_metrics['separation_ratio_centroids']:.3f}")
    print(f"  Improvement:                                     {delta_metrics['separation_ratio_centroids'] / raw_metrics['separation_ratio_centroids']:.2f}x")
    print()
    print(f"  Raw fingerprint separation ratio (all pairs):    {raw_metrics['separation_ratio_all_pairs']:.3f}")
    print(f"  Model delta separation ratio (all pairs):        {delta_metrics['separation_ratio_all_pairs']:.3f}")
    print(f"  Improvement:                                     {delta_metrics['separation_ratio_all_pairs'] / raw_metrics['separation_ratio_all_pairs']:.2f}x")
    print()
    print(f"  Raw within-variant cosine (self-consistency):    {raw_metrics['mean_within_cosine']:.3f}")
    print(f"  Delta within-variant cosine (self-consistency):  {delta_metrics['mean_within_cosine']:.3f}")
    print()
    print(f"  Raw between-variant cosine (inter-variant sim):  {raw_metrics['mean_between_cosine_centroids']:.3f}")
    print(f"  Delta between-variant cosine (inter-variant sim):{delta_metrics['mean_between_cosine_centroids']:.3f}")

    # Per-variant centroid analysis
    print()
    print("=" * 80)
    print("PER-VARIANT CENTROIDS (model delta)")
    print("=" * 80)
    for variant in VARIANTS:
        centroid = np.mean(delta_vectors[variant], axis=0)
        # Top 5 traits by absolute magnitude
        top_idx = np.argsort(np.abs(centroid))[::-1][:5]
        top_traits = [(trait_order[i], centroid[i]) for i in top_idx]
        print(f"\n  {variant}:")
        print(f"    Norm: {np.linalg.norm(centroid):.2f}")
        for t, v in top_traits:
            print(f"    {t:>35s}: {v:+.2f}")

    # Pairwise centroid cosine similarity matrix
    print()
    print("=" * 80)
    print("CENTROID COSINE SIMILARITY MATRIX (model delta)")
    print("=" * 80)
    centroids = {v: np.mean(delta_vectors[v], axis=0) for v in VARIANTS}
    short_names = [v.replace("_refusal", "").replace("em_", "em") for v in VARIANTS]
    header = f"{'':>18s}" + "".join(f"{n:>14s}" for n in short_names)
    print(header)
    for i, v1 in enumerate(VARIANTS):
        row = f"{short_names[i]:>18s}"
        for j, v2 in enumerate(VARIANTS):
            sim = cosine_sim(centroids[v1], centroids[v2])
            row += f"{sim:>14.3f}"
        print(row)


def print_metrics(metrics: dict):
    print("\n  Within-variant mean Euclidean distance (spread across eval sets):")
    for variant, dist in metrics["within_distances"].items():
        print(f"    {variant:>20s}: {dist:.3f}")
    print(f"    {'MEAN':>20s}: {metrics['mean_within_distance']:.3f}")

    print(f"\n  Between-variant mean Euclidean distance:")
    print(f"    Centroids:   {metrics['mean_between_distance_centroids']:.3f}")
    print(f"    All pairs:   {metrics['mean_between_distance_all_pairs']:.3f}")

    print(f"\n  Separation ratio (between / within):")
    print(f"    Centroids:   {metrics['separation_ratio_centroids']:.3f}")
    print(f"    All pairs:   {metrics['separation_ratio_all_pairs']:.3f}")

    print(f"\n  Within-variant cosine similarity:")
    for variant, sim in metrics["within_cosine"].items():
        print(f"    {variant:>20s}: {sim:.3f}")
    print(f"    {'MEAN':>20s}: {metrics['mean_within_cosine']:.3f}")

    print(f"\n  Between-variant cosine similarity (centroids): {metrics['mean_between_cosine_centroids']:.3f}")


if __name__ == "__main__":
    main()
