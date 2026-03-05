"""Analyze activation space coverage of trait vectors.

Computes effective dimensionality of the activation manifold per layer,
the effective rank of the trait vector subspace, and the overlap between them.

Input: analysis/coverage/activations_L{layer}.pt, trait vectors from extraction/
Output: analysis/coverage/coverage_results.json, coverage_plot.png

Usage:
    PYTHONPATH=. python experiments/aria_rl/analyze_coverage.py
"""

import json
import warnings
from pathlib import Path

import torch
import numpy as np

from utils.paths import discover_steering_entries, load_experiment_config
from utils.vectors import get_best_vector, load_vector

EXPERIMENT = "aria_rl"
BASE_DIR = Path(__file__).parent
COVERAGE_DIR = BASE_DIR / "analysis" / "coverage"


def load_trait_vectors_by_layer(layers):
    """Load all trait vectors, grouped by their best steering layer.

    Returns:
        by_layer: dict[layer] -> list of (trait_name, vector) tuples
        all_vectors: dict[trait_name] -> (layer, vector)
    """
    entries = discover_steering_entries(EXPERIMENT)
    traits = sorted(set(e["trait"] for e in entries))
    config = load_experiment_config(EXPERIMENT)
    extraction_variant = config.get("defaults", {}).get("extraction")

    by_layer = {l: [] for l in layers}
    all_vectors = {}

    for t in traits:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best = get_best_vector(EXPERIMENT, t, min_delta=20)
            layer = best["layer"]
            if layer not in layers:
                continue
            vector = load_vector(
                EXPERIMENT, t, layer, extraction_variant,
                method=best["method"],
                component=best.get("component", "residual"),
                position=best.get("position", "response[:5]"),
            )
            if vector is not None:
                v = vector.float()
                by_layer[layer].append((t, v))
                all_vectors[t] = (layer, v)
        except Exception:
            pass

    return by_layer, all_vectors


def effective_rank(S, threshold_frac=0.01):
    """Count singular values above threshold_frac * max."""
    thresh = S[0] * threshold_frac
    return int((S > thresh).sum().item())


def variance_explained_at(S, k):
    """Fraction of variance explained by top-k singular values."""
    total = (S ** 2).sum()
    topk = (S[:k] ** 2).sum()
    return (topk / total).item()


def subspace_overlap(U_acts, U_traits, S_acts):
    """Fraction of activation variance that lives in the trait subspace.

    For each activation PC, compute how much of it lies in the trait subspace.
    Weight by the variance (S^2) of that PC.

    Args:
        U_acts: [hidden_dim, n_acts_components] — left singular vectors of activations
        U_traits: [hidden_dim, n_trait_components] — left singular vectors of trait vectors
        S_acts: [n_acts_components] — singular values of activations
    """
    # Project each activation PC onto trait subspace
    # overlap_i = ||U_traits^T @ u_i||^2 (fraction of PC_i in trait subspace)
    projections = U_traits.T @ U_acts  # [n_trait, n_act]
    overlap_per_pc = (projections ** 2).sum(dim=0)  # [n_act], each in [0, 1]

    # Variance-weighted overlap
    variance = S_acts ** 2
    total_var = variance.sum()
    weighted_overlap = (overlap_per_pc * variance).sum() / total_var

    return weighted_overlap.item(), overlap_per_pc.numpy()


def analyze_layer(layer, acts_tensor, trait_pairs):
    """Analyze one layer.

    Args:
        layer: layer index
        acts_tensor: [n_items, hidden_dim] mean response activations
        trait_pairs: list of (trait_name, vector) at this layer

    Returns: dict with analysis results
    """
    hidden_dim = acts_tensor.shape[1]
    n_items = acts_tensor.shape[0]

    # Center activations
    acts_centered = acts_tensor - acts_tensor.mean(dim=0, keepdim=True)

    # SVD of activations
    U_a, S_a, _ = torch.linalg.svd(acts_centered.T, full_matrices=False)
    # U_a: [hidden_dim, min(hidden_dim, n_items)]
    # S_a: [min(hidden_dim, n_items)]

    act_eff_rank = effective_rank(S_a)
    act_var_50 = None
    act_var_90 = None
    for k in range(1, len(S_a) + 1):
        ve = variance_explained_at(S_a, k)
        if act_var_50 is None and ve >= 0.5:
            act_var_50 = k
        if act_var_90 is None and ve >= 0.9:
            act_var_90 = k

    result = {
        "layer": layer,
        "n_items": n_items,
        "hidden_dim": hidden_dim,
        "activation_effective_rank": act_eff_rank,
        "activation_dims_for_50pct_var": act_var_50,
        "activation_dims_for_90pct_var": act_var_90,
    }

    if not trait_pairs:
        result["n_traits_at_layer"] = 0
        return result

    # Stack trait vectors, SVD
    trait_mat = torch.stack([v for _, v in trait_pairs])  # [n_traits, hidden_dim]
    U_t, S_t, _ = torch.linalg.svd(trait_mat.T, full_matrices=False)
    trait_eff_rank = effective_rank(S_t)

    # Compute overlap
    weighted_overlap, per_pc_overlap = subspace_overlap(U_a, U_t, S_a)

    # Also compute overlap using all traits across all layers (not just this layer's)
    result.update({
        "n_traits_at_layer": len(trait_pairs),
        "trait_effective_rank": trait_eff_rank,
        "variance_weighted_overlap": round(weighted_overlap, 4),
        "trait_names": [t for t, _ in trait_pairs],
    })
    return result


def analyze_global_overlap(layers, all_vectors, coverage_dir):
    """At each layer, project activations onto ALL trait vectors (not just that layer's).

    This answers: if we evaluate all 152 traits at layer L, how much of the
    activation variance do they cover?
    """
    results = []
    all_vecs = list(all_vectors.values())  # [(layer, vector), ...]
    # Get vectors — they're all the same hidden_dim regardless of which layer they came from
    vec_list = [v for _, v in all_vecs]
    if not vec_list:
        return results

    trait_mat = torch.stack(vec_list)  # [n_traits, hidden_dim]
    U_t, S_t, _ = torch.linalg.svd(trait_mat.T, full_matrices=False)
    global_trait_rank = effective_rank(S_t)

    for layer in layers:
        act_path = coverage_dir / f"activations_L{layer}.pt"
        if not act_path.exists():
            continue
        acts = torch.load(act_path, weights_only=True)
        acts_centered = acts - acts.mean(dim=0, keepdim=True)
        U_a, S_a, _ = torch.linalg.svd(acts_centered.T, full_matrices=False)

        wo, _ = subspace_overlap(U_a, U_t, S_a)
        results.append({
            "layer": layer,
            "global_trait_rank": global_trait_rank,
            "global_variance_weighted_overlap": round(wo, 4),
        })

    return results


def plot_results(per_layer, global_results, output_path):
    """Plot coverage analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers_local = [r["layer"] for r in per_layer if r.get("variance_weighted_overlap") is not None]
    overlap_local = [r["variance_weighted_overlap"] for r in per_layer if r.get("variance_weighted_overlap") is not None]
    n_traits_local = [r["n_traits_at_layer"] for r in per_layer if r.get("variance_weighted_overlap") is not None]
    act_rank = [r["activation_effective_rank"] for r in per_layer]
    act_50 = [r["activation_dims_for_50pct_var"] for r in per_layer]
    act_90 = [r["activation_dims_for_90pct_var"] for r in per_layer]
    all_layers = [r["layer"] for r in per_layer]

    layers_global = [r["layer"] for r in global_results]
    overlap_global = [r["global_variance_weighted_overlap"] for r in global_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Activation effective dimensionality
    ax = axes[0, 0]
    ax.plot(all_layers, act_rank, "o-", label="Effective rank (>1% of max SV)")
    ax.plot(all_layers, act_50, "s--", label="Dims for 50% variance")
    ax.plot(all_layers, act_90, "^--", label="Dims for 90% variance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Dimensions")
    ax.set_title("Activation Manifold Dimensionality")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Local overlap (traits at their best layer)
    ax = axes[0, 1]
    ax.bar(layers_local, overlap_local, color="steelblue", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Variance-Weighted Overlap")
    ax.set_title("Trait Coverage (local: traits at their best layer)")
    ax.set_ylim(0, max(overlap_local) * 1.3 if overlap_local else 1)
    for i, (l, o, n) in enumerate(zip(layers_local, overlap_local, n_traits_local)):
        ax.text(l, o + 0.005, f"n={n}", ha="center", fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. Global overlap (all traits projected to each layer)
    ax = axes[1, 0]
    ax.plot(layers_global, overlap_global, "o-", color="darkred", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Variance-Weighted Overlap")
    ax.set_title(f"Trait Coverage (global: all {len(list(set(t for t,_ in [])))} traits at each layer)")
    ax.set_ylim(0, max(overlap_global) * 1.3 if overlap_global else 1)
    ax.grid(True, alpha=0.3)
    # Fix title
    n_global = global_results[0]["global_trait_rank"] if global_results else 0
    ax.set_title(f"Trait Coverage (global: all traits, eff. rank={n_global})")

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = "Summary\n" + "=" * 40
    if per_layer:
        mean_rank = np.mean(act_rank)
        mean_90 = np.mean(act_90) if act_90 else 0
        summary_text += f"\n\nActivation manifold:"
        summary_text += f"\n  Mean effective rank: {mean_rank:.0f}"
        summary_text += f"\n  Mean dims for 90% var: {mean_90:.0f}"
        summary_text += f"\n  Hidden dim: {per_layer[0]['hidden_dim']}"
    if overlap_global:
        mean_global = np.mean(overlap_global)
        summary_text += f"\n\nTrait coverage (global):"
        summary_text += f"\n  Mean overlap: {mean_global:.1%}"
        summary_text += f"\n  Range: {min(overlap_global):.1%} - {max(overlap_global):.1%}"
    if overlap_local:
        mean_local = np.mean(overlap_local)
        summary_text += f"\n\nTrait coverage (local):"
        summary_text += f"\n  Mean overlap: {mean_local:.1%}"
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    print("Loading trait vectors...")
    meta_path = COVERAGE_DIR / "metadata.json"
    if not meta_path.exists():
        print(f"ERROR: Run capture_activations.py first ({meta_path} not found)")
        return

    with open(meta_path) as f:
        meta = json.load(f)
    layers = meta["layers"]

    by_layer, all_vectors = load_trait_vectors_by_layer(layers)
    total_traits = sum(len(v) for v in by_layer.values())
    print(f"  {total_traits} traits across {len(layers)} layers")

    # Per-layer analysis
    per_layer = []
    for layer in layers:
        act_path = COVERAGE_DIR / f"activations_L{layer}.pt"
        if not act_path.exists():
            print(f"  Skipping L{layer}: no activation file")
            continue
        acts = torch.load(act_path, weights_only=True)
        result = analyze_layer(layer, acts, by_layer[layer])
        per_layer.append(result)
        n_t = result.get("n_traits_at_layer", 0)
        overlap = result.get("variance_weighted_overlap", None)
        overlap_str = f"{overlap:.1%}" if overlap is not None else "N/A"
        print(f"  L{layer}: act_rank={result['activation_effective_rank']}, "
              f"90%_var={result['activation_dims_for_90pct_var']}, "
              f"n_traits={n_t}, overlap={overlap_str}")

    # Global analysis
    print("\nGlobal overlap (all traits at each layer)...")
    global_results = analyze_global_overlap(layers, all_vectors, COVERAGE_DIR)
    for r in global_results:
        print(f"  L{r['layer']}: global_overlap={r['global_variance_weighted_overlap']:.1%}")

    # Save results
    output = {
        "per_layer": per_layer,
        "global_overlap": global_results,
        "n_traits_total": len(all_vectors),
    }
    # Clean non-serializable fields
    for r in output["per_layer"]:
        r.pop("trait_names", None)

    results_path = COVERAGE_DIR / "coverage_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Plot
    plot_results(per_layer, global_results, COVERAGE_DIR / "coverage_plot.png")


if __name__ == "__main__":
    main()
