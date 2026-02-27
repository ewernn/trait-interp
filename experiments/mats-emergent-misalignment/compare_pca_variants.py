"""Compare three PCA variants on the Assistant Axis.

Tests whether PC1 survives without the somewhat/fully confound (Eric Werner critique).

Variant (a) Paper-matching: PCA on all vectors (both fully + somewhat, duplicates per role)
Variant (b) Clean PCA: PCA on fully-only vectors (no somewhat/fully confound)
Variant (c) Current: PCA on deduplicated vectors (one per role, priority: fully > somewhat)

Also computes two axis definitions:
- Current axis: mean(default) - mean(best-per-role) [116 fully + 104 somewhat]
- Paper axis: mean(default) - mean(fully-only) [116 vectors]

Input: analysis/assistant_axis/vectors/ (316 .pt files)
Output: analysis/assistant_axis/validation/pca_variants.json

Usage:
    python experiments/mats-emergent-misalignment/compare_pca_variants.py
    python experiments/mats-emergent-misalignment/compare_pca_variants.py --layer 32
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

EXPERIMENT_DIR = Path(__file__).parent
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis" / "assistant_axis"
VECTORS_DIR = ANALYSIS_DIR / "vectors"
VALIDATION_DIR = ANALYSIS_DIR / "validation"


def load_all_vectors(layer: int):
    """Load all role vectors from .pt files, preserving category metadata.
    Returns (vectors_by_category, default_vec) where vectors_by_category is a list
    of (role_name, category, vector) tuples."""
    vectors = []
    default_vec = None

    for pt_file in sorted(VECTORS_DIR.glob("*.pt")):
        data = torch.load(pt_file, weights_only=True)
        role = data["role"]
        category = data.get("category", "all")
        vec = data["vector"][layer].float()

        if role == "default":
            default_vec = vec
        else:
            vectors.append((role, category, vec))

    return vectors, default_vec


def build_vector_sets(all_vectors):
    """Build the three vector sets for PCA comparison.
    Returns dict with 'paper_all', 'fully_only', 'deduplicated' keys."""
    # (a) Paper-matching: ALL vectors (both fully + somewhat)
    paper_all = [(role, cat, vec) for role, cat, vec in all_vectors]

    # (b) Fully-only
    fully_only = [(role, cat, vec) for role, cat, vec in all_vectors if cat == "fully"]

    # (c) Deduplicated: one per role, priority fully > somewhat > all
    PRIORITY = {"fully": 0, "somewhat": 1, "all": 2}
    best_per_role = {}
    for role, cat, vec in all_vectors:
        pri = PRIORITY.get(cat, 3)
        if role not in best_per_role or pri < best_per_role[role][0]:
            best_per_role[role] = (pri, role, cat, vec)

    deduplicated = [(role, cat, vec) for _, role, cat, vec in best_per_role.values()]

    return {
        "paper_all": paper_all,
        "fully_only": fully_only,
        "deduplicated": deduplicated,
    }


def build_axes(all_vectors, default_vec):
    """Build two axis definitions: current (mixed) and paper (fully-only).
    Returns dict with axis vectors and metadata."""
    PRIORITY = {"fully": 0, "somewhat": 1, "all": 2}

    # Current axis: best-per-role (matches extract_assistant_axis.py run_axis())
    best_per_role = {}
    for role, cat, vec in all_vectors:
        pri = PRIORITY.get(cat, 3)
        if role not in best_per_role or pri < best_per_role[role][0]:
            best_per_role[role] = (pri, cat, vec)

    current_vecs = [vec for _, _, vec in best_per_role.values()]
    current_cats = {}
    for _, cat, _ in best_per_role.values():
        current_cats[cat] = current_cats.get(cat, 0) + 1
    current_mean = torch.stack(current_vecs).mean(0)
    current_axis = default_vec - current_mean
    current_axis_normed = current_axis / current_axis.norm()

    # Paper axis: fully-only
    fully_vecs = [vec for _, cat, vec in all_vectors if cat == "fully"]
    fully_mean = torch.stack(fully_vecs).mean(0)
    paper_axis = default_vec - fully_mean
    paper_axis_normed = paper_axis / paper_axis.norm()

    cos_between = torch.nn.functional.cosine_similarity(
        current_axis_normed.unsqueeze(0), paper_axis_normed.unsqueeze(0)
    ).item()

    return {
        "current": {
            "axis_normed": current_axis_normed,
            "n_vectors": len(current_vecs),
            "category_breakdown": current_cats,
        },
        "paper": {
            "axis_normed": paper_axis_normed,
            "n_vectors": len(fully_vecs),
            "category_breakdown": {"fully": len(fully_vecs)},
        },
        "cos_between_axes": cos_between,
    }


def run_pca(vectors_list, axis_variants, label):
    """Run PCA on a vector set. Returns stats dict."""
    roles = [r for r, _, _ in vectors_list]
    cats = [c for _, c, _ in vectors_list]
    X = torch.stack([v for _, _, v in vectors_list]).numpy()
    n = len(X)

    # Count categories
    cat_counts = {}
    for c in cats:
        cat_counts[c] = cat_counts.get(c, 0) + 1

    # Mean-center
    X_centered = X - X.mean(axis=0)

    # PCA
    n_components = min(20, n)
    pca = PCA(n_components=n_components)
    pca.fit(X_centered)

    pc1 = torch.tensor(pca.components_[0], dtype=torch.float32)
    pc1_normed = pc1 / pc1.norm()

    # Project onto PC1
    projections = X_centered @ pca.components_[0]
    sorted_idx = np.argsort(projections)

    # Cosine with each axis variant
    cos_with_axes = {}
    for axis_name, axis_info in axis_variants.items():
        cos = abs((pc1_normed @ axis_info["axis_normed"]).item())
        cos_with_axes[axis_name] = cos

    # PCs for 70% variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_70 = int(np.searchsorted(cumsum, 0.70) + 1)

    # Between-category analysis (only meaningful if multiple categories)
    between_var_pct = None
    cohens_d = None
    if len(cat_counts) > 1 and "fully" in cat_counts and "somewhat" in cat_counts:
        fully_projs = [projections[i] for i in range(n) if cats[i] == "fully"]
        somewhat_projs = [projections[i] for i in range(n) if cats[i] == "somewhat"]
        if fully_projs and somewhat_projs:
            fully_mean = np.mean(fully_projs)
            somewhat_mean = np.mean(somewhat_projs)
            pooled_std = np.sqrt(
                (np.var(fully_projs) * len(fully_projs) + np.var(somewhat_projs) * len(somewhat_projs))
                / (len(fully_projs) + len(somewhat_projs))
            )
            if pooled_std > 0:
                cohens_d = (fully_mean - somewhat_mean) / pooled_std
            # Between-category variance as % of total PC1 variance
            grand_mean = np.mean(projections)
            between = (len(fully_projs) * (fully_mean - grand_mean)**2 +
                       len(somewhat_projs) * (somewhat_mean - grand_mean)**2)
            total = np.var(projections) * n
            between_var_pct = between / total if total > 0 else 0

    # Top/bottom roles on PC1
    top_roles = [(roles[i], cats[i], float(projections[i])) for i in sorted_idx[-5:][::-1]]
    bottom_roles = [(roles[i], cats[i], float(projections[i])) for i in sorted_idx[:5]]

    return {
        "label": label,
        "n_vectors": n,
        "category_breakdown": cat_counts,
        "pc1_variance_pct": float(pca.explained_variance_ratio_[0] * 100),
        "top5_variance_pct": [float(v * 100) for v in pca.explained_variance_ratio_[:5]],
        "cumulative_5pc_pct": float(cumsum[4] * 100) if len(cumsum) >= 5 else None,
        "n_pcs_for_70pct": n_70,
        "cos_with_axes": cos_with_axes,
        "between_category_var_pct": float(between_var_pct * 100) if between_var_pct is not None else None,
        "cohens_d_fully_vs_somewhat": float(cohens_d) if cohens_d is not None else None,
        "top_roles_pc1": top_roles,
        "bottom_roles_pc1": bottom_roles,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare PCA variants on Assistant Axis")
    parser.add_argument("--layer", type=int, default=24, help="Analysis layer (default: 24)")
    parser.add_argument("--multi-layer", action="store_true",
                        help="Run across multiple layers")
    args = parser.parse_args()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    def analyze_layer(layer):
        """Run full comparison at a single layer."""
        all_vectors, default_vec = load_all_vectors(layer)

        n_fully = sum(1 for _, c, _ in all_vectors if c == "fully")
        n_somewhat = sum(1 for _, c, _ in all_vectors if c == "somewhat")
        print(f"\n  Loaded {len(all_vectors)} vectors ({n_fully} fully, {n_somewhat} somewhat)")

        # Build vector sets and axes
        vector_sets = build_vector_sets(all_vectors)
        axes = build_axes(all_vectors, default_vec)

        print(f"  Current axis: {axes['current']['n_vectors']} vectors "
              f"({axes['current']['category_breakdown']})")
        print(f"  Paper axis: {axes['paper']['n_vectors']} vectors (fully-only)")
        print(f"  cos(current_axis, paper_axis): {axes['cos_between_axes']:.4f}")

        # Run three PCA variants
        axis_variants = {k: v for k, v in axes.items() if k != "cos_between_axes"}
        results = {}

        for key, label in [
            ("paper_all", "(a) Paper-matching: all vectors"),
            ("fully_only", "(b) Clean PCA: fully-only"),
            ("deduplicated", "(c) Current: deduplicated"),
        ]:
            print(f"\n  --- {label} ---")
            r = run_pca(vector_sets[key], axis_variants, label)
            results[key] = r
            print(f"  n={r['n_vectors']} ({r['category_breakdown']})")
            print(f"  PC1: {r['pc1_variance_pct']:.1f}% variance")
            for axis_name, cos in r["cos_with_axes"].items():
                print(f"  |cos(PC1, {axis_name}_axis)|: {cos:.3f}")
            if r["between_category_var_pct"] is not None:
                print(f"  Between-category variance on PC1: {r['between_category_var_pct']:.1f}%")
                print(f"  Cohen's d (fully vs somewhat on PC1): {r['cohens_d_fully_vs_somewhat']:+.2f}")

        return results, axes

    # Single layer analysis
    print(f"{'='*70}")
    print(f"PCA VARIANT COMPARISON — Layer {args.layer}")
    print(f"{'='*70}")

    results, axes = analyze_layer(args.layer)

    # Summary comparison table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Variant':40s} {'n':>5}  {'PC1%':>6}  {'cos(curr)':>10}  {'cos(paper)':>10}  {'btwn%':>6}")
    print(f"  {'-'*40} {'-'*5}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*6}")
    for key in ["paper_all", "fully_only", "deduplicated"]:
        r = results[key]
        btwn = f"{r['between_category_var_pct']:.1f}" if r["between_category_var_pct"] is not None else "n/a"
        print(f"  {r['label']:40s} {r['n_vectors']:>5}  {r['pc1_variance_pct']:>5.1f}%  "
              f"{r['cos_with_axes']['current']:>10.3f}  {r['cos_with_axes']['paper']:>10.3f}  "
              f"{btwn:>6}")

    print(f"\n  cos(current_axis, paper_axis): {axes['cos_between_axes']:.4f}")

    # Key question: does PC1 survive in fully-only?
    fully_pca = results["fully_only"]
    paper_pca = results["paper_all"]
    print(f"\n  KEY RESULT: Clean PCA (fully-only) PC1 explains {fully_pca['pc1_variance_pct']:.1f}% variance")
    print(f"  vs paper-matching PC1: {paper_pca['pc1_variance_pct']:.1f}%")
    cos_threshold = 0.71
    for axis_name in ["current", "paper"]:
        cos = fully_pca["cos_with_axes"][axis_name]
        status = "PASSES" if cos > cos_threshold else "FAILS"
        print(f"  |cos(clean_PC1, {axis_name}_axis)| = {cos:.3f} — {status} >0.71 threshold")

    # Role endpoints comparison
    print(f"\n  Top roles on PC1 (most assistant-like):")
    for key in ["paper_all", "fully_only", "deduplicated"]:
        r = results[key]
        top3 = [f"{role}_{cat}" for role, cat, _ in r["top_roles_pc1"][:3]]
        print(f"    {key:15s}: {', '.join(top3)}")

    print(f"\n  Bottom roles on PC1 (least assistant-like):")
    for key in ["paper_all", "fully_only", "deduplicated"]:
        r = results[key]
        bot3 = [f"{role}_{cat}" for role, cat, _ in r["bottom_roles_pc1"][:3]]
        print(f"    {key:15s}: {', '.join(bot3)}")

    # Multi-layer analysis
    if args.multi_layer:
        print(f"\n{'='*70}")
        print("MULTI-LAYER COMPARISON")
        print(f"{'='*70}")

        layers = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
        multi_results = {}

        print(f"\n  {'Layer':>5}  {'Paper PC1%':>10}  {'Clean PC1%':>10}  {'Dedup PC1%':>10}  "
              f"{'cos(clean,curr)':>16}  {'cos(clean,paper)':>17}")
        print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*16}  {'-'*17}")

        for layer in layers:
            lr, la = analyze_layer(layer)
            multi_results[layer] = {"results": lr, "axes": la}

            paper_pc1 = lr["paper_all"]["pc1_variance_pct"]
            clean_pc1 = lr["fully_only"]["pc1_variance_pct"]
            dedup_pc1 = lr["deduplicated"]["pc1_variance_pct"]
            cos_curr = lr["fully_only"]["cos_with_axes"]["current"]
            cos_paper = lr["fully_only"]["cos_with_axes"]["paper"]

            print(f"  L{layer:>3}  {paper_pc1:>9.1f}%  {clean_pc1:>9.1f}%  {dedup_pc1:>9.1f}%  "
                  f"{cos_curr:>16.3f}  {cos_paper:>17.3f}")

    # Save results
    output = {
        "layer": args.layer,
        "axes": {
            "cos_between_axes": axes["cos_between_axes"],
            "current": {
                "n_vectors": axes["current"]["n_vectors"],
                "category_breakdown": axes["current"]["category_breakdown"],
            },
            "paper": {
                "n_vectors": axes["paper"]["n_vectors"],
                "category_breakdown": axes["paper"]["category_breakdown"],
            },
        },
        "pca_variants": {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in results.items()
        },
    }

    output_file = VALIDATION_DIR / "pca_variants.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
